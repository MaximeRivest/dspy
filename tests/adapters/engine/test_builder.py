"""Builder unit tests: the plan records exactly what legacy preprocessing
decided. End-to-end byte-parity is adjudicated by the golden corpus; these
tests pin the RECORDS (transforms, tools, metadata deltas) and the legacy
contract surfaces."""

from typing import Any

import pytest
from golden.harness import Recorder, StubLM

import dspy
from dspy.adapters._engine.builder import assert_unrendered, build_plan
from dspy.adapters._engine.ir import AdapterPlan
from dspy.adapters._engine.transforms import HideOutputField
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.core.types import LMTextPart


class AgentStep(dspy.Signature):
    """Decide the next step."""

    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    next_thought: str = dspy.OutputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()


class ThoughtfulQA(dspy.Signature):
    """Think then answer."""

    question: str = dspy.InputField()
    reasoning: dspy.Reasoning = dspy.OutputField()
    answer: str = dspy.OutputField()


def _search(query: str) -> str:
    """Search."""
    return query


def _agent_inputs():
    return {"question": "Q?", "tools": [dspy.Tool(_search)]}


def _lm(**caps):
    return StubLM(Recorder(), **caps)


def test_native_fc_records_tools_and_hides_both_fields():
    adapter = ChatAdapter(use_native_function_calling=True)
    lm_kwargs: dict[str, Any] = {}
    built = build_plan(adapter, _lm(supports_function_calling=True), lm_kwargs, AgentStep, _agent_inputs())

    assert lm_kwargs["tools"][0]["function"]["name"] == "_search"
    assert built.plan.tools == lm_kwargs["tools"]
    hides = {(type(t).__name__, t.name, t.reason) for t in built.plan.field_transforms}
    assert hides == {
        ("HideInputField", "tools", "native_function_calling"),
        ("HideOutputField", "tool_calls", "native_function_calling"),
    }
    assert built.plan.find_field("input", "tools").hidden
    assert built.plan.find_field("output", "tool_calls").hidden
    # Legacy contract: the render signature lacks both fields.
    assert "tools" not in built.render_signature.input_fields
    assert "tool_calls" not in built.render_signature.output_fields


def test_fc_off_strips_provider_keys_in_place():
    adapter = ChatAdapter()
    lm_kwargs = {"tools": [{"type": "function"}], "tool_choice": "auto", "parallel_tool_calls": True, "n": 2}
    built = build_plan(adapter, _lm(), lm_kwargs, AgentStep, _agent_inputs())
    assert lm_kwargs == {"n": 2}
    assert built.plan.tools == []
    assert built.render_signature is AgentStep


def test_parallel_tool_calls_only_set_when_configured_and_absent():
    lm = _lm(supports_function_calling=True)
    lm_kwargs: dict[str, Any] = {}
    build_plan(
        ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True),
        lm,
        lm_kwargs,
        AgentStep,
        _agent_inputs(),
    )
    assert lm_kwargs["parallel_tool_calls"] is True

    presupplied = {"parallel_tool_calls": False}
    build_plan(
        ChatAdapter(use_native_function_calling=True, parallel_tool_calls=True),
        lm,
        presupplied,
        AgentStep,
        _agent_inputs(),
    )
    assert presupplied["parallel_tool_calls"] is False


def test_toolcalls_output_without_tools_input_raises_same_message():
    class Broken(dspy.Signature):
        question: str = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    with pytest.raises(ValueError, match="did not provide any tools as the input"):
        build_plan(
            ChatAdapter(use_native_function_calling=True),
            _lm(supports_function_calling=True),
            {},
            Broken,
            {"question": "Q?"},
        )


def test_native_reasoning_recorded_as_hidden_field_and_kwargs_delta():
    adapter = ChatAdapter()
    lm_kwargs: dict[str, Any] = {}
    built = build_plan(adapter, _lm(supports_reasoning=True), lm_kwargs, ThoughtfulQA, {"question": "Q?"})

    assert lm_kwargs.get("reasoning_effort") == "low"
    assert "reasoning" not in built.render_signature.output_fields
    reasoning_field = built.plan.find_field("output", "reasoning")
    assert reasoning_field.hidden
    hide = next(t for t in built.plan.field_transforms if isinstance(t, HideOutputField))
    assert hide.reason == "native:Reasoning"
    assert built.plan.metadata["native_feature_kwargs"]["Reasoning"] == {"reasoning_effort": "low"}


def test_generic_alias_annotations_never_reach_issubclass():
    class OpenEnded(dspy.Signature):
        text: str = dspy.InputField()
        info: dict[str, Any] = dspy.OutputField()

    built = build_plan(ChatAdapter(), _lm(), {}, OpenEnded, {"text": "t"})
    assert built.render_signature is OpenEnded


def test_plans_are_rebuilt_per_attempt_never_shared():
    adapter = ChatAdapter()

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    first = build_plan(adapter, _lm(), {}, QA, {"question": "Q?"})
    second = build_plan(adapter, _lm(), {}, QA, {"question": "Q?"})
    assert first.plan is not second.plan
    first.plan.warnings.append("local mutation")
    assert second.plan.warnings == []


def test_postprocess_plan_path_equals_legacy_path():
    adapter = ChatAdapter(use_native_function_calling=True)
    lm = _lm(supports_function_calling=True)
    lm_kwargs: dict[str, Any] = {}
    built = build_plan(adapter, lm, lm_kwargs, AgentStep, _agent_inputs())
    output = {
        "text": "",
        "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "_search", "arguments": '{"query": "x"}'}}
        ],
    }
    legacy = adapter._call_postprocess(built.render_signature, AgentStep, [output], lm, {})
    planned = adapter._call_postprocess(built.render_signature, AgentStep, [output], lm, {}, plan=built.plan)
    assert legacy == planned


def test_render_request_tripwire_rejects_unrendered_content():
    plan = AdapterPlan()
    assert_unrendered(plan)  # empty: fine
    plan.user_parts.append(LMTextPart(text="content nobody will render"))
    with pytest.raises(AssertionError, match="renderer is not wired"):
        assert_unrendered(plan)
