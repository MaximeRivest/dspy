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


# ---------------------------------------------------------------------------
# The bake-time triple check (ADP-006/ADP-007)
# ---------------------------------------------------------------------------


def _with_test_preset(name, messages, parser="chat"):
    """Temporarily register a preset so a format class can point at it."""
    from contextlib import contextmanager

    from dspy.adapters._engine.presets import PRESETS, _make_preset

    @contextmanager
    def scope():
        PRESETS[name] = _make_preset(
            name=name,
            template_messages=messages,
            parser=parser,
            codecs={"input": "text_pythonish", "output": "text_pythonish"},
            strategies={},
        )
        try:
            yield PRESETS[name]
        finally:
            del PRESETS[name]

    return scope()


def test_builtin_presets_never_trip_the_bake_checks():
    """Every builtin preset hosts every role and carries no explicit field
    slots — the triple check is invisible to existing programs."""
    from dspy.adapters._engine.builder import _check_template_capacity
    from dspy.adapters._engine.formats.baml import BAMLFormat
    from dspy.adapters._engine.formats.chat import ChatFormat
    from dspy.adapters._engine.formats.json import JSONFormat
    from dspy.adapters._engine.formats.xml import XMLFormat

    built = build_plan(ChatAdapter(), _lm(supports_reasoning=True), {}, ThoughtfulQA, {"question": "Q?"})
    for fmt in (ChatFormat(), JSONFormat(), XMLFormat(), BAMLFormat()):
        _check_template_capacity(fmt, _lm(), built.plan)


def test_bake_refuses_a_textual_role_with_no_lane():
    """ADP-006: a template that never iterates outputs cannot host a
    textually-served reasoning field; the refusal names field+role, the LM,
    and the template."""
    from dspy.adapters._engine.builder import _check_template_capacity
    from dspy.adapters._engine.formats.chat import ChatFormat

    messages = [
        {"role": "system", "content": "{instruction(style='raw')}"},
        {"role": "user", "content": "{% for f in inputs %}{f.value}{% endfor %}"},
    ]
    with _with_test_preset("_test_no_output_lane", messages):

        class _NoOutputLane(ChatFormat):
            preset_name = "_test_no_output_lane"
            system_template_message = None

        built = build_plan(ChatAdapter(), _lm(supports_reasoning=False), {}, ThoughtfulQA, {"question": "Q?"})
        with pytest.raises(ValueError, match="ADP-006") as excinfo:
            _check_template_capacity(_NoOutputLane(), _lm(), built.plan)
        message = str(excinfo.value)
        assert "'reasoning'" in message and "_test_no_output_lane" in message and "stub/golden-model" in message


def test_bake_refuses_an_explicit_slot_on_a_natively_hidden_field():
    """ADP-007: {reasoning} in the live lane, with reasoning served
    natively, refuses at bake — never an empty render."""
    from dspy.adapters._engine.builder import _check_template_capacity
    from dspy.adapters._engine.formats.chat import ChatFormat

    messages = [
        {"role": "system", "content": "{instruction(style='raw')}\n{field('reasoning')}"},
        {"role": "user", "content": "{% for f in inputs %}{f.value}{% endfor %}"},
    ]
    with _with_test_preset("_test_slotted", messages):

        class _Slotted(ChatFormat):
            preset_name = "_test_slotted"
            system_template_message = None

        built = build_plan(ChatAdapter(), _lm(supports_reasoning=True), {}, ThoughtfulQA, {"question": "Q?"})
        assert built.plan.find_field("output", "reasoning").hidden
        with pytest.raises(ValueError, match="ADP-007"):
            _check_template_capacity(_Slotted(), _lm(), built.plan)


def test_bake_refuses_an_example_lane_slot_on_a_natively_hidden_field():
    """ADP-007, example lane: a demos pattern spelling out a hidden field
    refuses at bake with the demo-specific message."""
    from dspy.adapters._engine.builder import _check_template_capacity
    from dspy.adapters._engine.formats.chat import ChatFormat

    messages = [
        {"role": "system", "content": "{instruction(style='raw')}"},
        {
            "role": "demos",
            "user": "{% for f in inputs %}{f.value}{% endfor %}",
            "assistant": "{field('reasoning')}",
        },
        {"role": "user", "content": "{% for f in inputs %}{f.value}{% endfor %}{% for f in outputs %}{f.marker}{% endfor %}"},
    ]
    with _with_test_preset("_test_demo_slotted", messages):

        class _DemoSlotted(ChatFormat):
            preset_name = "_test_demo_slotted"
            system_template_message = None

        built = build_plan(ChatAdapter(), _lm(supports_reasoning=True), {}, ThoughtfulQA, {"question": "Q?"})
        with pytest.raises(ValueError, match="example turns cannot show"):
            _check_template_capacity(_DemoSlotted(), _lm(), built.plan)


def test_preset_capacity_is_eager_data():
    from dspy.adapters._engine.formats.baml import BAMLFormat
    from dspy.adapters._engine.presets import effective_capacity, get_preset

    for name in ("chat", "json", "xml"):
        capacity = get_preset(name).capacity
        assert capacity.iterates_inputs and capacity.iterates_outputs
        assert capacity.fragment_targets == {"system", "user"}
    baml = effective_capacity(BAMLFormat())
    assert baml.iterates_inputs and baml.iterates_outputs
    assert baml.fragment_targets == {"system", "user"}
    assert effective_capacity(BAMLFormat()) is baml  # cached per pairing
