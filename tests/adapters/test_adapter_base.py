import dspy
from dspy.core.types import LMConfig, LMToolChoice
from tests.adapters.conftest import format_messages_and_lm_kwargs


def test_adapter_does_not_require_native_tool_call_format_kwarg_when_unused():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class OldStyleFormatAdapter(dspy.Adapter):
        def format(self, signature, demos, inputs):
            return [{"role": "user", "content": inputs["question"]}]

        def parse(self, signature, completion):
            return {"answer": completion}

    messages, lm_kwargs = format_messages_and_lm_kwargs(
        OldStyleFormatAdapter(),
        QA,
        [],
        {"question": "Q"},
    )

    assert messages == [{"role": "user", "content": "Q"}]
    assert lm_kwargs == {}


def test_call_context_does_not_carry_mutable_extra_state():
    context = dspy.Adapter(allow_parallel_tool_calls=False).build_call_context(
        dspy.utils.DummyLM([{}]),
        {"temperature": 0.2},
    )

    assert not hasattr(context, "extra")
    assert context.allow_parallel_tool_calls is False
    assert context.lm_kwargs == {"temperature": 0.2}
    assert context.lm_default_kwargs["temperature"] == 0.0


def test_adapter_merges_lm_config_patches_without_losing_nested_fields():
    adapter = dspy.Adapter()
    left = LMConfig.from_kwargs(reasoning={"effort": "low"}, cache=True, custom="left")
    right = LMConfig.from_kwargs(reasoning={"max_tokens": 64}, rollout_id="rollout", other="right")

    merged = adapter._merge_lm_config(left, right)

    assert merged.reasoning.effort == "low"
    assert merged.reasoning.max_tokens == 64
    assert merged.cache.enabled is True
    assert merged.cache.rollout_id == "rollout"
    assert merged.extensions == {"custom": "left", "other": "right"}


def test_adapter_lm_config_patch_preserves_explicit_tool_choice_mode():
    adapter = dspy.Adapter()
    merged_kwargs = adapter._merge_config_kwargs(
        {"tool_choice": "none"},
        LMConfig(tool_choice=LMToolChoice(parallel=False)),
    )

    assert merged_kwargs["tool_choice"]["mode"] == "none"
    assert merged_kwargs["tool_choice"]["parallel"] is False
