import dspy
from dspy.clients.language_models import LMOutput, LMTextPart, LMThinkingPart


def test_native_reasoning_renders_lm_reasoning_config_and_deletes_output_field():
    strategy = dspy.types.NativeReasoning(reasoning_effort="high", max_tokens=512, summary="auto")

    patch = strategy.render_output(field_name="reasoning", field=None, adapter=None)

    assert patch.delete_output_fields == ("reasoning",)
    assert patch.config.reasoning.effort == "high"
    assert patch.config.reasoning.max_tokens == 512
    assert patch.config.reasoning.summary == "auto"


def test_native_reasoning_accepts_effort_alias():
    strategy = dspy.types.NativeReasoning(effort="medium")

    patch = strategy.render_output(field_name="reasoning", field=None, adapter=None)

    assert patch.config.reasoning.effort == "medium"


def test_native_reasoning_parses_lm_thinking_part():
    strategy = dspy.types.NativeReasoning()
    output = LMOutput(parts=[LMThinkingPart(text="native reasoning"), LMTextPart(text="answer")])

    value = strategy.parse_output(field_name="reasoning", output=output)

    assert isinstance(value, dspy.Reasoning)
    assert value == "native reasoning"


def test_text_reasoning_keeps_field_in_adapter_visible_signature():
    strategy = dspy.types.TextReasoning()

    patch = strategy.render_output(field_name="reasoning", field=None, adapter=None)

    assert patch.delete_output_fields == ()
    assert patch.config is None
