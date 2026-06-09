"""IR shape tests, including plan sketches for the flows that historically
overfit single-adapter designs (JSONAdapter structured output, TwoStep
dual-LM extraction)."""

import dspy
from dspy.adapters._engine.ir import AdapterPlan, RenderField
from dspy.adapters._engine.parser_hook import ParseContext, ParserHook, ResponseView
from dspy.core.types import LMConfig


class GoldenQA(dspy.Signature):
    """Answer briefly."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


def test_from_signature_initializes_render_fields_one_to_one():
    plan = AdapterPlan.from_signature(GoldenQA, {"question": "Q?"})
    assert [(f.name, f.original_name, f.role) for f in plan.input_fields] == [("question", "question", "input")]
    assert plan.input_fields[0].value == "Q?"
    assert [(f.name, f.role, f.hidden) for f in plan.output_fields] == [("answer", "output", False)]


def test_destination_name_prefers_original_name():
    aux = RenderField(name="final", original_name="answer", role="output", annotation=str)
    assert aux.destination_name == "answer"
    synthetic = RenderField(name="draft", original_name=None, role="output", annotation=str)
    assert synthetic.destination_name == "draft"


def test_visible_fields_skip_hidden():
    plan = AdapterPlan.from_signature(GoldenQA)
    plan.output_fields[0].hidden = True
    assert plan.visible_output_fields() == []
    assert plan.find_field("output", "answer") is plan.output_fields[0]
    assert plan.find_field("output", "missing") is None


def test_plan_sketch_json_structured_output():
    """The IR must express JSONAdapter's call-level structured-output
    decision without per-field hacks: response_format lives in LMConfig."""
    plan = AdapterPlan.from_signature(GoldenQA, {"question": "Q?"})
    plan.config = LMConfig(response_format={"type": "json_object"})
    assert plan.config.response_format == {"type": "json_object"}
    # Visible fields drive the rendered JSON schema; nothing is hidden here.
    assert [f.name for f in plan.visible_output_fields()] == ["answer"]


class _ExtractionParser:
    """Sketch of TwoStep's extraction stage as a plan-carried parser: it is a
    ParserHook that uses ctx.lm (the extraction model)."""

    name = "two_step.extraction"

    def parse(self, response_view, ctx):
        assert ctx.lm is not None, "extraction parser requires ctx.lm"
        return {"answer": f"extracted from: {response_view.text}"}


def test_plan_sketch_two_step_extraction_parser():
    plan = AdapterPlan.from_signature(GoldenQA, {"question": "Q?"})
    parser = _ExtractionParser()
    plan.parsers.append(parser)
    assert isinstance(parser, ParserHook)

    ctx = ParseContext(plan=plan, signature=GoldenQA, lm=object())
    values = plan.parsers[0].parse(ResponseView("Berlin, obviously."), ctx)
    assert values == {"answer": "extracted from: Berlin, obviously."}


def test_response_view_facade_over_legacy_shapes():
    text_view = ResponseView("plain text")
    assert text_view.text == "plain text"
    assert text_view.tool_calls is None
    assert text_view.channel("reasoning_content") is None

    dict_view = ResponseView(
        {
            "text": "body",
            "tool_calls": [{"id": "call_1"}],
            "logprobs": {"content": []},
            "reasoning_content": "thinking",
        }
    )
    assert dict_view.text == "body"
    assert dict_view.tool_calls == [{"id": "call_1"}]
    assert dict_view.logprobs == {"content": []}
    assert dict_view.channel("reasoning_content") == "thinking"
    assert dict_view.raw["text"] == "body"
