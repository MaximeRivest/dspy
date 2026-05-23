from dataclasses import dataclass

import pytest

import dspy
from dspy.adapters._planning import (
    _InputFieldStrategyContext,
    _NativeImageStrategy,
    _plan_adapter_call,
    _Strategy,
    _StrategyContext,
    _StrategyPatch,
    _StrategyTrace,
    _TextReasoningStrategy,
    _TypeStrategy,
    _UserPartSegment,
)
from dspy.core.types import LMConfig, LMRequestPatch, LMTextPart


class ReasoningLM(dspy.utils.DummyLM):
    @property
    def supports_reasoning(self):
        return True


@dataclass(frozen=True)
class _CaptionImageAugment(_TypeStrategy[dspy.Image]):
    marker_type: type[dspy.Image] = dspy.Image

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        return _StrategyPatch(
            user_part_segments=(
                _UserPartSegment(ctx.field_name, [LMTextPart(text="caption: a test image")]),
            ),
            trace=(_StrategyTrace(ctx.field_name, type(self).__name__, "augment", "caption context"),),
        )


@dataclass(frozen=True)
class _ReplaceImageWithText(_TypeStrategy[dspy.Image]):
    marker_type: type[dspy.Image] = dspy.Image
    label: str = "replacement"

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        return _StrategyPatch(
            delete_input_fields=(ctx.field_name,),
            user_part_segments=(_UserPartSegment(ctx.field_name, [LMTextPart(text=self.label)]),),
            trace=(_StrategyTrace(ctx.field_name, type(self).__name__, "replace", self.label),),
        )


@dataclass(frozen=True)
class _StopStrategy(_Strategy):
    stop: str

    def plan_signature(self, ctx: _StrategyContext) -> _StrategyPatch:
        return _StrategyPatch(
            request=LMRequestPatch(config=LMConfig(stop=[self.stop])),
            trace=(_StrategyTrace(None, type(self).__name__, "configure", f"stop={self.stop}"),),
        )


def test_input_augment_strategy_composes_with_one_replacement_owner():
    class Inspect(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter(_type_strategies=[_CaptionImageAugment(), _NativeImageStrategy()])
    lm = dspy.utils.DummyLM([{}])

    plan = _plan_adapter_call(
        adapter,
        lm,
        {},
        Inspect,
        {"image": dspy.Image("https://example.com/a.png"), "question": "what?"},
    )

    assert "image" not in plan.render_signature.input_fields
    assert "question" in plan.render_signature.input_fields
    assert len(plan.user_part_segments) == 2
    assert isinstance(plan.user_part_segments[0].parts[0], LMTextPart)
    assert plan.user_part_segments[0].parts[0].text == "caption: a test image"
    assert plan.input_field_owners["image"] == "_NativeImageStrategy"


def test_two_replacement_strategies_for_same_input_field_conflict():
    class Inspect(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter(_type_strategies=[_ReplaceImageWithText(label="first"), _ReplaceImageWithText(label="second")])
    lm = dspy.utils.DummyLM([{}])

    with pytest.raises(ValueError, match="already owned"):
        _plan_adapter_call(adapter, lm, {}, Inspect, {"image": dspy.Image("https://example.com/a.png")})


def test_text_reasoning_strategy_blocks_native_reasoning_fallback():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter(_type_strategies=[_TextReasoningStrategy()])
    lm = ReasoningLM([{}])

    plan = _plan_adapter_call(adapter, lm, {}, QA, {"question": "why?"})

    assert "reasoning" in plan.render_signature.output_fields
    assert "reasoning_effort" not in plan.lm_kwargs
    assert plan.output_field_owners["reasoning"] == "text"
    assert any(trace.action == "skip" and trace.strategy == "_NativeReasoningStrategy" for trace in plan.strategy_trace)


def test_conflicting_strategy_lm_config_is_reported():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter(_type_strategies=[_StopStrategy("END"), _StopStrategy("\n```")])
    lm = dspy.utils.DummyLM([{}])

    with pytest.raises(ValueError, match="LM config field 'stop'"):
        _plan_adapter_call(adapter, lm, {}, QA, {"question": "q"})
