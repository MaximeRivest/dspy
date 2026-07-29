import asyncio
import logging

import pytest

import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import Hint, LMCritic, OfferHints, Refine, RefinementReport
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


class HintedQA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    hint: str = dspy.InputField(default="", desc="Optional advice from an earlier attempt; may be ignored.")
    answer: str = dspy.OutputField()


def make_tracking_adapter(calls):
    class TrackingChatAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            calls.append(
                {
                    "inputs": dict(inputs),
                    "input_fields": list(signature.input_fields),
                    "adapter_type": type(self),
                }
            )
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    return TrackingChatAdapter


def test_refine_forward_success_first_attempt():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    reward_call_count = [0]

    def reward_fn(kwargs, pred: Prediction) -> float:
        reward_call_count[0] += 1
        # The answer should always be one word.
        return 1.0 if len(pred.answer) == 1 else 0.0

    predict = DummyModule("question -> answer", count_calls)

    refine = Refine(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = refine(question="What is the capital of Belgium?")

    assert result.answer == "Brussels", "Result should be `Brussels`"
    assert reward_call_count[0] > 0, "Reward function should have been called"
    assert module_call_count[0] == 3, (
        "Module should have been called exactly 3 times, but was called %d times" % module_call_count[0]
    )


def test_refine_module_default_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")


def test_refine_module_custom_fail_count():
    lm = DummyLM([{"answer": "Brussels"}, {"answer": "City of Brussels"}, {"answer": "Brussels"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def raise_on_second_call(self, **kwargs):
        if module_call_count[0] < 2:
            module_call_count[0] += 1
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", raise_on_second_call)

    refine = Refine(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError):
        refine(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )


def test_refine_without_feedback_fn_is_best_of_n(caplog):
    """The honest default: no feedback_fn means resample-and-select, with no hidden feedback channel."""
    lm = DummyLM([{"answer": "one two"}, {"answer": "three"}, {"answer": "four five"}])
    dspy.configure(lm=lm, adapter=None)

    adapter_calls = []
    module = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    module.set_adapter(make_tracking_adapter(adapter_calls)())

    with caplog.at_level(logging.WARNING):
        refine = Refine(
            module=module,
            N=3,
            reward_fn=lambda _, pred: 1.0 if len(pred.answer.split()) == 1 else 0.0,
            threshold=2.0,  # unreachable: force all N attempts
        )
    assert "no longer generates LM feedback" in caplog.text

    result = refine(question="Pick one word.")

    assert result.answer == "three"  # best-of-N selection
    assert len(adapter_calls) == 3  # exactly N module runs, zero critic runs
    for call in adapter_calls:
        # No manufactured hint channel: the declared signature is all the adapter ever sees.
        assert call["input_fields"] == ["question"]
        assert "hint_" not in call["inputs"] and "hint" not in call["inputs"]


def test_feedback_reaches_declared_hint_field_through_configured_adapter():
    """Hints travel through the Signature's declared field; the Adapter is never wrapped or mutated."""
    adapter_calls = []
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule(HintedQA, lambda self, **kwargs: self.predictor(**kwargs))
    tracking_adapter_cls = make_tracking_adapter(adapter_calls)
    tracking_adapter = tracking_adapter_cls()
    module.set_adapter(tracking_adapter)

    feedback_calls = []

    def feedback_fn(inputs, prediction, reward):
        feedback_calls.append((dict(inputs), prediction.answer, reward))
        return "Answer with 'right'."

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=feedback_fn,
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    assert feedback_calls == [({"question": "What is the right answer?"}, "wrong", 0.0)]

    # First attempt sees the field's declared default; the retry sees the hint content.
    assert adapter_calls[0]["inputs"]["hint"] == ""
    assert adapter_calls[1]["inputs"]["hint"] == "Answer with 'right'."

    for call in adapter_calls:
        # Adapter independence: the exact configured adapter class runs every attempt — no dynamic
        # subclass, no appended undeclared field. The signature the adapter sees is the declared one.
        assert call["adapter_type"] is tracking_adapter_cls
        assert call["input_fields"] == ["question", "hint"]
        assert "hint_" not in call["inputs"]

    # The original module is untouched: same adapter object, unmutated signature default.
    assert module.predictor.adapter is tracking_adapter
    assert module.predictor.signature.fields["hint"].default == ""


def test_no_flexible_target_means_pure_resampling(caplog):
    """With no declared hint field, Refine must not manufacture one — feedback is never consulted."""
    lm = DummyLM([{"answer": "wrong"}, {"answer": "wrong again"}, {"answer": "still wrong"}])
    dspy.configure(lm=lm, adapter=None)

    adapter_calls = []
    module = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    module.set_adapter(make_tracking_adapter(adapter_calls)())

    feedback_calls = [0]

    def feedback_fn(inputs, prediction, reward):
        feedback_calls[0] += 1
        return "be right"

    with caplog.at_level(logging.WARNING):
        refine = Refine(
            module=module,
            N=3,
            reward_fn=lambda _, __: 0.0,
            threshold=1.0,
            feedback_fn=feedback_fn,
        )
    assert "no predictor" in caplog.text

    result = refine(question="What is the right answer?")

    assert result is not None
    assert feedback_calls[0] == 0, "feedback_fn must not run when hints have nowhere to go"
    for call in adapter_calls:
        assert call["input_fields"] == ["question"]
        assert "hint" not in call["inputs"] and "hint_" not in call["inputs"]


def test_module_supplied_hint_outranks_search_hint():
    """The code channel is authoritative: a hint value passed by the module's own forward wins."""
    adapter_calls = []
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=None)

    def forward_fn(self, **kwargs):
        return self.predictor(question=kwargs["question"], hint="from code")

    module = DummyModule(HintedQA, forward_fn)
    module.set_adapter(make_tracking_adapter(adapter_calls)())

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=lambda inputs, prediction, reward: "from search",
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    assert adapter_calls[0]["inputs"]["hint"] == "from code"
    assert adapter_calls[1]["inputs"]["hint"] == "from code", "search hints must not override module code"


def test_refinement_report_records_hints_with_provenance():
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule(HintedQA, lambda self, **kwargs: self.predictor(**kwargs))

    def my_feedback(inputs, prediction, reward):
        return "Answer with 'right'."

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=my_feedback,
    )
    result = refine(question="What is the right answer?")

    report = result._refinement
    assert isinstance(report, RefinementReport)
    assert len(report.attempts) == 2
    assert report.attempts[0].reward == 0.0
    assert report.attempts[0].hints_applied == ()
    (hint,) = report.attempts[1].hints_applied
    assert hint.content == "Answer with 'right'."
    assert hint.authority == "non_authoritative"
    assert hint.scope == "next_candidate"
    assert "my_feedback" in hint.provenance
    assert report.best_attempt == 1
    assert report.best_reward == 1.0
    # Metadata stays out of the prediction's fields, so it cannot leak into outputs or state.
    assert "_refinement" not in dict(result)


def test_targeted_hint_with_unknown_target_is_ignored_not_forced():
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule(HintedQA, lambda self, **kwargs: self.predictor(**kwargs))

    def feedback_fn(inputs, prediction, reward):
        return [
            Hint(content="for a predictor that does not exist", target="nope", provenance="test"),
            Hint(content="Answer with 'right'.", target="predictor", provenance="test"),
        ]

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=feedback_fn,
    )
    result = refine(question="What is the right answer?")

    report = result._refinement
    retry = report.attempts[1]
    assert [h.target for h in retry.hints_applied] == ["predictor"]
    assert [h.target for h in retry.hints_ignored] == ["nope"]


def test_trace_contains_winner_only_and_no_feedback_machinery():
    """Feedback LM calls stay out of the caller's trace; the declared hint field stays in it."""
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"advice": "Answer with 'right'."},  # consumed by the feedback predictor
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule(HintedQA, lambda self, **kwargs: self.predictor(**kwargs))

    def feedback_fn(inputs, prediction, reward):
        return dspy.Predict("question -> advice")(question=inputs["question"]).advice

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=feedback_fn,
    )

    with dspy.context(trace=[]):
        result = refine(question="What is the right answer?")
        trace = list(dspy.settings.trace)

    assert result.answer == "right"
    # Only the winning attempt's single predictor step: no losing attempts, no feedback predictor.
    assert len(trace) == 1
    _, step_inputs, step_outputs = trace[0]
    assert step_outputs.answer == "right"
    # The hint is a declared input of the winning call, so it belongs in the trace, undisguised.
    assert step_inputs["hint"] == "Answer with 'right'."


def test_offer_hints_signature_declares_no_source_archaeology():
    """The opt-in critic reasons from declared intent and run evidence, never from source code."""
    assert "objective" in OfferHints.input_fields
    assert "program_code" not in OfferHints.input_fields
    assert "reward_code" not in OfferHints.input_fields


def test_lm_critic_requires_declared_objective():
    with pytest.raises(ValueError):
        LMCritic(objective="")


def test_lm_critic_routes_advice_to_hint_capable_predictors():
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule(HintedQA, lambda self, **kwargs: self.predictor(**kwargs))

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=LMCritic(objective="The answer must be the word 'right'."),
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    (hint,) = result._refinement.attempts[1].hints_applied
    assert hint.target == "predictor"
    assert hint.content == "Answer with 'right'."
    assert hint.provenance == "LMCritic"


def test_fail_budget_is_per_call_not_per_instance():
    lm = DummyLM([{"answer": "unused"}])
    dspy.configure(lm=lm, adapter=None)

    def always_raise(self, **kwargs):
        raise ValueError("boom")

    module = DummyModule("question -> answer", always_raise)
    refine = Refine(module=module, N=2, reward_fn=lambda *_: 1.0, threshold=0.0)

    # N=2 with the default budget never raises within one call...
    assert refine(question="a") is None
    # ...and a second call must behave identically: the budget must not leak across calls.
    assert refine(question="b") is None
    assert refine.fail_count == 2


def test_failing_feedback_fn_degrades_to_resampling():
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule(HintedQA, lambda self, **kwargs: self.predictor(**kwargs))

    def broken_feedback(inputs, prediction, reward):
        raise RuntimeError("critic exploded")

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=broken_feedback,
    )
    result = refine(question="What is the right answer?")

    # The attempt's score survives, the retry happens without hints, and no budget is consumed.
    assert result.answer == "right"
    report = result._refinement
    assert report.attempts[0].reward == 0.0
    assert report.attempts[0].error is None
    assert report.attempts[1].hints_applied == ()


def test_refine_acall_delivers_hints_async():
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=None)

    class AsyncDummyModule(dspy.Module):
        def __init__(self, signature):
            super().__init__()
            self.predictor = Predict(signature)

        async def aforward(self, **kwargs) -> Prediction:
            return await self.predictor.acall(**kwargs)

    module = AsyncDummyModule(HintedQA)

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
        feedback_fn=lambda inputs, prediction, reward: "Answer with 'right'.",
    )
    result = asyncio.run(refine.acall(question="What is the right answer?"))

    assert result.answer == "right"
    (hint,) = result._refinement.attempts[1].hints_applied
    assert hint.content == "Answer with 'right'."
