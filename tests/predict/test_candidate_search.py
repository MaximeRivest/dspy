"""Tests for the candidate-search plane shared by `dspy.BestOfN` and `dspy.Refine`.

These tests pin the semantic contract of the search engine:

- hints are realized into the candidate program (signature + inputs) *before*
  any Adapter runs, so Adapters only ever see already-defined calls;
- the caller-visible trace contains only the selected execution, expressed in
  terms of the original program (hint fields stripped, predictors remapped);
- auxiliary search work (the feedback critic) never reaches the caller's trace;
- the failure budget is per-call, and a search with no successful attempt
  raises instead of returning `None`;
- provenance (attempts, rollout ids, rewards, hints and whether they applied)
  is attached to the returned prediction;
- the engine does not assume every refinable leaf is a `dspy.Predict`: modules
  with zero predictor call sites refine by resampling, with no critic call and
  no hint forced through transport.
"""

import asyncio

import pytest

import dspy
from dspy.predict.best_of_n import BestOfN
from dspy.predict.candidate_search import CandidateSearch, Hint, search_record
from dspy.predict.predict import Predict
from dspy.predict.refine import Refine
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn=None):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn or (lambda self, **kwargs: self.predictor(**kwargs))

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


def _wrong_then_right_lm():
    return DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )


def _right_reward(_, pred):
    return 1.0 if pred.answer == "right" else 0.0


def test_hint_is_a_defined_call_before_the_adapter_runs():
    """The retry's Adapter receives a signature that already declares `hint_` and inputs that
    already carry the hint value, through the plain configured adapter class (no dynamic
    wrapper subclass)."""
    adapter_calls = []

    class TrackingChatAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            adapter_calls.append((type(self), list(signature.input_fields), dict(inputs)))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    dspy.configure(lm=_wrong_then_right_lm(), adapter=None)

    module = DummyModule("question -> answer")
    tracking_adapter = TrackingChatAdapter()
    module.set_adapter(tracking_adapter)

    refine = Refine(module=module, N=2, reward_fn=_right_reward, threshold=1.0)
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    hinted = [(cls, fields, inputs) for cls, fields, inputs in adapter_calls if "hint_" in inputs]
    assert hinted, "The retry never carried the hint"
    cls, fields, inputs = hinted[0]
    # The call was fully defined before the adapter saw it...
    assert "hint_" in fields
    assert inputs["hint_"] == "Answer with 'right'."
    # ...and it went through the plain configured adapter class, not a wrapper subclass.
    assert cls is TrackingChatAdapter

    # The original program was never mutated.
    assert module.predictor.adapter is tracking_adapter
    assert "hint_" not in module.predictor.signature.input_fields


def test_selected_trace_is_clean_and_remapped_to_original_predictors():
    """The caller's trace shows only the winning execution: hint inputs stripped and entries
    remapped to the original module's predictors. The raw hinted trace stays available as
    refinement metadata on the search record."""
    dspy.configure(lm=_wrong_then_right_lm(), adapter=None)
    module = DummyModule("question -> answer")
    refine = Refine(module=module, N=2, reward_fn=_right_reward, threshold=1.0)

    with dspy.context(trace=[]):
        result = refine(question="What is the right answer?")
        caller_trace = list(dspy.settings.trace)

    assert result.answer == "right"
    assert len(caller_trace) == 1, "Only the selected execution belongs in the caller's trace"
    predictor, trace_inputs, trace_outputs = caller_trace[0]
    assert predictor is module.predictor, "Trace must reference the original program's predictor"
    assert "hint_" not in trace_inputs
    assert dict(trace_outputs) == {"answer": "right"}

    record = search_record(result)
    raw_inputs = [inputs for _, inputs, _ in record.best.trace]
    assert any("hint_" in inputs for inputs in raw_inputs), "Refinement metadata keeps the hinted execution"


def test_feedback_critic_stays_out_of_caller_trace():
    dspy.configure(lm=_wrong_then_right_lm(), adapter=None)
    module = DummyModule("question -> answer")
    refine = Refine(module=module, N=2, reward_fn=_right_reward, threshold=1.0)

    with dspy.context(trace=[]):
        refine(question="What is the right answer?")
        caller_trace = list(dspy.settings.trace)

    for predictor, _, _ in caller_trace:
        assert "advice" not in predictor.signature.output_fields, "The OfferFeedback critic leaked into the trace"


def test_failure_budget_is_per_call():
    """`fail_count` must not decay across calls: a module that fails once per call succeeds
    repeatedly under `fail_count=1`."""
    state = {"calls": 0}

    def fail_first_each_call(self, **kwargs):
        state["calls"] += 1
        if state["calls"] % 2 == 1:
            raise ValueError("flaky")
        return self.predictor(**kwargs)

    dspy.configure(lm=DummyLM([{"answer": "ok"}] * 5))
    module = DummyModule("question -> answer", fail_first_each_call)
    best_of_n = BestOfN(module=module, N=3, reward_fn=lambda *_: 1.0, threshold=1.0, fail_count=1)

    for _ in range(3):
        assert best_of_n(question="q").answer == "ok"
    assert best_of_n.fail_count == 1, "The configured budget must not be mutated"


def test_all_attempts_failing_raises_instead_of_returning_none():
    dspy.configure(lm=DummyLM([{"answer": "x"}] * 5))

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    module = DummyModule("question -> answer", always_raise)
    # Budget large enough that it is never exceeded: the raise comes from ending with no success.
    refine = Refine(module=module, N=2, reward_fn=lambda *_: 1.0, threshold=0.0, fail_count=10)
    with pytest.raises(ValueError, match="Deliberately failing"):
        refine(question="q")


def test_leaf_without_predictors_refines_by_resampling():
    """A refinable leaf need not expose `dspy.Predict` call sites (nor optimizer Parameters).
    The search still resamples and selects; no critic runs and no hint is manufactured."""
    counter = {"n": 0}

    class OpaqueLeaf(dspy.Module):
        def forward(self, **kwargs):
            counter["n"] += 1
            return Prediction(answer="good" if counter["n"] >= 2 else "bad")

    # An empty DummyLM: any LM call (module or critic) would produce "No more responses",
    # so a passing "good" answer proves no critic consumed the LM.
    dspy.configure(lm=DummyLM([]))
    leaf = OpaqueLeaf()
    refine = Refine(module=leaf, N=3, reward_fn=lambda _, p: 1.0 if p.answer == "good" else 0.0, threshold=1.0)
    result = refine(question="q")

    assert result.answer == "good"
    record = search_record(result)
    assert len(record.attempts) == 2
    assert all(attempt.hints == [] for attempt in record.attempts)
    # The wrapped leaf was never mutated into exposing predictors.
    assert leaf.named_predictors() == []


def test_unresolvable_hint_is_recorded_not_forced():
    """A hint whose target does not resolve on the candidate is kept as provenance with
    `applied=False` instead of being pushed through transport."""
    adapter_calls = []

    class TrackingChatAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            adapter_calls.append(dict(inputs))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    dspy.configure(lm=DummyLM([{"answer": "a"}, {"answer": "b"}]), adapter=None)
    module = DummyModule("question -> answer")
    module.set_adapter(TrackingChatAdapter())

    search = CandidateSearch(
        module=module,
        num_candidates=2,
        reward_fn=lambda *_: 0.0,
        threshold=1.0,
        propose=lambda attempt, inputs, candidate: [Hint(target="no_such_module", content="irrelevant")],
    )
    record = search.run({"question": "q"})

    hint = record.attempts[1].hints[0]
    assert hint.applied is False
    assert hint.authority == "non_authoritative"
    assert all("hint_" not in inputs for inputs in adapter_calls)


def test_search_record_provenance():
    dspy.configure(lm=DummyLM([{"answer": "one two"}, {"answer": "one"}]))
    module = DummyModule("question -> answer")
    best_of_n = BestOfN(
        module=module,
        N=3,
        reward_fn=lambda _, pred: 1.0 if len(pred.answer.split()) == 1 else 0.0,
        threshold=1.0,
    )
    result = best_of_n(question="q")

    record = search_record(result)
    assert [a.index for a in record.attempts] == [0, 1]
    assert [a.rollout_id for a in record.attempts] == [0, 1]
    assert [a.reward for a in record.attempts] == [0.0, 1.0]
    assert record.best_index == 1
    assert record.best.prediction is result
    assert record.best_reward == 1.0
    # BestOfN is the independent-sampling policy: no hints, ever.
    assert all(a.hints == [] for a in record.attempts)


def test_hint_proposal_failure_does_not_consume_the_search():
    """A crashing critic is a failed hint, not a failed attempt: the search keeps resampling
    and records the proposal error."""
    dspy.configure(lm=DummyLM([{"answer": "wrong"}, {"answer": "right"}]))
    module = DummyModule("question -> answer")

    def broken_propose(attempt, inputs, candidate):
        raise RuntimeError("critic crashed")

    search = CandidateSearch(
        module=module,
        num_candidates=2,
        reward_fn=_right_reward,
        threshold=1.0,
        propose=broken_propose,
    )
    record = search.run({"question": "q"})

    assert record.best.prediction.answer == "right"
    assert isinstance(record.attempts[0].proposal_error, RuntimeError)
    assert record.attempts[0].error is None, "The attempt itself succeeded"


def test_async_refine_carries_hint_into_retry():
    class AsyncDummyModule(dspy.Module):
        def __init__(self, signature):
            super().__init__()
            self.predictor = Predict(signature)

        async def aforward(self, **kwargs):
            return await self.predictor.acall(**kwargs)

    dspy.configure(lm=_wrong_then_right_lm(), adapter=None)
    module = AsyncDummyModule("question -> answer")
    refine = Refine(module=module, N=2, reward_fn=_right_reward, threshold=1.0)

    result = asyncio.run(refine.acall(question="What is the right answer?"))

    assert result.answer == "right"
    record = search_record(result)
    assert [h.content for h in record.attempts[1].hints] == ["Answer with 'right'."]
    assert record.attempts[1].hints[0].applied is True


def test_user_declared_hint_field_is_never_shadowed():
    """If the program itself declares an input named `hint_`, the search refuses to overwrite
    declared intent with search feedback."""
    dspy.configure(
        lm=DummyLM(
            [
                {"answer": "wrong"},
                {"discussion": "d", "advice": {"predictor": "Try again."}},
                {"answer": "right"},
            ]
        ),
        adapter=None,
    )
    module = DummyModule("question, hint_ -> answer")
    refine = Refine(module=module, N=2, reward_fn=_right_reward, threshold=1.0)
    result = refine(question="q", hint_="user-provided hint")

    assert result.answer == "right"
    record = search_record(result)
    hint = record.attempts[1].hints[0]
    assert hint.applied is False
    # The user's own hint_ input still flowed through, untouched.
    assert record.attempts[1].trace[0][1]["hint_"] == "user-provided hint"
