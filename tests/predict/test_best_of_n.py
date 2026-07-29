import pytest

import dspy
from dspy.predict.best_of_n import BestOfN
from dspy.predict.predict import Predict
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class DummyModule(dspy.Module):
    def __init__(self, signature, forward_fn):
        super().__init__()
        self.predictor = Predict(signature)
        self.forward_fn = forward_fn

    def forward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


class AsyncDummyModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def aforward(self, **kwargs) -> Prediction:
        return await self.predictor.acall(**kwargs)


class NoPredictorModule(dspy.Module):
    def forward(self, **kwargs) -> Prediction:
        return Prediction(answer="static")


def call_predictor(self, **kwargs):
    return self.predictor(**kwargs)


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

    best_of_n = BestOfN(module=predict, N=3, reward_fn=reward_fn, threshold=1.0)
    result = best_of_n(question="What is the capital of Belgium?")

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

    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0)
    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")


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

    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)
    with pytest.raises(ValueError):
        best_of_n(question="What is the capital of Belgium?")
    assert module_call_count[0] == 2, (
        "Module should have been called exactly 2 times, but was called %d times" % module_call_count[0]
    )


def test_best_of_n_budget_counts_failures_not_attempts():
    lm = DummyLM([{"answer": "one"}, {"answer": "two"}, {"answer": "four"}])
    dspy.configure(lm=lm)
    module_call_count = [0]

    def fail_third_call(self, **kwargs):
        module_call_count[0] += 1
        if module_call_count[0] == 3:
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", fail_third_call)
    best_of_n = BestOfN(module=predict, N=4, reward_fn=lambda *_: 0.0, threshold=1.0, fail_count=1)

    # Two low-reward successes precede the single failure: only the failure counts
    # against the budget, so the call must not raise.
    result = best_of_n(question="q")

    assert result is not None
    assert module_call_count[0] == 4


def test_best_of_n_fail_count_zero_raises_on_first_failure():
    dspy.configure(lm=DummyLM([{"answer": "x"}]))
    module_call_count = [0]

    def always_raise(self, **kwargs):
        module_call_count[0] += 1
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)
    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda *_: 1.0, threshold=0.0, fail_count=0)

    with pytest.raises(ValueError, match="Deliberately failing"):
        best_of_n(question="q")
    assert module_call_count[0] == 1


def test_best_of_n_failure_budget_is_per_call():
    lm = DummyLM({"q1": {"answer": "a1"}, "q2": {"answer": "a2"}})
    dspy.configure(lm=lm)
    counts = {}

    def fail_first(self, **kwargs):
        question = kwargs["question"]
        counts[question] = counts.get(question, 0) + 1
        if counts[question] == 1:
            raise ValueError(f"first attempt for {question} fails")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", fail_first)
    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda *_: 0.0, threshold=0.0, fail_count=1)

    # Each call fails exactly once; a shared, mutable budget would exhaust on the second call.
    assert best_of_n(question="q1").answer == "a1"
    assert best_of_n(question="q2").answer == "a2"


def test_best_of_n_all_attempts_failing_raises_last_error():
    dspy.configure(lm=DummyLM([]))

    def always_raise(self, **kwargs):
        raise RuntimeError("always fails")

    predict = DummyModule("question -> answer", always_raise)
    # The budget tolerates every failure; the all-failure path itself must raise
    # rather than return None.
    best_of_n = BestOfN(module=predict, N=3, reward_fn=lambda *_: 1.0, threshold=0.0, fail_count=5)

    with pytest.raises(RuntimeError, match="always fails"):
        best_of_n(question="q")


@pytest.mark.asyncio
async def test_best_of_n_async_all_attempts_failing_raises_last_error():
    dspy.configure(lm=DummyLM([]))

    class AsyncAlwaysFail(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict("question -> answer")

        async def aforward(self, **kwargs):
            raise RuntimeError("always fails")

    best_of_n = BestOfN(module=AsyncAlwaysFail(), N=3, reward_fn=lambda *_: 1.0, threshold=0.0, fail_count=5)

    with pytest.raises(RuntimeError, match="always fails"):
        await best_of_n.acall(question="q")


@pytest.mark.asyncio
async def test_best_of_n_async_awaits_async_reward_fn():
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm)

    async def reward(kwargs, pred):
        return 1.0 if pred.answer == "right" else 0.0

    best_of_n = BestOfN(module=AsyncDummyModule("question -> answer"), N=3, reward_fn=reward, threshold=1.0)
    result = await best_of_n.acall(question="q")

    assert result.answer == "right"


def test_best_of_n_exports_winning_trace_remapped_to_original_predictor():
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm)

    predict = DummyModule("question -> answer", call_predictor)
    best_of_n = BestOfN(
        module=predict, N=3, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )

    with dspy.context(trace=[]):
        best_of_n(question="q")
        trace = dspy.settings.trace
        # Only the winning attempt's entry is exported.
        assert len(trace) == 1
        traced_predictor, traced_inputs, traced_outputs = trace[0]
        # Remapped to the predictor the caller owns, so demo bootstrapping
        # attaches to the original program.
        assert traced_predictor is predict.predictor
        assert traced_inputs["question"] == "q"
        assert traced_outputs.answer == "right"


def test_best_of_n_safe_when_parent_trace_is_none():
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm)

    predict = DummyModule("question -> answer", call_predictor)
    best_of_n = BestOfN(
        module=predict, N=3, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )

    with dspy.context(trace=None):
        result = best_of_n(question="q")

    assert result.answer == "right"


def test_best_of_n_zero_predictor_module_resamples_without_crashing():
    lm = DummyLM([])
    dspy.configure(lm=lm)

    best_of_n = BestOfN(module=NoPredictorModule(), N=3, reward_fn=lambda *_: 0.0, threshold=1.0)
    result = best_of_n(question="q")

    assert result.answer == "static"
    assert len(lm.history) == 0


def test_best_of_n_zero_predictor_module_without_any_lm_raises_clear_error():
    best_of_n = BestOfN(module=NoPredictorModule(), N=3, reward_fn=lambda *_: 0.0, threshold=1.0)

    with pytest.raises(ValueError, match="No LM is loaded"):
        best_of_n(question="q")
