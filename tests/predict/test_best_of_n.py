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


def test_best_of_n_fail_budget_is_per_call_not_per_instance():
    """The failure budget must reset on every call instead of depleting shared instance state."""
    lm = DummyLM([{"answer": "ok"}] * 10)
    dspy.configure(lm=lm)
    call_count = [0]

    def flaky(self, **kwargs):
        call_count[0] += 1
        # Fail twice, then succeed on the third attempt of each forward call.
        if call_count[0] % 3 != 0:
            raise ValueError("flaky")
        return self.predictor(**kwargs)

    module = DummyModule("question -> answer", flaky)
    best_of_n = BestOfN(module=module, N=3, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=2)

    assert best_of_n(question="q").answer == "ok"
    # A second call gets a fresh failure budget instead of inheriting a depleted one.
    assert best_of_n(question="q").answer == "ok"


def test_best_of_n_raises_when_all_attempts_fail_within_budget():
    """With zero successful attempts there is no prediction: raise the last error, never return None."""
    lm = DummyLM([{"answer": "irrelevant"}])
    dspy.configure(lm=lm)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    module = DummyModule("question -> answer", always_raise)
    best_of_n = BestOfN(module=module, N=2, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=10)
    with pytest.raises(ValueError):
        best_of_n(question="q")


def test_best_of_n_trace_maps_to_original_predictors():
    """The exported winning trace must reference the wrapped module's own predictors, not the
    per-attempt deep copies, so trace consumers see the program the caller owns."""
    lm = DummyLM([{"answer": "Brussels"}])
    dspy.configure(lm=lm)

    module = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    best_of_n = BestOfN(module=module, N=2, reward_fn=lambda _, __: 1.0, threshold=1.0)

    with dspy.context(trace=[]):
        result = best_of_n(question="What is the capital of Belgium?")
        trace = list(dspy.settings.trace)

    assert result.answer == "Brussels"
    assert len(trace) == 1
    predictor, inputs, outputs = trace[0]
    assert predictor is module.predictor
    assert inputs["question"] == "What is the capital of Belgium?"
    assert outputs.answer == "Brussels"


class AsyncDummyModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs) -> Prediction:
        return self.predictor(**kwargs)

    async def aforward(self, **kwargs) -> Prediction:
        return await self.predictor.acall(**kwargs)


@pytest.mark.asyncio
async def test_best_of_n_async():
    """`acall` must follow the same sample-score-select contract as the sync path."""
    lm = DummyLM([{"answer": "City of Brussels"}, {"answer": "Brussels"}])
    with dspy.context(lm=lm):
        module = AsyncDummyModule("question -> answer")
        best_of_n = BestOfN(
            module=module,
            N=2,
            reward_fn=lambda _, pred: 1.0 if pred.answer == "Brussels" else 0.0,
            threshold=1.0,
        )
        result = await best_of_n.acall(question="What is the capital of Belgium?")
        assert result.answer == "Brussels"
