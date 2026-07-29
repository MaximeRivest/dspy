import pytest

import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import Refine
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


def test_refine_feedback_reaches_retry_with_instance_adapter():
    """Refine's feedback hint must reach the retry even when the module has an instance adapter via `set_adapter()`."""
    adapter_calls = []

    class TrackingChatAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            adapter_calls.append(dict(inputs))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    def forward_fn(self, **kwargs):
        return self.predictor(**kwargs)

    module = DummyModule("question -> answer", forward_fn)
    tracking_adapter = TrackingChatAdapter()
    module.set_adapter(tracking_adapter)

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    # The first attempt goes through the configured adapter without a hint.
    assert any("hint_" not in inputs for inputs in adapter_calls)
    # The retry must go through the configured adapter *and* carry the feedback hint.
    hinted = [inputs for inputs in adapter_calls if "hint_" in inputs]
    assert hinted, "Feedback hint never reached the retry through the configured adapter"
    assert hinted[0]["hint_"] == "Answer with 'right'."
    # The original module must not be mutated.
    assert module.predictor.adapter is tracking_adapter


def test_refine_feedback_retry_with_stateful_adapter_requiring_constructor_args():
    """Feedback delivery must not re-instantiate adapters whose constructors require arguments."""
    adapter_calls = []

    class StatefulAdapter(dspy.ChatAdapter):
        def __init__(self, tag):
            super().__init__()
            self.tag = tag
            self.calls = []

        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            call = dict(inputs)
            self.calls.append(call)
            adapter_calls.append((self.tag, call))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    stateful_adapter = StatefulAdapter(tag="configured")
    module.set_adapter(stateful_adapter)

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    # Refine runs a deep copy of the module. The external log proves that the copied
    # stateful adapter retained its required constructor state and received the hint.
    hinted = [(tag, inputs) for tag, inputs in adapter_calls if "hint_" in inputs]
    assert hinted == [("configured", {"question": "What is the right answer?", "hint_": "Answer with 'right'."})]
    # The original module and adapter remain unmodified.
    assert module.predictor.adapter is stateful_adapter
    assert type(stateful_adapter) is StatefulAdapter
    assert stateful_adapter.tag == "configured"
    assert stateful_adapter.calls == []


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


def test_refine_fail_budget_is_per_call_not_per_instance():
    """The failure budget must reset on every call instead of depleting shared instance state."""
    lm = DummyLM([{"answer": "ok"}] * 10)
    dspy.configure(lm=lm, adapter=None)
    call_count = [0]

    def flaky(self, **kwargs):
        call_count[0] += 1
        # Fail twice, then succeed on the third attempt of each forward call.
        if call_count[0] % 3 != 0:
            raise ValueError("flaky")
        return self.predictor(**kwargs)

    module = DummyModule("question -> answer", flaky)
    refine = Refine(module=module, N=3, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=2)

    assert refine(question="q").answer == "ok"
    # A second call gets a fresh failure budget instead of inheriting a depleted one.
    assert refine(question="q").answer == "ok"


def test_refine_raises_when_all_attempts_fail_within_budget():
    """With zero successful attempts there is no prediction: raise the last error, never return None."""
    lm = DummyLM([{"answer": "irrelevant"}])
    dspy.configure(lm=lm, adapter=None)

    def always_raise(self, **kwargs):
        raise ValueError("Deliberately failing")

    module = DummyModule("question -> answer", always_raise)
    refine = Refine(module=module, N=2, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=10)
    with pytest.raises(ValueError):
        refine(question="q")


def test_refine_hint_is_declared_on_candidate_signature_not_via_adapter_wrapping():
    """The retry must be fully defined before the adapter runs: the hint is an input field declared
    on the throwaway candidate's signature, and the adapter is never wrapped or subclassed."""
    seen = []

    class RecordingAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            seen.append((type(self), signature, dict(inputs)))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    adapter = RecordingAdapter()
    module.set_adapter(adapter)

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
    )
    result = refine(question="What is the right answer?")
    assert result.answer == "right"

    hinted = [(cls, sig, inputs) for cls, sig, inputs in seen if "hint_" in inputs]
    assert hinted, "The feedback hint never reached the retry"
    cls, sig, inputs = hinted[0]
    # The adapter runs unmodified: no dynamically-created subclass carries the hint.
    assert cls is RecordingAdapter
    # The hint is declared in the candidate's signature before the adapter formats the call.
    assert "hint_" in sig.input_fields
    assert inputs["hint_"] == "Answer with 'right'."
    # The wrapped module keeps its clean signature and its own adapter.
    assert "hint_" not in module.predictor.signature.input_fields
    assert module.predictor.adapter is adapter


def test_refine_exports_clean_trace_mapped_to_original_predictors():
    """The caller's trace must read as if the wrapped module produced the winning prediction directly:
    no critic steps, no failed attempts, no `hint_` input, and predictors mapped back to the original."""
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
    )

    with dspy.context(trace=[]):
        result = refine(question="What is the right answer?")
        trace = list(dspy.settings.trace)

    assert result.answer == "right"
    # Only the winning attempt's predictor call is reported: no critic call, no losing attempt.
    assert len(trace) == 1
    predictor, inputs, outputs = trace[0]
    assert predictor is module.predictor
    assert "hint_" not in inputs
    assert inputs["question"] == "What is the right answer?"
    assert outputs.answer == "right"


def test_refine_construction_survives_unavailable_source():
    """Source code is critic evidence, not a runtime requirement: constructing Refine around a class
    without retrievable source must degrade to a placeholder instead of raising."""
    DynamicModule = type("DynamicRefineModule", (DummyModule,), {})  # noqa: N806
    module = DynamicModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))

    refine = Refine(module=module, N=1, reward_fn=lambda _, __: 1.0, threshold=1.0)
    assert "source unavailable" in refine.module_code

    lm = DummyLM([{"answer": "ok"}])
    dspy.configure(lm=lm, adapter=None)
    assert refine(question="q").answer == "ok"


class AsyncDummyModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    def forward(self, **kwargs) -> Prediction:
        return self.predictor(**kwargs)

    async def aforward(self, **kwargs) -> Prediction:
        return await self.predictor.acall(**kwargs)


@pytest.mark.asyncio
async def test_refine_async_feedback_retry():
    """`acall` must follow the same contract as the sync path, including the feedback retry."""
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    with dspy.context(lm=lm, adapter=None):
        module = AsyncDummyModule("question -> answer")
        refine = Refine(
            module=module,
            N=2,
            reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
            threshold=1.0,
        )
        result = await refine.acall(question="What is the right answer?")
        assert result.answer == "right"


@pytest.mark.asyncio
async def test_refine_async_raises_when_all_attempts_fail():
    class FailingAsyncModule(AsyncDummyModule):
        async def aforward(self, **kwargs):
            raise ValueError("Deliberately failing")

    with dspy.context(lm=DummyLM([{"answer": "x"}]), adapter=None):
        module = FailingAsyncModule("question -> answer")
        refine = Refine(module=module, N=2, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=10)
        with pytest.raises(ValueError):
            await refine.acall(question="q")
