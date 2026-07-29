import pytest

import dspy
from dspy.predict.parameter import Parameter
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
    """Refine's hint wrapper must run even when the module has an instance adapter via `set_adapter()`."""
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
    """The hint wrapper must not re-instantiate adapters whose constructors require arguments."""
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


def test_refine_failure_budget_is_per_call():
    """The failure budget must reset each call: no cross-call or cross-thread decay."""
    lm = DummyLM([{"answer": "ok"}] * 4)
    dspy.configure(lm=lm)
    call_count = [0]

    def fail_first_attempt_each_call(self, **kwargs):
        call_count[0] += 1
        if call_count[0] % 2 == 1:
            raise ValueError("Deliberately failing the first attempt")
        return self.predictor(**kwargs)

    module = DummyModule("question -> answer", fail_first_attempt_each_call)
    refine = Refine(module=module, N=2, reward_fn=lambda _, __: 1.0, threshold=0.0, fail_count=1)

    # Each call fails once and then succeeds; a shared, decaying budget would
    # eventually raise. Four consecutive calls must all succeed.
    for _ in range(4):
        result = refine(question="What is the capital of Belgium?")
        assert result.answer == "ok"


def test_refine_keeps_hint_out_of_trace_and_records_provenance():
    """The winning trace shows the program as declared; search provenance lives in metadata."""
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
        trace = dspy.settings.trace.copy()

    assert result.answer == "right"

    # Only the winning attempt is narrated, without the hint and without critic steps.
    assert len(trace) == 1, "Critic calls and losing attempts must not appear in the program trace"
    _, traced_inputs, traced_outputs = trace[0]
    assert "hint_" not in traced_inputs
    assert "advice" not in traced_outputs

    # The hint is not erased: it is recorded as refinement metadata with provenance.
    metadata = result.get_refinement()
    assert metadata["best_reward"] == 1.0
    assert [a["rollout_id"] for a in metadata["attempts"]] == [0, 1]
    assert metadata["attempts"][0]["hints"] == []
    (hint,) = metadata["attempts"][1]["hints"]
    assert hint.target == "predictor"
    assert hint.content == "Answer with 'right'."
    assert hint.authority == "non_authoritative"
    assert hint.provenance["source"] == "dspy.Refine/OfferFeedback"

    # The original module is untouched: no adapter changes, no lingering hints.
    assert module.predictor.adapter is None
    assert module.predictor._hints == []


def test_refine_declines_feedback_without_hint_accepting_units():
    """With no declared hint channel, Refine resamples and selects; it never
    manufactures a feedback path through transport or runtime predictors."""

    class SealedLeaf(dspy.Module, Parameter):
        """Flex-shaped: the predictor is derived from the leaf's own state and
        must not become a refinement target; the leaf itself declines hints."""

        def __init__(self):
            super().__init__()
            self.derived = Predict("question -> answer")

        def forward(self, **kwargs):
            return self.derived(**kwargs)

    class Wrapper(dspy.Module):
        def __init__(self):
            super().__init__()
            self.leaf = SealedLeaf()

        def forward(self, **kwargs):
            return self.leaf(**kwargs)

    # Two module answers plus a sentinel: if Refine attempted feedback between
    # the two attempts, the critic would consume a response and shift the
    # sequence, leaving the sentinel consumed by the second attempt.
    lm = DummyLM([{"answer": "a"}, {"answer": "b"}, {"answer": "sentinel"}])
    dspy.configure(lm=lm, adapter=None)

    wrapper = Wrapper()
    refine = Refine(
        module=wrapper,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "a" else 0.0,
        threshold=2.0,  # unreachable: force both attempts
    )
    result = refine(question="q")

    assert result.answer == "a"
    metadata = result.get_refinement()
    assert len(metadata["attempts"]) == 2
    assert [attempt["hints"] for attempt in metadata["attempts"]] == [[], []]
    # No critic call happened: the sentinel response was never consumed.
    assert next(lm.answers)["answer"] == "sentinel"


@pytest.mark.asyncio
async def test_refine_async_feedback_reaches_retry():
    """aforward mirrors forward: hints flow through the leaf protocol in async too."""

    class AsyncModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict("question -> answer")

        async def aforward(self, **kwargs):
            return await self.predictor.acall(**kwargs)

    adapter_calls = []

    class TrackingChatAdapter(dspy.ChatAdapter):
        async def acall(self, lm, lm_kwargs, signature, demos, inputs):
            adapter_calls.append(dict(inputs))
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)

    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = AsyncModule()
    module.set_adapter(TrackingChatAdapter())

    refine = Refine(
        module=module,
        N=2,
        reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0,
        threshold=1.0,
    )
    result = await refine.acall(question="What is the right answer?")

    assert result.answer == "right"
    hinted = [inputs for inputs in adapter_calls if "hint_" in inputs]
    assert hinted and hinted[0]["hint_"] == "Answer with 'right'."
