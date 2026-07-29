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
    """Refine's hint must reach the retry through an instance adapter set via `set_adapter()`.

    The hint is delivered as a declared `hint_` input field on the retry candidate's signature,
    so it flows through whatever adapter the predictor resolves, without wrapping it.
    """
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
    """Hint delivery must not re-instantiate or wrap adapters whose constructors require arguments."""
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


class AsyncDummyModule(DummyModule):
    async def aforward(self, **kwargs) -> Prediction:
        return self.forward_fn(self, **kwargs)


class TrackingAdapter(dspy.ChatAdapter):
    """Records the (signature, inputs) pairs each predictor call resolves through this adapter."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        self.calls.append((signature, dict(inputs)))
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)


def make_qa_module():
    return DummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))


def right_or_feedback(args, pred):
    """A feedback-bearing eval: says *why* the score is low, in the GEPA score+feedback shape."""
    if pred.answer == "right":
        return dspy.Prediction(score=1.0, feedback="")
    return dspy.Prediction(score=0.0, feedback="Answer with 'right'.")


def test_refine_eval_feedback_used_directly_without_llm_critic():
    """A score+feedback reward result supplies the hint itself; the LM critic must not run.

    Only two LM responses are provided: if `OfferFeedback` ran, it would consume the second
    response and the retry could not answer "right".
    """
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    tracking = TrackingAdapter()
    dspy.configure(lm=lm, adapter=tracking)

    module = make_qa_module()
    refine = Refine(module=module, N=2, reward_fn=right_or_feedback, threshold=1.0)
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    # Exactly two predictor calls resolved through the adapter: no critic call happened.
    assert len(tracking.calls) == 2
    # The retry carried the eval's feedback verbatim as the hint value.
    hinted = [(sig, inputs) for sig, inputs in tracking.calls if "hint_" in inputs]
    assert len(hinted) == 1
    assert hinted[0][1]["hint_"] == "Answer with 'right'."


def test_refine_hint_field_declares_authority_and_provenance():
    """The hint travels as a declared input field whose description marks it non-authoritative
    and names its provenance, so the candidate's semantics stay inspectable above the adapter."""
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    tracking = TrackingAdapter()
    dspy.configure(lm=lm, adapter=tracking)

    module = make_qa_module()
    refine = Refine(module=module, N=2, reward_fn=right_or_feedback, threshold=1.0)
    refine(question="What is the right answer?")

    hinted_signature = next(sig for sig, inputs in tracking.calls if "hint_" in inputs)
    desc = hinted_signature.input_fields["hint_"].json_schema_extra["desc"]
    assert "non-authoritative" in desc.lower()
    assert "reward function" in desc  # provenance: the eval's own feedback, not an LM critique
    # The hint exists only on the retry candidate; the wrapped program's intent is untouched.
    assert "hint_" not in module.predictor.signature.input_fields


def test_refine_llm_critic_hint_provenance():
    """When the eval returns only a score, the "auto" policy falls back to the LM critic, and the
    resulting hint is marked as coming from an automatic critique."""
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    tracking = TrackingAdapter()
    dspy.configure(lm=lm, adapter=None)

    module = make_qa_module()
    module.set_adapter(tracking)
    refine = Refine(
        module=module, N=2, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    hinted_signature = next(sig for sig, inputs in tracking.calls if "hint_" in inputs)
    desc = hinted_signature.input_fields["hint_"].json_schema_extra["desc"]
    assert "critique" in desc
    attempts = result._refinement
    assert attempts[1].hints[0].provenance == "llm_critic"
    assert attempts[1].hints[0].target == "predictor"


def test_refine_feedback_policy_eval_never_invokes_critic():
    """With `feedback_policy="eval"`, a float-only reward yields no hints and no critic call."""
    lm = DummyLM([{"answer": "wrong"}, {"answer": "also wrong"}])
    tracking = TrackingAdapter()
    dspy.configure(lm=lm, adapter=tracking)

    module = make_qa_module()
    refine = Refine(
        module=module, N=2, reward_fn=lambda _, __: 0.0, threshold=1.0, feedback_policy="eval"
    )
    result = refine(question="What is the right answer?")

    # Two module attempts, no critic call, no hints anywhere.
    assert len(tracking.calls) == 2
    assert all("hint_" not in inputs for _, inputs in tracking.calls)
    # Ties keep the earliest attempt.
    assert result.answer == "wrong"


def test_refine_feedback_policy_none_disables_hints():
    """With `feedback_policy="none"`, even eval-provided feedback is not turned into hints."""
    lm = DummyLM([{"answer": "wrong"}, {"answer": "also wrong"}])
    tracking = TrackingAdapter()
    dspy.configure(lm=lm, adapter=tracking)

    module = make_qa_module()
    refine = Refine(
        module=module, N=2, reward_fn=right_or_feedback, threshold=1.0, feedback_policy="none"
    )
    refine(question="What is the right answer?")

    assert len(tracking.calls) == 2
    assert all("hint_" not in inputs for _, inputs in tracking.calls)


def test_refine_rejects_unknown_feedback_policy():
    with pytest.raises(ValueError):
        Refine(module=make_qa_module(), N=2, reward_fn=lambda _, __: 0.0, threshold=1.0, feedback_policy="sometimes")


def test_refine_merged_trace_is_hint_free_and_critic_free():
    """The caller's trace shows the winning run as the wrapped program declares it: the `hint_`
    refinement input is stripped, and the critic's own LM call never enters the program trace."""
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with 'right'."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    module = make_qa_module()
    refine = Refine(
        module=module, N=2, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )
    with dspy.context(trace=[]):
        result = refine(question="What is the right answer?")
        trace = list(dspy.settings.trace)

    assert result.answer == "right"
    # Only the winning attempt's predictor step is merged; the critic step is refinement machinery.
    assert len(trace) == 1
    _predictor, inputs, outputs = trace[0]
    assert "hint_" not in inputs
    assert inputs == {"question": "What is the right answer?"}
    assert "advice" not in outputs


def test_refine_result_carries_refinement_metadata():
    """The search record (scores, feedback, hints with authority/provenance) is attached to the
    returned prediction as `_refinement` instead of being erased or leaked into the trace."""
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=None)

    module = make_qa_module()
    refine = Refine(module=module, N=2, reward_fn=right_or_feedback, threshold=1.0)
    result = refine(question="What is the right answer?")

    attempts = result._refinement
    assert len(attempts) == 2
    assert attempts[0].score == 0.0
    assert attempts[0].feedback == "Answer with 'right'."
    assert attempts[0].hints == ()
    assert attempts[1].score == 1.0
    hint = attempts[1].hints[0]
    assert hint.content == "Answer with 'right'."
    assert hint.authority == "non_authoritative"
    assert hint.provenance == "reward_feedback"
    assert hint.scope == "next_candidate"


def test_refine_call_time_adapter_precedence_preserved():
    """A call-time `adapter=` override on the inner predictor still outranks everything on the
    hinted retry; the hint rides the resolved adapter instead of replacing it."""
    call_time = TrackingAdapter()
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

    module = DummyModule(
        "question -> answer", lambda self, **kwargs: self.predictor(adapter=call_time, **kwargs)
    )
    refine = Refine(module=module, N=2, reward_fn=right_or_feedback, threshold=1.0)
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    # Both attempts resolved the call-time adapter, and the retry carried the hint through it.
    assert len(call_time.calls) == 2
    assert [("hint_" in inputs) for _, inputs in call_time.calls] == [False, True]


def test_refine_fail_count_is_not_mutated_across_calls():
    """The failure budget is per-call state: one call's failures must not deplete the next call's
    budget (or race with it under threaded evaluation)."""
    lm = DummyLM([{"answer": "Brussels"}] * 8)
    dspy.configure(lm=lm, adapter=None)

    call_log = []

    def fail_first_attempt(self, **kwargs):
        call_log.append(1)
        if len(call_log) % 2 == 1:
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    module = DummyModule("question -> answer", fail_first_attempt)
    refine = Refine(module=module, N=2, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=1)

    for _ in range(3):
        result = refine(question="What is the capital of Belgium?")
        assert result.answer == "Brussels"
    assert refine.fail_count == 1


def test_refine_raises_last_error_instead_of_returning_none():
    """When every attempt fails within the failure budget, the last exception is raised rather
    than silently returning `None`."""
    lm = DummyLM([{"answer": "Brussels"}])
    dspy.configure(lm=lm, adapter=None)

    def always_raise(self, **kwargs):
        raise RuntimeError("Deliberately failing")

    module = DummyModule("question -> answer", always_raise)
    refine = Refine(module=module, N=2, reward_fn=lambda _, __: 1.0, threshold=1.0, fail_count=5)
    with pytest.raises(RuntimeError):
        refine(question="What is the capital of Belgium?")


@pytest.mark.asyncio
async def test_refine_async_eval_feedback():
    """`acall` follows the same feedback contract: eval feedback becomes the retry's hint and the
    critic is skipped, with the winning trace merged hint-free."""
    lm = DummyLM([{"answer": "wrong"}, {"answer": "right"}])
    tracking = TrackingAdapter()
    dspy.configure(lm=lm, adapter=tracking)

    module = AsyncDummyModule("question -> answer", lambda self, **kwargs: self.predictor(**kwargs))
    refine = Refine(module=module, N=2, reward_fn=right_or_feedback, threshold=1.0)
    with dspy.context(trace=[]):
        result = await refine.acall(question="What is the right answer?")
        trace = list(dspy.settings.trace)

    assert result.answer == "right"
    hinted = [inputs for _, inputs in tracking.calls if "hint_" in inputs]
    assert hinted and hinted[0]["hint_"] == "Answer with 'right'."
    assert all("hint_" not in inputs for _, inputs, _ in trace)
