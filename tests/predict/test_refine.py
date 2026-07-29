import json
import logging
import textwrap
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import dspy
from dspy.predict.predict import Predict
from dspy.predict.refine import HINT_FIELD_NAME, Refine
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


class TrackingChatAdapter(dspy.ChatAdapter):
    """A ChatAdapter that records the signature and inputs of every call it formats."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        self.calls.append({"signature": signature, "inputs": dict(inputs)})
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        self.calls.append({"signature": signature, "inputs": dict(inputs)})
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)

    def module_calls(self):
        return [call for call in self.calls if "question" in call["signature"].input_fields]

    def critic_calls(self):
        return [call for call in self.calls if "program_trajectory" in call["signature"].input_fields]


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


def test_refine_hinted_retry_appends_input_field_without_touching_adapter():
    adapter = TrackingChatAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with the word right."}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm, adapter=adapter)
    module_call_count = [0]

    def count_calls(self, **kwargs):
        module_call_count[0] += 1
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", count_calls)
    refine = Refine(
        module=predict, N=3, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    assert module_call_count[0] == 2

    # Calls in order: attempt 1, critic, attempt 2.
    assert len(adapter.calls) == 3
    first, _, retry = adapter.calls
    assert HINT_FIELD_NAME not in first["signature"].input_fields
    assert HINT_FIELD_NAME not in first["inputs"]
    assert HINT_FIELD_NAME in retry["signature"].input_fields
    assert retry["inputs"][HINT_FIELD_NAME] == "Answer with the word right."

    # The adapter is left alone: same instance, same class, no wrapping or subclassing.
    assert dspy.settings.adapter is adapter
    assert type(dspy.settings.adapter) is TrackingChatAdapter
    # The wrapped module is never mutated.
    assert HINT_FIELD_NAME not in predict.predictor.signature.input_fields


def test_refine_hinted_retry_with_json_adapter_and_demos():
    adapter = dspy.JSONAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "The answer was wrong.", "advice": {"predictor": "Answer with the word right."}},
            {"answer": "right"},
        ],
        adapter=adapter,
    )
    dspy.configure(lm=lm, adapter=adapter)

    predict = DummyModule("question -> answer", call_predictor)
    predict.predictor.demos = [dspy.Example(question="What is 1+1?", answer="two").with_inputs("question")]
    refine = Refine(
        module=predict, N=3, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )
    result = refine(question="What is the right answer?")

    assert result.answer == "right"
    assert HINT_FIELD_NAME not in predict.predictor.signature.input_fields


def test_refine_never_overwrites_user_declared_hint_field(caplog):
    adapter = TrackingChatAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "critic hint one"}},
            {"answer": "still wrong"},
            {"discussion": "d", "advice": {"predictor": "critic hint two"}},
            {"answer": "third"},
        ]
    )
    dspy.configure(lm=lm, adapter=adapter)

    predict = DummyModule("question, hint_ -> answer", call_predictor)
    refine = Refine(module=predict, N=3, reward_fn=lambda *_: 0.0, threshold=1.0)

    # The "dspy" logger does not propagate to the root logger (see
    # dspy/utils/logging_utils.py), where caplog's capture handler lives; propagate
    # temporarily so the records are observable.
    dspy_logger = logging.getLogger("dspy")
    original_propagate = dspy_logger.propagate
    dspy_logger.propagate = True
    try:
        with caplog.at_level(logging.WARNING, logger="dspy.predict.refine"):
            with dspy.context(trace=[]):
                refine(question="q", hint_="user hint")
                trace = dspy.settings.trace
                # A user-declared hint is a real input: it stays in the exported trace.
                assert len(trace) == 1
                assert trace[0][1][HINT_FIELD_NAME] == "user hint"
    finally:
        dspy_logger.propagate = original_propagate

    module_calls = adapter.module_calls()
    assert len(module_calls) == 3
    for call in module_calls:
        # The user's value flows through untouched on every attempt; no critic hint replaces it.
        assert call["inputs"][HINT_FIELD_NAME] == "user hint"

    shadow_warnings = [r for r in caplog.records if "already declares" in r.getMessage()]
    assert len(shadow_warnings) == 1, "The shadow warning should be logged once per call"


def test_refine_call_time_signature_override_receives_no_hint():
    adapter = TrackingChatAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "some hint"}},
            {"answer": "still wrong"},
        ]
    )
    dspy.configure(lm=lm, adapter=adapter)

    def override_signature(self, **kwargs):
        return self.predictor(signature="question -> answer", **kwargs)

    predict = DummyModule("question -> answer", override_signature)
    refine = Refine(module=predict, N=2, reward_fn=lambda *_: 0.0, threshold=1.0)
    result = refine(question="q")

    assert result.answer == "wrong"
    module_calls = adapter.module_calls()
    assert len(module_calls) == 2
    for call in module_calls:
        assert HINT_FIELD_NAME not in call["signature"].input_fields
        assert HINT_FIELD_NAME not in call["inputs"]


def test_refine_budget_counts_failures_not_attempts():
    lm = DummyLM(
        [
            {"answer": "one"},
            {"discussion": "d", "advice": {"predictor": "N/A"}},
            {"answer": "two"},
            {"discussion": "d", "advice": {"predictor": "N/A"}},
            {"answer": "four"},
        ]
    )
    dspy.configure(lm=lm)
    module_call_count = [0]

    def fail_third_call(self, **kwargs):
        module_call_count[0] += 1
        if module_call_count[0] == 3:
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", fail_third_call)
    refine = Refine(module=predict, N=4, reward_fn=lambda *_: 0.0, threshold=1.0, fail_count=1)

    # Two low-reward successes precede the single failure: only the failure counts
    # against the budget, so the call must not raise.
    result = refine(question="q")

    assert result is not None
    assert module_call_count[0] == 4


def test_refine_fail_count_zero_raises_on_first_failure():
    dspy.configure(lm=DummyLM([{"answer": "x"}]))
    module_call_count = [0]

    def always_raise(self, **kwargs):
        module_call_count[0] += 1
        raise ValueError("Deliberately failing")

    predict = DummyModule("question -> answer", always_raise)
    refine = Refine(module=predict, N=3, reward_fn=lambda *_: 1.0, threshold=0.0, fail_count=0)

    with pytest.raises(ValueError, match="Deliberately failing"):
        refine(question="q")
    assert module_call_count[0] == 1


def _fail_first_module():
    counts = {}
    lock = threading.Lock()

    def fail_first(self, **kwargs):
        question = kwargs["question"]
        with lock:
            counts[question] = counts.get(question, 0) + 1
            attempt = counts[question]
        if attempt == 1:
            raise ValueError(f"first attempt for {question} fails")
        return self.predictor(**kwargs)

    return DummyModule("question -> answer", fail_first)


def test_refine_failure_budget_is_per_call():
    lm = DummyLM({"q1": {"answer": "a1"}, "q2": {"answer": "a2"}})
    dspy.configure(lm=lm)

    refine = Refine(module=_fail_first_module(), N=3, reward_fn=lambda *_: 0.0, threshold=0.0, fail_count=1)

    # Each call fails exactly once; a shared, mutable budget would exhaust on the second call.
    assert refine(question="q1").answer == "a1"
    assert refine(question="q2").answer == "a2"


def test_refine_failure_budget_is_per_call_across_threads():
    questions = [f"tq{i}" for i in range(4)]
    lm = DummyLM({question: {"answer": question.upper()} for question in questions})
    dspy.configure(lm=lm)

    refine = Refine(module=_fail_first_module(), N=3, reward_fn=lambda *_: 0.0, threshold=0.0, fail_count=1)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda question: refine(question=question), questions))

    assert [result.answer for result in results] == [question.upper() for question in questions]


def test_refine_all_attempts_failing_raises_last_error():
    dspy.configure(lm=DummyLM([]))

    def always_raise(self, **kwargs):
        raise RuntimeError("always fails")

    predict = DummyModule("question -> answer", always_raise)
    # The budget tolerates every failure; the all-failure path itself must raise
    # rather than return None.
    refine = Refine(module=predict, N=3, reward_fn=lambda *_: 1.0, threshold=0.0, fail_count=5)

    with pytest.raises(RuntimeError, match="always fails"):
        refine(question="q")


@pytest.mark.asyncio
async def test_refine_async_all_attempts_failing_raises_last_error():
    class AsyncAlwaysFail(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = Predict("question -> answer")

        async def aforward(self, **kwargs):
            raise RuntimeError("always fails")

    refine = Refine(module=AsyncAlwaysFail(), N=3, reward_fn=lambda *_: 1.0, threshold=0.0, fail_count=5)

    # `dspy.configure` is forbidden from a non-owner async task; scope the settings instead.
    with dspy.context(lm=DummyLM([])):
        with pytest.raises(RuntimeError, match="always fails"):
            await refine.acall(question="q")


def test_refine_critic_failure_does_not_consume_budget_and_prior_advice_persists():
    adapter = TrackingChatAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "use the right answer"}},
            {"answer": "still wrong"},
            {"answer": "not the critic format"},  # Second critic call fails to parse.
            {"answer": "third"},
        ]
    )
    dspy.configure(lm=lm, adapter=adapter)

    predict = DummyModule("question -> answer", call_predictor)
    # fail_count=0: if the critic failure consumed the budget, this call would raise.
    refine = Refine(module=predict, N=3, reward_fn=lambda *_: 0.0, threshold=1.0, fail_count=0)
    result = refine(question="q")

    assert result.answer == "wrong"
    module_calls = adapter.module_calls()
    assert len(module_calls) == 3
    assert HINT_FIELD_NAME not in module_calls[0]["inputs"]
    assert module_calls[1]["inputs"][HINT_FIELD_NAME] == "use the right answer"
    # The next attempt proceeds with the advice that already exists.
    assert module_calls[2]["inputs"][HINT_FIELD_NAME] == "use the right answer"


def test_refine_reward_fn_error_counts_as_attempt_failure():
    dspy.configure(lm=DummyLM([{"answer": "a"}]))

    predict = DummyModule("question -> answer", call_predictor)

    def bad_reward(kwargs, pred):
        raise RuntimeError("reward exploded")

    refine = Refine(module=predict, N=3, reward_fn=bad_reward, threshold=1.0, fail_count=0)

    with pytest.raises(RuntimeError, match="reward exploded"):
        refine(question="q")


def test_refine_hint_persists_across_failed_hinted_attempt():
    adapter = TrackingChatAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "the advice"}},
            {"answer": "third"},
        ]
    )
    dspy.configure(lm=lm, adapter=adapter)
    module_call_count = [0]

    def fail_second_call(self, **kwargs):
        module_call_count[0] += 1
        if module_call_count[0] == 2:
            raise ValueError("Deliberately failing")
        return self.predictor(**kwargs)

    predict = DummyModule("question -> answer", fail_second_call)
    refine = Refine(module=predict, N=3, reward_fn=lambda *_: 0.0, threshold=1.0)
    refine(question="q")

    # Attempt 2 was hinted but failed before reaching the LM; attempt 3 still
    # receives the same advice.
    module_calls = adapter.module_calls()
    assert len(module_calls) == 2
    assert module_calls[1]["inputs"][HINT_FIELD_NAME] == "the advice"


def test_refine_zero_predictor_module_resamples_without_critic():
    lm = DummyLM([])
    dspy.configure(lm=lm)

    refine = Refine(module=NoPredictorModule(), N=3, reward_fn=lambda *_: 0.0, threshold=1.0)
    result = refine(question="q")

    assert result.answer == "static"
    # No predictors to hint: the critic is never constructed and no LM call is made.
    assert len(lm.history) == 0


def test_refine_zero_predictor_module_without_any_lm_raises_clear_error():
    refine = Refine(module=NoPredictorModule(), N=3, reward_fn=lambda *_: 0.0, threshold=1.0)

    with pytest.raises(ValueError, match="No LM is loaded"):
        refine(question="q")


def test_refine_exports_winning_trace_remapped_to_original_predictor():
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "answer right"}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm)

    predict = DummyModule("question -> answer", call_predictor)
    refine = Refine(
        module=predict, N=3, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )

    with dspy.context(trace=[]):
        refine(question="q")
        trace = dspy.settings.trace
        # Only the winning attempt's entry is exported; no OfferFeedback entry.
        assert len(trace) == 1
        traced_predictor, traced_inputs, traced_outputs = trace[0]
        # Remapped to the predictor the caller owns, so demo bootstrapping
        # attaches to the original program.
        assert traced_predictor is predict.predictor
        assert HINT_FIELD_NAME not in traced_inputs
        assert traced_inputs["question"] == "q"
        assert traced_outputs.answer == "right"


def test_refine_safe_when_parent_trace_is_none():
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "answer right"}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm)

    predict = DummyModule("question -> answer", call_predictor)
    refine = Refine(
        module=predict, N=3, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )

    with dspy.context(trace=None):
        result = refine(question="q")

    assert result.answer == "right"


def test_refine_critic_trajectory_contains_no_hint():
    adapter = TrackingChatAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "advice one"}},
            {"answer": "still wrong"},
            {"discussion": "d", "advice": {"predictor": "advice two"}},
            {"answer": "third"},
        ]
    )
    dspy.configure(lm=lm, adapter=adapter)

    predict = DummyModule("question -> answer", call_predictor)
    refine = Refine(module=predict, N=3, reward_fn=lambda *_: 0.0, threshold=1.0)
    refine(question="q")

    critic_calls = adapter.critic_calls()
    assert len(critic_calls) == 2
    # The second critic reviews a hinted attempt, but sees the same trajectory
    # shape as the first: no hint field.
    assert HINT_FIELD_NAME not in critic_calls[1]["inputs"]["program_trajectory"]


def test_refine_dump_state_unchanged_by_hinted_run():
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "answer right"}},
            {"answer": "right"},
        ]
    )
    dspy.configure(lm=lm)

    predict = DummyModule("question -> answer", call_predictor)
    refine = Refine(
        module=predict, N=3, reward_fn=lambda _, pred: 1.0 if pred.answer == "right" else 0.0, threshold=1.0
    )

    state_before = json.dumps(refine.dump_state(), sort_keys=True, default=repr)
    refine(question="q")
    state_after = json.dumps(refine.dump_state(), sort_keys=True, default=repr)

    assert state_before == state_after


def test_refine_constructs_with_exec_defined_module_and_reward():
    namespace = {"dspy": dspy, "Predict": Predict}
    exec(
        textwrap.dedent(
            """
            def reward(kwargs, pred):
                return 1.0 if pred.answer == "right" else 0.0

            class ExecModule(dspy.Module):
                def __init__(self):
                    super().__init__()
                    self.predictor = Predict("question -> answer")

                def forward(self, **kwargs):
                    return self.predictor(**kwargs)
            """
        ),
        namespace,
    )
    dspy.configure(lm=DummyLM([{"answer": "right"}]))

    refine = Refine(module=namespace["ExecModule"](), N=2, reward_fn=namespace["reward"], threshold=1.0)

    assert "<source unavailable for" in refine.module_code
    assert "<source unavailable for" in refine.reward_fn_code
    assert refine(question="q").answer == "right"


@pytest.mark.asyncio
async def test_refine_async_hinted_retry_awaits_async_reward_fn():
    adapter = TrackingChatAdapter()
    lm = DummyLM(
        [
            {"answer": "wrong"},
            {"discussion": "d", "advice": {"predictor": "say right"}},
            {"answer": "right"},
        ]
    )

    async def reward(kwargs, pred):
        return 1.0 if pred.answer == "right" else 0.0

    predict = AsyncDummyModule("question -> answer")
    refine = Refine(module=predict, N=3, reward_fn=reward, threshold=1.0)
    # `dspy.configure` is forbidden from a non-owner async task; scope the settings instead.
    with dspy.context(lm=lm, adapter=adapter):
        result = await refine.acall(question="q")

    assert result.answer == "right"
    module_calls = adapter.module_calls()
    assert len(module_calls) == 2
    assert module_calls[1]["inputs"][HINT_FIELD_NAME] == "say right"


def test_hinted_retry_prompt_matches_old_wrapper_formatting():
    # Before the rewrite, retries appended an undefaulted hint field at adapter-call
    # time and injected the value into `inputs`. The declared, defaulted field must
    # format identically.
    adapter = dspy.ChatAdapter()
    signature = dspy.make_signature("question -> answer")
    inputs = {"question": "q", HINT_FIELD_NAME: "advice text"}

    old_signature = signature.append(
        HINT_FIELD_NAME, dspy.InputField(desc="A hint to the module from an earlier run")
    )
    new_signature = signature.append(
        HINT_FIELD_NAME,
        dspy.InputField(desc="A hint to the module from an earlier run", default="advice text"),
        type_=str,
    )

    assert adapter.format(old_signature, [], inputs) == adapter.format(new_signature, [], inputs)
