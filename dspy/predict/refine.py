import inspect
import logging
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

import orjson

import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.predict.predict import Prediction
from dspy.signatures import InputField, OutputField, Signature

from .predict import Module

logger = logging.getLogger(__name__)

HINT_FIELD_NAME = "hint_"

FEEDBACK_POLICIES = ("auto", "eval", "none")

_PROVENANCE_LABELS = {
    "reward_feedback": "feedback returned by the reward function",
    "llm_critic": "an automatic LM critique of an earlier attempt",
}


def interpret_reward(value: Any) -> tuple[float, str | None]:
    """Interpret a reward result as an explicit ``(score, feedback)`` pair.

    Reward functions may return:

    - a plain ``float`` (or ``int``/``bool``) score, in which case there is no feedback;
    - any object with a ``score`` attribute and an optional ``feedback`` attribute, e.g.
      ``dspy.Prediction(score=0.0, feedback="The answer was too long.")`` — the same shape as
      GEPA's ``ScoreWithFeedback``;
    - a mapping with a ``"score"`` key and an optional ``"feedback"`` key.

    Returns:
        A ``(score, feedback)`` tuple where ``feedback`` is ``None`` when the eval did not
        provide any (empty strings count as no feedback).
    """
    score = getattr(value, "score", None)
    if score is None and isinstance(value, dict) and "score" in value:
        score = value["score"]
    if score is not None:
        feedback = getattr(value, "feedback", None)
        if feedback is None and isinstance(value, dict):
            feedback = value.get("feedback")
        return float(score), (str(feedback) if feedback else None)
    try:
        return float(value), None
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"reward_fn must return a float or a score-bearing object (e.g. "
            f"dspy.Prediction(score=..., feedback=...)), but returned {value!r}."
        ) from e


@dataclass(frozen=True)
class Hint:
    """A non-authoritative suggestion produced during refinement search.

    A hint is a fallible search hypothesis, not program intent: applying it must never change
    what the wrapped program *means*, only nudge how the next candidate attempts the same task.
    Hints are interpreted by ``Refine`` into the next candidate's declared signature (as an
    explicit, provenance-marked ``hint_`` input field) before any Adapter sees the call.

    Attributes:
        content: The hint text delivered to the targeted predictor(s).
        target: Name of the targeted predictor within the candidate (as reported by
            ``named_predictors()``), or ``None`` to target every predictor.
        authority: Always ``"non_authoritative"``; hints may be ignored by the model and are
            never promoted to program intent.
        scope: Lifetime of the hint. ``"next_candidate"`` means it applies only to the next
            attempt's disposable deep copy and is never persisted.
        provenance: Where the hint came from: ``"reward_feedback"`` (the eval said why the
            score was low) or ``"llm_critic"`` (the ``OfferFeedback`` fallback inferred it).
    """

    content: str
    target: str | None = None
    authority: str = "non_authoritative"
    scope: str = "next_candidate"
    provenance: str = "unknown"


@dataclass(frozen=True)
class RefinementAttempt:
    """Record of one refinement attempt, kept as metadata rather than in the program trace.

    Attributes:
        rollout_id: The rollout ID the attempt ran with.
        score: The interpreted reward score, or ``None`` when the attempt raised.
        feedback: Feedback the reward function returned for this attempt, if any.
        hints: The hints that were applied to this attempt's candidate.
        error: ``repr`` of the exception when the attempt failed, else ``None``.
    """

    rollout_id: int
    score: float | None
    feedback: str | None
    hints: tuple[Hint, ...] = ()
    error: str | None = None


class OfferFeedback(Signature):
    """
    In the discussion, assign blame to each module that contributed to the final reward being below the threshold, if
    any. Then, prescribe concrete advice of how the module should act on its future input when we retry the process, if
    it were to receive the same or similar inputs. If a module is not to blame, the advice should be N/A.
    The module will not see its own history, so it needs to rely on entirely concrete and actionable advice from you
    to avoid the same mistake on the same or similar inputs.
    """

    program_code: str = InputField(desc="The code of the program that we are analyzing")
    modules_defn: str = InputField(desc="The definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program that we are analyzing")
    program_trajectory: str = InputField(desc="The trajectory of the program's execution, showing each module's I/O")
    program_outputs: str = InputField(desc="The outputs of the program that we are analyzing")
    reward_code: str = InputField(
        desc="The source code of the reward function, as evidence of what the evaluation checks. "
        "It may not fully explain the score (e.g. closures or LM judges it calls are not shown)."
    )
    target_threshold: float = InputField(desc="The target threshold for the reward function")
    reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made "
        "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of"
        "scenario in the future. If the module is not to blame, write N/A."
    )


class Refine(Module):
    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float | Any],
        threshold: float,
        fail_count: int | None = None,
        feedback_policy: str = "auto",
    ):
        """
        Refines a module by running it up to N times with different rollout IDs at `temperature=1.0`
        and returns the best prediction.

        This module runs the provided module multiple times with varying rollout identifiers and selects
        either the first prediction that exceeds the specified threshold or the one with the highest reward.
        Between attempts, feedback about the failed attempt is turned into non-authoritative hints for
        the next attempt.

        **Feedback-bearing rewards.** `reward_fn` may return a plain float, or an explicit
        score-plus-feedback result: any object with a `score` attribute and an optional `feedback`
        attribute, e.g. `dspy.Prediction(score=0.0, feedback="The answer was two words; answer in one.")`
        (the same shape GEPA metrics use). When the eval provides feedback, that feedback is used
        directly as the hint for the next attempt and no LM critic is invoked.

        **Feedback policy.** When the eval returns only a score, whether an LM critic
        (`OfferFeedback`) is invoked to infer per-predictor advice is an explicit policy:

        - `"auto"` (default): use eval-provided feedback when present; otherwise fall back to the
          LM critic. This matches the historical behavior for float-returning rewards.
        - `"eval"`: only use feedback the eval itself provides; never invoke the LM critic.
        - `"none"`: never hint; `Refine` degrades to resampling and selecting the best attempt.

        **How hints are delivered.** Hints are interpreted *above* the Adapter: each attempt runs on
        a disposable deep copy of the module, and a hinted attempt's targeted predictors get an
        explicit `hint_` input field appended to their signature, whose description names the hint's
        provenance and non-authoritative status and whose default value carries the hint text.
        Adapters — resolved through the normal precedence (call > predictor > settings > default) —
        simply render an ordinary declared field; no adapter is wrapped, subclassed, or mutated. The
        original module is never modified.

        **Trace and metadata.** Only the winning attempt's trace is merged into the caller's trace,
        with the `hint_` input stripped so the trace matches the wrapped program's declared
        signatures. The full search record — per-attempt scores, feedback, hints (with provenance),
        and errors — is attached to the returned prediction as `prediction._refinement`, a tuple of
        `RefinementAttempt`.

        Args:
            module (Module): The module to refine.
            N (int): The maximum number of times to run the module.
            reward_fn (Callable): The reward function. Takes the call kwargs and the prediction and
                returns a float score or a score-bearing object with optional `feedback` (see above).
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of failed attempts tolerated before the
                failing exception is re-raised. Defaults to `N`. If every attempt fails, the last
                exception is raised rather than returning `None`.
            feedback_policy (str, optional): One of `"auto"`, `"eval"`, or `"none"` (see above).

        Examples:
            ```python
            import dspy

            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

            # Define a QA module with chain of thought
            qa = dspy.ChainOfThought("question -> answer")

            # A reward that says *why* it scored low, so no LM critic is needed:
            def one_word_answer(args, pred):
                if len(pred.answer.split()) == 1:
                    return dspy.Prediction(score=1.0, feedback="")
                return dspy.Prediction(score=0.0, feedback="The answer must be exactly one word.")

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)

            # Use the refined module
            result = best_of_3(question="What is the capital of Belgium?").answer
            # Returns: Brussels
            ```
        """
        if feedback_policy not in FEEDBACK_POLICIES:
            raise ValueError(
                f"feedback_policy must be one of {FEEDBACK_POLICIES}, but received: {feedback_policy!r}."
            )
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided
        self.feedback_policy = feedback_policy
        self.module_code = _safe_getsource(module.__class__)
        self.reward_fn_code = _safe_getsource(reward_fn)
        if self.reward_fn_code == _SOURCE_UNAVAILABLE:
            self.reward_fn_code = _safe_getsource(reward_fn.__class__)

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_score = None, None, -float("inf")
        hints: list[Hint] = []
        attempts: list[RefinementAttempt] = []
        failures, last_error = 0, None

        for idx in range(self.N):
            rid = start + idx
            mod = self._make_candidate(lm, rid, hints)
            predictor2name = {predictor: name for name, predictor in mod.named_predictors()}
            module_names = [name for name, _ in mod.named_predictors()]

            try:
                with dspy.context(trace=[]):
                    outputs = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    score, feedback = interpret_reward(self.reward_fn(kwargs, outputs))

                attempts.append(RefinementAttempt(rollout_id=rid, score=score, feedback=feedback, hints=tuple(hints)))

                if score > best_score:
                    best_score, best_pred, best_trace = score, outputs, trace

                if self.threshold is not None and score >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                if self._wants_critic(feedback):
                    advise_kwargs = self._build_advise_kwargs(
                        mod, kwargs, outputs, trace, predictor2name, module_names, score
                    )
                    try:
                        # The critic is search machinery, not program execution: keep it out of
                        # the caller's trace.
                        with dspy.context(trace=[]):
                            advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice
                        hints = self._hints_from_advice(advice, module_names)
                    except Exception:
                        logger.warning("Refine: feedback critic failed; retrying without new hints.", exc_info=True)
                else:
                    hints = self._hints_from_feedback(feedback)

            except Exception as e:
                failures += 1
                last_error = e
                attempts.append(RefinementAttempt(rollout_id=rid, score=None, feedback=None,
                                                  hints=tuple(hints), error=repr(e)))
                logger.warning("Refine: attempt with rollout id %s failed: %s", rid, e)
                if failures > self.fail_count:
                    raise e

        return self._finalize(best_pred, best_trace, attempts, last_error)

    async def aforward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_score = None, None, -float("inf")
        hints: list[Hint] = []
        attempts: list[RefinementAttempt] = []
        failures, last_error = 0, None

        for idx in range(self.N):
            rid = start + idx
            mod = self._make_candidate(lm, rid, hints)
            predictor2name = {predictor: name for name, predictor in mod.named_predictors()}
            module_names = [name for name, _ in mod.named_predictors()]

            try:
                with dspy.context(trace=[]):
                    outputs = await mod.acall(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    score, feedback = interpret_reward(self.reward_fn(kwargs, outputs))

                attempts.append(RefinementAttempt(rollout_id=rid, score=score, feedback=feedback, hints=tuple(hints)))

                if score > best_score:
                    best_score, best_pred, best_trace = score, outputs, trace

                if self.threshold is not None and score >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                if self._wants_critic(feedback):
                    advise_kwargs = self._build_advise_kwargs(
                        mod, kwargs, outputs, trace, predictor2name, module_names, score
                    )
                    try:
                        with dspy.context(trace=[]):
                            advice = (await dspy.Predict(OfferFeedback).acall(**advise_kwargs)).advice
                        hints = self._hints_from_advice(advice, module_names)
                    except Exception:
                        logger.warning("Refine: feedback critic failed; retrying without new hints.", exc_info=True)
                else:
                    hints = self._hints_from_feedback(feedback)

            except Exception as e:
                failures += 1
                last_error = e
                attempts.append(RefinementAttempt(rollout_id=rid, score=None, feedback=None,
                                                  hints=tuple(hints), error=repr(e)))
                logger.warning("Refine: attempt with rollout id %s failed: %s", rid, e)
                if failures > self.fail_count:
                    raise e

        return self._finalize(best_pred, best_trace, attempts, last_error)

    def _make_candidate(self, lm, rollout_id: int, hints: list[Hint]) -> Module:
        """Build the disposable candidate for one attempt: a deep copy of the wrapped module with
        a diversified LM and, when hinted, an explicit hint field applied to its signatures."""
        mod = self.module.deepcopy()
        mod.set_lm(lm.copy(rollout_id=rollout_id, temperature=1.0))
        if hints:
            apply_hints(mod, hints)
        return mod

    def _wants_critic(self, feedback: str | None) -> bool:
        """Whether the LM critic fallback should run, per the explicit feedback policy."""
        return self.feedback_policy == "auto" and not feedback

    def _hints_from_feedback(self, feedback: str | None) -> list[Hint]:
        if self.feedback_policy == "none" or not feedback:
            return []
        return [Hint(content=feedback, target=None, provenance="reward_feedback")]

    def _hints_from_advice(self, advice: dict[str, str] | None, module_names: list[str]) -> list[Hint]:
        hints = []
        for name, text in (advice or {}).items():
            if name not in module_names or not text or text.strip().upper() == "N/A":
                continue
            hints.append(Hint(content=text, target=name, provenance="llm_critic"))
        return hints

    def _build_advise_kwargs(self, mod, kwargs, outputs, trace, predictor2name, module_names, score) -> dict[str, str]:
        modules = {"program_code": self.module_code, "modules_defn": inspect_modules(mod)}
        trajectory = [
            {"module_name": predictor2name.get(p, "<unnamed>"), "inputs": i, "outputs": dict(o)} for p, i, o in trace
        ]
        trajectory = {
            "program_inputs": kwargs,
            "program_trajectory": trajectory,
            "program_outputs": dict(outputs),
        }
        reward = {
            "reward_code": self.reward_fn_code,
            "target_threshold": self.threshold,
            "reward_value": score,
        }
        advise_kwargs = dict(**modules, **trajectory, **reward, module_names=module_names)
        # only dumps if it's a list or dict
        return {
            k: v if isinstance(v, str) else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
            for k, v in advise_kwargs.items()
        }

    def _finalize(self, best_pred, best_trace, attempts, last_error):
        if best_pred is None and last_error is not None:
            raise last_error
        if best_trace and dspy.settings.trace is not None:
            # The program trace shows the winning run as the wrapped program declares it: the
            # hint input is refinement metadata, not part of the program's signatures.
            dspy.settings.trace.extend(strip_hints_from_trace(best_trace))
        if best_pred is not None:
            best_pred._refinement = tuple(attempts)
        return best_pred


def apply_hints(module: Module, hints: list[Hint]) -> None:
    """Interpret hints into a candidate module before execution.

    Each targeted predictor's signature gains an explicit ``hint_`` input field whose description
    names the hint's provenance and non-authoritative status, and whose default value carries the
    hint text. The candidate is thereby a fully declared program: adapters render an ordinary
    field and are never wrapped or mutated. Only call this on a disposable copy — hints have
    ``next_candidate`` scope and must never be persisted onto the original module.
    """
    for name, predictor in module.named_predictors():
        selected = [h for h in hints if h.target is None or h.target == name]
        if not selected:
            continue
        content = "\n\n".join(h.content for h in selected)
        sources = sorted({_PROVENANCE_LABELS.get(h.provenance, h.provenance) for h in selected})
        desc = (
            f"A non-authoritative hint carried over from an earlier attempt at this task "
            f"(source: {'; '.join(sources)}). It is a fallible suggestion, not part of the task "
            "definition: follow the instructions above, and ignore the hint wherever it conflicts with them."
        )
        predictor.signature = predictor.signature.delete(HINT_FIELD_NAME).append(
            HINT_FIELD_NAME, InputField(default=content, desc=desc), type_=str
        )


def strip_hints_from_trace(trace):
    """Return a copy of a trace with the ``hint_`` refinement input removed from each step."""
    return [
        (predictor, {k: v for k, v in inputs.items() if k != HINT_FIELD_NAME}, prediction)
        for predictor, inputs, prediction in trace
    ]


_SOURCE_UNAVAILABLE = "<source unavailable>"


def _safe_getsource(obj) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return _SOURCE_UNAVAILABLE


def inspect_modules(program):
    separator = "-" * 80
    output = [separator]

    for _, (name, predictor) in enumerate(program.named_predictors()):
        signature = predictor.signature
        instructions = textwrap.dedent(signature.instructions)
        instructions = ("\n" + "\t" * 2).join([""] + instructions.splitlines())

        output.append(f"Module {name}")
        output.append("\n\tInput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.input_fields).splitlines()))
        output.append("\tOutput Fields:")
        output.append(("\n" + "\t" * 2).join([""] + get_field_description_string(signature.output_fields).splitlines()))
        output.append(f"\tOriginal Instructions: {instructions}")
        output.append(separator)

    return "\n".join([o.strip("\n") for o in output])


def recursive_mask(o):
    # If the object is already serializable, return it.
    try:
        orjson.dumps(o)
        return o
    except TypeError:
        pass

    # If it's a dictionary, apply recursively to its values.
    if isinstance(o, dict):
        return {k: recursive_mask(v) for k, v in o.items()}
    # If it's a list, apply recursively.
    elif isinstance(o, list):
        return [recursive_mask(v) for v in o]
    # If it's a tuple, apply recursively.
    elif isinstance(o, tuple):
        return tuple(recursive_mask(v) for v in o)
    # Otherwise, replace it with a placeholder string (or use repr(o)).
    else:
        return f"<non-serializable: {type(o).__name__}>"
