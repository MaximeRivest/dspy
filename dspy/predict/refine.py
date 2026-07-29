import copy
import inspect
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import orjson

import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.predict.predict import Prediction
from dspy.signatures import InputField, OutputField, Signature

from .predict import Module

logger = logging.getLogger(__name__)

DEFAULT_HINT_FIELD = "hint"

# The optional keyword arguments Refine passes to a `feedback_fn` that declares them.
_FEEDBACK_OPTIONAL_KWARGS = ("trace", "module", "threshold", "hint_field")


@dataclass(frozen=True)
class Hint:
    """A non-authoritative nudge for the next refinement attempt.

    A hint is search-derived advice, not program intent. It may be wrong, it may be ignored, and it
    never changes what the program *means*: it is delivered only to predictors whose Signature
    explicitly declares an input field for it (see `Refine`'s `hint_field`). A hint that has no
    declared destination is recorded as ignored, never smuggled in through an Adapter or an
    undeclared field.

    Attributes:
        content: The advice text.
        target: The name of the predictor (as returned by `named_predictors()`) the hint is
            addressed to, or `None` to address every hint-capable predictor.
        authority: Always `"non_authoritative"`. Hints cannot be promoted to intent; the type
            admits no other value.
        scope: Always `"next_candidate"`. A hint applies to exactly one retry and is then
            discarded; it is never persisted into the module's state.
        provenance: Where the hint came from (e.g. the `feedback_fn` name or `"LMCritic"`), kept
            so refinement metadata never presents inferred advice as declared intent.
    """

    content: str
    target: str | None = None
    authority: Literal["non_authoritative"] = "non_authoritative"
    scope: Literal["next_candidate"] = "next_candidate"
    provenance: str = "unknown"


@dataclass
class RefinementAttempt:
    """Metadata for one attempt inside a `Refine` call."""

    rollout_id: int
    reward: float | None = None
    hints_applied: tuple[Hint, ...] = ()
    hints_ignored: tuple[Hint, ...] = ()
    error: str | None = None


@dataclass
class RefinementReport:
    """Search provenance for one `Refine` call.

    Attached to the returned `Prediction` as the `_refinement` attribute. It is deliberately not a
    field of the prediction: refinement bookkeeping is metadata about the search, not program
    output, so it stays out of the prediction's `_store`, out of serialization, and out of the
    trace.
    """

    attempts: list[RefinementAttempt] = field(default_factory=list)
    best_attempt: int | None = None
    best_reward: float | None = None


def hint_capable_predictors(module: Module, hint_field: str = DEFAULT_HINT_FIELD) -> list[tuple[str, Any]]:
    """Return the `(name, predictor)` pairs whose Signature declares the hint input field.

    Declaring the field is how a program author marks a predictor as open to search-time advice.
    Predictors that do not declare it are committed as-is: `Refine` may resample them but will not
    alter what they receive.
    """
    return [(name, pred) for name, pred in module.named_predictors() if hint_field in pred.signature.input_fields]


def _deliver_hints(module: Module, hints: list[Hint], hint_field: str) -> tuple[list[Hint], list[Hint]]:
    """Apply hints to a disposable module copy by filling the *declared* hint field's default.

    This happens in the program/search plane, before any Adapter sees the call: the effective
    candidate is the copied module whose declared `hint_field` now defaults to the hint content.
    `Predict` populates input-field defaults for inputs the caller omits, so a hint only takes
    effect when the module's own code does not supply the field — code outranks hints.

    Returns `(applied, ignored)`. Hints whose target does not exist or does not declare the field
    are ignored, never force-fitted.
    """
    targets = dict(hint_capable_predictors(module, hint_field))
    applied, ignored = [], []
    for hint in hints:
        if hint.target is None:
            names = list(targets)
        elif hint.target in targets:
            names = [hint.target]
        else:
            names = []
        if not names:
            ignored.append(hint)
            continue
        for name in names:
            predictor = targets[name]
            fields = copy.deepcopy(predictor.signature.fields)
            fields[hint_field].default = hint.content
            predictor.signature = Signature(fields, predictor.signature.instructions)
        applied.append(hint)
    return applied, ignored


def _normalize_hints(raw: Any, provenance: str | None) -> list[Hint]:
    """Normalize a `feedback_fn` return value into a list of `Hint` objects with provenance."""
    provenance = provenance or "unknown"
    if raw is None:
        return []
    if isinstance(raw, Hint):
        return [raw]
    if isinstance(raw, str):
        return [Hint(content=raw, provenance=provenance)] if raw.strip() else []
    if isinstance(raw, dict):
        hints = []
        for target, content in raw.items():
            if content is None:
                continue
            content = str(content)
            if not content.strip() or content.strip().upper() in ("N/A", "NA"):
                continue
            hints.append(Hint(content=content, target=target, provenance=provenance))
        return hints
    if isinstance(raw, (list, tuple)):
        hints = []
        for item in raw:
            hints.extend(_normalize_hints(item, provenance))
        return hints
    raise TypeError(
        "feedback_fn must return None, a str, a dict mapping predictor names to advice, a "
        f"dspy.Hint, or a list of these, but returned {type(raw).__name__}."
    )


def _optional_feedback_params(fn: Callable) -> frozenset[str]:
    """Which of the optional keyword arguments the feedback callable declares."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return frozenset()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return frozenset(_FEEDBACK_OPTIONAL_KWARGS)
    return frozenset(k for k in _FEEDBACK_OPTIONAL_KWARGS if k in params)


class OfferHints(Signature):
    """
    A program was run once and its output scored below the target. You are given the program
    author's declared objective, the declared signature of each module, and the evidence of this
    one run. Propose concrete, self-contained advice for each listed module to use on an
    independent retry with the same or similar inputs. The retried module will not see its own
    history, so the advice must stand alone. Your advice is a fallible suggestion derived from one
    observation, not a rule: never contradict the declared objective or the modules' declared
    signatures. If a module needs no advice, write N/A.
    """

    objective: str = InputField(desc="The program author's own statement of what a good output looks like")
    modules_defn: str = InputField(desc="The declared definition of each module in the program, including its I/O")
    program_inputs: str = InputField(desc="The inputs to the program run being analyzed")
    program_trajectory: str = InputField(desc="The trajectory of the program's execution, showing each module's I/O")
    program_outputs: str = InputField(desc="The outputs of the program run being analyzed")
    target_threshold: float | None = InputField(desc="The target threshold for the reward")
    reward_value: float = InputField(desc="The reward assigned to this run's outputs")
    module_names: list[str] = InputField(desc="The modules that can receive advice on the retry")
    discussion: str = OutputField(desc="Discussion of where each listed module likely fell short, if it did")
    advice: dict[str, str] = OutputField(
        desc="For each listed module, concrete self-contained advice for the retry, or N/A if none is needed"
    )


class LMCritic:
    """Opt-in LM critic usable as `Refine`'s `feedback_fn`.

    Unlike the retired automatic critic, `LMCritic` performs no source-code archaeology: it never
    reads the program's or the reward function's source. What the reward *means* must be declared
    by the author through `objective` — a scalar reward does not explain why a run failed, and the
    critic does not pretend to reverse-engineer that explanation from code. The critic sees only
    declared intent (the objective and each module's signature) and evidence from the scored run
    (inputs, trajectory, outputs, reward, threshold), and returns per-module advice for the
    hint-capable predictors.

    Args:
        objective: Human-authored statement of what a good output looks like. Required.
        lm: Optional LM to run the critic on. Defaults to the ambient `dspy.settings.lm`.
        hint_field: The declared input field name that marks a predictor as hint-capable.

    Example:
        ```python
        critic = dspy.LMCritic(objective="The answer must be a single word.")
        refine = dspy.Refine(module=qa, N=3, reward_fn=one_word, threshold=1.0, feedback_fn=critic)
        ```
    """

    def __init__(self, objective: str, lm=None, hint_field: str = DEFAULT_HINT_FIELD):
        if not objective or not str(objective).strip():
            raise ValueError(
                "LMCritic requires a human-authored `objective` describing what a good output looks "
                "like. A scalar reward does not explain why a run failed, and LMCritic will not "
                "guess by reading source code."
            )
        self.objective = objective
        self.lm = lm
        self.hint_field = hint_field

    def __call__(
        self,
        inputs: dict,
        prediction: Prediction,
        reward: float,
        *,
        trace=None,
        module: Module | None = None,
        threshold: float | None = None,
    ) -> dict[str, str] | None:
        if module is None or trace is None:
            return None
        module_names = [name for name, _ in hint_capable_predictors(module, self.hint_field)]
        if not module_names:
            return None

        predictor2name = {predictor: name for name, predictor in module.named_predictors()}
        trajectory = [
            {"module_name": predictor2name.get(p, repr(p)), "inputs": i, "outputs": dict(o)} for p, i, o in trace
        ]

        def dump(value):
            return orjson.dumps(recursive_mask(value), option=orjson.OPT_INDENT_2).decode()

        critic = dspy.Predict(OfferHints)
        kwargs = {
            "objective": self.objective,
            "modules_defn": inspect_modules(module),
            "program_inputs": dump(inputs),
            "program_trajectory": dump(trajectory),
            "program_outputs": dump(dict(prediction)),
            "target_threshold": threshold,
            "reward_value": reward,
            "module_names": module_names,
        }
        if self.lm is not None:
            with dspy.context(lm=self.lm):
                return critic(**kwargs).advice
        return critic(**kwargs).advice


class Refine(Module):
    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
        *,
        feedback_fn: Callable | None = None,
        hint_field: str = DEFAULT_HINT_FIELD,
    ):
        """
        Runs a module up to N times with different rollout IDs at `temperature=1.0` and returns the
        first prediction whose reward meets `threshold`, or the highest-reward prediction.

        Refinement is explicit. By default (no `feedback_fn`), Refine only resamples and selects —
        exactly `dspy.BestOfN`. A scalar reward does not explain *why* an attempt failed, so Refine
        no longer manufactures an explanation automatically. To guide retries:

        1. Declare where advice may go: give a predictor's Signature an input field named
           `hint_field` (default `"hint"`), e.g. `hint: str = dspy.InputField(default="")`. Only
           predictors that declare the field can receive hints; the rest are committed code that
           Refine will resample but never alter. If no predictor declares the field, `feedback_fn`
           is never invoked and Refine degrades to pure resampling.
        2. Say where advice comes from: pass `feedback_fn(inputs, prediction, reward)`, returning
           `None`, a hint string, a `{predictor_name: advice}` dict, a `dspy.Hint`, or a list of
           these. The callable may additionally declare any of the keyword-only parameters `trace`
           (the attempt's isolated trace), `module` (the attempt's disposable module copy),
           `threshold`, and `hint_field`, which Refine then supplies. For an opt-in LM critic, pass
           `dspy.LMCritic(objective=...)`.

        Hints are non-authoritative and scoped to the next attempt. They are delivered by filling
        the declared hint field's default value on the per-attempt module copy — never by wrapping
        or subclassing an Adapter, and never by appending undeclared fields to a Signature. Because
        delivery uses input-field defaults, a value the module's own code passes for the field
        always wins over a hint. Feedback generation runs inside an isolated trace, so critic calls
        never appear in the caller's trace; the delivered hint does appear in the winning trace as
        a value of the field the Signature genuinely declares.

        The returned `Prediction` carries a `RefinementReport` on its `_refinement` attribute:
        per-attempt rollout IDs, rewards, applied/ignored hints with provenance, and errors.

        Args:
            module (Module): The module to refine.
            N (int): The maximum number of times to run the module.
            reward_fn (Callable): The reward function `(inputs_dict, prediction) -> float`.
            threshold (float): The reward threshold at which an attempt is accepted immediately.
            fail_count (Optional[int]): The number of times attempts may fail before raising.
                The budget is local to each call; it does not deplete across calls or threads.
            feedback_fn (Optional[Callable]): Explicit source of retry hints; see above.
            hint_field (str): The declared input field name that marks a predictor as hint-capable.

        Examples:
            ```python
            import dspy

            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

            class HintedQA(dspy.Signature):
                question: str = dspy.InputField()
                hint: str = dspy.InputField(default="", desc="Advice from an earlier attempt.")
                answer: str = dspy.OutputField()

            qa = dspy.ChainOfThought(HintedQA)

            def one_word_answer(args, pred):
                return 1.0 if len(pred.answer.split()) == 1 else 0.0

            def one_word_feedback(args, pred, reward):
                if reward < 1.0:
                    return f"Your previous answer had {len(pred.answer.split())} words. Answer with exactly one word."

            refine = dspy.Refine(
                module=qa, N=3, reward_fn=one_word_answer, threshold=1.0, feedback_fn=one_word_feedback
            )
            result = refine(question="What is the capital of Belgium?")
            ```
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided
        self.hint_field = hint_field

        if feedback_fn is None:
            self.feedback_fn = None
            self._feedback_params = frozenset()
            self._feedback_provenance = None
            logger.warning(
                "dspy.Refine no longer generates LM feedback automatically. Without `feedback_fn`, "
                "Refine resamples and returns the best attempt, exactly like dspy.BestOfN. To guide "
                "retries, declare a %r input field on the predictors that may receive advice and pass "
                "`feedback_fn=` (e.g. `dspy.LMCritic(objective=...)` for an opt-in LM critic).",
                hint_field,
            )
        else:
            self._feedback_params = _optional_feedback_params(feedback_fn)
            self._feedback_provenance = getattr(feedback_fn, "__qualname__", None) or type(feedback_fn).__name__
            # Wrapped in a lambda to keep module-valued feedback out of named_parameters().
            self.feedback_fn = lambda *args, **kw: feedback_fn(*args, **kw)
            if not hint_capable_predictors(module, hint_field):
                logger.warning(
                    "dspy.Refine received `feedback_fn`, but no predictor in the wrapped module "
                    "declares a %r input field, so hints have nowhere to go. Refine will only "
                    "resample. Declare the field (e.g. `hint: str = dspy.InputField(default='')`) "
                    "on the predictors that should receive advice.",
                    hint_field,
                )

    def _prepare_attempt(self, lm, rid, hints):
        """Build the disposable module copy for one attempt and apply pending hints to it."""
        mod = self.module.deepcopy()
        mod.set_lm(lm.copy(rollout_id=rid, temperature=1.0))
        applied, ignored = _deliver_hints(mod, hints, self.hint_field) if hints else ([], [])
        return mod, tuple(applied), tuple(ignored)

    def _gather_hints(self, inputs, prediction, reward, trace, module) -> list[Hint]:
        """Invoke `feedback_fn` in an isolated trace and normalize its result into hints.

        A failing feedback function degrades Refine to plain resampling for the next attempt; it
        neither consumes the failure budget nor discards the already-scored attempt.
        """
        if self.feedback_fn is None or not hint_capable_predictors(self.module, self.hint_field):
            return []
        optional = {"trace": trace, "module": module, "threshold": self.threshold, "hint_field": self.hint_field}
        kwargs = {k: v for k, v in optional.items() if k in self._feedback_params}
        try:
            with dspy.context(trace=[]):
                raw = self.feedback_fn(inputs, prediction, reward, **kwargs)
        except Exception as e:
            logger.warning("Refine: feedback_fn raised (%s); continuing with plain resampling.", e)
            return []
        return _normalize_hints(raw, self._feedback_provenance)

    def _finalize(self, best_pred, best_trace, report):
        if best_trace and dspy.settings.trace is not None:
            dspy.settings.trace.extend(best_trace)
        if best_pred is not None:
            try:
                best_pred._refinement = report
            except Exception:  # a module may return an object that rejects attributes
                logger.debug("Refine: could not attach RefinementReport to the returned prediction.")
        return best_pred

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_reward = None, None, -float("inf")
        report = RefinementReport()
        hints: list[Hint] = []
        fail_budget = self.fail_count  # local budget: no cross-call or cross-thread mutation

        for idx in range(self.N):
            rid = start + idx
            mod, applied, ignored = self._prepare_attempt(lm, rid, hints)
            hints = []  # scope="next_candidate": a hint is applied to exactly one attempt
            attempt = RefinementAttempt(rollout_id=rid, hints_applied=applied, hints_ignored=ignored)
            report.attempts.append(attempt)

            try:
                with dspy.context(trace=[]):
                    outputs = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)
            except Exception as e:
                attempt.error = str(e)
                logger.warning("Refine: attempt with rollout id %s failed: %s", rid, e)
                if idx > fail_budget:
                    raise e
                fail_budget -= 1
                continue

            attempt.reward = reward

            if reward > best_reward:
                best_reward, best_pred, best_trace = reward, outputs, trace
                report.best_attempt, report.best_reward = idx, reward

            if self.threshold is not None and reward >= self.threshold:
                break

            if idx == self.N - 1:
                break

            # Outside the attempt's try/except: a feedback contract violation raises loudly instead
            # of being miscounted as a failed attempt, and a feedback *runtime* error is downgraded
            # to plain resampling inside `_gather_hints`.
            hints = self._gather_hints(kwargs, outputs, reward, trace, mod)

        return self._finalize(best_pred, best_trace, report)

    async def aforward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_reward = None, None, -float("inf")
        report = RefinementReport()
        hints: list[Hint] = []
        fail_budget = self.fail_count  # local budget: no cross-call or cross-thread mutation

        for idx in range(self.N):
            rid = start + idx
            mod, applied, ignored = self._prepare_attempt(lm, rid, hints)
            hints = []  # scope="next_candidate": a hint is applied to exactly one attempt
            attempt = RefinementAttempt(rollout_id=rid, hints_applied=applied, hints_ignored=ignored)
            report.attempts.append(attempt)

            try:
                with dspy.context(trace=[]):
                    outputs = await mod.acall(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)
            except Exception as e:
                attempt.error = str(e)
                logger.warning("Refine: attempt with rollout id %s failed: %s", rid, e)
                if idx > fail_budget:
                    raise e
                fail_budget -= 1
                continue

            attempt.reward = reward

            if reward > best_reward:
                best_reward, best_pred, best_trace = reward, outputs, trace
                report.best_attempt, report.best_reward = idx, reward

            if self.threshold is not None and reward >= self.threshold:
                break

            if idx == self.N - 1:
                break

            # NOTE: feedback_fn is invoked synchronously, even on the async path. Feedback contract
            # violations raise; feedback runtime errors degrade to plain resampling.
            hints = self._gather_hints(kwargs, outputs, reward, trace, mod)

        return self._finalize(best_pred, best_trace, report)


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
