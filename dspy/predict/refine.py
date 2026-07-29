import inspect
import logging
from typing import Callable

import orjson

import dspy
from dspy.predict.parameter import Hint
from dspy.predict.predict import Prediction
from dspy.signatures import InputField, OutputField, Signature

from .predict import Module

logger = logging.getLogger(__name__)


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
    reward_code: str = InputField(desc="The code of the reward function that we are analyzing")
    target_threshold: float = InputField(desc="The target threshold for the reward function")
    reward_value: float = InputField(desc="The reward value assigned to the program's outputs")
    module_names: list[str] = InputField(desc="The names of the modules in the program, for which we seek advice")
    discussion: str = OutputField(desc="Discussing blame of where each module went wrong, if it did")
    advice: dict[str, str] = OutputField(
        desc="For each module, describe very concretely, in this order: the specific scenarios in which it has made "
        "mistakes in the past and what each mistake was, followed by what it should do differently in that kind of"
        "scenario in the future. If the module is not to blame, write N/A."
    )


def _source_of(obj) -> str:
    try:
        return inspect.getsource(obj)
    except (TypeError, OSError):
        try:
            return inspect.getsource(obj.__class__)
        except (TypeError, OSError):
            return "<source unavailable>"


class Refine(Module):
    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ):
        """
        Refines a module by running it up to N times with different rollout IDs at `temperature=1.0`
        and returns the best prediction.

        This module runs the provided module multiple times with varying rollout identifiers and selects
        either the first prediction that exceeds the specified threshold or the one with the highest reward.
        If no prediction meets the threshold, it asks an LM critic for per-unit advice and carries that
        advice into the next attempt.

        Feedback targets *declared refinement units* — the ``dspy.predict.parameter.Parameter`` leaves
        reported by ``module.named_refinement_units()`` whose ``accepts_hint()`` returns ``True``
        (``dspy.Predict`` opts in by default). Advice is delivered as an explicit, non-authoritative
        ``dspy.predict.parameter.Hint`` with provenance, installed on a throwaway deep copy of the
        module; each hinted unit interprets it into its own effective call before the resolved adapter
        formats it. Refine never wraps or mutates adapters, never mutates the original module, and
        keeps hints out of the recorded trace. If a program declares no hint-accepting unit (for
        example, a leaf whose predictors are derived runtime state), Refine degrades to
        resample-and-select and does not manufacture a feedback channel.

        The trace observed by the caller contains only the winning attempt. Search provenance —
        per-attempt rollout ids, rewards, errors, and the hints applied — is attached to the returned
        prediction and available via ``prediction.get_refinement()``.

        Args:
            module (Module): The module to refine.
            N (int): The maximum number of times to run the module.
            reward_fn (Callable): The reward function.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of failed attempts tolerated before the
                last error is raised. Defaults to ``N``. If every attempt fails, the last error is
                raised even when the budget is not exhausted.

        Examples:
            ```python
            import dspy

            dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

            # Define a QA module with chain of thought
            qa = dspy.ChainOfThought("question -> answer")

            # Define a reward function that checks for one-word answers
            def one_word_answer(args, pred):
                return 1.0 if len(pred.answer.split()) == 1 else 0.0

            # Create a refined module that tries up to 3 times
            best_of_3 = dspy.Refine(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)

            # Use the refined module
            result = best_of_3(question="What is the capital of Belgium?").answer
            # Returns: Brussels
            ```
        """
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count if fail_count is not None else N
        self.module_code = _source_of(module.__class__)
        self.reward_fn_code = _source_of(reward_fn)

    def _resolve_lm(self):
        """Resolve the LM used to diversify attempts.

        A module may expose zero runtime predictors (e.g. a leaf that constructs
        its predictors from its own state, or makes no LM calls); fall back to
        the configured LM instead of requiring a runtime-predictor traversal.
        """
        lm = self.module.get_lm() if self.module.predictors() else None
        lm = lm or dspy.settings.lm
        if lm is None:
            raise ValueError(
                "No LM is loaded. Please configure one with `dspy.configure(lm=dspy.LM(...))` or set it on "
                "the wrapped module."
            )
        return lm

    def _make_candidate(self, lm, rollout_id, hints):
        """Build one throwaway candidate: a deep copy with its LM set and hints installed.

        Returns the candidate module and the mapping of hint-accepting refinement
        units (name -> unit). Hints addressed to units that do not accept them are
        dropped, never rerouted through transport.
        """
        lm_ = lm.copy(rollout_id=rollout_id, temperature=1.0)
        mod = self.module.deepcopy()
        mod.set_lm(lm_)

        hintable = {name: unit for name, unit in mod.named_refinement_units() if unit.accepts_hint()}
        for name, hint in (hints or {}).items():
            unit = hintable.get(name)
            if unit is not None and unit.accepts_hint(hint):
                unit.apply_hint(hint)
        return mod, hintable

    def _advice_kwargs(self, mod, hintable, kwargs, trace, outputs, reward):
        """Assemble the critic's evidence. Trajectory entries are labeled with runtime
        predictor names (execution evidence); advice is requested only for declared,
        hint-accepting units."""
        predictor2name = {id(predictor): name for name, predictor in mod.named_predictors()}
        trajectory = [
            {
                "module_name": predictor2name.get(id(p), f"<runtime:{type(p).__name__}>"),
                "inputs": i,
                "outputs": dict(o),
            }
            for p, i, o in trace
        ]
        advise_kwargs = {
            "program_code": self.module_code,
            "modules_defn": inspect_modules(mod),
            "program_inputs": kwargs,
            "program_trajectory": trajectory,
            "program_outputs": dict(outputs),
            "reward_code": self.reward_fn_code,
            "target_threshold": self.threshold,
            "reward_value": reward,
            "module_names": list(hintable),
        }
        # only dumps if it's a list or dict
        return {
            k: v if isinstance(v, str) else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
            for k, v in advise_kwargs.items()
        }

    def _hints_from_advice(self, advice, hintable, reward, rollout_id):
        """Turn critic advice into explicit non-authoritative hints with provenance.

        Advice for undeclared units and "N/A" verdicts is dropped: search feedback
        is a fallible hint, not intent, and only declared flexible state may
        receive it.
        """
        hints = {}
        for name, content in (advice or {}).items():
            if name not in hintable:
                logger.debug("Refine: dropping advice for undeclared or non-hintable unit %r.", name)
                continue
            if not content or content.strip().upper() in ("N/A", "NA"):
                continue
            hints[name] = Hint(
                target=name,
                content=content,
                provenance={
                    "source": "dspy.Refine/OfferFeedback",
                    "rollout_id": rollout_id,
                    "reward": reward,
                    "threshold": self.threshold,
                },
            )
        return hints or None

    def forward(self, **kwargs):
        lm = self._resolve_lm()
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_reward = None, None, -float("inf")
        hints = None
        attempts = []
        failures, last_error = 0, None
        fail_budget = self.fail_count

        for idx in range(self.N):
            rid = start + idx
            mod, hintable = self._make_candidate(lm, rid, hints)

            try:
                with dspy.context(trace=[]):
                    outputs = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)

                attempts.append({"rollout_id": rid, "reward": reward, "hints": list((hints or {}).values())})

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, outputs, trace

                if self.threshold is not None and reward >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                if not hintable:
                    logger.info(
                        "Refine: no refinement unit accepts hints; retrying without feedback "
                        "(resample-and-select only)."
                    )
                    hints = None
                    continue

                advise_kwargs = self._advice_kwargs(mod, hintable, kwargs, trace, outputs, reward)
                # The critic is search-plane work: it runs outside the caller's trace, and
                # its output is recorded as hint provenance rather than program history.
                with dspy.context(trace=[]):
                    advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice
                hints = self._hints_from_advice(advice, hintable, reward, rid)

            except Exception as e:
                failures += 1
                last_error = e
                attempts.append({"rollout_id": rid, "error": str(e), "hints": list((hints or {}).values())})
                logger.warning("Refine: attempt with rollout id %s failed: %s", rid, e)
                if failures > fail_budget:
                    raise e

        if best_pred is None:
            if last_error is not None:
                raise last_error
            return None

        if hasattr(best_pred, "set_refinement"):
            best_pred.set_refinement({"attempts": attempts, "best_reward": best_reward})
        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred

    async def aforward(self, **kwargs):
        lm = self._resolve_lm()
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_reward = None, None, -float("inf")
        hints = None
        attempts = []
        failures, last_error = 0, None
        fail_budget = self.fail_count

        for idx in range(self.N):
            rid = start + idx
            mod, hintable = self._make_candidate(lm, rid, hints)

            try:
                with dspy.context(trace=[]):
                    outputs = await mod.acall(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)

                attempts.append({"rollout_id": rid, "reward": reward, "hints": list((hints or {}).values())})

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, outputs, trace

                if self.threshold is not None and reward >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                if not hintable:
                    logger.info(
                        "Refine: no refinement unit accepts hints; retrying without feedback "
                        "(resample-and-select only)."
                    )
                    hints = None
                    continue

                advise_kwargs = self._advice_kwargs(mod, hintable, kwargs, trace, outputs, reward)
                with dspy.context(trace=[]):
                    advice = (await dspy.Predict(OfferFeedback).acall(**advise_kwargs)).advice
                hints = self._hints_from_advice(advice, hintable, reward, rid)

            except Exception as e:
                failures += 1
                last_error = e
                attempts.append({"rollout_id": rid, "error": str(e), "hints": list((hints or {}).values())})
                logger.warning("Refine: attempt with rollout id %s failed: %s", rid, e)
                if failures > fail_budget:
                    raise e

        if best_pred is None:
            if last_error is not None:
                raise last_error
            return None

        if hasattr(best_pred, "set_refinement"):
            best_pred.set_refinement({"attempts": attempts, "best_reward": best_reward})
        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred


def inspect_modules(program):
    """Describe each declared refinement unit of ``program`` for the critic."""
    separator = "-" * 80
    output = [separator]

    for name, unit in program.named_refinement_units():
        output.append(f"Module {name}")
        output.append(unit.describe_refinement_unit())
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
