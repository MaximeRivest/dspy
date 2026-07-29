import inspect
import logging
import textwrap
from typing import Callable

import orjson

import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.predict.predict import Prediction
from dspy.signatures import InputField, OutputField, Signature

from .predict import Module

logger = logging.getLogger(__name__)

# Name of the input field through which feedback reaches a retry. The field is appended to the
# per-attempt copy's predictor signatures (never the wrapped module's), and is stripped from the
# trace that `Refine` reports to its caller.
HINT_FIELD_NAME = "hint_"


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


def _safe_getsource(obj) -> str:
    """Best-effort source retrieval for the feedback prompt.

    Source code is evidence shown to the feedback LM, not a requirement for running the wrapped
    module, so unavailable source (REPL/exec-defined classes, builtins, partials) degrades to a
    placeholder instead of failing construction.
    """
    targets = [obj]
    if not inspect.isclass(obj):
        targets.append(type(obj))
    for target in targets:
        try:
            return inspect.getsource(target)
        except (OSError, TypeError):
            continue
    name = getattr(obj, "__name__", type(obj).__name__)
    return f"<source unavailable for {name}>"


def _apply_feedback_hints(candidate: Module, advice: dict[str, str]) -> None:
    """Interpret critic advice into the throwaway candidate, above the adapter boundary.

    The advice produced by ``OfferFeedback`` is a fallible, non-authoritative search hint. It is
    applied by appending an explicit ``hint_`` input field, whose default value is that predictor's
    slice of the advice, to each predictor's signature on the per-attempt deep copy. The effective
    candidate is therefore fully defined before any adapter formats a call: adapters stay pure
    renderers (never wrapped or subclassed), each predictor's own adapter resolution is untouched,
    and the wrapped module is never mutated.

    Only ``dspy.Predict`` leaves discovered through ``named_predictors()`` receive hints; other
    kinds of sub-modules are resampled and selected but not hinted.
    """
    for name, predictor in candidate.named_predictors():
        predictor.signature = predictor.signature.append(
            HINT_FIELD_NAME,
            InputField(
                desc="A hint to the module from an earlier run",
                default=advice.get(name, "N/A"),
            ),
            type_=str,
        )


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
        If no prediction meets the threshold, it automatically generates feedback to improve future predictions.

        The feedback is a non-authoritative search hint: it is applied to the throwaway per-attempt copy of
        the module as an explicit `hint_` input field on each predictor's signature, so a retry is fully
        defined before any adapter formats a call. The wrapped module is never mutated, adapters are never
        wrapped or subclassed, and the hint is stripped from the trace reported to the caller. Only
        `dspy.Predict` leaves found via `named_predictors()` receive hints; other sub-modules are resampled
        and selected but not hinted.

        Args:
            module (Module): The module to refine.
            N (int): The maximum number of times to run the module.
            reward_fn (Callable): The reward function.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of failed attempts tolerated per call before
                the most recent error is raised. Defaults to `N`. Independently of this budget, if every
                attempt fails, the last error is raised instead of returning `None`.

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
        self.fail_count = fail_count or N  # default to N if fail_count is not provided
        self.module_code = _safe_getsource(module.__class__)
        self.reward_fn_code = _safe_getsource(reward_fn)

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_map, best_reward = None, None, None, -float("inf")
        advice, failures, last_error = None, 0, None

        for idx in range(self.N):
            mod, predictor2name = self._prepare_attempt(lm, start + idx, advice)

            try:
                with dspy.context(trace=[]):
                    outputs = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)

                logger.debug("Refine: attempt %d/%d achieved reward %s.", idx + 1, self.N, reward)

                if reward > best_reward:
                    best_reward, best_pred, best_trace, best_map = reward, outputs, trace, predictor2name

                if self.threshold is not None and reward >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                advise_kwargs = self._advice_kwargs(predictor2name, kwargs, outputs, trace, reward)
                # The critic call is refinement machinery, not part of the wrapped program:
                # keep it out of the caller's trace.
                with dspy.context(trace=[]):
                    advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice
                logger.debug("Refine: advice for each module: %s", advice)

            except Exception as e:
                failures += 1
                last_error = e
                logger.warning(
                    "Refine: attempt %d/%d failed with rollout id %s: %s", idx + 1, self.N, start + idx, e
                )
                if failures > self.fail_count:
                    raise e

        return self._conclude(best_pred, best_trace, best_map, last_error)

    async def aforward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_map, best_reward = None, None, None, -float("inf")
        advice, failures, last_error = None, 0, None

        for idx in range(self.N):
            mod, predictor2name = self._prepare_attempt(lm, start + idx, advice)

            try:
                with dspy.context(trace=[]):
                    outputs = await mod.acall(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, outputs)

                logger.debug("Refine: attempt %d/%d achieved reward %s.", idx + 1, self.N, reward)

                if reward > best_reward:
                    best_reward, best_pred, best_trace, best_map = reward, outputs, trace, predictor2name

                if self.threshold is not None and reward >= self.threshold:
                    break

                if idx == self.N - 1:
                    break

                advise_kwargs = self._advice_kwargs(predictor2name, kwargs, outputs, trace, reward)
                # The critic call is refinement machinery, not part of the wrapped program:
                # keep it out of the caller's trace.
                with dspy.context(trace=[]):
                    advice = (await dspy.Predict(OfferFeedback).acall(**advise_kwargs)).advice
                logger.debug("Refine: advice for each module: %s", advice)

            except Exception as e:
                failures += 1
                last_error = e
                logger.warning(
                    "Refine: attempt %d/%d failed with rollout id %s: %s", idx + 1, self.N, start + idx, e
                )
                if failures > self.fail_count:
                    raise e

        return self._conclude(best_pred, best_trace, best_map, last_error)

    def _prepare_attempt(self, lm, rollout_id, advice):
        """Build the throwaway candidate for one attempt.

        The candidate is a deep copy of the wrapped module with a diversified LM and, when
        advice is available, the feedback hints applied to its predictor signatures.
        """
        mod = self.module.deepcopy()
        mod.set_lm(lm.copy(rollout_id=rollout_id, temperature=1.0))
        if advice:
            _apply_feedback_hints(mod, advice)
        return mod, {predictor: name for name, predictor in mod.named_predictors()}

    def _advice_kwargs(self, predictor2name, kwargs, outputs, trace, reward):
        modules = {"program_code": self.module_code, "modules_defn": inspect_modules(self.module)}
        trajectory = [
            {"module_name": predictor2name.get(p, type(p).__name__), "inputs": i, "outputs": dict(o)}
            for p, i, o in trace
        ]
        trajectory = {
            "program_inputs": kwargs,
            "program_trajectory": trajectory,
            "program_outputs": dict(outputs),
        }
        reward_info = {
            "reward_code": self.reward_fn_code,
            "target_threshold": self.threshold,
            "reward_value": reward,
        }
        module_names = [name for name, _ in self.module.named_predictors()]

        advise_kwargs = dict(**modules, **trajectory, **reward_info, module_names=module_names)
        # only dumps if it's a list or dict
        return {
            k: v if isinstance(v, str) else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
            for k, v in advise_kwargs.items()
        }

    def _conclude(self, best_pred, best_trace, best_map, last_error):
        if best_trace:
            self._export_trace(best_trace, best_map)
        if best_pred is None and last_error is not None:
            raise last_error
        return best_pred

    def _export_trace(self, trace, predictor2name):
        """Append the winning attempt's trace to the caller's trace.

        The winning attempt is reported as if the wrapped module had produced the prediction
        directly: the refinement hint is stripped from recorded inputs and the per-attempt
        predictor copies are mapped back to the original module's predictors, so trace consumers
        (e.g. demo bootstrapping) see the program the caller actually owns.
        """
        parent_trace = dspy.settings.trace
        if parent_trace is None:
            return
        original_by_name = dict(self.module.named_predictors())
        predictor2name = predictor2name or {}
        for predictor, inputs, outputs in trace:
            name = predictor2name.get(predictor)
            original = original_by_name.get(name, predictor)
            inputs = {k: v for k, v in inputs.items() if k != HINT_FIELD_NAME}
            parent_trace.append((original, inputs, outputs))


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
