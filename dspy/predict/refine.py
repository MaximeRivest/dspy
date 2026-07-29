import inspect
import logging
import textwrap
from typing import Callable

import orjson

import dspy
from dspy.adapters.utils import get_field_description_string
from dspy.predict.candidate_search import Attempt, CandidateSearch, Hint
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


def _safe_getsource(obj, fallback_name: str) -> str:
    try:
        return inspect.getsource(obj)
    except (TypeError, OSError):
        return f"<source unavailable for {fallback_name}>"


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
        When an attempt scores below the threshold, an LM critic reads the attempt (program source, per-module
        trajectory, reward-function source, and score) and proposes per-module advice. That advice is a
        non-authoritative hint: it is realized into the *next* attempt's throwaway program copy as a defaulted
        `hint_` input field on each blamed predictor's signature, so the retry's calls are fully defined before
        any Adapter formats them. Adapter configuration and resolution are never modified.

        The wrapped module is never mutated, and only the winning attempt's execution is added to the caller's
        trace (with the transient `hint_` input stripped and trace entries mapped back to the wrapped module's
        own predictors). The full search provenance, including every attempt, its reward, and the hints it
        received, is attached to the returned prediction and can be read with
        `dspy.predict.candidate_search.search_record(prediction)`.

        If the wrapped module exposes no `dspy.Predict` call sites (for example, a leaf whose implementation is
        opaque state), refinement degrades to resample-and-select: no critic is invoked and no hint is forced
        into the program.

        Args:
            module (Module): The module to refine.
            N (int): The maximum number of times to run the module.
            reward_fn (Callable): The reward function.
            threshold (float): The threshold for the reward function.
            fail_count (Optional[int], optional): The number of failed attempts tolerated per call before the
                last error is raised. Defaults to `N`. Independently of this budget, if every attempt fails,
                the last error is raised rather than returning `None`.

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
        self.module_code = _safe_getsource(module.__class__, type(module).__name__)
        try:
            self.reward_fn_code = inspect.getsource(reward_fn)
        except (TypeError, OSError):
            self.reward_fn_code = _safe_getsource(reward_fn.__class__, type(reward_fn).__name__)

    def _build_search(self) -> CandidateSearch:
        return CandidateSearch(
            module=self.module,
            num_candidates=self.N,
            reward_fn=self.reward_fn,
            threshold=self.threshold,
            fail_count=self.fail_count,
            propose=self._propose_hints,
        )

    def forward(self, **kwargs):
        search = self._build_search()
        return search.finalize(search.run(kwargs))

    async def aforward(self, **kwargs):
        search = self._build_search()
        return search.finalize(await search.arun(kwargs))

    def _propose_hints(self, attempt: Attempt, inputs: dict, candidate: Module) -> list[Hint]:
        """Turn one below-threshold attempt into per-module hints via the `OfferFeedback` critic.

        The critic's advice is search feedback, not intent: entries of "N/A" (or advice for unknown
        module names) never reach the program, and applied hints remain inspectable on the search
        record. When the candidate exposes no predictor call sites, the critic is skipped entirely
        and the search resamples.
        """
        named_predictors = candidate.named_predictors()
        if not named_predictors:
            logger.info("Refine: module exposes no predictor call sites; retrying by resampling only.")
            return []

        predictor2name = {id(predictor): name for name, predictor in named_predictors}
        module_names = [name for name, _ in named_predictors]

        trajectory = [
            {"module_name": predictor2name.get(id(p), type(p).__name__), "inputs": i, "outputs": dict(o)}
            for p, i, o in attempt.trace
        ]
        advise_kwargs = {
            "program_code": self.module_code,
            "modules_defn": inspect_modules(candidate),
            "program_inputs": inputs,
            "program_trajectory": trajectory,
            "program_outputs": dict(attempt.prediction),
            "reward_code": self.reward_fn_code,
            "target_threshold": self.threshold,
            "reward_value": attempt.reward,
            "module_names": module_names,
        }
        # only dumps if it's a list or dict
        advise_kwargs = {
            k: v if isinstance(v, str) else orjson.dumps(recursive_mask(v), option=orjson.OPT_INDENT_2).decode()
            for k, v in advise_kwargs.items()
        }

        advice = dspy.Predict(OfferFeedback)(**advise_kwargs).advice
        if not isinstance(advice, dict):
            logger.warning("Refine: feedback predictor returned %r instead of an advice dict; ignoring.", advice)
            return []

        provenance = {
            "source": "dspy.Refine/OfferFeedback",
            "attempt": attempt.index,
            "rollout_id": attempt.rollout_id,
            "reward": attempt.reward,
        }
        return [
            Hint(target=name, content=content, provenance=dict(provenance))
            for name, content in advice.items()
            if isinstance(content, str) and content.strip() and content.strip().lower() != "n/a"
        ]


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
