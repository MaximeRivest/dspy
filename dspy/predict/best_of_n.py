import inspect
import logging
from typing import Callable

import dspy
from dspy.predict.predict import Module, Prediction

logger = logging.getLogger(__name__)


class BestOfN(Module):
    """Run a module up to N times and return the highest-scoring prediction.

    Each attempt runs a deep copy of `module` with a distinct rollout ID at `temperature=1.0` so
    the LM resamples instead of replaying a cached response. `reward_fn(kwargs, prediction)` scores
    each attempt; the first prediction whose reward meets `threshold` is returned immediately,
    otherwise the highest-scoring one wins.

    Args:
        module: The module to run repeatedly.
        N: The maximum number of attempts.
        reward_fn: A callable `(kwargs, prediction) -> float` that scores an attempt against the
            call's inputs. In `aforward`, an async `reward_fn` is awaited.
        threshold: Return immediately once an attempt's reward meets this value.
        fail_count: Number of failed attempts tolerated per call before the most recent error is
            raised. Defaults to `N`; `fail_count=0` raises on the first failure. Independently of
            this budget, if every attempt fails the last error is raised instead of returning
            `None`.

    Examples:
        ```python
        import dspy

        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

        qa = dspy.ChainOfThought("question -> answer")

        def one_word_answer(args, pred):
            return 1.0 if len(pred.answer.split()) == 1 else 0.0

        best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)

        result = best_of_3(question="What is the capital of Belgium?").answer
        # Returns: Brussels
        ```

    See Also:
        [`dspy.Refine`][dspy.Refine]: the same sampling loop with LM feedback between attempts.
    """

    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ):
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = N if fail_count is None else fail_count
        self._reward_fn_is_async = inspect.iscoroutinefunction(reward_fn)

    def forward(self, **kwargs):
        lm = self._resolve_lm()
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_map, best_reward = None, None, None, -float("inf")
        failures, last_error = 0, None

        for idx in range(self.N):
            mod, predictor2name = self._prepare_attempt(lm, start + idx)

            try:
                with dspy.context(trace=[]):
                    pred = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, pred)
            except Exception as e:
                failures += 1
                last_error = e
                logger.warning("BestOfN: attempt %d/%d failed with rollout id %s: %s", idx + 1, self.N, start + idx, e)
                if failures > self.fail_count:
                    raise e
                continue

            logger.debug("BestOfN: attempt %d/%d achieved reward %s.", idx + 1, self.N, reward)

            if reward > best_reward:
                best_reward, best_pred, best_trace, best_map = reward, pred, trace, predictor2name

            if reward >= self.threshold:
                break

        return self._conclude(best_pred, best_trace, best_map, last_error)

    async def aforward(self, **kwargs):
        lm = self._resolve_lm()
        start = lm.kwargs.get("rollout_id", 0)
        best_pred, best_trace, best_map, best_reward = None, None, None, -float("inf")
        failures, last_error = 0, None

        for idx in range(self.N):
            mod, predictor2name = self._prepare_attempt(lm, start + idx)

            try:
                with dspy.context(trace=[]):
                    pred = await mod.acall(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward = self.reward_fn(kwargs, pred)
                    if self._reward_fn_is_async:
                        reward = await reward
            except Exception as e:
                failures += 1
                last_error = e
                logger.warning("BestOfN: attempt %d/%d failed with rollout id %s: %s", idx + 1, self.N, start + idx, e)
                if failures > self.fail_count:
                    raise e
                continue

            logger.debug("BestOfN: attempt %d/%d achieved reward %s.", idx + 1, self.N, reward)

            if reward > best_reward:
                best_reward, best_pred, best_trace, best_map = reward, pred, trace, predictor2name

            if reward >= self.threshold:
                break

        return self._conclude(best_pred, best_trace, best_map, last_error)

    def _resolve_lm(self):
        lm = (self.module.get_lm() if self.module.predictors() else None) or dspy.settings.lm
        if lm is None:
            raise ValueError(
                "No LM is loaded. Configure one with `dspy.configure(lm=dspy.LM(...))` or set an LM on the "
                "wrapped module with `module.set_lm(...)`."
            )
        return lm

    def _prepare_attempt(self, lm, rollout_id):
        """Build the throwaway candidate for one attempt: a deep copy with a diversified LM."""
        mod = self.module.deepcopy()
        mod.set_lm(lm.copy(rollout_id=rollout_id, temperature=1.0))
        return mod, {predictor: name for name, predictor in mod.named_predictors()}

    def _conclude(self, best_pred, best_trace, best_map, last_error):
        if best_trace:
            self._export_trace(best_trace, best_map)
        if best_pred is None and last_error is not None:
            raise last_error
        return best_pred

    def _export_trace(self, trace, predictor2name):
        """Append the winning attempt's trace to the caller's trace.

        The per-attempt predictor copies are mapped back to the original module's predictors, so
        trace consumers (e.g. demo bootstrapping) see the program the caller actually owns.
        """
        parent_trace = dspy.settings.trace
        if parent_trace is None:
            return
        original_by_name = dict(self.module.named_predictors())
        predictor2name = predictor2name or {}
        # Like the previous `trace.extend(...)`, this appends without honoring
        # `settings.max_trace_size`; enforcing the cap here is a separate change.
        for predictor, inputs, outputs in trace:
            original = original_by_name.get(predictor2name.get(predictor), predictor)
            parent_trace.append((original, inputs, outputs))
