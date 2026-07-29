from typing import Any, Callable

import dspy
from dspy.predict.predict import Module, Prediction
from dspy.predict.refine import interpret_reward


class BestOfN(Module):
    """Runs a module up to N times and returns the highest-scoring prediction.

    At each attempt, the module is called with a different rollout ID at
    ``temperature=1.0`` to encourage output diversity. The attempt with the
    highest reward is returned, or the first attempt whose reward meets or
    exceeds ``threshold`` (whichever comes first).

    Args:
        module: The DSPy module to run repeatedly.
        N: Maximum number of attempts.
        reward_fn: A callable that takes the input kwargs dict and a
            ``Prediction``, and returns a scalar float reward score or a
            score-bearing object such as ``dspy.Prediction(score=...)``
            (any ``feedback`` it carries is ignored here; use ``dspy.Refine``
            to act on feedback between attempts).
        threshold: If an attempt's reward is at or above this value, that
            prediction is returned immediately without further attempts.
        fail_count: Number of allowed failures before raising an exception.
            Defaults to ``N`` if not provided.

    Example:
        >>> import dspy
        >>> dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
        >>> qa = dspy.ChainOfThought("question -> answer")
        >>> def one_word_answer(args, pred):
        ...     return 1.0 if len(pred.answer.split()) == 1 else 0.0
        >>> best_of_3 = dspy.BestOfN(module=qa, N=3, reward_fn=one_word_answer, threshold=1.0)
        >>> result = best_of_3(question="What is the capital of Belgium?")
        >>> print(result.answer)  # Brussels
    """

    def __init__(
        self,
        module: Module,
        N: int,  # noqa: N803
        reward_fn: Callable[[dict, Prediction], float | Any],
        threshold: float,
        fail_count: int | None = None,
    ):
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided

    def forward(self, **kwargs):
        lm = self.module.get_lm() or dspy.settings.lm
        start = lm.kwargs.get("rollout_id", 0)
        rollout_ids = [start + i for i in range(self.N)]
        best_pred, best_trace, best_reward = None, None, -float("inf")

        for idx, rid in enumerate(rollout_ids):
            lm_ = lm.copy(rollout_id=rid, temperature=1.0)
            mod = self.module.deepcopy()
            mod.set_lm(lm_)

            try:
                with dspy.context(trace=[]):
                    pred = mod(**kwargs)
                    trace = dspy.settings.trace.copy()

                    # NOTE: Not including the trace of reward_fn.
                    reward, _ = interpret_reward(self.reward_fn(kwargs, pred))

                if reward > best_reward:
                    best_reward, best_pred, best_trace = reward, pred, trace

                if reward >= self.threshold:
                    break

            except Exception as e:
                print(f"BestOfN: Attempt {idx + 1} failed with rollout id {rid}: {e}")
                if idx > self.fail_count:
                    raise e
                self.fail_count -= 1

        if best_trace:
            dspy.settings.trace.extend(best_trace)
        return best_pred
