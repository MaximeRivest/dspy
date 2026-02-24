from typing import Callable

import dspy
from dspy.predict.predict import Module, Prediction


class BestOfN(Module):
    """Run a module up to *N* times and return the best prediction.

    ``BestOfN`` is an inference-time scaling strategy. It executes the wrapped
    module multiple times (each with a distinct ``rollout_id`` at
    ``temperature=1.0``) and keeps the prediction with the highest reward. If any
    attempt meets or exceeds the ``threshold``, execution stops early.

    Example:

        ```python
        import dspy

        dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

        qa = dspy.ChainOfThought("question -> answer")

        def one_word_answer(args, pred):
            return 1.0 if len(pred.answer.split()) == 1 else 0.0

        best_of_3 = dspy.BestOfN(
            module=qa, N=3, reward_fn=one_word_answer, threshold=1.0,
        )

        result = best_of_3(question="What is the capital of Belgium?")
        print(result.answer)
        ```

    Args:
        module: The DSPy module to run repeatedly.
        N: Maximum number of attempts.
        reward_fn: A callable ``(kwargs, prediction) -> float`` that scores each
            prediction. Higher is better.
        threshold: If a prediction's reward meets or exceeds this value, return
            it immediately without remaining attempts.
        fail_count: Maximum tolerated exceptions before re-raising. Defaults to
            ``N`` when not provided.

    Note:
        **Rollout IDs:** Each attempt uses a different ``rollout_id`` passed to the
        underlying LM, ensuring diverse outputs even with the same prompt.

        **Early stopping:** Execution stops as soon as any prediction reaches the
        ``threshold``, which can save significant compute.

        **Comparison with Refine:** ``BestOfN`` generates candidates independently.
        ``dspy.Refine`` goes further by generating feedback after each failed
        attempt and injecting it as a hint into the next try.
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
                    reward = self.reward_fn(kwargs, pred)

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
