from typing import Callable

from dspy.predict.candidate_search import CandidateSearch
from dspy.predict.predict import Module, Prediction


class BestOfN(Module):
    """Runs a module up to N times and returns the highest-scoring prediction.

    At each attempt, the module is called on a fresh deep copy with a different
    rollout ID at ``temperature=1.0`` to encourage output diversity; attempts
    are independent (no feedback flows between them). The attempt with the
    highest reward is returned, or the first attempt whose reward meets or
    exceeds ``threshold`` (whichever comes first).

    Only the winning attempt's execution is added to the caller's trace,
    expressed in terms of the wrapped module's own predictors. The full search
    provenance (every attempt, its rollout ID, reward, and error, if any) is
    attached to the returned prediction and can be read with
    ``dspy.predict.candidate_search.search_record(prediction)``.

    Args:
        module: The DSPy module to run repeatedly.
        N: Maximum number of attempts.
        reward_fn: A callable that takes the input kwargs dict and a
            ``Prediction``, and returns a scalar float reward score.
        threshold: If an attempt's reward is at or above this value, that
            prediction is returned immediately without further attempts.
        fail_count: Number of failed attempts tolerated per call before the
            last error is raised. Defaults to ``N`` if not provided. If every
            attempt fails, the last error is raised rather than returning
            ``None``.

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
        reward_fn: Callable[[dict, Prediction], float],
        threshold: float,
        fail_count: int | None = None,
    ):
        self.module = module
        self.reward_fn = lambda *args: reward_fn(*args)  # to prevent this from becoming a parameter
        self.threshold = threshold
        self.N = N
        self.fail_count = fail_count or N  # default to N if fail_count is not provided

    def _build_search(self) -> CandidateSearch:
        # BestOfN is the independent-sampling policy: no `propose` hook, so no
        # feedback flows between attempts.
        return CandidateSearch(
            module=self.module,
            num_candidates=self.N,
            reward_fn=self.reward_fn,
            threshold=self.threshold,
            fail_count=self.fail_count,
        )

    def forward(self, **kwargs):
        search = self._build_search()
        return search.finalize(search.run(kwargs))

    async def aforward(self, **kwargs):
        search = self._build_search()
        return search.finalize(await search.arun(kwargs))
