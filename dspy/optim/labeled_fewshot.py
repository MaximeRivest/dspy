"""LabeledFewShot: the trivial mutation — k labeled demos per predictor.

No scoring, no search: sample k trainset examples and assign them as
every predictor's demos. The mutation surface is exactly
`predictor.demos`; the compiled artifact carries the same records as
component `3b_demos`. This is the plumbing proof every other optimizer
stands on.
"""

from __future__ import annotations

import random
from typing import Any

from dspy.modules.module import Module
from dspy.optim.base import Checkpointer, Optimizer, check_trainset

__all__ = ["LabeledFewShot"]


class LabeledFewShot(Optimizer):
    """Attach k labeled trainset examples to every predictor.

    Args:
        k: Demos per predictor (fewer when the trainset is smaller).
        seed: Seed for the sampling rng; one rng draws sequentially
            across predictors, so each predictor can get a different
            sample deterministically.

    Examples:
        ```python
        optimizer = dspy.LabeledFewShot(k=4)
        compiled = optimizer.compile(program, trainset=trainset)
        assert all(len(p.demos) == 4 for _, p in compiled.named_predictors())
        ```
    """

    def __init__(self, k: int = 16, *, seed: int = 0):
        self.k = k
        self.seed = seed

    def compile(
        self,
        program: Module,
        *,
        trainset: Any,
        sample: bool = True,
        checkpoint_dir: str | None = None,
    ) -> Module:
        """Assign demos and return the program (mutated in place).

        Args:
            program: The program to optimize.
            trainset: `dspy.Example` values with declared inputs.
            sample: Sample demos with the seeded rng (True) or take the
                first k in trainset order (False).
            checkpoint_dir: When given, save the accepted candidate as
                `<checkpoint_dir>/candidate-000/` plus a `scores.json`.
        """
        trainset = check_trainset(trainset)
        rng = random.Random(self.seed)
        for _, predictor in program.named_predictors():
            count = min(self.k, len(trainset))
            predictor.demos = rng.sample(trainset, count) if sample else list(trainset[:count])
        if checkpoint_dir is not None:
            Checkpointer(checkpoint_dir).accept(program, score=None, label="labeled-fewshot")
        return program
