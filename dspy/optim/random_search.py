"""BootstrapFewShotWithRandomSearch: N seeded candidates, keep the argmax.

The candidate schedule is the classic one: seed -3 is zero-shot, -2 is
labeled demos only, -1 is the unshuffled bootstrap, and every seed >= 0
bootstraps over a seed-shuffled trainset with a seed-drawn demo budget.
Each candidate is a `ProgramState` proposed on the one live program,
scored by engine replay over the valset, and reverted; the first
strictly-best score is kept. Evaluation is sequential — candidates run
one after another, deterministically.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from dspy.core.example import Example
from dspy.modules.module import Module
from dspy.optim.base import (
    Checkpointer,
    Optimizer,
    apply_state,
    check_trainset,
    evaluate,
    snapshot_state,
)
from dspy.optim.bootstrap import BootstrapFewShot
from dspy.optim.labeled_fewshot import LabeledFewShot

__all__ = ["BootstrapFewShotWithRandomSearch"]


class BootstrapFewShotWithRandomSearch(Optimizer):
    """Search over seeded bootstrap candidates; keep the best scorer.

    Args:
        metric: `metric(example, prediction) -> bool | float`, used both
            to filter bootstrap traces and to score candidates.
        metric_threshold: Numeric acceptance bar for bootstrap traces.
        max_bootstrapped_demos: The demo-budget ceiling; shuffled
            candidates draw their own budget in [1, this] per seed.
        max_labeled_demos: Total demo cap per predictor.
        num_candidate_programs: Candidates with seed >= 0, on top of the
            three fixed ones (zero-shot, labeled-only, unshuffled).
        stop_at_score: Stop the search early at this score or better.
        seed: Seed for the rngs INSIDE each candidate's bootstrap; the
            candidate seeds themselves are the schedule (-3..N-1) and
            drive the shuffles and budgets.

    Attributes:
        trajectory: After `compile`, one record per evaluated candidate:
            `{"seed", "label", "score", "lm_calls", "accepted"}` (plus
            `"candidate"` when a checkpoint directory was written).

    Examples:
        ```python
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=exact_match, num_candidate_programs=4
        )
        compiled = optimizer.compile(program, trainset=trainset)
        ```
    """

    def __init__(
        self,
        metric: Callable[[Example, Any], Any],
        *,
        metric_threshold: float | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        num_candidate_programs: int = 16,
        stop_at_score: float | None = None,
        seed: int = 0,
    ):
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.num_candidate_programs = num_candidate_programs
        self.stop_at_score = stop_at_score
        self.seed = seed
        self.trajectory: list[dict[str, Any]] = []

    def compile(
        self,
        program: Module,
        *,
        trainset: Any,
        valset: Any = None,
        teacher: Module | None = None,
        checkpoint_dir: str | None = None,
    ) -> Module:
        """Search the candidate schedule; return the program at its best state.

        Args:
            program: The student program (mutated in place; the winning
                candidate's state is applied on return).
            trainset: `dspy.Example` values with declared inputs.
            valset: Examples to score candidates on; None scores on the
                trainset.
            teacher: Optional structurally identical teacher for the
                bootstrap candidates.
            checkpoint_dir: When given, every ACCEPTED candidate (each
                new best) is saved as `<checkpoint_dir>/candidate-NNN/`,
                with `scores.json` recording the trajectory.
        """
        trainset = check_trainset(trainset)
        devset = check_trainset(valset, name="valset") if valset is not None else trainset
        checkpointer = Checkpointer(checkpoint_dir) if checkpoint_dir is not None else None

        baseline = snapshot_state(program)
        best_score: float | None = None
        best_state = baseline
        self.trajectory = []
        for candidate_seed in range(-3, self.num_candidate_programs):
            apply_state(program, baseline)
            label = self._propose(program, candidate_seed, trainset, teacher)
            result = evaluate(program, devset, self.metric)
            entry: dict[str, Any] = {
                "seed": candidate_seed,
                "label": label,
                "score": result.score,
                "lm_calls": result.lm_calls,
                "accepted": False,
            }
            if best_score is None or result.score > best_score:
                best_score = result.score
                best_state = snapshot_state(program)
                entry["accepted"] = True
                if checkpointer is not None:
                    entry["candidate"] = checkpointer.accept(program, score=result.score, label=label)
            self.trajectory.append(entry)
            if self.stop_at_score is not None and result.score >= self.stop_at_score:
                break

        apply_state(program, best_state)
        return program

    def _propose(self, program: Module, seed: int, trainset: list[Example], teacher: Module | None) -> str:
        """Mutate the program into candidate `seed`'s state; return its label."""
        if seed == -3:
            for _, predictor in program.named_predictors():
                predictor.demos = []
            return "zero-shot"
        if seed == -2:
            LabeledFewShot(self.max_labeled_demos, seed=self.seed).compile(program, trainset=trainset)
            return "labeled-only"
        if seed == -1:
            shuffled, budget = list(trainset), self.max_bootstrapped_demos
            label = "bootstrap-unshuffled"
        else:
            shuffled = list(trainset)
            random.Random(seed).shuffle(shuffled)
            budget = random.Random(seed).randint(1, self.max_bootstrapped_demos)
            label = f"bootstrap-shuffle-{seed}"
        optimizer = BootstrapFewShot(
            metric=self.metric,
            metric_threshold=self.metric_threshold,
            max_bootstrapped_demos=budget,
            max_labeled_demos=self.max_labeled_demos,
            seed=self.seed,
        )
        optimizer.compile(program, trainset=shuffled, teacher=teacher)
        return label
