"""BootstrapFewShot: demos from the teacher's own metric-passing runs.

The teacher (the student itself by default, or a structurally identical
program you pass) replays the trainset THROUGH THE ENGINE. Each run's
per-predictor call records ride the prediction's
`_trajectory["predictor_calls"]` exhaust; when the run's prediction
passes the metric, those records become demos — real inputs, real
outputs, per predictor path. The student then gets up to
`max_bootstrapped_demos` of these augmented demos plus labeled trainset
examples up to `max_labeled_demos`.
"""

from __future__ import annotations

import random
from typing import Any, Callable

from dspy.core.errors import CatchableError
from dspy.core.example import Example
from dspy.modules.module import Module
from dspy.optim.base import (
    Checkpointer,
    Optimizer,
    apply_state,
    check_trainset,
    run_engine,
    snapshot_state,
)
from dspy.optim.labeled_fewshot import LabeledFewShot

__all__ = ["BootstrapFewShot"]


class BootstrapFewShot(Optimizer):
    """Bootstrap demos from metric-passing teacher traces.

    Args:
        metric: `metric(example, prediction) -> bool | float`; a trace is
            kept when the value is truthy (or `>= metric_threshold` when
            a threshold is set). None keeps every completed trace.
        metric_threshold: Numeric acceptance bar for the metric value.
        max_bootstrapped_demos: Cap on trainset examples turned into
            augmented demos (one accepted run per example).
        max_labeled_demos: Total demo cap per predictor; the space left
            after augmented demos fills with labeled trainset examples
            the bootstrap did not consume.
        seed: One explicit seed drives every rng in the pass: the
            teacher's labeled demos, the validation shuffle, and the
            labeled fill sampling.

    Examples:
        ```python
        optimizer = dspy.BootstrapFewShot(metric=exact_match, max_bootstrapped_demos=2)
        compiled = optimizer.compile(program, trainset=trainset)
        ```
    """

    def __init__(
        self,
        metric: Callable[[Example, Any], Any] | None = None,
        *,
        metric_threshold: float | None = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        seed: int = 0,
    ):
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.seed = seed

    def compile(
        self,
        program: Module,
        *,
        trainset: Any,
        teacher: Module | None = None,
        checkpoint_dir: str | None = None,
    ) -> Module:
        """Bootstrap demos onto the program (mutated in place) and return it.

        Args:
            program: The student program.
            trainset: `dspy.Example` values with declared inputs.
            teacher: A structurally identical program to gather traces
                with; None runs the student itself as teacher.
            checkpoint_dir: When given, save the accepted candidate as
                `<checkpoint_dir>/candidate-000/` plus a `scores.json`.
        """
        trainset = check_trainset(trainset)
        student = program
        if teacher is None:
            teacher = student
        else:
            _check_same_structure(student, teacher)

        student_baseline = snapshot_state(student)
        teacher_baseline = snapshot_state(teacher)
        traces: dict[str, list[Example]] = {path: [] for path, _ in student.named_predictors()}
        bootstrapped: set[int] = set()
        try:
            if self.max_labeled_demos:
                LabeledFewShot(self.max_labeled_demos, seed=self.seed).compile(teacher, trainset=trainset)
            for index, example in enumerate(trainset):
                if len(bootstrapped) >= self.max_bootstrapped_demos:
                    break
                success, records = self._attempt(teacher, example)
                if not success:
                    continue
                bootstrapped.add(index)
                for path, call in records.items():
                    if path not in traces:
                        continue
                    demo = Example(augmented=True, **call["inputs"], **call["outputs"])
                    traces[path].append(demo.with_inputs(*call["inputs"]))
        finally:
            apply_state(teacher, teacher_baseline)
            apply_state(student, student_baseline)

        validation = [example for index, example in enumerate(trainset) if index not in bootstrapped]
        random.Random(self.seed).shuffle(validation)
        rng = random.Random(self.seed)
        for path, predictor in student.named_predictors():
            augmented = traces[path][: self.max_bootstrapped_demos]
            fill = max(0, min(self.max_labeled_demos - len(augmented), len(validation)))
            predictor.demos = augmented + rng.sample(validation, fill)

        if checkpoint_dir is not None:
            Checkpointer(checkpoint_dir).accept(student, score=None, label="bootstrap-fewshot")
        return student

    def _attempt(self, teacher: Module, example: Example) -> tuple[bool, dict[str, dict[str, Any]]]:
        """Run one trainset example through the teacher, on the engine.

        Returns `(success, records)` where records maps each predictor
        path to its LAST engine call record for this run.
        """
        saved = [(predictor, predictor.demos) for _, predictor in teacher.named_predictors()]
        for predictor, demos in saved:
            # The example must not teach itself: drop it from the demos.
            predictor.demos = [demo for demo in demos if demo != example]
        try:
            inputs = {key: example[key] for key in example.inputs().keys()}
            prediction = run_engine(teacher, inputs)
        except CatchableError:
            return False, {}
        finally:
            for predictor, demos in saved:
                predictor.demos = demos

        if self.metric is not None:
            value = self.metric(example, prediction)
            success = value >= self.metric_threshold if self.metric_threshold is not None else bool(value)
        else:
            success = True
        if not success:
            return False, {}

        records: dict[str, dict[str, Any]] = {}
        for call in prediction._trajectory.get("predictor_calls", []):
            records[call["predictor"]] = call  # the last call per predictor wins
        return True, records


def _check_same_structure(student: Module, teacher: Module) -> None:
    student_paths = student.named_predictors()
    teacher_paths = teacher.named_predictors()
    if [path for path, _ in student_paths] != [path for path, _ in teacher_paths]:
        raise ValueError(
            "student and teacher must share the same predictor structure; student has "
            f"{[path for path, _ in student_paths]}, teacher has {[path for path, _ in teacher_paths]}"
        )
    for (path, student_predictor), (_, teacher_predictor) in zip(student_paths, teacher_paths, strict=True):
        if student_predictor is teacher_predictor:
            raise ValueError(
                f"student and teacher share the predictor instance at {path!r}; a teacher must be a "
                "separately constructed program (or pass teacher=None to run the student as teacher)"
            )
        if not student_predictor.signature.equals(teacher_predictor.signature):
            raise ValueError(f"student and teacher disagree on the signature of predictor {path!r}")
