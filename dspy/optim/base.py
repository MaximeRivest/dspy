"""Optimizers as IR mutations: propose, score, keep.

An optimizer never rewrites a program's code. It mutates the program's
DATA — the exact state the artifact carries as components `3a_instructions`
and `3b_demos`: on a live program that surface is `predictor.demos` (a
list of `dspy.Example`) and `predictor.signature` (instructions, swapped
via `with_instructions`). The compile fingerprint watches both, so a
mutated program recompiles itself on its next call — propose is a plain
assignment, score is a replay, keep is the best state.

checkpoint == save: an accepted candidate is written with the ordinary
`program.save(dir)` path, so every checkpoint directory is a loadable
ProgramIR artifact. `scores.json` beside them records the acceptance
trajectory in order.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dspy.core.errors import CatchableError
from dspy.core.example import Example
from dspy.modules.module import Module

__all__ = [
    "Checkpointer",
    "EvaluationResult",
    "Optimizer",
    "ProgramState",
    "apply_state",
    "check_trainset",
    "evaluate",
    "run_engine",
    "snapshot_state",
]


# ---------------------------------------------------------------------------
# The mutation surface: program state as data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProgramState:
    """One candidate's mutable state: demos and instructions per predictor.

    This is the live mirror of manifest components `3b_demos` and
    `3a_instructions`, keyed by the same dotted predictor paths.
    """

    demos: dict[str, list[Example]] = field(default_factory=dict)
    instructions: dict[str, str] = field(default_factory=dict)


def snapshot_state(program: Module) -> ProgramState:
    """Capture a program's demos and instructions, per predictor path."""
    demos: dict[str, list[Example]] = {}
    instructions: dict[str, str] = {}
    for path, predictor in program.named_predictors():
        demos[path] = list(predictor.demos)
        instructions[path] = predictor.signature.instructions
    return ProgramState(demos=demos, instructions=instructions)


def apply_state(program: Module, state: ProgramState) -> None:
    """Write a captured state back onto a program's predictors.

    Refuses when the program's predictor paths no longer match the
    state's — an optimizer mutates data, never structure.
    """
    paths = [path for path, _ in program.named_predictors()]
    missing = sorted(set(paths) - set(state.demos))
    if missing:
        raise ValueError(
            f"cannot apply program state: predictor path(s) {missing} have no captured state "
            f"(captured: {sorted(state.demos)}). Optimizer states apply to the program they came from."
        )
    for path, predictor in program.named_predictors():
        predictor.demos = list(state.demos[path])
        if predictor.signature.instructions != state.instructions[path]:
            predictor.signature = predictor.signature.with_instructions(state.instructions[path])


def check_trainset(examples: Any, *, name: str = "trainset") -> list[Example]:
    """Refuse loudly unless every entry is an Example with declared inputs."""
    examples = list(examples)
    for index, example in enumerate(examples):
        if not isinstance(example, Example):
            raise ValueError(
                f"{name}[{index}] is {type(example).__name__}, not dspy.Example; optimizers take dspy.Example values"
            )
        if example._input_keys is None:
            raise ValueError(f"{name}[{index}] has no input designation; call .with_inputs(...) on every example")
    return examples


# ---------------------------------------------------------------------------
# Score: engine replay over a devset
# ---------------------------------------------------------------------------


def run_engine(program: Any, inputs: dict[str, Any]):
    """Run one program over one input dict, through the engine.

    A `Module` (composite or leaf) compiles and runs its IR; anything
    else — a loaded `ExecutableProgram`, say — is called directly, which
    already IS the engine path.
    """
    if isinstance(program, Module):
        return program._executable()(**inputs)
    return program(**inputs)


@dataclass(frozen=True)
class EvaluationResult:
    """One devset pass: the mean score, per-example detail, LM-call count.

    Attributes:
        score: Mean metric value over the devset (a failed run scores 0.0).
        results: One `(example, prediction, score)` triple per example;
            `prediction` is None when the run raised a catchable error.
        lm_calls: Predictor exchanges recorded in the returned
            predictions' exhaust; a failed run's calls are not counted.
    """

    score: float
    results: list[tuple[Example, Any, float]]
    lm_calls: int


def evaluate(program: Any, devset: Any, metric: Callable[[Example, Any], Any]) -> EvaluationResult:
    """Score one program over a devset, sequentially, through the engine.

    Args:
        program: A `dspy.Module` (or any callable program).
        devset: `dspy.Example` values with declared inputs.
        metric: `metric(example, prediction) -> bool | float`.

    Returns:
        An `EvaluationResult`; `result.score` is the mean metric value.
        A run that raises a catchable error scores 0.0 (prediction None);
        a metric that raises on one example scores THAT example 0.0 and the
        pass continues (brief 1.f) — one bad example never aborts the run.

    Raises:
        ValueError: On an empty devset or an undeclared-input example.
    """
    devset = check_trainset(devset, name="devset")
    if not devset:
        raise ValueError("evaluate() needs a non-empty devset; there is nothing to score")
    results: list[tuple[Example, Any, float]] = []
    lm_calls = 0
    for example in devset:
        inputs = {key: example[key] for key in example.inputs().keys()}
        try:
            prediction = run_engine(program, inputs)
        except CatchableError:
            results.append((example, None, 0.0))
            continue
        lm_calls += _count_lm_calls(prediction)
        # A metric that raises scores THAT example 0.0 and the run continues
        # (brief 1.f): one bad example must not abort the whole devset pass,
        # or a single flaky metric call would kill an optimization run. The
        # prediction itself succeeded (it ran through the engine), so its
        # lm_calls are already counted above. Only a real metric exception is
        # swallowed here; typed infrastructure errors from the ENGINE surface
        # as CatchableError above and keep their semantics.
        try:
            value = float(metric(example, prediction))
        except Exception as error:
            warnings.warn(
                f"metric raised on one example ({type(error).__name__}: {error}); scoring it 0.0 and "
                "continuing the devset pass",
                stacklevel=2,
            )
            value = 0.0
        results.append((example, prediction, value))
    score = sum(value for _, _, value in results) / len(results)
    return EvaluationResult(score=score, results=results, lm_calls=lm_calls)


def _count_lm_calls(prediction: Any) -> int:
    exhaust = getattr(prediction, "_trajectory", None) or {}
    calls = exhaust.get("predictor_calls")
    if calls is not None:
        return len(calls)
    return 1 if "completion" in exhaust else 0


# ---------------------------------------------------------------------------
# Keep: accepted candidates as ordinary artifacts
# ---------------------------------------------------------------------------


class Checkpointer:
    """Write accepted candidates under one directory, as artifacts.

    Every accepted candidate is saved through the one save path
    (`program.save`), so `<directory>/candidate-NNN/` loads back with
    `dspy.load(dir, bindings=...)` like any program. `scores.json` at the
    directory root is the acceptance trajectory, in order.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def accept(self, program: Module, *, score: float | None, label: str | None = None) -> str:
        """Save one accepted candidate; returns its directory name."""
        name = f"candidate-{len(self.records):03d}"
        program.save(self.directory / name)
        record: dict[str, Any] = {"candidate": name, "score": score}
        if label is not None:
            record["label"] = label
        self.records.append(record)
        path = self.directory / "scores.json"
        path.write_text(json.dumps(self.records, indent=2) + "\n", encoding="utf-8")
        return name


# ---------------------------------------------------------------------------
# The loop shape
# ---------------------------------------------------------------------------


class Optimizer:
    """Base of the mutation loop: propose a state, score it, keep the best.

    `compile(program, *, trainset, ...)` mutates the given program's
    demos/instructions and returns it with the winning state applied.
    There is no program cloning: a candidate is a `ProgramState`, applied
    and reverted on the one live program (`snapshot_state`/`apply_state`).
    Construct a fresh program if you need the unoptimized original.
    """

    def compile(self, program: Module, *, trainset: Any, **kwargs: Any) -> Module:
        raise NotImplementedError(
            f"{type(self).__name__} declares no compile(); an Optimizer implements "
            "compile(program, *, trainset, ...) -> the program with the best state applied"
        )
