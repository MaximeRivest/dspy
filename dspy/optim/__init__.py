"""Optimizers as IR mutations: propose = mutate data, score = engine replay, keep = the artifact.

The mutation surface is the program's own state — demos (component
`3b_demos`) and instructions (`3a_instructions`) on live predictors —
never its code. `evaluate` replays a program over a devset through the
engine; `Checkpointer` writes accepted candidates with the one save path,
so every checkpoint is a loadable artifact.
"""

from dspy.optim.base import (
    Checkpointer,
    EvaluationResult,
    Optimizer,
    ProgramState,
    apply_state,
    evaluate,
    snapshot_state,
)
from dspy.optim.bootstrap import BootstrapFewShot
from dspy.optim.flex import FlexIR
from dspy.optim.labeled_fewshot import LabeledFewShot
from dspy.optim.random_search import BootstrapFewShotWithRandomSearch

__all__ = [
    "BootstrapFewShot",
    "BootstrapFewShotWithRandomSearch",
    "Checkpointer",
    "EvaluationResult",
    "FlexIR",
    "LabeledFewShot",
    "Optimizer",
    "ProgramState",
    "apply_state",
    "evaluate",
    "snapshot_state",
]
