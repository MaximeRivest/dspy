"""Compile, validate, write, and load portable DSPy program artifacts."""

import os
from typing import Any

from dspy.programir.compile import compile
from dspy.programir.engine import ExecutableProgram, materialize
from dspy.programir.export import export
from dspy.programir.link import link
from dspy.programir.model import ProgramIR
from dspy.programir.read import read
from dspy.programir.write import write

__all__ = [
    "ExecutableProgram",
    "ProgramIR",
    "compile",
    "export",
    "link",
    "load",
    "materialize",
    "read",
    "write",
]


def load(path: str | os.PathLike[str], bindings: dict[str, dict[str, Any]] | None = None) -> ExecutableProgram:
    """Load one artifact directory into a callable program.

    The one load path: read (parse + validate) + link + materialize.
    Resolution is loud — an LM or interpreter pool entry the caller does
    not bind refuses by name.

    Args:
        path: Artifact directory written by `Module.save` / `export`.
        bindings: Receiver bindings, keyed by kind (`"lm"`, `"adapter"`,
            `"tool"`, `"interpreter"`), then by pool entry name.

    Returns:
        An `ExecutableProgram`; call it with the program's inputs.
    """
    return materialize(read(path), bindings)
