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


def load(
    path: str | os.PathLike[str],
    bindings: dict[str, dict[str, Any]] | None = None,
    *,
    envelope: Any = None,
) -> ExecutableProgram:
    """Load one artifact directory into a callable program.

    The one load path: read (parse + validate) + link + materialize.
    Resolution is loud — an LM or interpreter pool entry the caller does
    not bind refuses by name.

    Args:
        path: Artifact directory written by `Module.save` / `export`.
        bindings: Receiver bindings, keyed by kind (`"lm"`, `"adapter"`,
            `"tool"`, `"interpreter"`), then by pool entry name.
        envelope: An optional `dspy.Envelope` (or `IsolationPolicy`) — the
            receiver's isolation envelope; one meeting a leaf's floor
            satisfies its `isolation_required` grant (D-042).

    Returns:
        An `ExecutableProgram`; call it with the program's inputs.
    """
    if envelope is not None:
        bindings = dict(bindings or {})
        policy = envelope.policy if hasattr(envelope, "policy") else envelope
        isolation = dict(bindings.get("isolation") or {})
        isolation["envelope"] = policy
        bindings["isolation"] = isolation
    return materialize(read(path), bindings)
