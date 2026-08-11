"""Typed runtime errors for the ProgramIR node-set engine.

The table now lives in `dspy.core.errors` (one class identity for the
whole system); this module re-exports it so engine-internal imports and
the contract-facing path keep working unchanged.
"""

from __future__ import annotations

from dspy.core.errors import (
    CATCHABLE_NAMES,
    HANDLER_NAMES,
    RAISEABLE,
    AdapterParseError,
    CatchableError,
    InterpreterArithmeticError,
    InterpreterError,
    InterpreterKeyError,
    InterpreterTypeError,
    LMError,
    LoopCapError,
    MalformedNodeError,
    PirError,
    ToolError,
    UncatchableError,
    handler_matches,
)

__all__ = [
    "PirError",
    "CatchableError",
    "ToolError",
    "InterpreterError",
    "AdapterParseError",
    "LMError",
    "InterpreterTypeError",
    "InterpreterKeyError",
    "InterpreterArithmeticError",
    "UncatchableError",
    "LoopCapError",
    "MalformedNodeError",
    "RAISEABLE",
    "CATCHABLE_NAMES",
    "HANDLER_NAMES",
    "handler_matches",
]
