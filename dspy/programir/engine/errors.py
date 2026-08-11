"""Typed runtime errors for the ProgramIR node-set engine.

This module mirrors the programir-contract reference `errors.py` (SEM-3):
the typed-error table, which names a program may `Raise`, and how
`ExceptHandler` types match. Two channels, deliberately separate:

- **Catchable** (`CatchableError`): the program-level errors a forward's
  `Try` can handle. Matching is by exact name, or the `Exception`
  handler, which catches every catchable error.
- **Uncatchable** (`UncatchableError`): harness guards. `LoopCapError`
  (SEM-6 caps) and `MalformedNodeError` (input the compiler should have
  refused). No program handler ever sees these.
"""

from __future__ import annotations

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


class PirError(Exception):
    """Base of every typed error the engine emits.

    Message text is non-normative; equivalence compares the error's type
    name, never its wording.
    """


class CatchableError(PirError):
    """Base of the program-catchable channel (SEM-3)."""


class ToolError(CatchableError):
    """A tool leaf failed or an unknown tool name was dispatched."""


class InterpreterError(CatchableError):
    """A code-interpreter leaf (component 7) failed."""


class AdapterParseError(CatchableError):
    """An adapter could not parse an LM completion into the signature."""


class LMError(CatchableError):
    """An LM leaf failed."""


class InterpreterTypeError(CatchableError):
    """A value had the wrong type for the operation (SEM-1, SEM-4, SEM-7)."""


class InterpreterKeyError(CatchableError):
    """A name or field was absent (SEM-7). Typed and catchable."""


class InterpreterArithmeticError(CatchableError):
    """An arithmetic operation left the value model (v0.2, D-034)."""


class UncatchableError(PirError):
    """Base of the harness-guard channel; program handlers never match it."""


class LoopCapError(UncatchableError):
    """An interpretation-side cap was breached (SEM-6)."""


class MalformedNodeError(UncatchableError):
    """Input the compiler is required to refuse reached the interpreter."""


#: Names a program's `Raise` may use, per SEM-3 — exactly the table.
RAISEABLE: dict[str, type[CatchableError]] = {
    "ToolError": ToolError,
    "InterpreterError": InterpreterError,
    "AdapterParseError": AdapterParseError,
    "LMError": LMError,
}

#: Every typed error a program `Try` can catch.
CATCHABLE_NAMES: frozenset[str] = frozenset(RAISEABLE) | frozenset(
    {"InterpreterTypeError", "InterpreterKeyError", "InterpreterArithmeticError"}
)

#: Legal `ExceptHandler` type names: the catchable set plus `Exception`.
HANDLER_NAMES: frozenset[str] = CATCHABLE_NAMES | frozenset({"Exception"})


def handler_matches(err: BaseException, handler_type: str) -> bool:
    """Decide whether one handler catches one error (SEM-3).

    Exact-name matching — no subtype table — except `Exception`, which
    catches every catchable error.

    Args:
        err: The in-flight error.
        handler_type: The handler's declared `type` string.

    Returns:
        True when the handler catches this error.
    """
    if not isinstance(err, CatchableError):
        return False
    if handler_type == "Exception":
        return True
    return handler_type in CATCHABLE_NAMES and type(err).__name__ == handler_type
