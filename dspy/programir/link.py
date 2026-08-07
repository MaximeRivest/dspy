"""Resolve ProgramIR internal pool bindings without materializing code."""

from __future__ import annotations

from typing import Any

from dspy.programir.model import ProgramIR
from dspy.programir.validate import resolve_bindings


def link(ir: ProgramIR) -> dict[str, dict[str, Any]]:
    """Return the pure predictor binding table for a ProgramIR value.

    Args:
        ir: ProgramIR whose internal references should resolve.

    Returns:
        Predictor paths mapped to their adapter, LM, and delta names.
    """
    if not isinstance(ir, ProgramIR):
        raise TypeError(f"programir.link() takes a ProgramIR, got {type(ir).__name__}")
    return resolve_bindings(ir.to_manifest())
