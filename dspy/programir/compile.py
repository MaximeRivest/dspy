"""Compile frontend programs into canonical ProgramIR values."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from dspy.programir.model import ProgramIR


def compile(program: Any, *, metric: Any = None, devset: Any = None) -> ProgramIR:
    """Compile a supported frontend program into one ProgramIR value.

    `ProgramIR` is the only in-memory program representation produced by this
    API. Frontend-specific code may collect temporary component fragments, but
    no second frontend IR crosses this boundary.

    Args:
        program: A program from a supported frontend.
        metric: Optional evaluation metric. Metric extraction lands in I-4.
        devset: Optional evaluation examples. Evaluation extraction lands in
            I-4.

    Returns:
        A detached ProgramIR value.

    Raises:
        TypeError: If no frontend recognizes `program`.
        NotImplementedError: If evaluation extraction is requested before I-4.
    """
    if metric is not None or devset is not None:
        raise NotImplementedError("ProgramIR metric and devset extraction lands in exporter step I-4")

    from dspy.primitives.module import Module

    if isinstance(program, Module):
        from dspy.programir._dspy import compile_program

        return compile_program(program)

    raise TypeError(f"programir.compile() does not support {type(program).__name__}")


def build_program_ir(
    *,
    versions: dict[str, str],
    components: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    sidecars: dict[str, bytes] | None = None,
) -> ProgramIR:
    """Assemble validated component fragments into a ProgramIR value.

    This private-facing builder is the common endpoint for framework bridges
    and direct graph frontends. It accepts plain data and performs no framework
    introspection, filesystem access, environment reads, or clock reads.
    """
    manifest: dict[str, Any] = {
        "versions": deepcopy(versions),
        "components": deepcopy(components),
    }
    if provenance is not None:
        manifest["provenance"] = deepcopy(provenance)
    _require_json_data(manifest)
    return ProgramIR(manifest=manifest, sidecars=deepcopy(sidecars or {}))


def _require_json_data(value: Any) -> None:
    """Refuse host objects and non-finite floats before they reach a writer."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"ProgramIR contains a non-JSON value: {error}") from error
