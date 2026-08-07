"""Read and validate self-contained ProgramIR artifact directories."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dspy.programir.link import link
from dspy.programir.model import ProgramIR
from dspy.programir.validate import validate_manifest


def read(path: str | os.PathLike[str]) -> ProgramIR:
    """Read one artifact directory into a validated ProgramIR value.

    Reading is grade 1: it parses bytes, checks versions and internal links,
    and executes no authored code or receiver binding.

    Args:
        path: Artifact directory containing `manifest.json`.

    Returns:
        The validated ProgramIR and its uninterpreted sidecar bytes.
    """
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"ProgramIR path must be an artifact directory: {root}")
    manifest_path = root / "manifest.json"
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_refuse_json_constant,
        )
    except OSError as error:
        raise ValueError(f"ProgramIR artifact has no readable manifest.json: {root}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"ProgramIR manifest.json is not canonical JSON: {error}") from error

    validate_manifest(manifest)
    sidecars = {
        file.relative_to(root).as_posix(): file.read_bytes()
        for file in sorted(root.rglob("*"))
        if file.is_file() and file != manifest_path
    }
    ir = ProgramIR(manifest=manifest, sidecars=sidecars)
    link(ir)
    return ir


def _refuse_json_constant(value: str) -> Any:
    raise ValueError(f"ProgramIR JSON refuses non-finite number {value}")
