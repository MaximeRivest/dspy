"""Write ProgramIR values as deterministic artifact directories."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from dspy.programir.model import ProgramIR


def write(
    ir: ProgramIR,
    path: str | os.PathLike[str],
    *,
    credential_values: Mapping[str, str] | None = None,
) -> None:
    """Write one ProgramIR artifact directory.

    The destination must not already exist. The writer stages every file,
    scans the finished bytes for credential values, and only then publishes
    the directory.

    Args:
        ir: The compiled ProgramIR value to write.
        path: Destination directory.
        credential_values: Credential names and their live values. Values are
            scanned but never written; a match refuses and names only the
            credential and file.
    """
    if not isinstance(ir, ProgramIR):
        raise TypeError(f"programir.write() takes a ProgramIR, got {type(ir).__name__}")

    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"ProgramIR destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{destination.name}.", dir=destination.parent) as temporary:
        root = Path(temporary) / "artifact"
        root.mkdir()
        (root / "manifest.json").write_bytes(canonical_json_bytes(ir.to_manifest()))
        for relative, content in ir.sidecars.items():
            target = root / _safe_relative_path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(content))

        _scan_credentials(root, credential_values or {})
        os.replace(root, destination)


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value with the ProgramIR canonical byte profile."""
    return (_encode(value) + "\n").encode("utf-8")


def _encode(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value) or (value == 0.0 and math.copysign(1.0, value) < 0):
            raise ValueError(f"ProgramIR canonical JSON refuses non-finite or negative-zero float {value!r}")
        rendered = repr(value)
        if "." not in rendered and "e" not in rendered.lower():
            rendered += ".0"
        return rendered
    if isinstance(value, str):
        _refuse_lone_surrogates(value)
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item) for item in value) + "]"
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("ProgramIR JSON object keys must be strings")
        return "{" + ",".join(f"{_encode(key)}:{_encode(value[key])}" for key in sorted(value)) + "}"
    raise TypeError(f"ProgramIR contains a non-JSON value of type {type(value).__name__}")


def _refuse_lone_surrogates(value: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("ProgramIR canonical JSON refuses strings containing lone surrogates")


def _safe_relative_path(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"ProgramIR sidecar path must be a safe relative POSIX path, got {value!r}")
    return Path(*path.parts)


def _scan_credentials(root: Path, credential_values: Mapping[str, str]) -> None:
    needles = [
        (name, value.encode("utf-8"))
        for name, value in credential_values.items()
        if isinstance(value, str) and value
    ]
    for file in sorted(path for path in root.rglob("*") if path.is_file()):
        content = file.read_bytes()
        for name, needle in needles:
            if needle in content:
                relative = file.relative_to(root).as_posix()
                raise ValueError(f"ProgramIR artifact contains credential {name!r} in {relative!r}")
