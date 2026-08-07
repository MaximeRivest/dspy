"""Write ProgramIR values as deterministic artifact directories."""

from __future__ import annotations

import json
import math
import os
import subprocess
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
) -> ProgramIR:
    """Write one ProgramIR artifact directory and return its finalized value.

    The destination must not already exist. The writer stages every file,
    scans the finished bytes for credential values, and only then publishes
    the directory.

    Args:
        ir: The compiled ProgramIR value to write.
        path: Destination directory.
        credential_values: Credential names and their live values. Values are
            scanned but never written; a match refuses and names only the
            credential and file.

    Returns:
        The finalized ProgramIR, including lockfiles generated during writing.
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
        manifest = ir.to_manifest()
        (root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        for relative, content in ir.sidecars.items():
            target = root / _safe_relative_path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bytes(content))

        _materialize_environment_locks(root, manifest)
        _scan_credentials(root, credential_values or {})
        finalized = ProgramIR(
            manifest=manifest,
            sidecars={
                file.relative_to(root).as_posix(): file.read_bytes()
                for file in sorted(root.rglob("*"))
                if file.is_file() and file != root / "manifest.json"
            },
        )
        os.replace(root, destination)
        return finalized


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


def _materialize_environment_locks(root: Path, manifest: Mapping[str, Any]) -> None:
    environments = manifest.get("components", {}).get("9_environment", {})
    python = environments.get("python") if isinstance(environments, Mapping) else None
    if not isinstance(python, Mapping):
        return
    entry_name = python.get("pep723_entry")
    lock_name = python.get("lock")
    if not isinstance(entry_name, str) or not isinstance(lock_name, str):
        raise ValueError("ProgramIR Python environment must declare pep723_entry and lock paths")
    entry = root / _safe_relative_path(entry_name)
    lock = root / _safe_relative_path(lock_name)
    if not entry.is_file():
        raise ValueError(f"ProgramIR Python environment entry is missing: {entry_name!r}")
    if lock.is_file():
        return
    try:
        result = subprocess.run(
            ["uv", "lock", "--script", str(entry)],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError("ProgramIR could not run `uv lock --script` for the Python environment") from error
    if result.returncode != 0 or not lock.is_file():
        raise ValueError("ProgramIR `uv lock --script` failed to produce the declared lockfile")


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
