"""Extract and validate structural interpreter identity profiles.

Also home to `InProcessInterpreter`, the reference declared-interpreter
runtime: an exact structural profile (D-033) plus a one-call contract.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from typing import Any

_STRUCTURAL_KEYS = {
    "language",
    "runtime",
    "contract",
    "namespace_policy",
    "result_convention",
    "vars_marshaling",
    "packages",
    "resource_limits",
    "isolation_floor",
    "placement",
}
_ISOLATION_ORDER = {
    "none": 0,
    "process": 1,
    "wasm": 2,
    "os_sandbox": 3,
    "remote_sandbox": 4,
}


def extract_interpreter(value: Any, *, name: str) -> dict[str, Any]:
    """Extract one custom interpreter's declared ProgramIR profile.

    Custom interpreters declare identity as plain data through
    `programir_profile()`. The hook is descriptive only: calling it must not
    start the interpreter or inspect the host environment.
    """
    from dspy.primitives.python_interpreter import PythonInterpreter

    if isinstance(value, PythonInterpreter):
        raise ValueError(
            f"ProgramIR cannot yet export PythonInterpreter {name!r}: its Deno and npm:pyodide "
            "runtime versions are not pinned on the live object, so an exact interpreter identity "
            "cannot be declared (D-033)"
        )
    hook = getattr(value, "programir_profile", None)
    if not callable(hook):
        raise ValueError(
            f"ProgramIR interpreter {name!r} must provide programir_profile() returning plain identity data"
        )
    profile = hook()
    validate_interpreter_profile(profile, name=name)
    return deepcopy(profile)


def validate_interpreter_profile(profile: Any, *, name: str) -> None:
    """Validate D-033's fixed structural profile envelope."""
    if not isinstance(profile, dict):
        raise ValueError(f"ProgramIR interpreter {name!r} profile must be an object")
    missing = sorted(_STRUCTURAL_KEYS - set(profile))
    unknown = sorted(set(profile) - _STRUCTURAL_KEYS)
    if missing or unknown:
        raise ValueError(
            f"ProgramIR interpreter {name!r} profile has missing keys {missing} and unknown keys {unknown}"
        )
    if not _nonempty_string(profile["language"]):
        raise ValueError(f"ProgramIR interpreter {name!r} language must be a non-empty tag")
    runtime = profile["runtime"]
    if (
        not isinstance(runtime, dict)
        or set(runtime) != {"identity", "version"}
        or not _nonempty_string(runtime.get("identity"))
        or not _nonempty_string(runtime.get("version"))
    ):
        raise ValueError(
            f"ProgramIR interpreter {name!r} runtime must be exact {{identity, version}} strings"
        )
    for field in ("contract", "namespace_policy", "result_convention", "vars_marshaling"):
        if not isinstance(profile[field], (str, dict)):
            raise ValueError(f"ProgramIR interpreter {name!r} {field} must be a string or object")
    packages = profile["packages"]
    if not isinstance(packages, list) or not all(isinstance(package, str) for package in packages):
        raise ValueError(f"ProgramIR interpreter {name!r} packages must be a string list")
    if not isinstance(profile["resource_limits"], dict):
        raise ValueError(f"ProgramIR interpreter {name!r} resource_limits must be an object")
    floor = profile["isolation_floor"]
    placement = profile["placement"]
    if not _nonempty_string(floor):
        raise ValueError(f"ProgramIR interpreter {name!r} isolation_floor must be a non-empty string")
    if not isinstance(placement, dict):
        raise ValueError(f"ProgramIR interpreter {name!r} placement must be an object")
    actual = placement.get("isolation")
    if not _nonempty_string(actual):
        raise ValueError(f"ProgramIR interpreter {name!r} placement isolation must be a non-empty string")
    if floor in _ISOLATION_ORDER and actual in _ISOLATION_ORDER:
        if _ISOLATION_ORDER[actual] < _ISOLATION_ORDER[floor]:
            raise ValueError(
                f"ProgramIR interpreter {name!r} placement isolation {actual!r} is inward of "
                f"its declared floor {floor!r}"
            )
    elif actual != floor:
        raise ValueError(
            f"ProgramIR interpreter {name!r} uses unknown isolation ordering "
            f"{floor!r} -> {actual!r}; use the floor itself or a standard outward isolation"
        )
    if not _nonempty_string(placement.get("rung")) or not isinstance(placement.get("contract"), (str, dict)):
        raise ValueError(f"ProgramIR interpreter {name!r} placement must declare rung and contract")


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


class InProcessInterpreter:
    """The in-process Python interpreter leaf: exec, no builtins, `result`.

    The reference runtime for declared interpreter leaves (RLM's default).
    Code runs under `exec` with an EMPTY builtins table in a fresh
    namespace and must assign its final value to a variable named
    `result`; the call returns `str(result)`. Every failure — bad code, a
    missing `result`, anything the code raises — surfaces as the typed
    `InterpreterError`, so `except InterpreterError` behaves identically
    on the native path and under the engine.

    The declared profile is structural identity, not an implementation:
    a receiver loading an artifact binds its own runtime for the pool
    entry (D-033).
    """

    def programir_profile(self) -> dict[str, Any]:
        return {
            "language": "python",
            "runtime": {
                "identity": "cpython",
                "version": f"{sys.version_info.major}.{sys.version_info.minor}",
            },
            "contract": "call(code)->result",
            "namespace_policy": "fresh-empty-no-builtins",
            "result_convention": "result variable",
            "vars_marshaling": "str-values",
            "packages": [],
            "resource_limits": {},
            "isolation_floor": "none",
            "placement": {
                "rung": "in_process",
                "contract": "call(code)->result",
                "endpoint_ref": None,
                "isolation": "none",
                "credential_ref": None,
            },
        }

    def __call__(self, *, code: str) -> str:
        from dspy.core.errors import InterpreterError

        namespace: dict[str, Any] = {"__builtins__": {}}
        try:
            exec(code, namespace)
        except InterpreterError:
            raise
        except Exception as error:
            raise InterpreterError(f"code execution failed: {error}") from error
        if "result" not in namespace:
            raise InterpreterError("the code must assign the final value to a variable named `result`")
        return str(namespace["result"])
