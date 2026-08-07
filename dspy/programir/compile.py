"""Lower framework-neutral snapshots into ProgramIR values."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from dspy.programir.model import FrontendProgram, ProgramIR

_REQUIRED_VERSION_KEYS = (
    "ir_version",
    "node_set",
    "roles",
    "strategies",
    "codecs",
    "adapter_ir",
    "lm15",
)


def compile(frontend: FrontendProgram) -> ProgramIR:
    """Compile a framework-neutral snapshot into one ProgramIR value.

    This function is pure: it reads no settings, filesystem state, clock,
    environment variables, or credentials. Framework frontends must resolve
    those inputs before calling it.

    Args:
        frontend: Plain component data from a framework frontend.

    Returns:
        A ProgramIR value ready for validation or writing.
    """
    if not isinstance(frontend, FrontendProgram):
        raise TypeError(
            "programir.compile() takes a FrontendProgram snapshot; live framework "
            "objects must pass through their frontend bridge first"
        )

    missing_versions = [key for key in _REQUIRED_VERSION_KEYS if key not in frontend.versions]
    if missing_versions:
        raise ValueError(f"FrontendProgram versions block is missing required entries: {missing_versions}")

    components: dict[str, Any] = {
        "1_module_tree": deepcopy(dict(frontend.module_tree)),
        "2_signature": deepcopy(dict(frontend.signatures)),
        "3a_instructions": deepcopy(dict(frontend.instructions)),
        "3b_demos": deepcopy(dict(frontend.demos)),
        "3c_predictor_config": deepcopy(dict(frontend.predictor_config)),
        "4_adapter": deepcopy(dict(frontend.adapters)),
        "5_forward": deepcopy(dict(frontend.forwards)),
        "6_tools": deepcopy(dict(frontend.tools)),
        "7_interpreter": deepcopy(dict(frontend.interpreters)),
        "8_lm": deepcopy(dict(frontend.lms)),
        "9_environment": deepcopy(dict(frontend.environment)),
        "10_credentials": deepcopy(list(frontend.credentials)),
        "11_ambient_policy": deepcopy(dict(frontend.ambient_policy)),
    }
    if frontend.evaluation is not None:
        components["12_metric"] = deepcopy(dict(frontend.evaluation))

    manifest: dict[str, Any] = {
        "versions": deepcopy(dict(frontend.versions)),
        "components": components,
    }
    if frontend.provenance is not None:
        manifest["provenance"] = deepcopy(dict(frontend.provenance))

    _require_json_data(manifest)
    return ProgramIR(manifest=manifest)


def _require_json_data(value: Any) -> None:
    """Refuse host objects and non-finite floats before they reach a writer."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"FrontendProgram contains a non-JSON value: {error}") from error
