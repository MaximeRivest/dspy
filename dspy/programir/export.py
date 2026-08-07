"""Public ProgramIR export composition."""

from __future__ import annotations

import os
import re
from typing import Any

from dspy.clients.base_lm import SENSITIVE_HEADER_NAMES, SENSITIVE_LM_KWARG_NAMES
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.programir.compile import compile
from dspy.programir.model import ProgramIR
from dspy.programir.write import write

_CREDENTIAL_ENV = re.compile(r"(?:API[_-]?KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.IGNORECASE)


def export(
    program: Module,
    path: str | os.PathLike[str],
    *,
    metric: Any = None,
    devset: Any = None,
) -> ProgramIR:
    """Compile and write a portable DSPy program artifact.

    Args:
        program: The DSPy module or predictor to export.
        path: Destination directory for the artifact.
        metric: Optional named Python metric to bake with the program.
        devset: Optional evaluation examples carrying `input_keys`.

    Returns:
        The finalized ProgramIR, including generated environment lockfiles.
    """
    credential_values = _credential_values(program)
    ir = compile(program, metric=metric, devset=devset)
    return write(ir, path, credential_values=credential_values)


def _credential_values(program: Module) -> dict[str, str]:
    found = {
        name: value
        for name, value in os.environ.items()
        if _CREDENTIAL_ENV.search(name) and isinstance(value, str) and value
    }
    predictors = [program] if isinstance(program, Predict) else [predictor for _, predictor in program.named_predictors()]
    seen_lms: set[int] = set()
    lm_index = 0
    for predictor in predictors:
        lm = predictor.lm or settings.lm
        if lm is None or id(lm) in seen_lms:
            continue
        seen_lms.add(id(lm))
        lm_index += 1
        suffix = "" if lm_index == 1 else f"_{lm_index}"
        for key, value in getattr(lm, "kwargs", {}).items():
            if key in SENSITIVE_LM_KWARG_NAMES and isinstance(value, str) and value:
                found[f"LM_API_KEY{suffix}"] = value
            elif key == "headers" and isinstance(value, dict):
                for header, header_value in value.items():
                    if str(header).lower() in SENSITIVE_HEADER_NAMES and isinstance(header_value, str) and header_value:
                        found[f"LM_HEADER{suffix}_{str(header).upper()}"] = header_value
        for attribute in ("api_key", "_api_key"):
            value = getattr(lm, attribute, None)
            if isinstance(value, str) and value:
                found[f"LM_API_KEY{suffix}"] = value
    return found
