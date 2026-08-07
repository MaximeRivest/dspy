"""DSPy frontend for ProgramIR compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.programir.compile import build_program_ir
from dspy.programir.model import ProgramIR
from dspy.programir.versions import IMPLEMENTED_VERSIONS
from dspy.signatures.roles import resolve_semantic_role


def compile_predict(program: Predict) -> ProgramIR:
    """Compile a bare DSPy predictor without introducing a frontend IR."""
    path = "self"
    lm = program.lm or settings.lm
    if lm is None:
        raise ValueError("ProgramIR compile cannot resolve an LM for predictor 'self'")
    if not isinstance(lm, LM):
        raise ValueError(
            f"ProgramIR phase 1 cannot compile {type(lm).__name__} bound to predictor 'self'; "
            "only declared dspy.LM entries are supported before weights baking"
        )

    adapter = getattr(program, "adapter", None) or settings.adapter or ChatAdapter()
    try:
        adapter_entry = adapter.dump_entry()
    except (AttributeError, ValueError) as error:
        raise ValueError(
            f"ProgramIR compile cannot serialize adapter {type(adapter).__name__} "
            "bound to predictor 'self'"
        ) from error

    adapter_name = adapter_entry["name"]
    lm_name = _pool_name(lm.model)
    input_names = list(program.signature.input_fields)
    components = {
        "1_module_tree": {
            "kind": "Predict",
            "name": path,
            "children": [],
            "bindings": {"adapter": adapter_name, "lm": lm_name, "delta": None},
            "forward_ref": f"5_forward/{path}",
        },
        "2_signature": {
            path: {
                "fields": [_field_record(name, field) for name, field in program.signature.fields.items()]
            }
        },
        "3a_instructions": {path: program.signature.instructions},
        "3b_demos": {path: [_example_record(example, input_names) for example in program.demos]},
        "3c_predictor_config": {path: deepcopy(program.config)},
        "4_adapter": {adapter_name: adapter_entry},
        "5_forward": {path: _predict_forward(input_names)},
        "6_tools": {},
        "7_interpreter": {},
        "8_lm": {lm_name: _lm_entry(lm)},
        "9_environment": {},
        "10_credentials": [{"name": "LM_API_KEY", "scope": f"LM {lm_name}"}],
        "11_ambient_policy": {
            "max_errors": settings.max_errors,
            "async_max_workers": settings.async_max_workers,
            "allow_tool_async_sync_conversion": settings.allow_tool_async_sync_conversion,
        },
    }
    return build_program_ir(
        versions=dict(IMPLEMENTED_VERSIONS),
        components=components,
        provenance={"source": "dspy.export", "evidence": "dspy frontend compile"},
    )


def _field_record(name: str, field: Any) -> dict[str, Any]:
    extra = field.json_schema_extra or {}
    try:
        shape = TypeAdapter(field.annotation).json_schema()
    except Exception as error:  # Pydantic exposes several schema refusal types.
        raise ValueError(f"ProgramIR cannot derive JSON Schema for field {name!r}: {error}") from error
    return {
        "name": name,
        "direction": extra["__dspy_field_type"],
        "prefix": extra.get("prefix"),
        "desc": extra.get("desc"),
        "shape": shape,
        "semantic_role": resolve_semantic_role(field, field_name=name),
    }


def _example_record(example: Any, input_names: list[str]) -> dict[str, Any]:
    if not hasattr(example, "toDict"):
        raise ValueError("ProgramIR predictor demos must be dspy.Example values")
    values = deepcopy(example.toDict())
    declared = getattr(example, "_input_keys", None)
    if declared is None:
        raise ValueError("ProgramIR demo is missing input designation; call .with_inputs(...)")
    values["input_keys"] = [name for name in input_names if name in declared]
    return values


def _lm_entry(lm: LM) -> dict[str, Any]:
    return {
        "forward_contract": "typed_lm",
        "weights_identity": lm.model,
        "placement": {
            "rung": "http_remote",
            "contract": "forward(LMRequest)->LMResponse",
            "endpoint_ref": "LM_ENDPOINT",
            "isolation": "none",
            "credential_ref": "LM_API_KEY",
        },
    }


def _predict_forward(input_names: list[str]) -> dict[str, Any]:
    return {
        "language": "restricted-python-ast",
        "args": input_names,
        "body": [
            {
                "node": "Assign",
                "target": "prediction",
                "value": {
                    "node": "Call",
                    "leaf": {"kind": "predict", "ref": "self"},
                    "kwargs": {name: {"node": "Var", "name": name} for name in input_names},
                },
            },
            {"node": "Return", "value": {"node": "Var", "name": "prediction"}},
        ],
    }


def _pool_name(identity: str) -> str:
    rendered = "".join(character if character.isalnum() or character in "_-" else "-" for character in identity)
    rendered = rendered.strip("-") or "lm"
    if rendered[0].isdigit():
        rendered = f"lm-{rendered}"
    return rendered
