"""DSPy frontend for ProgramIR compilation."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.clients.lm import LM
from dspy.dsp.utils.settings import settings
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.programir.compile import build_program_ir
from dspy.programir.forward import LeafRef, compile_forward
from dspy.programir.model import ProgramIR
from dspy.programir.versions import IMPLEMENTED_VERSIONS
from dspy.signatures.roles import resolve_semantic_role


def compile_program(program: Module) -> ProgramIR:
    """Compile one DSPy module directly into a ProgramIR value."""
    return _DSPyCompiler().compile(program)


class _DSPyCompiler:
    def __init__(self):
        self.signatures: dict[str, Any] = {}
        self.instructions: dict[str, str] = {}
        self.demos: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.adapters: dict[str, Any] = {}
        self.lms: dict[str, Any] = {}
        self.forwards: dict[str, Any] = {}
        self.credentials: list[dict[str, str]] = []
        self._adapter_names: dict[int, str] = {}
        self._lm_names: dict[int, str] = {}
        self._module_owners: dict[int, str] = {}
        self._predictor_owners: dict[int, str] = {}
        self._default_adapter = ChatAdapter()

    def compile(self, program: Module) -> ProgramIR:
        if isinstance(program, Predict):
            tree = self.predict_node(program, path="self", name="self", root=True)
        else:
            tree = self.module_node(program, path="self", name="self")
        components = {
            "1_module_tree": tree,
            "2_signature": self.signatures,
            "3a_instructions": self.instructions,
            "3b_demos": self.demos,
            "3c_predictor_config": self.config,
            "4_adapter": self.adapters,
            "5_forward": self.forwards,
            "6_tools": {},
            "7_interpreter": {},
            "8_lm": self.lms,
            "9_environment": {},
            "10_credentials": self.credentials,
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

    def module_node(self, module: Module, *, path: str, name: str) -> dict[str, Any]:
        previous = self._module_owners.get(id(module))
        if previous is not None and previous != path:
            raise ValueError(f"DSPy module instance is shared by {previous!r} and {path!r}; shared modules are not ratified")
        self._module_owners[id(module)] = path

        children: list[dict[str, Any]] = []
        leaves: dict[str, LeafRef] = {}
        for child_name, child in module.__dict__.items():
            if not _is_identifier(child_name) or not isinstance(child, Module):
                continue
            child_path = child_name if path == "self" else f"{path}.{child_name}"
            if isinstance(child, Predict):
                children.append(self.predict_node(child, path=child_path, name=child_name))
                leaves[child_name] = LeafRef("predict", child_name)
            else:
                children.append(self.module_node(child, path=child_path, name=child_name))
                leaves[child_name] = LeafRef("module", child_name)

        self.forwards[path] = compile_forward(type(module).forward, leaves)
        class_name = type(module).__name__
        return {
            "kind": class_name,
            "name": name,
            "module_class": class_name,
            "forward_ref": f"5_forward/{path}",
            "children": children,
        }

    def predict_node(self, predictor: Predict, *, path: str, name: str, root: bool = False) -> dict[str, Any]:
        previous = self._predictor_owners.get(id(predictor))
        if previous is not None and previous != path:
            raise ValueError(
                f"DSPy predictor instance is shared by {previous!r} and {path!r}; "
                "shared predictor state is not ratified"
            )
        self._predictor_owners[id(predictor)] = path
        lm = predictor.lm or settings.lm
        if lm is None:
            raise ValueError(f"ProgramIR compile cannot resolve an LM for predictor {path!r}")
        if not isinstance(lm, LM):
            raise ValueError(
                f"ProgramIR phase 1 cannot compile {type(lm).__name__} bound to predictor {path!r}; "
                "only declared dspy.LM entries are supported before weights baking"
            )
        adapter = predictor._resolve_adapter(default=self._default_adapter)
        try:
            adapter_entry = adapter.dump_entry()
        except (AttributeError, ValueError) as error:
            raise ValueError(
                f"ProgramIR compile cannot serialize adapter {type(adapter).__name__} bound to predictor {path!r}"
            ) from error

        adapter_name = self.adapter_name(adapter, adapter_entry)
        lm_name = self.lm_name(lm)
        input_names = list(predictor.signature.input_fields)
        self.signatures[path] = {
            "fields": [_field_record(field_name, field) for field_name, field in predictor.signature.fields.items()]
        }
        self.instructions[path] = predictor.signature.instructions
        self.demos[path] = [_example_record(example, input_names, path=path) for example in predictor.demos]
        self.config[path] = deepcopy(predictor.config)

        node: dict[str, Any] = {
            "kind": "Predict",
            "name": name,
            "children": [],
            "bindings": {"adapter": adapter_name, "lm": lm_name, "delta": None},
        }
        if root:
            node["forward_ref"] = "5_forward/self"
            self.forwards["self"] = _predict_forward(input_names)
        return node

    def adapter_name(self, adapter: Any, entry: dict[str, Any]) -> str:
        existing = self._adapter_names.get(id(adapter))
        if existing is not None:
            return existing
        name = _allocate_name(entry["name"], self.adapters)
        self._adapter_names[id(adapter)] = name
        self.adapters[name] = entry
        return name

    def lm_name(self, lm: LM) -> str:
        existing = self._lm_names.get(id(lm))
        if existing is not None:
            return existing
        name = _allocate_name(_pool_name(lm.model), self.lms)
        index = len(self.lms) + 1
        endpoint_ref = "LM_ENDPOINT" if index == 1 else f"LM_ENDPOINT_{index}"
        credential_ref = "LM_API_KEY" if index == 1 else f"LM_API_KEY_{index}"
        self._lm_names[id(lm)] = name
        self.lms[name] = _lm_entry(lm, endpoint_ref=endpoint_ref, credential_ref=credential_ref)
        self.credentials.append({"name": credential_ref, "scope": f"LM {name}"})
        return name


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


def _example_record(example: Any, input_names: list[str], *, path: str) -> dict[str, Any]:
    if not hasattr(example, "toDict"):
        raise ValueError(f"ProgramIR demos for predictor {path!r} must be dspy.Example values")
    values = deepcopy(example.toDict())
    declared = getattr(example, "_input_keys", None)
    if declared is None:
        raise ValueError(f"ProgramIR demo for predictor {path!r} is missing input designation; call .with_inputs(...)")
    values["input_keys"] = [name for name in input_names if name in declared]
    return values


def _lm_entry(lm: LM, *, endpoint_ref: str, credential_ref: str) -> dict[str, Any]:
    return {
        "forward_contract": "typed_lm",
        "weights_identity": lm.model,
        "placement": {
            "rung": "http_remote",
            "contract": "forward(LMRequest)->LMResponse",
            "endpoint_ref": endpoint_ref,
            "isolation": "none",
            "credential_ref": credential_ref,
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


def _allocate_name(base: str, pool: dict[str, Any]) -> str:
    if base not in pool:
        return base
    index = 2
    while f"{base}-{index}" in pool:
        index += 1
    return f"{base}-{index}"


def _pool_name(identity: str) -> str:
    rendered = "".join(character if character.isalnum() or character in "_-" else "-" for character in identity)
    rendered = rendered.strip("-") or "lm"
    if rendered[0].isdigit():
        rendered = f"lm-{rendered}"
    return rendered


def _is_identifier(value: str) -> bool:
    return value.isidentifier() and value.isascii()
