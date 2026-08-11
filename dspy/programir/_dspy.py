"""DSPy frontend for ProgramIR compilation.

Compiles the IR-native `Module`/`Predict` tree into one ProgramIR value.
Every predictor's LM and adapter resolve through explicit bindings
(per-predictor first, then the default table `dspy.configure` writes) —
there is no ambient settings object anywhere on this path.

`compile_with_live` additionally returns the live pool bindings (pool
name -> the live LM/adapter/tool/interpreter captured at compile time),
which is exactly what `materialize` needs to execute the freshly
compiled IR in this process.
"""

from __future__ import annotations

import inspect
from copy import deepcopy
from typing import Any

from pydantic import TypeAdapter

from dspy.adapters.types.tool import Tool
from dspy.lm.bindings import BindingError
from dspy.lm.lm import LM
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.programir.compile import build_program_ir
from dspy.programir.environment import python_environment
from dspy.programir.forward import LeafRef, admit_forward, compile_forward
from dspy.programir.interpreters import extract_interpreter
from dspy.programir.leaves import extract_metric, extract_tool
from dspy.programir.model import ProgramIR
from dspy.programir.versions import IMPLEMENTED_VERSIONS
from dspy.programir.weights import bake_lm, has_weight_spec
from dspy.signatures.roles import resolve_semantic_role


def compile_program(program: Module, *, metric: Any = None, devset: Any = None) -> ProgramIR:
    """Compile one DSPy module directly into a ProgramIR value."""
    ir, _ = compile_with_live(program, metric=metric, devset=devset)
    return ir


def compile_with_live(
    program: Module, *, metric: Any = None, devset: Any = None
) -> tuple[ProgramIR, dict[str, dict[str, Any]]]:
    """Compile one module and return `(ir, live_bindings)`.

    `live_bindings` maps binding kinds (`lm`, `adapter`, `tool`,
    `interpreter`) to pool-name -> live-object tables — the bindings
    `materialize` needs to run the IR against the very objects the
    program held at compile time.
    """
    compiler = _DSPyCompiler(metric=metric, devset=devset)
    ir = compiler.compile(program)
    return ir, compiler.live


class _DSPyCompiler:
    def __init__(self, *, metric: Any = None, devset: Any = None):
        self.signatures: dict[str, Any] = {}
        self.instructions: dict[str, str] = {}
        self.demos: dict[str, Any] = {}
        self.config: dict[str, Any] = {}
        self.adapters: dict[str, Any] = {}
        self.lms: dict[str, Any] = {}
        self.forwards: dict[str, Any] = {}
        self.tools: dict[str, Any] = {}
        self.interpreters: dict[str, Any] = {}
        self.sidecars: dict[str, bytes] = {}
        self.credentials: list[dict[str, str]] = []
        self.live: dict[str, dict[str, Any]] = {"lm": {}, "adapter": {}, "tool": {}, "interpreter": {}}
        self.metric = metric
        self.devset = devset
        self._adapter_names: dict[int, str] = {}
        self._lm_names: dict[int, str] = {}
        self._module_owners: dict[int, str] = {}
        self._predictor_owners: dict[int, str] = {}
        self._interpreter_names: dict[int, str] = {}
        self._baked_lm_count = 0

    def compile(self, program: Module) -> ProgramIR:
        if isinstance(program, Predict):
            tree = self.predict_node(program, path="self", name="self", root=True)
        else:
            tree = self.module_node(program, path="self", name="self")
        evaluation = None
        metrics = {}
        if self.metric is not None:
            metric_name = self.metric.__name__ if inspect.isfunction(self.metric) else type(self.metric).__name__
            extracted = extract_metric(self.metric, name=metric_name)
            metrics[metric_name] = extracted.entry
            self.sidecars[extracted.source_path] = extracted.source
        if self.metric is not None or self.devset is not None:
            evaluation = {
                "metrics": metrics,
                "devset": [_devset_record(example) for example in (self.devset or [])],
            }

        authored_lms = [
            entry["class"] for entry in self.lms.values() if entry.get("class", {}).get("origin") == "authored"
        ]
        authored_python = [*self.tools.values(), *metrics.values(), *authored_lms]
        dependencies = [dependency for entry in authored_python for dependency in entry.get("deps", [])]
        python_block, entry_source = python_environment(dependencies)
        environment = {"python": python_block}
        self.sidecars[python_block["pep723_entry"]] = entry_source

        components = {
            "1_module_tree": tree,
            "2_signature": self.signatures,
            "3a_instructions": self.instructions,
            "3b_demos": self.demos,
            "3c_predictor_config": self.config,
            "4_adapter": self.adapters,
            "5_forward": self.forwards,
            "6_tools": self.tools,
            "7_interpreter": self.interpreters,
            "8_lm": self.lms,
            "9_environment": environment,
            "10_credentials": self.credentials,
            # Contents loose until ratified; the greenfield core has no
            # ambient settings to record.
            "11_ambient_policy": {},
        }
        if evaluation is not None:
            components["12_metric"] = evaluation
        return build_program_ir(
            versions=dict(IMPLEMENTED_VERSIONS),
            components=components,
            provenance={"source": "dspy.export", "evidence": "dspy frontend compile"},
            sidecars=self.sidecars,
        )

    def module_node(self, module: Module, *, path: str, name: str) -> dict[str, Any]:
        previous = self._module_owners.get(id(module))
        if previous is not None and previous != path:
            raise ValueError(
                f"DSPy module instance is shared by {previous!r} and {path!r}; shared modules are not ratified"
            )
        self._module_owners[id(module)] = path

        children: list[dict[str, Any]] = []
        leaves: dict[str, LeafRef] = {}
        module_tools: list[str] = []
        uses_interpreter = False
        for child_name, child in module.__dict__.items():
            if not _is_identifier(child_name) or child_name == "forward":
                continue
            if isinstance(child, Module):
                child_path = child_name if path == "self" else f"{path}.{child_name}"
                if isinstance(child, Predict):
                    children.append(self.predict_node(child, path=child_path, name=child_name))
                    leaves[child_name] = LeafRef("predict", child_name)
                else:
                    children.append(self.module_node(child, path=child_path, name=child_name))
                    leaves[child_name] = LeafRef("module", child_name)
            elif isinstance(child, Tool) or inspect.isfunction(child):
                tool_name = self.register_tool(child, name=child_name)
                leaves[child_name] = LeafRef("tool", tool_name)
                module_tools.append(tool_name)
            elif (
                isinstance(child, dict)
                and child
                and all(
                    isinstance(key, str)
                    and _is_identifier(key)
                    and (isinstance(value, Tool) or inspect.isfunction(value))
                    for key, value in child.items()
                )
            ):
                for tool_name, tool in child.items():
                    module_tools.append(self.register_tool(tool, name=tool_name))
                leaves[child_name] = LeafRef("tool")
            elif _is_interpreter(child):
                interpreter_name = self.register_interpreter(child, name=child_name)
                leaves[child_name] = LeafRef("interpreter", interpreter_name)
                uses_interpreter = True

        # IR-first door: a module that BUILDS its forward (build.py
        # constructors) hands the tree over directly — no source parse.
        # `build_forward_ir` bakes the current declared-literal values and
        # refreshes the printed native twin; the shared admission runs
        # here with the declared leaf table, exactly as compile_forward
        # runs it on parsed source.
        builder = getattr(module, "build_forward_ir", None)
        if callable(builder):
            self.forwards[path] = admit_forward(builder(), leaves)
        else:
            # A generated forward bound on the instance wins over the class's.
            forward_fn = module.__dict__.get("forward") or type(module).forward
            self.forwards[path] = compile_forward(
                forward_fn,
                leaves,
                literals=_declared_literals(module),
                signature=_declared_signature(module),
            )
        class_name = type(module).__name__
        node = {
            "kind": class_name,
            "name": name,
            "module_class": class_name,
            "forward_ref": f"5_forward/{path}",
            "children": children,
        }
        if module_tools:
            node["tools"] = list(dict.fromkeys(module_tools))
        if uses_interpreter:
            node["uses_interpreter"] = True
        return node

    def predict_node(self, predictor: Predict, *, path: str, name: str, root: bool = False) -> dict[str, Any]:
        previous = self._predictor_owners.get(id(predictor))
        if previous is not None and previous != path:
            raise ValueError(
                f"DSPy predictor instance is shared by {previous!r} and {path!r}; "
                "shared predictor state is not ratified"
            )
        self._predictor_owners[id(predictor)] = path
        try:
            lm = predictor.resolve_lm()
        except BindingError as error:
            raise BindingError(f"ProgramIR compile cannot resolve an LM for predictor {path!r}: {error}") from error
        if not isinstance(lm, LM) and not has_weight_spec(lm):
            raise ValueError(
                f"ProgramIR cannot compile {type(lm).__name__} bound to predictor {path!r}; "
                "bind a dspy.LM, or a custom class declaring programir_weight_spec()"
            )
        adapter = predictor.resolve_adapter()
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

    def register_interpreter(self, interpreter: Any, *, name: str) -> str:
        existing = self._interpreter_names.get(id(interpreter))
        if existing is not None:
            return existing
        pool_name = _allocate_name(_pool_name(name), self.interpreters)
        self.interpreters[pool_name] = extract_interpreter(interpreter, name=pool_name)
        self._interpreter_names[id(interpreter)] = pool_name
        self.live["interpreter"][pool_name] = interpreter
        return pool_name

    def register_tool(self, tool: Tool | Any, *, name: str) -> str:
        extracted = extract_tool(tool, name=name)
        existing = self.tools.get(name)
        if existing is not None:
            if existing != extracted.entry or self.sidecars[extracted.source_path] != extracted.source:
                raise ValueError(f"ProgramIR tool name {name!r} refers to multiple function identities")
            return name
        self.tools[name] = extracted.entry
        self.sidecars[extracted.source_path] = extracted.source
        self.live["tool"][name] = tool
        return name

    def adapter_name(self, adapter: Any, entry: dict[str, Any]) -> str:
        existing = self._adapter_names.get(id(adapter))
        if existing is not None:
            return existing
        name = _allocate_name(entry["name"], self.adapters)
        self._adapter_names[id(adapter)] = name
        self.adapters[name] = entry
        self.live["adapter"][name] = adapter
        return name

    def lm_name(self, lm: LM) -> str:
        existing = self._lm_names.get(id(lm))
        if existing is not None:
            return existing
        if has_weight_spec(lm):
            spec = lm.programir_weight_spec()
            identity = spec.get("weights_identity") if isinstance(spec, dict) else type(lm).__name__
            name = _allocate_name(_pool_name(identity if isinstance(identity, str) else type(lm).__name__), self.lms)
            root = "weights" if self._baked_lm_count == 0 else f"weights/{name}"
            baked = bake_lm(lm, name=name, weights_root=root)
            self._baked_lm_count += 1
            self._lm_names[id(lm)] = name
            self.lms[name] = baked.entry
            self.sidecars.update(baked.sidecars)
            self.live["lm"][name] = lm
            return name

        name = _allocate_name(_pool_name(lm.model), self.lms)
        index = len(self.credentials) + 1
        endpoint_ref = "LM_ENDPOINT" if index == 1 else f"LM_ENDPOINT_{index}"
        credential_ref = "LM_API_KEY" if index == 1 else f"LM_API_KEY_{index}"
        self._lm_names[id(lm)] = name
        self.lms[name] = _lm_entry(lm, endpoint_ref=endpoint_ref, credential_ref=credential_ref)
        self.credentials.append({"name": credential_ref, "scope": f"LM {name}"})
        self.live["lm"][name] = lm
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


def _devset_record(example: Any) -> dict[str, Any]:
    if not hasattr(example, "toDict"):
        raise ValueError("ProgramIR devset values must be dspy.Example values")
    values = deepcopy(example.toDict())
    declared = getattr(example, "_input_keys", None)
    if declared is None:
        raise ValueError("ProgramIR devset example is missing input designation; call .with_inputs(...)")
    values["input_keys"] = [name for name in values if name in declared]
    return values


def _lm_entry(lm: LM, *, endpoint_ref: str, credential_ref: str) -> dict[str, Any]:
    return {
        "forward_contract": "typed_lm",
        "weights_identity": lm.model,
        # `config` is the schema's loose entry-level slot; the declared
        # capability facts ride here as provenance. The receiver's bound
        # LM brings its own facts at materialize time.
        "config": {"lm_capabilities": lm.capabilities.to_dict()},
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


def _declared_literals(module: Module) -> dict[str, Any]:
    """Resolve the class's `ir_literals` names to instance values.

    `ir_literals` is a tuple of attribute names on the module class; each
    named attribute must hold a JSON scalar at compile time. The compiler
    bakes these values into the forward as `Const` nodes, which is how a
    loop cap like `max_iters` stays configuration on the instance yet a
    literal in the artifact (zero-reach-back holds at run time).
    """
    names = getattr(type(module), "ir_literals", ())
    if isinstance(names, str) or not all(isinstance(name, str) for name in names):
        raise ValueError(
            f"{type(module).__name__}.ir_literals must be an iterable of attribute-name strings, got {names!r}"
        )
    literals: dict[str, Any] = {}
    for name in names:
        if not hasattr(module, name):
            raise ValueError(
                f"{type(module).__name__}.ir_literals names {name!r}, but the instance has no such attribute"
            )
        value = getattr(module, name)
        if not (value is None or isinstance(value, (str, int, float, bool))):
            raise ValueError(
                f"{type(module).__name__}.{name} is declared in ir_literals and must be a JSON scalar "
                f"to bake into the forward, got {type(value).__name__}"
            )
        literals[name] = value
    return literals


def _declared_signature(module: Module) -> list[str] | None:
    """The module's declared input field names, or None when it has none.

    The v0.4 record envelope (D-041) needs the module's own input
    signature (component 2, D-036): a signature-polymorphic forward
    (`def forward(self, inputs)` or `**kwargs`) threads exactly these
    fields. A module without a `signature.input_fields` attribute (a
    plain composite whose forward names its inputs) has no record
    envelope available — the compiler stays byte-identical to v0.3.
    """
    signature = getattr(module, "signature", None)
    if signature is None or not hasattr(signature, "input_fields"):
        return None
    return list(signature.input_fields)


def _is_identifier(value: str) -> bool:
    return value.isidentifier() and value.isascii()


def _is_interpreter(value: Any) -> bool:
    from dspy.primitives.python_interpreter import PythonInterpreter

    return isinstance(value, PythonInterpreter) or callable(getattr(value, "programir_profile", None))
