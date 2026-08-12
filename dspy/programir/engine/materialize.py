"""Materialize a ProgramIR into an executable program (grade 2).

`materialize(ir, bindings)` resolves every pool entry into a live
component and returns an `ExecutableProgram` that runs the compiled
forward trees through the engine interpreter. Resolution is loud: an
entry the artifact cannot rebuild and the caller did not bind refuses by
name. There is no ambient fallback — the default binding table is never
consulted here; what runs is the artifact plus the bindings you pass.

Resolution per pool:

- **LMs** (component 8): receiver bindings only. An LM entry declares
  its credential as a name, never a value, so the caller must supply the
  live LM under the entry's pool name.
- **Adapters** (component 4): receiver binding, else
  `dspy.adapters.load_entry` on the carried entry.
- **Tools** (component 6): receiver binding, else the carried source
  sidecar executed in a fresh namespace.
- **Interpreters** (component 7): receiver bindings only. The carried
  profile is structural identity, not an implementation; the engine
  never invents a runtime (D-033).
"""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any, Callable

from dspy.adapters.adapter import Adapter
from dspy.adapters.serde import load_entry
from dspy.core.example import Example
from dspy.core.prediction import Prediction
from dspy.lm.lm import LM
from dspy.modules.predict import Predict
from dspy.programir.engine import interpret
from dspy.programir.engine.errors import InterpreterError, ToolError
from dspy.programir.leaves import ISOLATION_REQUIRED_RUNG
from dspy.programir.link import link
from dspy.programir.model import ProgramIR
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import make_signature

__all__ = ["ExecutableProgram", "materialize"]

_BINDING_KINDS = ("lm", "adapter", "tool", "interpreter", "isolation")

_SHAPE_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ExecutableProgram:
    """One materialized ProgramIR, runnable through the engine interpreter.

    Call it with the root forward's keyword inputs; it returns a
    `dspy.Prediction` built from the record the interpreter produced.
    Every predict call runs a real reconstructed `dspy.Predict`, so the
    adapter exchange, demos, and config behave exactly as in native
    execution. The per-run predictor call sequence rides in the
    prediction's `_trajectory["predictor_calls"]` exhaust.
    """

    def __init__(
        self,
        *,
        forwards: dict[str, Any],
        predictors: dict[str, Predict],
        tools: dict[str, Callable[..., Any]],
        interpreters: dict[str, Any],
    ):
        self.forwards = forwards
        self.predictors = predictors
        self.tools = tools
        self.interpreters = interpreters

    def __call__(self, **inputs: Any) -> Prediction:
        leaves = _Leaves(self)
        record = interpret.run_forward(self.forwards, "", dict(inputs), leaves)
        prediction = Prediction(**record)
        prediction._trajectory["predictor_calls"] = leaves.trace
        return prediction


class _Leaves:
    """Dispatch interpreter leaf calls to the materialized components."""

    def __init__(self, program: ExecutableProgram):
        self.program = program
        self.trace: list[dict[str, Any]] = []

    def predict(self, path: str, kwargs: dict) -> dict:
        predictor = self.program.predictors.get(path)
        if predictor is None:
            raise interpret.MalformedNodeError(f"no materialized predictor at path {path!r}")
        record = predictor(**kwargs).toDict()
        self.trace.append({"predictor": path, "inputs": dict(kwargs), "outputs": dict(record)})
        return record

    def tool(self, name: str, kwargs: dict) -> Any:
        tool = self.program.tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}")
        try:
            return tool(**kwargs)
        except ToolError:
            raise
        except Exception as error:
            raise ToolError(f"tool {name!r} failed: {error}") from error

    def interpreter(self, ref: str, code: str) -> Any:
        runtime = self.program.interpreters.get(ref)
        if runtime is None:
            raise interpret.MalformedNodeError(f"no materialized interpreter for pool entry {ref!r}")
        try:
            return runtime(code=code)
        except InterpreterError:
            raise
        except Exception as error:
            raise InterpreterError(f"interpreter {ref!r} failed: {error}") from error


def materialize(ir: ProgramIR, bindings: dict[str, dict[str, Any]] | None = None) -> ExecutableProgram:
    """Materialize one ProgramIR into an executable program.

    Args:
        ir: A compiled or read ProgramIR value.
        bindings: Receiver bindings, keyed first by kind (`"lm"`,
            `"adapter"`, `"tool"`, `"interpreter"`), then by pool entry
            name. Bindings override the artifact's pools.

    Returns:
        An `ExecutableProgram` whose call runs the IR.

    Raises:
        TypeError: If `ir` is not a ProgramIR.
        ValueError: If a pool entry cannot resolve to a live component;
            the error names the entry.
    """
    if not isinstance(ir, ProgramIR):
        raise TypeError(f"programir.materialize() takes a ProgramIR, got {type(ir).__name__}")
    bindings = bindings or {}
    unknown = sorted(set(bindings) - set(_BINDING_KINDS))
    if unknown:
        raise ValueError(
            f"programir.materialize() got unknown binding kinds {unknown}; legal kinds are {list(_BINDING_KINDS)}"
        )

    manifest = ir.to_manifest()
    components = manifest["components"]
    binding_table = link(ir)

    lms = _resolve_lms(components.get("8_lm", {}), bindings.get("lm", {}))
    adapters = _resolve_adapters(components.get("4_adapter", {}), bindings.get("adapter", {}))
    tools = _resolve_tools(
        components.get("6_tools", {}),
        dict(ir.sidecars),
        bindings.get("tool", {}),
        envelope=(bindings.get("isolation") or {}).get("envelope"),
    )
    interpreters = _resolve_interpreters(components.get("7_interpreter", {}), bindings.get("interpreter", {}))

    predictors = {
        path: _build_predictor(path, components, lms[names["lm"]], adapters[names["adapter"]])
        for path, names in binding_table.items()
    }
    return ExecutableProgram(
        forwards=deepcopy(components["5_forward"]),
        predictors=predictors,
        tools=tools,
        interpreters=interpreters,
    )


def _resolve_lms(pool: dict[str, Any], bound: dict[str, Any]) -> dict[str, LM]:
    _check_bound_names("lm", pool, bound)
    resolved: dict[str, LM] = {}
    for name in pool:
        lm = bound.get(name)
        if lm is None:
            raise ValueError(
                f"programir.materialize() cannot resolve LM pool entry {name!r}: the artifact declares "
                "credentials by name only; pass bindings={'lm': {" + repr(name) + ": <dspy.LM>}}"
            )
        if not isinstance(lm, LM):
            raise ValueError(f"binding for LM pool entry {name!r} must be a dspy.LM, got {type(lm).__name__}")
        resolved[name] = lm
    return resolved


def _resolve_adapters(pool: dict[str, Any], bound: dict[str, Any]) -> dict[str, Adapter]:
    _check_bound_names("adapter", pool, bound)
    resolved: dict[str, Adapter] = {}
    for name, entry in pool.items():
        if name in bound:
            adapter = bound[name]
            if not isinstance(adapter, Adapter):
                raise ValueError(
                    f"binding for adapter pool entry {name!r} must be a dspy Adapter, got {type(adapter).__name__}"
                )
            resolved[name] = adapter
            continue
        try:
            resolved[name] = load_entry(entry)
        except Exception as error:
            raise ValueError(f"programir.materialize() cannot link adapter pool entry {name!r}: {error}") from error
    return resolved


def _envelope_satisfies_floor(envelope: Any) -> bool:
    """True when a bound isolation envelope meets the authored-leaf floor.

    The envelope may be an `IsolationPolicy`, or a plain dict/name from a
    JSON binding. Anything that resolves to level >= fork_ratchet is a
    grant for isolation-required leaves (D-042).
    """
    if envelope is None:
        return False
    from dspy.programir.engine.isolation import AUTHORED_LEAF_FLOOR, IsolationPolicy, parse_level

    if isinstance(envelope, IsolationPolicy):
        return envelope.satisfies(AUTHORED_LEAF_FLOOR)
    level = envelope.get("level") if isinstance(envelope, dict) else envelope
    try:
        return parse_level(level) >= AUTHORED_LEAF_FLOOR
    except ValueError:
        return False


def _resolve_tools(
    pool: dict[str, Any], sidecars: dict[str, bytes], bound: dict[str, Any], *, envelope: Any = None
) -> dict[str, Callable[..., Any]]:
    _check_bound_names("tool", pool, bound)
    resolved: dict[str, Callable[..., Any]] = {}
    for name, entry in pool.items():
        if name in bound:
            # A receiver binding IS the grant. It supersedes the sidecar for
            # any tool — and it is the ONLY lawful way to run a tool whose
            # placement demands isolation (below): the receiver consciously
            # supplies a live callable at a rung of its choosing.
            if not callable(bound[name]):
                raise ValueError(f"binding for tool pool entry {name!r} must be callable")
            resolved[name] = bound[name]
            continue
        # The trust pairing rule (spec/trust.md): authored-origin code runs
        # at a rung whose isolation the placement ENFORCES. An optimizer-
        # authored leaf carries the `isolation_required` rung, and this is
        # where `materialize` KEEPS that promise: it FAILS CLOSED rather
        # than rebuild-and-run the sidecar in-process. Running it would be
        # exactly the silent in-process execution of unreviewed machine-
        # written code the rule forbids. The receiver must GRANT it (bind a
        # callable) after reviewing the sidecar. (Owed: a true sandbox rung
        # that runs it under enforced isolation without a full-trust grant —
        # BUILD-STATE A10 fix wave, "enforced-isolation owed".)
        if isinstance(entry, dict) and entry.get("placement", {}).get("rung") == ISOLATION_REQUIRED_RUNG:
            # D-042: an ISOLATION ENVELOPE that meets the authored-leaf
            # floor IS a grant. When the receiver binds an envelope at
            # level >= fork_ratchet, the sidecar becomes lawfully runnable
            # — the wall the leaf demanded is present, so materialize
            # rebuilds and runs it instead of failing closed. The explicit
            # per-leaf callable grant still works (handled above).
            if _envelope_satisfies_floor(envelope):
                source_path = entry.get("source")
                source = sidecars.get(source_path) if isinstance(source_path, str) else None
                if source is None:
                    raise ValueError(
                        f"programir.materialize() cannot resolve tool pool entry {name!r}: "
                        f"source sidecar {source_path!r} is absent"
                    )
                resolved[name] = _load_function(source.decode("utf-8"), entry_name=name)
                continue
            raise ValueError(
                f"programir.materialize() refuses to run tool pool entry {name!r} in-process: it is "
                f"{entry.get('authored_by', 'authored')}-authored code whose placement requires isolation "
                "(the trust pairing rule — authored code runs at an isolation rung, never silently in-process "
                f"from its sidecar). Review tools/{name}.py, then GRANT it explicitly with "
                "bindings={'tool': {" + repr(name) + ": <reviewed callable>}}, or bind an isolation envelope "
                "at level >= fork_ratchet (the envelope IS the grant, D-042)."
            )
        source_path = entry.get("source")
        source = sidecars.get(source_path) if isinstance(source_path, str) else None
        if source is None:
            raise ValueError(
                f"programir.materialize() cannot resolve tool pool entry {name!r}: "
                f"source sidecar {source_path!r} is absent"
            )
        resolved[name] = _load_function(source.decode("utf-8"), entry_name=name)
    return resolved


def _resolve_interpreters(pool: dict[str, Any], bound: dict[str, Any]) -> dict[str, Any]:
    _check_bound_names("interpreter", pool, bound)
    resolved: dict[str, Any] = {}
    for name in pool:
        runtime = bound.get(name)
        if runtime is None:
            raise ValueError(
                f"programir.materialize() cannot resolve interpreter pool entry {name!r}: the profile is "
                "structural identity, not an implementation (D-033); pass bindings={'interpreter': {"
                + repr(name)
                + ": <runtime>}}"
            )
        if not callable(runtime):
            raise ValueError(f"binding for interpreter pool entry {name!r} must be callable")
        resolved[name] = runtime
    return resolved


def _check_bound_names(kind: str, pool: dict[str, Any], bound: dict[str, Any]) -> None:
    dangling = sorted(set(bound) - set(pool))
    if dangling:
        raise ValueError(f"programir.materialize() got {kind} bindings {dangling} that name no pool entry")


def _load_function(source: str, *, entry_name: str) -> Callable[..., Any]:
    """Rebuild one carried tool function from its baked source."""
    tree = ast.parse(source)
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(names) != 1:
        raise ValueError(f"tool pool entry {entry_name!r} source must hold exactly one function definition")
    namespace: dict[str, Any] = {}
    exec(compile(tree, filename=f"<programir tool {entry_name}>", mode="exec"), namespace)
    return namespace[names[0]]


def _build_predictor(path: str, components: dict[str, Any], lm: LM, adapter: Adapter) -> Predict:
    """Rebuild one live `dspy.Predict` from its manifest components."""
    signature_entry = components["2_signature"].get(path)
    if signature_entry is None:
        raise ValueError(f"programir.materialize() found no signature for predictor {path!r}")
    fields: dict[str, tuple[type, Any]] = {}
    for record in signature_entry["fields"]:
        annotation = _shape_type(record, path=path)
        # The manifest carries `prefix` for rendering fidelity; the live
        # field API deprecated the argument, so only `desc` maps back.
        keywords: dict[str, Any] = {}
        if record.get("desc") is not None:
            keywords["desc"] = record["desc"]
        # The semantic role must survive the rebuild: the adapter's
        # strategies predicate on roles (reasoning, tools, ...), so a
        # role-blind rebuilt signature would render different messages.
        if record.get("semantic_role") not in (None, "plain"):
            keywords["role"] = record["semantic_role"]
        maker = InputField if record["direction"] == "input" else OutputField
        fields[record["name"]] = (annotation, maker(**keywords))
    signature = make_signature(fields, components["3a_instructions"].get(path))

    predictor = Predict(signature, lm=lm, adapter=adapter, **deepcopy(components["3c_predictor_config"].get(path, {})))
    predictor.demos = [_demo(record) for record in components["3b_demos"].get(path, [])]
    return predictor


def _shape_type(record: dict[str, Any], *, path: str) -> type:
    shape_type = record.get("shape", {}).get("type")
    annotation = _SHAPE_TYPES.get(shape_type)
    if annotation is None:
        raise ValueError(
            f"programir.materialize() cannot rebuild field {record['name']!r} of predictor {path!r}: "
            f"shape type {shape_type!r} has no live Python mapping yet"
        )
    return annotation


def _demo(record: dict[str, Any]) -> Example:
    values = deepcopy(record)
    input_keys = values.pop("input_keys", [])
    return Example(**values).with_inputs(*input_keys)
