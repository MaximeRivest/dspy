"""Extract authored Python leaves into ProgramIR pool entries."""

from __future__ import annotations

import ast
import builtins
import inspect
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import TypeAdapter

from dspy.adapters.types.tool import Tool

_DEPS_PATTERN = re.compile(r"^\s*#\s*deps:\s*(.*?)\s*$")
_DIST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class AuthoredLeaf:
    """Hold one extracted pool entry and its source sidecar."""

    name: str
    entry: dict[str, Any]
    source_path: str
    source: bytes


def extract_tool(value: Tool | Callable[..., Any], *, name: str) -> AuthoredLeaf:
    """Extract one self-contained Python function as a tool entry."""
    tool = (
        value
        if isinstance(value, Tool) and value.name == name
        else Tool(value.func if isinstance(value, Tool) else value, name=name)
    )
    function = tool.func
    signature = inspect.signature(function)
    missing = [
        parameter.name for parameter in signature.parameters.values() if parameter.annotation is inspect.Parameter.empty
    ]
    if missing:
        raise ValueError(f"ProgramIR tool {name!r} has parameters without type hints: {missing}")
    if signature.return_annotation is inspect.Signature.empty:
        raise ValueError(f"ProgramIR tool {name!r} is missing a return type hint")
    source = _source(function, subject=f"tool {name!r}")
    _check_self_contained(function, source, subject=f"tool {name!r}")
    path = f"tools/{name}.py"
    # PIR-014: machine-written code carries its provenance into the
    # artifact. A function tagged `_dspy_authored_by` (FlexIR stamps
    # "optimizer" on generated code leaves) surfaces it here so the
    # provenance survives every recompile — a receiver can audit or
    # re-place the leaf before trusting it.
    authored_by = getattr(function, "_dspy_authored_by", None)
    # The trust pairing rule (spec/trust.md): authored-origin code runs at
    # an isolation rung the placement ENFORCES, not one admission promises.
    # Optimizer-authored code therefore does NOT get the in-process
    # placement a builtin tool gets — its placement records that it must be
    # placed under isolation the receiver supplies. `materialize` fails
    # closed on this rung: it refuses to run such a leaf in-process from its
    # sidecar unless the receiver explicitly GRANTS it (binds a callable
    # under `bindings["tool"][name]`). The binding IS the grant. NOTE: true
    # sandbox isolation (subprocess/wasm, no ambient network/host globals)
    # is NOT yet expressible in this engine — see BUILD-STATE A10 fix wave,
    # "enforced-isolation owed"; this delivers the consent gate, not the
    # sandbox rung.
    # The rung is PARAMETRIC but the default is the law above: an
    # optimizer-authored leaf is isolation-required unless the function
    # carries an explicit `_dspy_placement_rung = "in_process"` stamp —
    # the optimizing USER's own recorded choice (FlexIR code_trust). Like
    # `_dspy_authored_by`, the stamp lives on the function so the choice
    # survives every recompile, and `authored_by` stays in the entry
    # either way so a receiver can audit or re-place the leaf.
    placement_rung = getattr(function, "_dspy_placement_rung", None)
    placement = (
        _isolation_required_placement("call(kwargs)->result")
        if authored_by == "optimizer" and placement_rung != "in_process"
        else _in_process_placement("call(kwargs)->result")
    )
    # D-042 floor (PIR-016): `@dspy.tool(isolation=...)` stamps
    # `_dspy_isolation_floor` — the minimum level a receiver envelope must
    # meet, recorded on the placement as a manifest-level floor. Absent =
    # no explicit floor (the rung's authored-origin default still binds).
    floor = getattr(function, "_dspy_isolation_floor", None)
    if isinstance(floor, str):
        placement = {**placement, "isolation_floor": floor}
    lowered = getattr(function, "_dspy_floor_lowered_by", None)
    if isinstance(lowered, dict):
        placement = {**placement, "floor_lowered_by": dict(lowered)}
    entry = {
        "name": name,
        "description": tool.desc or "",
        "parameters": deepcopy_json(tool.args or {}),
        "return_schema": TypeAdapter(signature.return_annotation).json_schema(),
        "source": path,
        "deps": parse_deps(source),
        "language": "python",
        "placement": placement,
    }
    if authored_by is not None:
        entry["authored_by"] = authored_by
    # PIR-021 unified leaf substrate (D-043): a leaf may carry the
    # invocation discriminant `kind` ("call" default — absent — or
    # "session") and the closed static effect row `grants[]`. Both are
    # ADDITIVE: absent means today's call-kind, grant-free tool, so every
    # existing manifest stays byte-identical. The stamps live on the
    # function (`_dspy_leaf_kind`, `_dspy_leaf_grants`) so they survive
    # every recompile, like `_dspy_authored_by`.
    kind = getattr(function, "_dspy_leaf_kind", None)
    if kind == "session":
        entry["kind"] = "session"
    grants = getattr(function, "_dspy_leaf_grants", None)
    if grants:
        entry["grants"] = [dict(grant) for grant in grants]
    return AuthoredLeaf(name=name, entry=entry, source_path=path, source=source.encode("utf-8"))


#: Legacy prefix of the pre-contract bridge encoding: before the contract
#: landed the `leaf` grant kind (contract-sync wave 2026-08-12), the
#: bridge rode as an `fd`-kind grant named `"leaf:<pool_name>"`. New
#: manifests emit the real `leaf` kind; the prefix survives read-side only
#: so previously exported artifacts keep loading.
LEAF_GRANT_PREFIX = "leaf:"


def leaf_grant(pool_name: str) -> dict[str, str]:
    """One pool-leaf bridge grant, in contract `{kind: "leaf", name}` bytes.

    The PIR-021 grant-bridge byte shape (contract-sync wave 2026-08-12):
    a `leaf`-kind grant names a pool leaf the granted (session) leaf may
    call back into through the engine bridge — never directly.
    """
    return {"kind": "leaf", "name": pool_name}


def granted_leaf_name(grant: dict[str, Any]) -> str | None:
    """The pool leaf a bridge grant names, or None for a non-bridge grant.

    Reads the contract `leaf` kind, plus the legacy `fd`/`"leaf:<name>"`
    private overload (write path retired) so old artifacts keep loading.
    """
    name = grant.get("name", "")
    if grant.get("kind") == "leaf" and isinstance(name, str) and name:
        return name
    if grant.get("kind") == "fd" and isinstance(name, str) and name.startswith(LEAF_GRANT_PREFIX):
        return name[len(LEAF_GRANT_PREFIX) :]
    return None


def extract_metric(function: Callable[..., Any], *, name: str) -> AuthoredLeaf:
    """Extract one introspectable metric as a tool-shaped entry."""
    if not inspect.isfunction(function) or function.__name__ == "<lambda>":
        raise ValueError(f"ProgramIR metric {name!r} must be a named Python function")
    source = _source(function, subject=f"metric {name!r}")
    _check_self_contained(function, source, subject=f"metric {name!r}")
    signature = inspect.signature(function)
    if signature.return_annotation is inspect.Signature.empty:
        raise ValueError(f"ProgramIR metric {name!r} is missing a return type hint")
    path = f"metric/{name}.py"
    entry = {
        "name": name,
        "description": inspect.getdoc(function) or "",
        "return_schema": TypeAdapter(signature.return_annotation).json_schema(),
        "source": path,
        "deps": parse_deps(source),
        "language": "python",
        "placement": _in_process_placement("metric(example,prediction)->score"),
    }
    return AuthoredLeaf(name=name, entry=entry, source_path=path, source=source.encode("utf-8"))


def parse_deps(source: str) -> list[str]:
    """Return the first valid `# deps:` declaration inside authored source."""
    for line in source.splitlines()[1:]:
        match = _DEPS_PATTERN.match(line)
        if match is None:
            continue
        text = match.group(1)
        if not text:
            return []
        names = [item.strip() for item in text.split(",")]
        if any(not _DIST_NAME.match(item) for item in names):
            raise ValueError(f"invalid # deps: declaration {text!r}")
        return names
    return []


def _source(function: Callable[..., Any], *, subject: str) -> str:
    try:
        source = textwrap.dedent(inspect.getsource(function)).strip() + "\n"
    except (OSError, TypeError) as error:
        raise ValueError(f"ProgramIR {subject} source is not introspectable") from error
    tree = ast.parse(source)
    if not tree.body or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError(f"ProgramIR {subject} source is not one function definition")
    definition = tree.body[0]
    if definition.decorator_list:
        # The `@dspy.tool(...)` marker is a DECLARATION decorator: it only
        # attaches `_dspy_*` metadata this module already reads and returns
        # the SAME callable unchanged. So it is not the "hidden behavior"
        # the decorator rule forbids — strip it and keep the undecorated
        # source (which reparses cleanly). Any OTHER decorator still
        # refuses: it may wrap or replace the callable, so the baked source
        # would not be what runs.
        source = _strip_tool_decorators(source, definition, function, subject=f"ProgramIR {subject}")
    return source


_MISSING = object()


def _strip_tool_decorators(
    source: str,
    definition: ast.FunctionDef | ast.AsyncFunctionDef,
    function: Callable[..., Any],
    *,
    subject: str,
) -> str:
    """Strip genuine `@dspy.tool` markers, or refuse naming the decorator.

    Matching by NAME alone is forgeable: a foreign decorator literally
    named `tool` (another library's) would be silently stripped, changing
    semantics. So each decorator expression must RESOLVE — against the
    function's globals (and closure) — to `dspy.tooling.tool` itself.
    Anything else refuses loudly, naming the decorator: it may wrap or
    replace the callable, so the baked source would not be what runs.
    """
    from dspy.tooling import tool as marker

    for decorator in definition.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if _resolve_decorator(target, function) is not marker:
            raise ValueError(
                f"{subject} uses decorator @{ast.unparse(target)}, which is not dspy's @tool declaration "
                "marker; bake the undecorated function instead"
            )
    return _strip_decorators(source, definition)


def _resolve_decorator(target: ast.expr, function: Callable[..., Any]) -> Any:
    """Resolve a (possibly dotted) decorator expression to a live object."""
    parts: list[str] = []
    node = target
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return _MISSING
    value = _lookup_name(node.id, function)
    for attribute in reversed(parts):
        if value is _MISSING:
            return _MISSING
        value = getattr(value, attribute, _MISSING)
    return value


def _lookup_name(name: str, function: Callable[..., Any]) -> Any:
    """Look a name up the way the decoration site did: closure, globals, builtins."""
    code = getattr(function, "__code__", None)
    closure = getattr(function, "__closure__", None)
    if code is not None and closure and name in code.co_freevars:
        try:
            return closure[code.co_freevars.index(name)].cell_contents
        except ValueError:  # an empty cell — unresolvable
            return _MISSING
    namespace = getattr(function, "__globals__", None) or {}
    if name in namespace:
        return namespace[name]
    return getattr(builtins, name, _MISSING)


def _strip_decorators(source: str, definition: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Drop the decorator lines, returning the bare `def ...:` source."""
    lines = source.splitlines()
    start = definition.lineno - 1  # 1-based to 0-based; the `def` line
    return "\n".join(lines[start:]).strip() + "\n"


def _check_self_contained(function: Callable[..., Any], source: str, *, subject: str) -> None:
    if function.__closure__:
        names = sorted(function.__code__.co_freevars)
        raise ValueError(f"ProgramIR {subject} closes over values {names}; authored leaves must be self-contained")
    node = ast.parse(source).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    local = {argument.arg for argument in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
    if node.args.vararg:
        local.add(node.args.vararg.arg)
    if node.args.kwarg:
        local.add(node.args.kwarg.arg)
    for item in ast.walk(node):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(item, ast.Lambda):
                local.add(item.name)
            inner = item.args
            local.update(argument.arg for argument in (*inner.posonlyargs, *inner.args, *inner.kwonlyargs))
            if inner.vararg:
                local.add(inner.vararg.arg)
            if inner.kwarg:
                local.add(inner.kwarg.arg)
        elif isinstance(item, ast.ExceptHandler) and item.name:
            local.add(item.name)
        elif isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            local.add(item.id)
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            for alias in item.names:
                local.add(alias.asname or alias.name.split(".")[0])
    loaded = {item.id for item in ast.walk(node) if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)}
    allowed = local | set(dir(builtins))
    global_reads = sorted(loaded - allowed)
    if global_reads:
        raise ValueError(
            f"ProgramIR {subject} reads globals {global_reads}; import or define dependencies inside the function"
        )


#: The placement rung an optimizer-authored code leaf carries. It is NOT
#: `in_process`: the trust pairing rule (spec/trust.md) says authored code
#: must run at a rung whose isolation the placement ENFORCES. `materialize`
#: fails closed on this rung — it refuses to rebuild-and-run such a leaf
#: from its sidecar in-process unless the receiver explicitly grants it a
#: live callable. (Full sandbox isolation is owed, not yet delivered.)
ISOLATION_REQUIRED_RUNG = "isolation_required"


def _in_process_placement(contract: str) -> dict[str, Any]:
    return {
        "rung": "in_process",
        "contract": contract,
        "endpoint_ref": None,
        "isolation": "none",
        "credential_ref": None,
    }


def _isolation_required_placement(contract: str) -> dict[str, Any]:
    return {
        "rung": ISOLATION_REQUIRED_RUNG,
        "contract": contract,
        "endpoint_ref": None,
        "isolation": "required",
        "credential_ref": None,
    }


def deepcopy_json(value: Any) -> Any:
    """Copy inferred schemas while keeping this module independent of DSPy state."""
    import copy

    return copy.deepcopy(value)
