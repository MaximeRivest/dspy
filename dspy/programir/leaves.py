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
    placement = (
        _isolation_required_placement("call(kwargs)->result")
        if authored_by == "optimizer"
        else _in_process_placement("call(kwargs)->result")
    )
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
    return AuthoredLeaf(name=name, entry=entry, source_path=path, source=source.encode("utf-8"))


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
    if tree.body[0].decorator_list:
        raise ValueError(f"ProgramIR {subject} uses decorators; bake the undecorated function instead")
    return source


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
