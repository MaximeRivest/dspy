"""Shared manifest plumbing for the ProgramIR workbench tools.

Every tool in this package is a pure function of the manifest dict: no LM
calls, no execution of authored code, no filesystem access beyond loading
the manifest the caller names. This module holds the loading seam and the
small amount of graph plumbing (predictor paths, call-site resolution,
constant folding) that all three tools consume.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

WIDTH = 74


def rule(title: str) -> str:
    """Render one explain-style section rule."""
    return f"== {title} " + "=" * max(0, WIDTH - len(title) - 4)


def load_manifest(source: Any) -> dict[str, Any]:
    """Resolve one tool target into a plain manifest dict.

    Accepts, in order of checks: a ProgramIR value, a manifest mapping, an
    artifact directory (or its `manifest.json`), or a `module:attribute`
    import spec naming a DSPy module, a factory, or a ProgramIR value.

    Args:
        source: The target named by a caller or on the command line.

    Returns:
        A detached manifest dict with a `components` key.

    Raises:
        ValueError: If the target cannot be resolved to a manifest.
    """
    if hasattr(source, "to_manifest"):
        return source.to_manifest()
    if isinstance(source, Mapping):
        if "components" not in source:
            raise ValueError("manifest mapping has no 'components' key")
        return json.loads(json.dumps(dict(source)))
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            path = path / "manifest.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        if isinstance(source, str) and ":" in source:
            return _import_target(source)
        raise ValueError(f"no artifact directory or manifest.json at {source!r}")
    raise ValueError(f"cannot load a manifest from {type(source).__name__}")


def _import_target(spec: str) -> dict[str, Any]:
    """Import `module:attribute`, compile it if needed, and return the manifest."""
    module_name, _, attribute = spec.partition(":")
    value = getattr(importlib.import_module(module_name), attribute)
    if callable(value) and not hasattr(value, "to_manifest"):
        value = value()
    if hasattr(value, "to_manifest"):
        return value.to_manifest()
    from dspy.programir.compile import compile as compile_ir

    return compile_ir(value).to_manifest()


# ─── Module tree plumbing ────────────────────────────────────────────


def predictor_nodes(tree: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Map each predictor path (component-map key) to its tree node."""
    out: dict[str, dict[str, Any]] = {}

    def walk(node: Mapping[str, Any], prefix: str) -> None:
        if node.get("kind") == "Predict":
            out[prefix or "self"] = dict(node)
            return
        for child in node.get("children") or []:
            name = child.get("name", "?")
            walk(child, name if not prefix else f"{prefix}.{name}")

    walk(tree, "")
    return out


def module_paths(tree: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Map each non-predictor module path to its tree node."""
    out: dict[str, dict[str, Any]] = {}

    def walk(node: Mapping[str, Any], path: str) -> None:
        if node.get("kind") == "Predict" and not node.get("forward_ref"):
            return
        out[path] = dict(node)
        for child in node.get("children") or []:
            name = child.get("name", "?")
            walk(child, name if path == "self" else f"{path}.{name}")

    walk(tree, "self")
    return out


def child_path(module_path: str, child_name: str) -> str:
    """Join one child name onto a module path with the compiler's convention."""
    return child_name if module_path == "self" else f"{module_path}.{child_name}"


# ─── Node walking ────────────────────────────────────────────────────


def walk_statements(body: list[Any] | None, prefix: str) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield `(manifest_path, statement)` for every statement, depth first.

    Paths use the `body[i]` indexing style, e.g. `5_forward/self/body[1]/
    orelse[0]`, so a finding points into the manifest like a file:line.
    """
    for index, statement in enumerate(body or []):
        if not isinstance(statement, dict):
            continue
        path = f"{prefix}[{index}]"
        yield path, statement
        for key in ("body", "orelse"):
            if statement.get(key):
                yield from walk_statements(statement[key], f"{path}/{key}")
        for h_index, handler in enumerate(statement.get("handlers") or []):
            yield from walk_statements(handler.get("body"), f"{path}/handlers[{h_index}]/body")


def walk_expressions(value: Any) -> Iterator[dict[str, Any]]:
    """Yield every expression node reachable inside one statement or expr."""
    if isinstance(value, dict):
        if "node" in value:
            yield value
        for item in value.values():
            yield from walk_expressions(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_expressions(item)


def statement_expressions(statement: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the expression nodes owned by one statement, not its sub-blocks."""
    for key in ("value", "test"):
        if key in statement:
            yield from walk_expressions(statement[key])


def const_test(test: Any) -> bool | None:
    """Fold one test expression to a constant, or return None.

    Folds `Const` truthiness and `Compare` between two `Const` operands —
    exactly the shapes an author can write by accident.
    """
    if not isinstance(test, dict):
        return None
    if test.get("node") == "Const":
        return bool(test.get("value"))
    if test.get("node") == "Compare":
        left, right = test.get("left"), test.get("right")
        if (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.get("node") == "Const"
            and right.get("node") == "Const"
        ):
            equal = left.get("value") == right.get("value")
            return equal if test.get("op") == "eq" else not equal
    return None


def render_expr(value: Any) -> str:
    """Render one expression the way explain (View 1) prints it."""
    from dspy.programir import explain_view

    return explain_view._render_expr(value)
