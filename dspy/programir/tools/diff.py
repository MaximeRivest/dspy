"""Semantic diff of two ProgramIR manifests (Tier 1, staging-lessons §B).

An optimization step is a labeled diff between two complete programs
(IR-program-spec §e1, View 3): which optimizable field changed, where.
This tool renders that diff as readable explain-style text, not raw JSON:

- per component pool (adapters, tools, interpreters, LMs, metrics):
  added / removed / changed entries, with the changed sub-keys named;
- per predictor: instruction delta (old -> new), demo delta (added /
  removed / edited, summarized by input/label keys), config and binding
  deltas;
- per module forward: a node-level tree diff — which statement was
  added, removed, or rewritten, under which branch — using the same
  statement rendering as explain (View 1).

Pure over the two manifest dicts: no LM calls, no execution.

Run from Python (`diff(a, b)` / `build_text(a, b)`) or the shell:

    python -m dspy.programir.tools.diff <old-target> <new-target>
"""

from __future__ import annotations

import difflib
import json
import sys
from typing import Any, Mapping

from dspy.programir.tools._common import (
    WIDTH,
    load_manifest,
    predictor_nodes,
    render_expr,
    rule,
)

__all__ = ["diff", "build_text", "main"]

_POOLS = (
    ("4_adapter", "adapter"),
    ("6_tools", "tool"),
    ("7_interpreter", "interpreter"),
    ("8_lm", "lm"),
)


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def diff(old: Any, new: Any) -> list[str]:
    """Compute the semantic diff between two manifests as report lines.

    Args:
        old: The baseline target (ProgramIR, manifest dict, path, or spec).
        new: The candidate target, same forms accepted.

    Returns:
        The rendered diff lines; empty when the manifests are equivalent.
    """
    a = load_manifest(old)
    b = load_manifest(new)
    if a == b:
        return []
    lines: list[str] = []
    lines += _diff_versions(a.get("versions", {}), b.get("versions", {}))
    components_a = a.get("components", {})
    components_b = b.get("components", {})
    lines += _diff_predictors(components_a, components_b)
    lines += _diff_forwards(components_a, components_b)
    lines += _diff_pools(components_a, components_b)
    lines += _diff_metric(components_a, components_b)
    return lines


def build_text(old: Any, new: Any) -> str:
    """Render the semantic diff as the full report text."""
    out = ["=" * WIDTH, " ProgramIR diff — old -> new (semantic, per component)", "=" * WIDTH]
    lines = diff(old, new)
    if not lines:
        out.append("  (no differences — the two manifests are equivalent)")
    else:
        out += lines
    out.append("=" * WIDTH)
    return "\n".join(out)


# ─── Sections ────────────────────────────────────────────────────────


def _diff_versions(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    if a == b:
        return []
    lines = [rule("VERSIONS")]
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            lines.append(f"  ~ {key}: {a.get(key, '(absent)')} -> {b.get(key, '(absent)')}")
    return lines


def _demo_summary(demo: Mapping[str, Any]) -> str:
    input_keys = demo.get("input_keys", []) or []
    rendered = ", ".join(f"{key}={_dumps(demo.get(key))}" for key in input_keys)
    labels = sorted(set(demo) - set(input_keys) - {"input_keys"})
    label_text = ", ".join(f"{key}={_dumps(demo.get(key))}" for key in labels)
    return f"inputs({rendered}) -> labels({label_text})"


def _diff_predictors(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    bindings_a = {p: n.get("bindings") for p, n in predictor_nodes(a.get("1_module_tree", {})).items()}
    bindings_b = {p: n.get("bindings") for p, n in predictor_nodes(b.get("1_module_tree", {})).items()}
    signatures_a, signatures_b = a.get("2_signature", {}), b.get("2_signature", {})
    paths = sorted(set(signatures_a) | set(signatures_b))
    body: list[str] = []
    for path in paths:
        entry: list[str] = []
        if path not in signatures_a:
            body.append(f"  + predictor '{path}' added")
            continue
        if path not in signatures_b:
            body.append(f"  - predictor '{path}' removed")
            continue

        instruction_a = a.get("3a_instructions", {}).get(path)
        instruction_b = b.get("3a_instructions", {}).get(path)
        if instruction_a != instruction_b:
            entry.append(f"    ~ instructions (3a): {_dumps(instruction_a)}")
            entry.append(f"                      -> {_dumps(instruction_b)}")

        demos_a = a.get("3b_demos", {}).get(path) or []
        demos_b = b.get("3b_demos", {}).get(path) or []
        if demos_a != demos_b:
            entry.append(f"    ~ demos (3b): {len(demos_a)} -> {len(demos_b)} baked")
            matcher = difflib.SequenceMatcher(
                a=[_dumps(demo) for demo in demos_a],
                b=[_dumps(demo) for demo in demos_b],
            )
            for tag, i1, i2, j1, j2 in matcher.get_opcodes():
                if tag in ("delete", "replace"):
                    for demo in demos_a[i1:i2]:
                        entry.append(f"      - {_demo_summary(demo)}")
                if tag in ("insert", "replace"):
                    for demo in demos_b[j1:j2]:
                        entry.append(f"      + {_demo_summary(demo)}")

        config_a = a.get("3c_predictor_config", {}).get(path) or {}
        config_b = b.get("3c_predictor_config", {}).get(path) or {}
        for key in sorted(set(config_a) | set(config_b)):
            if config_a.get(key) != config_b.get(key):
                entry.append(f"    ~ config (3c) {key}: {_dumps(config_a.get(key))} -> {_dumps(config_b.get(key))}")

        bind_a = bindings_a.get(path) or {}
        bind_b = bindings_b.get(path) or {}
        for key in sorted(set(bind_a) | set(bind_b)):
            if bind_a.get(key) != bind_b.get(key):
                entry.append(f"    ~ binding {key}: {bind_a.get(key)} -> {bind_b.get(key)}")

        if signatures_a.get(path) != signatures_b.get(path):
            fields_a = {f["name"]: f for f in signatures_a[path].get("fields", [])}
            fields_b = {f["name"]: f for f in signatures_b[path].get("fields", [])}
            for name in sorted(set(fields_a) | set(fields_b)):
                if name not in fields_b:
                    entry.append(f"    - field .{name} removed")
                elif name not in fields_a:
                    field = fields_b[name]
                    entry.append(f"    + field .{name} added ({field.get('direction')})")
                elif fields_a[name] != fields_b[name]:
                    changed = [
                        key
                        for key in set(fields_a[name]) | set(fields_b[name])
                        if fields_a[name].get(key) != fields_b[name].get(key)
                    ]
                    entry.append(f"    ~ field .{name}: {', '.join(sorted(changed))} changed")

        if entry:
            body.append(f"  predictor '{path}'")
            body += entry
    if not body:
        return []
    return [rule("PREDICTORS (2, 3a, 3b, 3c, bindings)")] + body


def _diff_forwards(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    forwards_a, forwards_b = a.get("5_forward", {}), b.get("5_forward", {})
    if forwards_a == forwards_b:
        return []
    lines = [rule("FORWARD (5) — node-level tree diff")]
    for module in sorted(set(forwards_a) | set(forwards_b)):
        spec_a, spec_b = forwards_a.get(module), forwards_b.get(module)
        if spec_a == spec_b:
            continue
        if spec_a is None:
            lines.append(f"  + forward '{module}' added")
            continue
        if spec_b is None:
            lines.append(f"  - forward '{module}' removed")
            continue
        lines.append(f"  forward '{module}'")
        if spec_a.get("args") != spec_b.get("args"):
            lines.append(f"    ~ args: ({', '.join(spec_a.get('args', []))}) -> ({', '.join(spec_b.get('args', []))})")
        lines += _diff_body(spec_a.get("body") or [], spec_b.get("body") or [], f"5_forward/{module}/body")
    return lines


def _statement_head(statement: Mapping[str, Any]) -> str:
    """One-line explain-style rendering of a statement's own clause."""
    node = statement.get("node")
    if node == "Assign":
        return f"{statement.get('target')} = {render_expr(statement.get('value'))}"
    if node == "Return":
        return f"return {render_expr(statement.get('value'))}"
    if node in ("If", "While"):
        return f"{node.lower()} {render_expr(statement.get('test'))}:"
    if node == "For":
        return f"for {statement.get('target')} in range({statement.get('range')}):"
    if node == "Try":
        return "try:"
    if node == "Raise":
        return f"raise {statement.get('exc')}({_dumps(statement.get('message'))})"
    return str(node).lower() if node else _dumps(statement)


def _diff_body(old: list[Any], new: list[Any], path: str) -> list[str]:
    """Node-level statement-list diff: added / removed / rewritten, where."""
    lines: list[str] = []
    matcher = difflib.SequenceMatcher(
        a=[_dumps(statement) for statement in old],
        b=[_dumps(statement) for statement in new],
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            # Pairwise: same slot rewritten — recurse into matching shapes.
            for offset in range(i2 - i1):
                lines += _diff_statement(old[i1 + offset], new[j1 + offset], f"{path}[{i1 + offset}]")
            continue
        for index in range(i1, i2):
            lines.append(f"    - {path}[{index}]: {_statement_head(old[index])}")
        for index in range(j1, j2):
            lines.append(f"    + {path}[{index}]: {_statement_head(new[index])}")
    return lines


def _diff_statement(old: Mapping[str, Any], new: Mapping[str, Any], path: str) -> list[str]:
    if old.get("node") != new.get("node"):
        return [f"    - {path}: {_statement_head(old)}", f"    + {path}: {_statement_head(new)}"]
    lines: list[str] = []
    head_a, head_b = _statement_head(old), _statement_head(new)
    if head_a != head_b:
        lines.append(f"    ~ {path}: {head_a}")
        lines.append(f"      {' ' * len(path)}-> {head_b}")
    for key in ("body", "orelse"):
        if old.get(key) != new.get(key):
            lines += _diff_body(old.get(key) or [], new.get(key) or [], f"{path}/{key}")
    if old.get("handlers") != new.get("handlers"):
        handlers_a, handlers_b = old.get("handlers") or [], new.get("handlers") or []
        for index in range(max(len(handlers_a), len(handlers_b))):
            handler_a = handlers_a[index] if index < len(handlers_a) else None
            handler_b = handlers_b[index] if index < len(handlers_b) else None
            handler_path = f"{path}/handlers[{index}]"
            if handler_a is None:
                lines.append(f"    + {handler_path}: except {handler_b.get('type')}")
            elif handler_b is None:
                lines.append(f"    - {handler_path}: except {handler_a.get('type')}")
            elif handler_a != handler_b:
                if handler_a.get("type") != handler_b.get("type"):
                    lines.append(
                        f"    ~ {handler_path}: except {handler_a.get('type')} -> except {handler_b.get('type')}"
                    )
                lines += _diff_body(handler_a.get("body") or [], handler_b.get("body") or [], f"{handler_path}/body")
    return lines


def _diff_pools(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for pool_key, label in _POOLS:
        pool_a, pool_b = a.get(pool_key, {}) or {}, b.get(pool_key, {}) or {}
        if pool_a == pool_b:
            continue
        lines.append(rule(f"{label.upper()} POOL ({pool_key.split('_')[0]})"))
        for name in sorted(set(pool_a) | set(pool_b)):
            if name not in pool_b:
                lines.append(f"  - {label} '{name}' removed")
            elif name not in pool_a:
                lines.append(f"  + {label} '{name}' added")
            elif pool_a[name] != pool_b[name]:
                entry_a, entry_b = pool_a[name], pool_b[name]
                if isinstance(entry_a, Mapping) and isinstance(entry_b, Mapping):
                    changed = sorted(key for key in set(entry_a) | set(entry_b) if entry_a.get(key) != entry_b.get(key))
                    lines.append(f"  ~ {label} '{name}': {', '.join(changed)} changed")
                    for key in changed:
                        lines.append(f"      {key}: {_dumps(entry_a.get(key))} -> {_dumps(entry_b.get(key))}")
                else:
                    lines.append(f"  ~ {label} '{name}' changed")
    return lines


def _diff_metric(a: Mapping[str, Any], b: Mapping[str, Any]) -> list[str]:
    metric_a, metric_b = a.get("12_metric"), b.get("12_metric")
    if metric_a == metric_b:
        return []
    lines = [rule("METRIC (12)")]
    if metric_a is None:
        lines.append("  + evaluation block added")
        metric_a = {}
    elif metric_b is None:
        lines.append("  - evaluation block removed (ship object)")
        metric_b = {}
    metrics_a = (metric_a or {}).get("metrics", {})
    metrics_b = (metric_b or {}).get("metrics", {})
    for name in sorted(set(metrics_a) | set(metrics_b)):
        if name not in metrics_b:
            lines.append(f"  - metric '{name}' removed")
        elif name not in metrics_a:
            lines.append(f"  + metric '{name}' added")
        elif metrics_a[name] != metrics_b[name]:
            lines.append(f"  ~ metric '{name}' changed")
    devset_a = (metric_a or {}).get("devset", [])
    devset_b = (metric_b or {}).get("devset", [])
    if devset_a != devset_b:
        lines.append(f"  ~ devset: {len(devset_a)} -> {len(devset_b)} examples")
    return lines


def main(argv: list[str] | None = None) -> int:
    """CLI entry: diff two artifact directories, manifests, or import specs."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 2:
        print("usage: python -m dspy.programir.tools.diff <old-target> <new-target>", file=sys.stderr)
        return 2
    print(build_text(args[0], args[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
