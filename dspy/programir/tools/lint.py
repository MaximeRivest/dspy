"""Static lint over a ProgramIR manifest (Tier 1, staging-lessons §B).

Every check is a pure read of the manifest JSON: no LM calls, no authored
code runs. Findings speak in the teaching-error voice of the compiler's
refusals — each one says what the manifest declares, what the forward
graph actually does, and why the gap will bite — and each carries a
file:line-style path into the manifest (`5_forward/self/body[2]/test`).

Checks (code — meaning):
    PIR-L-FIELD-IN    input signature field never passed at any call site
    PIR-L-FIELD-OUT   output signature field never read from any result
    PIR-L-UNREACH     pool entry (predictor/tool/interpreter/adapter/lm)
                      never used on any path from the root forward
    PIR-L-DEADBRANCH  If/While test folds to a constant; a block never runs
    PIR-L-DEMO        baked demo keys do not match the signature fields
    PIR-L-WHILE       While whose test no body statement can change
    PIR-L-EXCEPT      except handler with an empty body (error swallowed)

Run from Python (`lint(manifest)` / `build_text(manifest)`) or the shell:

    python -m dspy.programir.tools.lint <artifact-dir | manifest.json | pkg.mod:obj>
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

from dspy.programir.tools._common import (
    WIDTH,
    child_path,
    const_test,
    load_manifest,
    module_paths,
    predictor_nodes,
    render_expr,
    rule,
    statement_expressions,
    walk_expressions,
    walk_statements,
)

__all__ = ["Finding", "lint", "build_text", "main"]

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Finding:
    """One lint finding: severity, stable code, manifest path, teaching text."""

    severity: str
    code: str
    path: str
    message: str

    def render(self) -> str:
        return f"[{self.severity:<7}] {self.code:<16} {self.path}\n    {self.message}"


def lint(source: Any) -> list[Finding]:
    """Run every static check over one manifest and return the findings.

    Args:
        source: A ProgramIR value, manifest dict, artifact path, or import
            spec (see `_common.load_manifest`).

    Returns:
        Findings sorted by severity, then manifest path.
    """
    manifest = load_manifest(source)
    components = manifest.get("components", {})
    findings: list[Finding] = []
    findings += _check_dead_branches(components)
    findings += _check_while_state(components)
    findings += _check_empty_except(components)
    findings += _check_demo_keys(components)
    calls = _collect_live_calls(components)
    findings += _check_signature_fields(components, calls)
    findings += _check_unreachable(components, calls)
    findings.sort(key=lambda f: (_SEVERITY_ORDER.get(f.severity, 9), f.path, f.code))
    return findings


def build_text(source: Any) -> str:
    """Render the findings as the lint report text."""
    findings = lint(source)
    out = ["=" * WIDTH, " ProgramIR lint — static findings over the manifest", "=" * WIDTH]
    if not findings:
        out.append("  (clean — no findings)")
    else:
        counts: dict[str, int] = {}
        for finding in findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
            out.append(finding.render())
        out.append(rule("SUMMARY"))
        out.append(
            "  "
            + "  ".join(
                f"{severity}: {counts[severity]}" for severity in ("error", "warning", "info") if severity in counts
            )
        )
    out.append("=" * WIDTH)
    return "\n".join(out)


# ─── Call-site collection ────────────────────────────────────────────


@dataclass(frozen=True)
class CallSite:
    """One resolved leaf call inside a live (non-dead) statement."""

    module: str  # module path owning the forward
    path: str  # manifest path of the enclosing statement
    kind: str  # predict | module | tool | interpreter
    target: str | None  # resolved component-map / pool key; None = dynamic
    kwargs: tuple[str, ...]
    assign_target: str | None


def _live_statements(body: list[Any] | None, prefix: str) -> Iterator[tuple[str, dict]]:
    """Walk statements, skipping blocks a constant test proves dead."""
    for index, statement in enumerate(body or []):
        if not isinstance(statement, dict):
            continue
        path = f"{prefix}[{index}]"
        yield path, statement
        node = statement.get("node")
        folded = const_test(statement.get("test")) if node in ("If", "While") else None
        if not (node == "If" and folded is False) and not (node == "While" and folded is False):
            yield from _live_statements(statement.get("body"), f"{path}/body")
        if node == "If" and folded is not True:
            yield from _live_statements(statement.get("orelse"), f"{path}/orelse")
        elif node != "If" and statement.get("orelse"):
            yield from _live_statements(statement.get("orelse"), f"{path}/orelse")
        for h_index, handler in enumerate(statement.get("handlers") or []):
            yield from _live_statements(handler.get("body"), f"{path}/handlers[{h_index}]/body")


def _collect_live_calls(components: Mapping[str, Any]) -> list[CallSite]:
    """Collect every leaf call reachable from the root forward.

    The walk starts at `5_forward/self` and follows module calls only, so
    a sub-module nobody calls contributes no call sites — which is exactly
    what makes its own leaves report as unreachable.
    """
    forwards = components.get("5_forward", {})
    calls: list[CallSite] = []
    visited: set[str] = set()
    queue = ["self"] if "self" in forwards else sorted(forwards)[:1]
    while queue:
        module = queue.pop(0)
        if module in visited:
            continue
        visited.add(module)
        spec = forwards.get(module) or {}
        for path, statement in _live_statements(spec.get("body"), f"5_forward/{module}/body"):
            assign_target = statement.get("target") if statement.get("node") == "Assign" else None
            for expr in statement_expressions(statement):
                if expr.get("node") != "Call":
                    continue
                leaf = expr.get("leaf", {})
                kind = leaf.get("kind", "?")
                ref = leaf.get("ref")
                target = None
                if kind in ("predict", "module") and ref is not None:
                    target = child_path(module, ref)
                elif ref is not None:
                    target = ref
                calls.append(
                    CallSite(
                        module=module,
                        path=path,
                        kind=kind,
                        target=target,
                        kwargs=tuple(expr.get("kwargs", {})),
                        assign_target=assign_target,
                    )
                )
                if kind == "module" and target is not None:
                    queue.append(target)
    return calls


# ─── Checks ──────────────────────────────────────────────────────────


def _check_signature_fields(components: Mapping[str, Any], calls: list[CallSite]) -> list[Finding]:
    """Flag declared signature fields the forward graph never touches."""
    findings: list[Finding] = []
    signatures = components.get("2_signature", {})
    forwards = components.get("5_forward", {})
    by_target: dict[str, list[CallSite]] = {}
    for call in calls:
        if call.kind == "predict" and call.target is not None:
            by_target.setdefault(call.target, []).append(call)

    for predictor in sorted(signatures):
        sites = by_target.get(predictor, [])
        if not sites:
            continue  # PIR-L-UNREACH owns the no-call-site story.
        fields = signatures[predictor].get("fields", [])
        passed = {key for site in sites for key in site.kwargs}
        for index, field in enumerate(fields):
            if field.get("direction") == "input" and field["name"] not in passed:
                findings.append(
                    Finding(
                        "warning",
                        "PIR-L-FIELD-IN",
                        f"2_signature/{predictor}/fields[{index}]",
                        f"predictor '{predictor}' declares input field "
                        f"'{field['name']}', but no call site passes it; the "
                        "rendered prompt will carry an empty slot the model "
                        "must guess around. Drop the field or pass a value.",
                    )
                )

        # Output use: scan each calling module's forward for reads of the
        # variable each call site assigns.
        read_attrs: set[str] = set()
        used_whole = False
        for site in sites:
            if site.assign_target is None:
                used_whole = True  # result feeds an expression directly
                continue
            variable = site.assign_target
            spec = forwards.get(site.module) or {}
            for _path, statement in walk_statements(spec.get("body"), "body"):
                for expr in statement_expressions(statement):
                    if expr.get("node") == "Attr" and expr.get("obj") == variable:
                        read_attrs.add(expr.get("attr"))
                    elif expr.get("node") == "Var" and expr.get("name") == variable:
                        used_whole = True
        if used_whole:
            continue  # whole-value flow: every output may be read downstream
        for index, field in enumerate(fields):
            if field.get("direction") == "output" and field["name"] not in read_attrs:
                findings.append(
                    Finding(
                        "warning",
                        "PIR-L-FIELD-OUT",
                        f"2_signature/{predictor}/fields[{index}]",
                        f"predictor '{predictor}' declares output field "
                        f"'{field['name']}', but no forward statement reads "
                        "it from any result; the model spends tokens producing "
                        "a value the program throws away.",
                    )
                )
    return findings


def _check_unreachable(components: Mapping[str, Any], calls: list[CallSite]) -> list[Finding]:
    """Flag pool entries no path from the root forward can use."""
    findings: list[Finding] = []
    tree = components.get("1_module_tree", {})
    predictors = predictor_nodes(tree)
    modules = module_paths(tree)

    called = {c.target for c in calls if c.target is not None}
    dynamic_modules = {c.module for c in calls if c.kind == "tool" and c.target is None}

    for path in sorted(predictors):
        if path == "self" and not calls:
            continue  # a bare Predict root calls itself; nothing to flag
        if path not in called and path != "self":
            findings.append(
                Finding(
                    "error",
                    "PIR-L-UNREACH",
                    f"1_module_tree -> predictor '{path}'",
                    f"predictor '{path}' is declared in the module tree with a "
                    "full signature and bindings, but no statement on any path "
                    "from 5_forward/self calls it; it ships dead weight and "
                    "its instructions can drift unreviewed.",
                )
            )

    tools = components.get("6_tools", {})
    tool_refs = {c.target for c in calls if c.kind == "tool" and c.target is not None}
    dynamic_pool: set[str] = set()
    for module in dynamic_modules:
        dynamic_pool.update((modules.get(module) or {}).get("tools") or [])
    for name in sorted(tools):
        if name not in tool_refs and name not in dynamic_pool:
            findings.append(
                Finding(
                    "warning",
                    "PIR-L-UNREACH",
                    f"6_tools/{name}",
                    f"tool '{name}' sits in the pool, but no forward calls it "
                    "by name and no dynamic dispatch site can reach it; the "
                    "artifact carries source the program never runs.",
                )
            )

    interpreters = components.get("7_interpreter", {})
    interpreter_refs = {c.target for c in calls if c.kind == "interpreter"}
    for name in sorted(interpreters):
        if name not in interpreter_refs:
            findings.append(
                Finding(
                    "warning",
                    "PIR-L-UNREACH",
                    f"7_interpreter/{name}",
                    f"interpreter '{name}' is declared but no forward leaf "
                    "calls it; its runtime profile is dead configuration.",
                )
            )

    bound_adapters: set[str] = set()
    bound_lms: set[str] = set()
    for path, node in predictors.items():
        if path == "self" or path in called or not calls:
            bindings = node.get("bindings") or {}
            bound_adapters.add(bindings.get("adapter"))
            bound_lms.add(bindings.get("lm"))
    for pool_key, bound, label in (("4_adapter", bound_adapters, "adapter"), ("8_lm", bound_lms, "lm")):
        for name in sorted(components.get(pool_key, {})):
            if name not in bound:
                findings.append(
                    Finding(
                        "info",
                        "PIR-L-UNREACH",
                        f"{pool_key}/{name}",
                        f"{label} '{name}' is in the pool but no reachable "
                        "predictor binds it; harmless, but it bloats the "
                        "artifact and invites stale-config confusion.",
                    )
                )
    return findings


def _check_dead_branches(components: Mapping[str, Any]) -> list[Finding]:
    """Flag If/While tests that fold to a constant."""
    findings: list[Finding] = []
    for module, spec in sorted(components.get("5_forward", {}).items()):
        for path, statement in walk_statements(spec.get("body"), f"5_forward/{module}/body"):
            node = statement.get("node")
            if node not in ("If", "While"):
                continue
            folded = const_test(statement.get("test"))
            if folded is None:
                continue
            test_text = render_expr(statement.get("test"))
            if node == "While" and folded is False:
                verdict = "the loop body never runs"
            elif node == "While":
                continue  # `while True` is PIR-L-WHILE territory, not a dead branch
            elif folded:
                verdict = "the else branch never runs" if statement.get("orelse") else "the test is decoration"
            else:
                verdict = "the if body never runs"
            findings.append(
                Finding(
                    "error",
                    "PIR-L-DEADBRANCH",
                    f"{path}/test",
                    f"test `{test_text}` is constant ({str(folded).lower()}): "
                    f"{verdict}. A branch that cannot flip is either a stub "
                    "left in, or a comparison against the wrong literal.",
                )
            )
    return findings


def _check_while_state(components: Mapping[str, Any]) -> list[Finding]:
    """Flag While loops whose body cannot change the tested state or exit."""
    findings: list[Finding] = []
    for module, spec in sorted(components.get("5_forward", {}).items()):
        for path, statement in walk_statements(spec.get("body"), f"5_forward/{module}/body"):
            if statement.get("node") != "While":
                continue
            if const_test(statement.get("test")) is False:
                continue  # already a dead branch
            tested = {expr.get("name") for expr in walk_expressions(statement.get("test")) if expr.get("node") == "Var"}
            can_exit = False
            for _inner_path, inner in walk_statements(statement.get("body"), "body"):
                node = inner.get("node")
                if node == "Assign" and inner.get("target") in tested:
                    can_exit = True
                elif node in ("Break", "Return", "Raise"):
                    can_exit = True
            if not can_exit:
                test_text = render_expr(statement.get("test"))
                findings.append(
                    Finding(
                        "error",
                        "PIR-L-WHILE",
                        path,
                        f"while `{test_text}` re-tests state no body statement "
                        "assigns, and the body holds no break, return, or "
                        "raise: once entered, this loop cannot terminate.",
                    )
                )
    return findings


def _check_empty_except(components: Mapping[str, Any]) -> list[Finding]:
    """Flag except handlers with an empty body."""
    findings: list[Finding] = []
    for module, spec in sorted(components.get("5_forward", {}).items()):
        for path, statement in walk_statements(spec.get("body"), f"5_forward/{module}/body"):
            if statement.get("node") != "Try":
                continue
            for index, handler in enumerate(statement.get("handlers") or []):
                if not handler.get("body"):
                    findings.append(
                        Finding(
                            "warning",
                            "PIR-L-EXCEPT",
                            f"{path}/handlers[{index}]",
                            f"except {handler.get('type', '?')} has an empty "
                            "body: the error vanishes and the program continues "
                            "with whatever state the failed statement left "
                            "behind. Assign a fallback value or re-raise.",
                        )
                    )
    return findings


def _check_demo_keys(components: Mapping[str, Any]) -> list[Finding]:
    """Flag baked demos whose keys do not match their signature."""
    findings: list[Finding] = []
    signatures = components.get("2_signature", {})
    for predictor, demos in sorted(components.get("3b_demos", {}).items()):
        fields = (signatures.get(predictor) or {}).get("fields", [])
        declared = {f["name"] for f in fields}
        inputs = {f["name"] for f in fields if f.get("direction") == "input"}
        for index, demo in enumerate(demos or []):
            input_keys = demo.get("input_keys", []) or []
            keys = set(demo) - {"input_keys"}
            for key in sorted(keys - declared):
                findings.append(
                    Finding(
                        "error",
                        "PIR-L-DEMO",
                        f"3b_demos/{predictor}[{index}]/{key}",
                        f"demo[{index}] of predictor '{predictor}' carries key "
                        f"'{key}', which is not a field of its signature "
                        f"({', '.join(sorted(declared)) or 'no fields'}); the "
                        "adapter cannot place it, so the example silently "
                        "teaches nothing for that value.",
                    )
                )
            for key in sorted(set(input_keys) - inputs):
                findings.append(
                    Finding(
                        "error",
                        "PIR-L-DEMO",
                        f"3b_demos/{predictor}[{index}]/input_keys",
                        f"demo[{index}] of predictor '{predictor}' marks "
                        f"'{key}' as an input, but the signature declares no "
                        "such input field; the demo's input/label split is "
                        "wrong and the rendered example will be malformed.",
                    )
                )
    return findings


def main(argv: list[str] | None = None) -> int:
    """CLI entry: lint one artifact directory, manifest, or import spec."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(
            "usage: python -m dspy.programir.tools.lint <artifact-dir | manifest.json | pkg.mod:obj>", file=sys.stderr
        )
        return 2
    text = build_text(args[0])
    print(text)
    return 1 if "PIR-L-" in text else 0


if __name__ == "__main__":
    raise SystemExit(main())
