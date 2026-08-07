"""View 1 (static print) over a RATIFIED ProgramIR manifest.

The grade-1 `explain` op's renderer: pure function of the manifest JSON,
no filesystem, no sizes, no timestamps — deterministic text a fixture can
pin byte-for-byte. Ported from the pre-ratification examples renderer
(examples-mirror `explain/explain.py`) and RE-KEYED to the ratified
one-box shape (spec/manifest.md D-029 rulings):

- the versions block prints FIRST (spec/versions.md: "the first thing
  `explain` prints");
- named pools everywhere: `4_adapter`, `6_tools`, `7_interpreter`,
  `8_lm` (one object per LM, nested weights when baked — no 8a/8b),
  `12_metric`;
- signature fields carry {name, direction, prefix, desc, shape,
  semantic_role}; instructions render from 3a only;
- placement lines use the ratified endpoint_ref/default_endpoint keys;
- forwards render the SEM-8 v0 encoding (kwargs-only calls, range loops,
  dynamic tool dispatch, interpreter leaves with a pool ref).

The output shape is 1.x-PROVISIONAL (spec/SCOPE.md: "the `explain`
normalized form (View 1)" may change with a changes/ entry). Per-
predictor and pool maps iterate in sorted key order so equivalent
manifests with different key orders print identically.

Assumes a schema-valid manifest — the shim validates before rendering.
"""

from __future__ import annotations

import json
from typing import Any

WIDTH = 74

__all__ = ["build_text"]


def _rule(title: str) -> str:
    return f"== {title} " + "=" * max(0, WIDTH - len(title) - 4)


def _kv(label: str, value: Any, indent: int = 2, width: int = 22) -> str:
    return f"{' ' * indent}{label:<{width}} {value}"


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _shape_short(shape: Any) -> str:
    """Compact rendering of an embedded JSON Schema (signature `shape`)."""
    if shape is True:
        return "any"
    if shape is False:
        return "never"
    if isinstance(shape, dict):
        if set(shape) == {"type"} and isinstance(shape["type"], str):
            return shape["type"]
        return _dumps(shape)
    return _dumps(shape)


def _placement_line(placement: Any) -> str:
    if not isinstance(placement, dict):
        return "(no placement block)"
    bits = [f"rung={placement.get('rung', '?')}"]
    contract = placement.get("contract")
    if contract:
        bits.append(f"contract={contract if isinstance(contract, str) else _dumps(contract)}")
    ref = placement.get("endpoint_ref")
    if ref:
        default = placement.get("default_endpoint")
        bits.append(f"endpoint_ref={ref}"
                    + (f" (default {default})" if default else " (no default)"))
    isolation = placement.get("isolation")
    if isolation and isolation != "none":
        bits.append(f"isolation={isolation}")
    if placement.get("credential_ref"):
        bits.append(f"credential={placement['credential_ref']}")
    return "  ".join(bits)


# ─── Forward rendering (SEM-8 v0 encoding) ───────────────────────────

def _render_expr(v: Any) -> str:
    if not isinstance(v, dict):
        return _dumps(v)
    node = v.get("node")
    if node == "Call":
        leaf = v.get("leaf", {})
        kind = leaf.get("kind", "?")
        if kind == "tool" and "ref" not in leaf:
            target = f"tool[{_render_expr(v.get('name'))}]"
        else:
            target = f"{kind}[{leaf.get('ref', '?')}]"
        args = ", ".join(f"{k}={_render_expr(x)}"
                         for k, x in v.get("kwargs", {}).items())
        return f"{target}({args})"
    if node == "Attr":
        return f"{v.get('obj')}.{v.get('attr')}"
    if node == "Var":
        return str(v.get("name"))
    if node == "Const":
        return _dumps(v.get("value"))
    if node == "Compare":
        op = "==" if v.get("op") == "eq" else "!="
        return f"{_render_expr(v.get('left'))} {op} {_render_expr(v.get('right'))}"
    if node == "BinOp":
        return f"{_render_expr(v.get('left'))} + {_render_expr(v.get('right'))}"
    return _dumps(v)


def _render_stmts(body: Any, indent: int) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    for st in body or []:
        node = st.get("node") if isinstance(st, dict) else None
        if node == "Assign":
            lines.append(f"{pad}{st.get('target')} = {_render_expr(st.get('value'))}")
        elif node == "Return":
            lines.append(f"{pad}return {_render_expr(st.get('value'))}")
        elif node in ("If", "While"):
            lines.append(f"{pad}{node.lower()} {_render_expr(st.get('test'))}:")
            lines += _render_stmts(st.get("body"), indent + 4)
            if st.get("orelse"):
                lines.append(f"{pad}else:")
                lines += _render_stmts(st.get("orelse"), indent + 4)
        elif node == "For":
            lines.append(f"{pad}for {st.get('target')} in range({st.get('range')}):")
            lines += _render_stmts(st.get("body"), indent + 4)
        elif node == "Try":
            lines.append(f"{pad}try:")
            lines += _render_stmts(st.get("body"), indent + 4)
            for handler in st.get("handlers", []):
                lines.append(f"{pad}except {handler.get('type', '?')}:")
                lines += _render_stmts(handler.get("body"), indent + 4)
        elif node == "Raise":
            lines.append(f"{pad}raise {st.get('exc')}({_dumps(st.get('message'))})")
        elif node in ("Break", "Continue"):
            lines.append(f"{pad}{node.lower()}")
        else:
            lines.append(f"{pad}# <unrenderable node: {_dumps(st)}>")
    return lines


# ─── Module tree ─────────────────────────────────────────────────────

def _render_tree(node: dict, indent: int = 2, lines: list[str] | None = None) -> list[str]:
    if lines is None:
        lines = []
    suffix = ""
    ref = node.get("forward_ref")
    if ref:
        suffix += f"   [forward -> {ref}]"
    bindings = node.get("bindings")
    if bindings:
        bits = [f"adapter={bindings.get('adapter')}", f"lm={bindings.get('lm')}"]
        if bindings.get("delta") is not None:
            bits.append(f"delta={bindings['delta']}")
        suffix += "   {" + ", ".join(bits) + "}"
    lines.append(" " * indent + f"{node.get('name', '?')}: {node.get('kind', '?')}{suffix}")
    for child in node.get("children") or []:
        _render_tree(child, indent + 4, lines)
    return lines


def _predictor_paths(tree: dict) -> list[str]:
    out: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        if node.get("kind") == "Predict":
            out.append(prefix or "self")
            return
        for child in node.get("children", []):
            name = child.get("name", "?")
            walk(child, f"{prefix}.{name}" if prefix else name)

    walk(tree, "")
    return out


# ─── The view ────────────────────────────────────────────────────────

def build_text(manifest: dict) -> str:
    """Render one schema-valid manifest as the View-1 static print.

    Args:
        manifest: A manifest conforming to schema/manifest.schema.json.

    Returns:
        The multi-line explain text, deterministic for a given manifest.
    """
    components = manifest["components"]
    out: list[str] = []
    absent: list[str] = []

    out.append("=" * WIDTH)
    out.append(" ProgramIR explain — View 1 (static print, grade 1)")
    out.append("=" * WIDTH)

    # Versions first (spec/versions.md: the first thing explain prints).
    out.append(_rule("VERSIONS"))
    versions = manifest["versions"]
    required_order = ["ir_version", "node_set", "roles", "strategies",
                      "codecs", "adapter_ir", "lm15"]
    for entry in required_order + sorted(set(versions) - set(required_order)):
        out.append(_kv(entry, versions[entry]))

    # 1 module tree.
    out.append(_rule("MODULE TREE (1)"))
    tree = components["1_module_tree"]
    out += _render_tree(tree)
    predictors = _predictor_paths(tree)
    out.append(f"  ({len(predictors)} predictor{'s' if len(predictors) != 1 else ''};"
               " leaves bind {adapter, lm, delta} by name)")

    # 2/3a/3b/3c per predictor (sorted paths).
    out.append(_rule("PREDICTORS (2, 3a, 3b, 3c)"))
    signatures = components["2_signature"]
    instructions = components["3a_instructions"]
    demos_map = components["3b_demos"]
    config_map = components["3c_predictor_config"]
    for path in sorted(predictors):
        out.append(f"  predictor '{path}'")
        signature = signatures.get(path)
        if not signature:
            out.append(_kv("signature", "(no signature record)", 4))
            absent.append(f"signature (2) [{path}]")
        else:
            fields = signature["fields"]
            ins = [f for f in fields if f.get("direction") == "input"]
            outs = [f for f in fields if f.get("direction") == "output"]
            arrow = (", ".join(f"{f['name']}: {_shape_short(f.get('shape'))}" for f in ins)
                     + " -> "
                     + ", ".join(f"{f['name']}: {_shape_short(f.get('shape'))}" for f in outs))
            out.append(_kv("signature", arrow, 4))
            for f in fields:
                out.append(_kv(
                    f"  .{f['name']}",
                    f"{f.get('direction')}  role={f.get('semantic_role')}  "
                    f"prefix={f.get('prefix')!r}  desc={f.get('desc')!r}", 4))
        instruction = instructions.get(path)
        if instruction is None or instruction == "":
            out.append(_kv("instructions (3a)", "(none)", 4))
            absent.append(f"instructions (3a) [{path}]")
        else:
            out.append(_kv("instructions (3a)", _dumps(instruction), 4))
        demos = demos_map.get(path) or []
        out.append(_kv("demos (3b)", f"{len(demos)} baked", 4))
        for i, demo in enumerate(demos):
            input_keys = demo.get("input_keys", [])
            labels = sorted(set(demo) - set(input_keys) - {"input_keys"})
            out.append(_kv(f"  demo[{i}]",
                           f"inputs={input_keys} labels={labels}", 4))
        config = config_map.get(path)
        if not config:
            out.append(_kv("config (3c)", "(none)", 4))
            absent.append(f"config (3c) [{path}]")
        else:
            out.append(_kv("config (3c)",
                           "  ".join(f"{k}={_dumps(v)}"
                                     for k, v in sorted(config.items())), 4))

    # 5 forward.
    out.append(_rule("FORWARD (5) — node-set v0"))
    forwards = components["5_forward"]
    for module in sorted(forwards):
        spec = forwards[module]
        args = ", ".join(spec.get("args", []))
        out.append(f"  def forward[{module}]({args}):        # {spec.get('language', '?')}")
        out += _render_stmts(spec.get("body"), 6)

    # 4 adapter pool.
    out.append(_rule("ADAPTER POOL (4)"))
    adapters = components["4_adapter"]
    if not adapters:
        out.append("  (empty pool)")
        absent.append("adapters (4)")
    for name in sorted(adapters):
        entry = adapters[name]
        if isinstance(entry, dict) and "preset" in entry:
            out.append(_kv(name, f"builtin preset '{entry['preset']}'"
                                 f"  versions={_dumps(entry.get('versions', {}))}"))
        else:
            keys = ", ".join(sorted(entry)) if isinstance(entry, dict) else "?"
            out.append(_kv(name, "full preset (internals owned by the "
                                 f"adapter-ir contract; keys: {keys})"))

    # 6 tools / 7 interpreter pools.
    out.append(_rule("TOOL POOL (6)"))
    tools = components["6_tools"]
    if not tools:
        out.append("  (empty pool)")
        absent.append("tools (6)")
    for name in sorted(tools):
        entry = tools[name]
        out.append(_kv(name, f"language={entry.get('language', '?')}  "
                             f"source={entry.get('source', '?')}"))
        if entry.get("description"):
            out.append(_kv("  description", entry["description"], 4))
        if entry.get("parameters") is not None:
            out.append(_kv("  params", _dumps(entry.get("parameters")), 4))
        out.append(_kv("  return", _shape_short(entry.get("return_schema")), 4))
        out.append(_kv("  placement", _placement_line(entry.get("placement")), 4))

    out.append(_rule("INTERPRETER POOL (7)"))
    interpreters = components["7_interpreter"]
    if not interpreters:
        out.append("  (empty pool)")
        absent.append("interpreters (7)")
    for name in sorted(interpreters):
        entry = interpreters[name]
        if "kind" in entry:
            out.append(_kv(name, f"{entry['kind']}  (legacy fused profile)"))
        else:
            runtime = entry.get("runtime", {})
            runtime_text = f"{runtime.get('identity', '?')}@{runtime.get('version', '?')}"
            out.append(_kv(name, f"{entry.get('language', '?')} · {runtime_text}"))
            out.append(_kv("  isolation floor", entry.get("isolation_floor", "?"), 4))
            out.append(_kv("  packages", _dumps(entry.get("packages", [])), 4))
            out.append(_kv("  namespace", _dumps(entry.get("namespace_policy")), 4))
            out.append(_kv("  result", _dumps(entry.get("result_convention")), 4))
            out.append(_kv("  vars", _dumps(entry.get("vars_marshaling")), 4))
        out.append(_kv("  placement", _placement_line(entry.get("placement")), 4))

    # 8 LM pool (one-box entries, D-029 ruling 1).
    out.append(_rule("LM POOL (8)"))
    lms = components["8_lm"]
    if not lms:
        out.append("  (empty pool)")
        absent.append("lms (8)")
    for name in sorted(lms):
        entry = lms[name]
        out.append(f"  [{name}]")
        out.append(_kv("contract", entry.get("forward_contract", "?"), 4))
        out.append(_kv("weights identity", entry.get("weights_identity", "?"), 4))
        aliases = entry.get("served_aliases")
        if aliases is not None:
            out.append(_kv("served aliases", ", ".join(aliases) or "(empty)", 4))
        cls = entry.get("class")
        if cls:
            out.append(_kv("class",
                           f"{cls.get('identity', '?')}  [{cls.get('origin', '?')}, "
                           f"{cls.get('language', '?')}]  (advisory for declared "
                           "entries, D-023)", 4))
        engine = entry.get("engine")
        if engine:
            out.append(_kv("engine", f"{engine}  (baked entry)", 4))
        weights = entry.get("weights")
        if weights is None:
            out.append(_kv("weights", "(none — declared LM, not baked)", 4))
        else:
            out.append(_kv("weights", f"format={weights.get('format', '?')}", 4))
            for file_key in sorted(weights.get("files", {})):
                out.append(_kv(f"  files.{file_key}", weights["files"][file_key], 4))
            if "frozen" in weights:
                out.append(_kv("  frozen", _dumps(weights["frozen"]), 4))
            if "weight_ref" in weights:
                out.append(_kv("  weight_ref", weights["weight_ref"], 4))
        config = entry.get("config")
        if config:
            out.append(_kv("config", _dumps(config), 4))
        out.append(_kv("placement", _placement_line(entry.get("placement")), 4))

    # Placement census across every placeable entry.
    out.append(_rule("PLACEMENT"))
    placeable: list[tuple[str, dict]] = []
    for name in sorted(lms):
        entry = lms[name]
        if isinstance(entry.get("placement"), dict):
            placeable.append((f"lm:{name}", entry["placement"]))
        weights = entry.get("weights")
        if isinstance(weights, dict) and isinstance(weights.get("placement"), dict):
            placeable.append((f"lm:{name}.weights", weights["placement"]))
    for pool_key, label in (("6_tools", "tool"), ("7_interpreter", "interpreter"),
                            ("12_metric", "metric")):
        pool = components.get(pool_key) or {}
        if pool_key == "12_metric":
            pool = pool.get("metrics", {})
        for name in sorted(pool):
            placement = pool[name].get("placement")
            if isinstance(placement, dict):
                placeable.append((f"{label}:{name}", placement))
    placeable.sort(key=lambda lp: lp[0])
    if not placeable:
        out.append("  (no placement blocks)")
    else:
        for label, placement in placeable:
            out.append(_kv(label, _placement_line(placement)))
        rungs = sorted({p.get("rung") or "?" for _l, p in placeable})
        if rungs == ["in_process"]:
            out.append("  => every placeable component is rung 0 (in-process);"
                       " fully self-contained")
        else:
            out.append(f"  => rungs present: {rungs}")

    # 9 environment / 10 credentials.
    out.append(_rule("ENVIRONMENT (9) & CREDENTIALS (10)"))
    environment = components["9_environment"]
    if not environment:
        out.append(_kv("environment", "(no env blocks)"))
        absent.append("environment (9)")
    for language in sorted(environment):
        block = environment[language]
        rendered = ("  ".join(f"{k}={_dumps(v)}" for k, v in sorted(block.items()))
                    if isinstance(block, dict) and block else "(empty block)")
        out.append(_kv(f"language '{language}'", rendered))
    credentials = components["10_credentials"]
    if not credentials:
        out.append(_kv("credentials", "(none — no secrets baked, none required)"))
    for cred in credentials:
        out.append(_kv("credential",
                       f"{cred.get('name', '?')}  scope={cred.get('scope', '?')}  "
                       "(name only; value byte-absent, PIR-005)"))

    # 11 ambient policy.
    out.append(_rule("AMBIENT POLICY (11)"))
    policy = components["11_ambient_policy"]
    if not policy:
        out.append("  (empty)")
        absent.append("ambient policy (11)")
    else:
        out.append("  " + "  ".join(f"{k}={_dumps(v)}"
                                    for k, v in sorted(policy.items())))

    # 12 metric pool (the droppable component).
    out.append(_rule("METRIC POOL (12)"))
    evaluation = components.get("12_metric")
    metrics = evaluation.get("metrics", {}) if evaluation else {}
    if evaluation is None:
        out.append("  (none baked — ship object, droppable component)")
        absent.append("metrics (12)")
    else:
        if not metrics:
            out.append("  metrics                (empty pool)")
        for name in sorted(metrics):
            entry = metrics[name]
            out.append(_kv(name, f"language={entry.get('language', '?')}  "
                                 f"source={entry.get('source', '?')}"))
            out.append(_kv("  placement", _placement_line(entry.get("placement")), 4))
        out.append(_kv("devset", f"{len(evaluation['devset'])} ordered examples"))

    # Summary.
    out.append(_rule("SUMMARY"))
    out.append(_kv("ir_version", versions["ir_version"]))
    out.append(_kv("predictors", f"{len(predictors)} ({', '.join(sorted(predictors))})"))
    out.append(_kv("pools", f"adapters {len(adapters)} · tools {len(tools)} · "
                            f"interpreters {len(interpreters)} · lms {len(lms)} · "
                            f"metrics {len(metrics or {})}"))
    out.append(_kv("credentials", str(len(credentials))))
    out.append(_kv("empty/absent", ", ".join(absent) if absent else "(all present)"))
    out.append("=" * WIDTH)
    return "\n".join(out)
