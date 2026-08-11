"""Static cost estimate over a ProgramIR manifest (Tier 1, staging-lessons §B).

Bounded loops plus a closed call graph make cost decidable before any run:
walk the forward node tree from `5_forward/self`, count LM-call sites per
path, and multiply loop bodies by their caps. `For` carries a literal
range; a `While` cap is read from a `if <var> == <int>: break` guard when
one exists, else the loop reports an unbounded maximum.

Two estimates come out, both pure over the manifest:

- **calls** per predictor and total: min / expected / max invocations per
  one program invocation. Branches take min/max over their arms and the
  midpoint as "expected"; loops with an early `break` may exit after one
  pass, so their minimum is one pass, not the cap.
- **prompt tokens** per call: rendered from the instruction, signature
  field names/prefixes/descs, and baked demos with the chars/4 heuristic
  (the adapter preview machinery needs live adapter objects, which a
  manifest does not hold — so this is a labeled approximation, not the
  adapter's exact bytes). Output tokens are read from the predictor's
  `max_tokens` config when present.

Run from Python (`estimate(manifest)` / `build_text(manifest)`) or:

    python -m dspy.programir.tools.cost <artifact-dir | manifest.json | pkg.mod:obj>
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from typing import Any, Mapping

from dspy.programir.tools._common import (
    WIDTH,
    child_path,
    const_test,
    load_manifest,
    rule,
    walk_expressions,
)

__all__ = ["Bounds", "estimate", "build_text", "main"]

_OVERHEAD_TOKENS = 60  # adapter scaffold per rendered request (heuristic)


@dataclass(frozen=True)
class Bounds:
    """Min / expected / max of one counted quantity per program invocation."""

    minimum: float = 0.0
    expected: float = 0.0
    maximum: float = 0.0

    def __add__(self, other: Bounds) -> Bounds:
        return Bounds(self.minimum + other.minimum, self.expected + other.expected, self.maximum + other.maximum)

    def scaled(self, low: float, mid: float, high: float) -> Bounds:
        return Bounds(self.minimum * low, self.expected * mid, self.maximum * high)

    def render(self) -> str:
        high = "unbounded" if math.isinf(self.maximum) else _number(self.maximum)
        return f"min {_number(self.minimum)} / exp {_number(self.expected)} / max {high}"


def _number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def estimate(source: Any) -> dict[str, Any]:
    """Estimate calls and token bounds for one manifest.

    Args:
        source: A ProgramIR value, manifest dict, artifact path, or import
            spec (see `_common.load_manifest`).

    Returns:
        A dict with `calls` (predictor path -> Bounds), `tokens_per_call`
        (predictor path -> {prompt_tokens, output_cap}), `prompt_tokens`
        (predictor path -> Bounds), and `total_calls` / `total_prompt_tokens`.
    """
    manifest = load_manifest(source)
    components = manifest.get("components", {})
    forwards = components.get("5_forward", {})
    root = "self" if "self" in forwards else (sorted(forwards)[0] if forwards else None)
    calls: dict[str, Bounds] = {}
    if root is not None:
        calls = _module_calls(root, forwards, seen=())

    tokens: dict[str, dict[str, Any]] = {}
    prompt_bounds: dict[str, Bounds] = {}
    for predictor in sorted(calls):
        per_call = _prompt_tokens(components, predictor)
        tokens[predictor] = per_call
        prompt = per_call["prompt_tokens"]
        prompt_bounds[predictor] = calls[predictor].scaled(prompt, prompt, prompt)

    zero = Bounds()
    return {
        "calls": calls,
        "tokens_per_call": tokens,
        "prompt_tokens": prompt_bounds,
        "total_calls": sum(calls.values(), zero),
        "total_prompt_tokens": sum(prompt_bounds.values(), zero),
    }


# ─── Call counting ───────────────────────────────────────────────────


def _merge(base: dict[str, Bounds], extra: Mapping[str, Bounds]) -> None:
    for key, bounds in extra.items():
        base[key] = base.get(key, Bounds()) + bounds


def _module_calls(module: str, forwards: Mapping[str, Any], seen: tuple[str, ...]) -> dict[str, Bounds]:
    """Count per-predictor call bounds for one invocation of `module`."""
    if module in seen:
        return {}  # defensive: the compiler refuses shared/recursive modules
    spec = forwards.get(module) or {}
    return _body_calls(spec.get("body") or [], module, forwards, seen + (module,))


def _body_calls(body: list[Any], module: str, forwards: Mapping[str, Any], seen: tuple[str, ...]) -> dict[str, Bounds]:
    total: dict[str, Bounds] = {}
    for statement in body:
        if not isinstance(statement, dict):
            continue
        _merge(total, _statement_calls(statement, module, forwards, seen))
    return total


def _statement_calls(
    statement: Mapping[str, Any], module: str, forwards: Mapping[str, Any], seen: tuple[str, ...]
) -> dict[str, Bounds]:
    node = statement.get("node")
    counts: dict[str, Bounds] = {}
    for key in ("value", "test"):
        if key in statement:
            _merge(counts, _expression_calls(statement[key], module, forwards, seen))

    if node == "If":
        folded = const_test(statement.get("test"))
        then = _body_calls(statement.get("body") or [], module, forwards, seen)
        other = _body_calls(statement.get("orelse") or [], module, forwards, seen)
        if folded is True:
            _merge(counts, then)
        elif folded is False:
            _merge(counts, other)
        else:
            for key in set(then) | set(other):
                a = then.get(key, Bounds())
                b = other.get(key, Bounds())
                counts[key] = counts.get(key, Bounds()) + Bounds(
                    min(a.minimum, b.minimum),
                    (a.expected + b.expected) / 2,
                    max(a.maximum, b.maximum),
                )
    elif node == "For":
        cap = statement.get("range") or 0
        inner = _body_calls(statement.get("body") or [], module, forwards, seen)
        breaks = _has_break(statement.get("body") or [])
        low = min(1, cap) if breaks else cap
        mid = max(cap / 2, min(1, cap)) if breaks else cap
        for key, bounds in inner.items():
            counts[key] = counts.get(key, Bounds()) + bounds.scaled(low, mid, cap)
    elif node == "While":
        if const_test(statement.get("test")) is False:
            return counts
        cap = _while_cap(statement)
        inner = _body_calls(statement.get("body") or [], module, forwards, seen)
        high = cap if cap is not None else math.inf
        mid = (cap / 2 if cap is not None else 1.0) or 0.0
        for key, bounds in inner.items():
            counts[key] = counts.get(key, Bounds()) + bounds.scaled(0, max(mid, 0.5), high)
    elif node == "Try":
        _merge(counts, _body_calls(statement.get("body") or [], module, forwards, seen))
        for handler in statement.get("handlers") or []:
            fallback = _body_calls(handler.get("body") or [], module, forwards, seen)
            for key, bounds in fallback.items():
                counts[key] = counts.get(key, Bounds()) + Bounds(0, 0, bounds.maximum)
    return counts


def _expression_calls(value: Any, module: str, forwards: Mapping[str, Any], seen: tuple[str, ...]) -> dict[str, Bounds]:
    counts: dict[str, Bounds] = {}
    for expr in walk_expressions(value):
        if expr.get("node") != "Call":
            continue
        leaf = expr.get("leaf", {})
        kind, ref = leaf.get("kind"), leaf.get("ref")
        if kind == "predict" and ref is not None:
            target = child_path(module, ref)
            counts[target] = counts.get(target, Bounds()) + Bounds(1, 1, 1)
        elif kind == "module" and ref is not None:
            _merge(counts, _module_calls(child_path(module, ref), forwards, seen))
    return counts


def _has_break(body: list[Any]) -> bool:
    for statement in body:
        if not isinstance(statement, dict):
            continue
        if statement.get("node") == "Break":
            return True
        for key in ("body", "orelse"):
            if _has_break(statement.get(key) or []):
                return True
        for handler in statement.get("handlers") or []:
            if _has_break(handler.get("body") or []):
                return True
    return False


def _while_cap(statement: Mapping[str, Any]) -> int | None:
    """Read a While iteration cap from an `if <var> == <int>: break` guard."""
    for inner in statement.get("body") or []:
        if not isinstance(inner, dict) or inner.get("node") != "If":
            continue
        test = inner.get("test") or {}
        if test.get("node") != "Compare" or test.get("op") != "eq":
            continue
        sides = [test.get("left") or {}, test.get("right") or {}]
        constant = next(
            (
                side.get("value")
                for side in sides
                if side.get("node") == "Const"
                and isinstance(side.get("value"), int)
                and not isinstance(side.get("value"), bool)
            ),
            None,
        )
        if constant is None:
            continue
        if any(item.get("node") == "Break" for item in inner.get("body") or [] if isinstance(item, dict)):
            return max(int(constant), 0) + 1  # guard fires after `cap` full passes
    return None


# ─── Token estimation (chars/4 heuristic) ────────────────────────────


def _prompt_tokens(components: Mapping[str, Any], predictor: str) -> dict[str, Any]:
    """Estimate prompt tokens for one call of `predictor` (chars/4)."""
    characters = 0
    signature = components.get("2_signature", {}).get(predictor) or {}
    for field in signature.get("fields", []):
        characters += len(str(field.get("name", "")))
        characters += len(str(field.get("prefix") or ""))
        characters += len(str(field.get("desc") or ""))
    instruction = components.get("3a_instructions", {}).get(predictor) or ""
    characters += len(instruction)
    demos = components.get("3b_demos", {}).get(predictor) or []
    for demo in demos:
        characters += len(json.dumps(demo, ensure_ascii=False))
    config = components.get("3c_predictor_config", {}).get(predictor) or {}
    output_cap = config.get("max_tokens")
    return {
        "prompt_tokens": characters // 4 + _OVERHEAD_TOKENS,
        "output_cap": output_cap,
        "demos": len(demos),
        "basis": "chars/4 heuristic + fixed scaffold; excludes runtime inputs",
    }


# ─── Rendering ───────────────────────────────────────────────────────


def build_text(source: Any) -> str:
    """Render the static cost estimate as the report text."""
    result = estimate(source)
    out = ["=" * WIDTH, " ProgramIR cost — static estimate per program invocation", "=" * WIDTH]
    out.append(rule("LM CALLS PER PREDICTOR"))
    calls: dict[str, Bounds] = result["calls"]
    if not calls:
        out.append("  (no LM-call sites reachable from 5_forward/self)")
    for predictor in sorted(calls):
        out.append(f"  {predictor:<28} {calls[predictor].render()}")
    out.append(f"  {'TOTAL':<28} {result['total_calls'].render()}")

    out.append(rule("PROMPT TOKENS (chars/4 heuristic — not adapter-exact)"))
    for predictor in sorted(calls):
        per_call = result["tokens_per_call"][predictor]
        cap = per_call["output_cap"]
        out.append(
            f"  {predictor:<28} ~{per_call['prompt_tokens']} tok/call "
            f"({per_call['demos']} demos)" + (f"  output cap {cap}" if cap is not None else "")
        )
        out.append(f"  {'':<28} bounds: {result['prompt_tokens'][predictor].render()}")
    out.append(f"  {'TOTAL':<28} bounds: {result['total_prompt_tokens'].render()}")
    out.append(rule("BASIS"))
    out.append("  calls: loop caps read statically (For range, While break-guard);")
    out.append("  branches take min/max over arms, midpoint as expected.")
    out.append("  tokens: instruction + field prefixes/descs + baked demos, chars/4,")
    out.append(f"  plus {_OVERHEAD_TOKENS}-token scaffold; runtime inputs excluded. Heuristic only.")
    out.append("=" * WIDTH)
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: estimate one artifact directory, manifest, or import spec."""
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print(
            "usage: python -m dspy.programir.tools.cost <artifact-dir | manifest.json | pkg.mod:obj>", file=sys.stderr
        )
        return 2
    print(build_text(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
