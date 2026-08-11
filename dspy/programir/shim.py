#!/usr/bin/env python3
"""The grade-1 reference shim (harness/PROTOCOL.md, JSONL loop).

One request per stdin line, one reply per stdout line, same order, exit 0
on EOF. The shim only transforms — all comparison happens harness-side —
and every grade-1 op here is pure by construction: stdlib only, no
network, no filesystem beyond loading the two schema files
(schema/README.md; the lm15 zero-dep discipline).

Ops implemented (grade 1, D-028): ``capabilities``, ``load_manifest``,
``check_versions``, ``link``, ``profile_check``, ``diff``,
``node_compile``, ``explain``. Grade-2 ops (``node_execute``, ``verify``)
and ``serde_roundtrip`` are deliberately absent from capabilities so the
harness skips their fixtures honestly (CASES-FORMAT.md: skips are never
passes).

Refusal discipline (spec/errors.md): artifact refusals are
``{code, component, detail}`` with non-normative ``message`` text;
shim-internal failures use the reserved ``PIR-E-SHIM-*`` family —
``-001`` internal error (carries the native error name), ``-002``
unknown op, ``-003`` malformed request — so "artifact refused" and
"shim broke" never blur.

Provisional shapes follow the fixture corpus where it pins them
(AUTHORITY.md: the corpus is the oracle; cases/ + tools/migrate_v0.py)
and are otherwise recorded here, not improvised (spec/SCOPE.md 1.x):

- ``load_manifest`` returns ``{"manifest": <echo>}`` — normalization is
  provisionally the identity over the already-normalized form
  (cases/artifacts-provisional/).
- ``node_compile`` returns ``{"forward": <echo>}`` likewise
  (cases/node-set/compile/). Optional ``declared`` context enables leaf
  resolution (PIR-E-NODE-002 with ``leaf`` + ``entry``).
- ``link`` returns ``{"bindings": {predictor path: {adapter, lm,
  delta}}}`` (linking.md open item 1) and, per the linking order's step
  1, runs the PIR-001 version gate before resolving anything. Refusals
  carry ``predictor``/``binding``/``pool``/``entry`` top-level
  (cases/link-errors/).
- ``check_versions`` applies the ratified comparison rule
  (spec/versions.md, 2026-08-06) against `IMPLEMENTED_VERSIONS` (all
  "0.1" pre-1.0): while an entry's major is 0 versions compare EXACT;
  from 1.0 on same-major accepts. Mismatch refuses PIR-E-VERSION-001
  with ``entry`` + ``declared`` (cases/versions/); an entry whose major
  cannot be parsed refuses too (a reader that cannot establish the rule
  must not accept, PIR-002); entries this reference does not implement
  are ignored (versions.md: extra entries are ignored). Success returns
  ``{"versions": <the block, echoed>}`` and nothing else.
- ``profile_check`` knows one profile, ``declared-tier`` (D-023), with
  clause ids DT-001..DT-005 and ``{clause, subject}`` violation records
  (cases/profile/); an unknown profile name is the op-input case —
  PIR-E-PROFILE-001, the family's "nothing valid presented" code
  (spec/errors.md rulings 2026-08-06; spec/placement.md).
- ``diff`` returns ``{"equal", "diff": [{path, op, left?, right?}]}`` —
  JSON-pointer paths, ``op`` in added/removed/changed, number types
  preserved (cases/diff/; envelope 1.x-provisional, spec/SCOPE.md).
- ``explain`` returns ``{"view": 1, "text": ...}`` (View 1, provisional;
  no fixtures yet).

Missing op input (PROTOCOL.md, ratified 2026-08-06): a request lacking
its op's required payload field — or carrying it wrongly typed —
refuses with the op's OWN family ``-001`` code naming the field in
``detail`` (`_OP_FAMILY`); ``PIR-E-SHIM-003`` stays reserved for
framing breaks (non-JSON, non-object, missing ``op``).

Self-drive (manual testing only, never the conformance path):

    python3 reference/shim.py --selfdrive path/to/manifest[.json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

from dspy.programir import contract_validate as validate
from dspy.programir import explain_view
from dspy.programir.versions import IMPLEMENTED_VERSIONS, supported_versions

JsonObject = dict[str, Any]

IMPL_VERSION = "0.1"

#: What this reference implements per versions entry (spec/versions.md).
#: All majors are 0 pre-1.0; the MAJOR.MINOR grammar is illustrative
#: (open item 0) and only the major is compared.
# Version support is single-sourced with the compiler and reader.

#: Required-entry check order: the schema's declared order, then extras
#: sorted — deterministic first-mismatch selection.
_REQUIRED_ORDER = ("ir_version", "node_set", "roles", "strategies",
                   "codecs", "adapter_ir", "lm15")

class Refusal(Exception):  # noqa: N818 — "refusal" is the contract's own word for this act
    """An artifact refusal: carries the {code, component, detail} object."""

    def __init__(self, error: JsonObject):
        super().__init__(error.get("code", "?"))
        self.error = error


#: Each op's refusal family for the missing-op-input rule (PROTOCOL.md,
#: ratified 2026-08-06): a request lacking its op's required payload field
#: (or carrying it wrongly typed) refuses with the op's OWN family -001 —
#: "nothing valid presented" — naming the missing field in ``detail``.
#: PIR-E-SHIM-003 stays reserved for framing breaks. ``diff`` and
#: ``explain`` have no family of their own in spec/errors.md (open item
#: 1); their payloads are manifests, so MANIFEST is the provisional
#: choice, recorded here not improvised.
_OP_FAMILY: dict[str, str] = {
    "load_manifest": "MANIFEST",
    "check_versions": "VERSION",
    "link": "LINK",
    "profile_check": "PROFILE",
    "node_compile": "NODE",
    "diff": "MANIFEST",
    "explain": "MANIFEST",
}


def _input_refusal(op: str, detail: JsonObject, message: str) -> Refusal:
    return Refusal({
        "code": f"PIR-E-{_OP_FAMILY[op]}-001",
        "component": op,
        "detail": detail,
        "message": message,  # non-normative, never compared
    })


def _op_input(payload: JsonObject, op: str, name: str,
              type_ok: Callable[[Any], bool] | None = None,
              type_name: str = "") -> Any:
    """The op's required payload field, or its family's -001 refusal."""
    if name not in payload:
        raise _input_refusal(
            op, {"missing_field": name},
            f"request lacks {op}'s required field {name!r} — nothing valid "
            "presented (PROTOCOL.md missing-op-input rule)")
    value = payload[name]
    if type_ok is not None and not type_ok(value):
        raise _input_refusal(
            op, {"field": name, "type": validate.json_type(value)},
            f"{op}'s field {name!r} is {validate.json_type(value)}, "
            f"not {type_name} — wrongly typed op payload "
            "(PROTOCOL.md missing-op-input rule)")
    return value


def _refuse_if(error: JsonObject | None) -> None:
    if error is not None:
        raise Refusal(error)


# ─── Version comparison (PIR-001) ────────────────────────────────────

def _major(version: str) -> str | None:
    match = re.match(r"^(\d+)(?:\.|$)", version)
    return match.group(1) if match else None


def _versions_gate(manifest: JsonObject) -> None:
    """Per-entry acceptance; refuses PIR-E-VERSION-001 on the first mismatch.

    The ratified comparison rule (spec/versions.md, 2026-08-06): while an
    entry's major is 0, versions compare EXACT (pre-1.0 is all-breaking);
    from 1.0 on, same-major accepts. "Major" = the numeric segment before
    the first dot; an entry whose major cannot be parsed refuses too (a
    reader that cannot establish the rule must not accept, PIR-002).
    Entries this reference does not implement are ignored (versions.md:
    extra entries are ignored). Assumes the block passed
    `validate.versions_refusal`.
    """
    block = manifest["versions"]
    order = list(_REQUIRED_ORDER) + sorted(set(block) - set(_REQUIRED_ORDER))
    for name in order:
        artifact_version = block[name]
        implemented = IMPLEMENTED_VERSIONS.get(name)
        if implemented is None:
            continue  # extra entry — ignored (spec/versions.md)
        artifact_major = _major(artifact_version)
        same_major = (artifact_major is not None
                      and artifact_major == _major(implemented))
        pre_1_0_exact = same_major and artifact_major == "0"
        accepted = supported_versions(name) or ()
        if not same_major or (pre_1_0_exact
                              and artifact_version not in accepted):
            raise Refusal({
                "code": "PIR-E-VERSION-001",
                "component": "versions",
                "entry": name,
                "declared": artifact_version,
                "implemented": implemented,
                "message": f"versions.{name}: declared {artifact_version!r} "
                           f"vs implemented {implemented!r} — pre-1.0 "
                           "compares exact, from 1.0 same-major accepts "
                           "(PIR-001; spec/versions.md 2026-08-06)",
            })


# ─── Op handlers ─────────────────────────────────────────────────────

def op_capabilities(payload: JsonObject) -> JsonObject:
    return {
        "language": "python",
        "impl": "dspy-programir",
        "impl_version": IMPL_VERSION,
        "grades": [1],
        "profiles": ["declared-tier"],
        "versions": dict(IMPLEMENTED_VERSIONS),
        "ops": sorted(HANDLERS),
    }


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def op_load_manifest(payload: JsonObject) -> JsonObject:
    manifest = _op_input(payload, "load_manifest", "manifest")
    _refuse_if(validate.manifest_refusal(manifest))
    # Normalization is provisionally the identity over the already-
    # normalized form; the WHOLE result is pinned by fixtures, so no
    # derived index tables ride along (cases/artifacts-provisional/).
    return {"manifest": manifest}


def op_check_versions(payload: JsonObject) -> JsonObject:
    manifest = _op_input(payload, "check_versions", "manifest",
                         _is_object, "an object")
    _refuse_if(validate.versions_refusal(manifest))
    _versions_gate(manifest)
    # Ratified success shape (spec/versions.md, 2026-08-06): the block,
    # echoed — and nothing else (whole-result pinning, CASES-FORMAT.md).
    return {"versions": manifest["versions"]}


def op_link(payload: JsonObject) -> JsonObject:
    manifest = _op_input(payload, "link", "manifest", _is_object, "an object")
    _refuse_if(validate.manifest_refusal(manifest))
    _versions_gate(manifest)  # linking.md step 1: versions before anything
    table, refusal = validate.resolve_bindings(manifest)
    _refuse_if(refusal)
    return {"bindings": table}


def op_profile_check(payload: JsonObject) -> JsonObject:
    manifest = _op_input(payload, "profile_check", "manifest",
                         _is_object, "an object")
    profile = _op_input(payload, "profile_check", "profile",
                        lambda v: isinstance(v, str), "a string")
    if profile != "declared-tier":
        # An unknown profile name is the op-input case (spec/errors.md
        # rulings 2026-08-06; spec/placement.md): the op never refuses on
        # PROFILE grounds, but "nothing valid presented" is its family -001.
        raise _input_refusal(
            "profile_check",
            {"field": "profile", "unknown": profile,
             "implemented": ["declared-tier"]},
            f"profile {profile!r} is not implemented; implemented "
            "profiles: ['declared-tier'] (spec/placement.md D-023)")
    _refuse_if(validate.manifest_refusal(manifest))
    violations = validate.declared_tier_violations(manifest)
    return {"profile": profile,
            "in_profile": not violations,
            "violations": violations}


def op_diff(payload: JsonObject) -> JsonObject:
    left = _op_input(payload, "diff", "left")
    right = _op_input(payload, "diff", "right")
    records = _diff_records(left, right, "")
    return {"equal": not records, "diff": records}


def op_node_compile(payload: JsonObject) -> JsonObject:
    forward = _op_input(payload, "node_compile", "forward")
    declared = payload.get("declared")
    if declared is not None:
        if not isinstance(declared, dict):
            raise _input_refusal(
                "node_compile",
                {"field": "declared", "type": validate.json_type(declared)},
                "declared must be an object (wrongly typed op payload)")
        for key, value in declared.items():
            if key not in ("predict", "module", "tools", "interpreters"):
                raise _input_refusal(
                    "node_compile", {"field": "declared", "unknown_key": key},
                    f"declared has unknown key {key!r}")
            if not isinstance(value, list) or not all(
                    isinstance(item, str) for item in value):
                raise _input_refusal(
                    "node_compile", {"field": f"declared.{key}"},
                    f"declared.{key} must be a string list")
    _refuse_if(validate.forward_refusal(forward, declared))
    # Identity normalization, whole-result pinning (cases/node-set/compile/).
    return {"forward": forward}


def op_explain(payload: JsonObject) -> JsonObject:
    manifest = _op_input(payload, "explain", "manifest",
                         _is_object, "an object")
    _refuse_if(validate.manifest_refusal(manifest))
    return {"view": 1, "text": explain_view.build_text(manifest)}


HANDLERS: dict[str, Callable[[JsonObject], JsonObject]] = {
    "capabilities": op_capabilities,
    "load_manifest": op_load_manifest,
    "check_versions": op_check_versions,
    "link": op_link,
    "profile_check": op_profile_check,
    "diff": op_diff,
    "node_compile": op_node_compile,
    "explain": op_explain,
}


# ─── diff (View-3; envelope 1.x-provisional, spec/SCOPE.md) ──────────

def _pointer_seg(seg: Any) -> str:
    """One JSON-pointer reference token (RFC 6901 escaping)."""
    if isinstance(seg, int):
        return str(seg)
    return seg.replace("~", "~0").replace("/", "~1")


def _diff_records(a: Any, b: Any, path: str) -> list[JsonObject]:
    """Deepest-changed-path records: ``{path, op, left?, right?}``.

    The result SHAPE {equal, diff: [{path, op, left?, right?}]} is
    1.x-provisional (spec/SCOPE.md), pinned by cases/diff/ as far as the
    ratified number rule implies it: paths are JSON pointers; ``op`` is
    one of added/removed/changed; ``changed`` carries both sides with
    their number types preserved, ``added`` only ``right``, ``removed``
    only ``left``. Equality is the strict typed discipline
    (`validate.json_equal`): an integer never equals a float; floats
    compare by numeric value (PROTOCOL.md number typing, 2026-08-06).
    Objects recurse per key (sorted); equal-length arrays recurse per
    index; arrays whose length changed are one ``changed`` record at the
    array path.
    """
    if isinstance(a, dict) and isinstance(b, dict):
        records: list[JsonObject] = []
        for key in sorted(set(a) | set(b)):
            here = f"{path}/{_pointer_seg(key)}"
            if key not in b:
                records.append({"path": here, "op": "removed", "left": a[key]})
            elif key not in a:
                records.append({"path": here, "op": "added", "right": b[key]})
            else:
                records.extend(_diff_records(a[key], b[key], here))
        return records
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        records = []
        for i, (item_a, item_b) in enumerate(zip(a, b, strict=False)):
            records.extend(_diff_records(item_a, item_b, f"{path}/{i}"))
        return records
    if not validate.json_equal(a, b):
        return [{"path": path, "op": "changed", "left": a, "right": b}]
    return []


# ─── Framing (harness/PROTOCOL.md wire format) ───────────────────────

def _error_reply(req_id: Any, code: str, component: str, detail: Any,
                 message: str = "") -> JsonObject:
    error: JsonObject = {"code": code, "component": component, "detail": detail}
    if message:
        error["message"] = message  # non-normative, never compared
    return {"id": req_id, "ok": False, "error": error}


def handle_line(line: str) -> JsonObject:
    try:
        request = json.loads(line)
    except json.JSONDecodeError as exc:
        return _error_reply(None, "PIR-E-SHIM-003", "shim",
                            {"reason": "request line is not JSON"}, str(exc))
    if not isinstance(request, dict):
        return _error_reply(None, "PIR-E-SHIM-003", "shim",
                            {"reason": "request is not an object"})
    req_id = request.get("id")
    op = request.get("op")
    if not isinstance(op, str):
        return _error_reply(req_id, "PIR-E-SHIM-003", "shim",
                            {"reason": "op missing or not a string"})
    handler = HANDLERS.get(op)
    if handler is None:
        return _error_reply(req_id, "PIR-E-SHIM-002", "shim", {"op": op})
    payload = {k: v for k, v in request.items() if k not in ("op", "id")}
    try:
        result = handler(payload)
    except Refusal as refusal:
        return {"id": req_id, "ok": False, "error": refusal.error}
    except Exception as exc:
        return _error_reply(req_id, "PIR-E-SHIM-001", "shim",
                            {"native": type(exc).__name__}, str(exc))
    return {"id": req_id, "ok": True, "result": result}


def serve() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        reply = handle_line(line)
        sys.stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        sys.stdout.flush()
    return 0


# ─── Self-drive (manual smoke testing only) ──────────────────────────

def selfdrive(target: str) -> int:
    """Run the grade-1 ops over one on-disk manifest and print results.

    Args:
        target: A manifest.json path, or a directory containing one.

    Returns:
        0 when every op accepted; 1 when any op refused.
    """
    path = Path(target)
    if path.is_dir():
        path = path / "manifest.json"
    try:
        manifest = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"selfdrive: cannot read manifest at {path}: {exc}",
              file=sys.stderr)
        return 1

    failed = False

    def run(op: str, payload: JsonObject) -> JsonObject | None:
        nonlocal failed
        reply = handle_line(json.dumps({"op": op, "id": f"drive:{op}",
                                        **payload}))
        if reply["ok"]:
            print(f"-- {op}: ok")
            return reply["result"]
        failed = True
        print(f"-- {op}: REFUSED {json.dumps(reply['error'], indent=2)}")
        return None

    loaded = run("load_manifest", {"manifest": manifest})
    if loaded is not None:
        tree = manifest["components"]["1_module_tree"]
        predictors = sorted(p for p, _n in validate.collect_predict_leaves(tree))
        pools = {key: sorted(manifest["components"].get(key) or {})
                 for key in ("4_adapter", "6_tools", "7_interpreter",
                             "8_lm", "12_metric")}
        print(f"   predictors: {predictors}")
        print(f"   modules:    {sorted(validate.collect_module_paths(tree))}")
        print(f"   pools:      {json.dumps(pools)}")
    versions = run("check_versions", {"manifest": manifest})
    if versions is not None:
        print(f"   versions echoed: {json.dumps(versions['versions'])}")
    linked = run("link", {"manifest": manifest})
    if linked is not None:
        print(f"   bindings: {json.dumps(linked['bindings'], indent=2)}")
    profile = run("profile_check", {"manifest": manifest,
                                    "profile": "declared-tier"})
    if profile is not None:
        print(f"   in declared-tier: {profile['in_profile']}"
              + (f"; violations: {json.dumps(profile['violations'])}"
                 if profile["violations"] else ""))

    if loaded is not None:
        components = manifest["components"]
        tools = sorted(components["6_tools"])
        interpreters = sorted(components["7_interpreter"])
        tree = components["1_module_tree"]
        children_of: dict[str, tuple[list[str], list[str]]] = {}

        def walk(node: JsonObject, prefix: str) -> None:
            if node.get("kind") == "Predict":
                return
            predicts = [c["name"] for c in node.get("children", [])
                        if c.get("kind") == "Predict"]
            modules = [c["name"] for c in node.get("children", [])
                       if c.get("kind") != "Predict"]
            children_of[prefix or "self"] = (predicts, modules)
            for child in node.get("children", []):
                name = child.get("name", "?")
                walk(child, f"{prefix}.{name}" if prefix else name)

        walk(tree, "")
        for module in sorted(components["5_forward"]):
            predicts, modules = children_of.get(module, ([], []))
            declared = {"predict": predicts, "module": modules,
                        "tools": tools, "interpreters": interpreters}
            compiled = run("node_compile",
                           {"forward": components["5_forward"][module],
                            "declared": declared})
            if compiled is not None:
                census = validate.leaf_census(components["5_forward"][module])
                print(f"   [{module}] leaves: {json.dumps(census)}")

    explained = run("explain", {"manifest": manifest})
    if explained is not None:
        print(explained["text"])
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--selfdrive", metavar="MANIFEST", default=None,
        help="run the grade-1 ops over one manifest file/dir and print "
             "results (manual testing; the conformance path is "
             "harness/check.py)")
    args = parser.parse_args(argv)
    if args.selfdrive:
        return selfdrive(args.selfdrive)
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
