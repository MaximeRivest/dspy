"""Hand-rolled structural validation for the grade-1 reference shim.

Implements the two contract schemas (`schema/manifest.schema.json`,
`schema/node-set.schema.json`) as a stdlib-only subset evaluator — the
lm15 zero-dep discipline; no jsonschema library exists in a conforming
grade-1 implementation (schema/README.md). The schema FILES stay the
authority: this module interprets them, it never re-states them, so a
schema change is a behavior change here with zero code edits
(AUTHORITY.md: code conforms to spec, never the reverse).

Supported subset — exactly the restricted feature set schema/README.md
pins: ``type, enum, const, properties, required, additionalProperties,
propertyNames, items, minItems, maxItems, uniqueItems, pattern, minimum,
minLength, oneOf, anyOf, allOf, not, if/then/else, dependentRequired``,
boolean schemas, and internal ``$ref`` (``#/$defs/...``) only.

Two deliberate strictness choices, recorded not improvised:

- ``type: "integer"`` admits only an int, never ``1.0`` (vanilla 2020-12
  would admit integral floats). PIR-004 serde exactness — ``1 != 1.0`` —
  wins over the draft's numeric laxity.
- ``enum``/``const``/``uniqueItems`` compare with strict typed equality
  (``true != 1``, ``1 != 1.0``), the same five-values discipline the
  harness comparator enforces (harness/check.py).

Refusal mapping (spec/errors.md; every detail cites the schema path):

- manifest not an object            -> PIR-E-MANIFEST-001
- versions block / required entry absent -> PIR-E-MANIFEST-003
- any other manifest schema violation    -> PIR-E-MANIFEST-002
- forward schema violation          -> PIR-E-NODE-001 (node kind + location)
- unresolvable declared leaf ref    -> PIR-E-NODE-002 (target + location)
- dangling binding                  -> PIR-E-LINK-001 (binding, pool, entry)
- delta binding (no ratified pool)  -> PIR-E-LINK-002

Message text everywhere is non-normative (spec/errors.md); fixtures pin
codes and structured detail fields only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"

_SCHEMA_CACHE: dict[str, dict] = {}

#: The declared-tier profile predicate as data (spec/placement.md D-023,
#: open item 1: a checkable rule list with clause ids). PROVISIONAL ids
#: DT-001..DT-005 — the decomposition the profile fixture family pins
#: (cases/profile/, emitted by tools/migrate_v0.py): the D-023 bullets,
#: with the advisory-class bullet as an interpretation rule, not a
#: clause. The corpus is the oracle here (AUTHORITY.md).
DECLARED_TIER_CLAUSES: dict[str, str] = {
    "DT-001": "every LM entry is served at a receiver-bound endpoint rung "
              "(endpoint_ref bound; never in_process)",
    "DT-002": "adapter entries are data-only: builtin preset references or full canonical entries",
    "DT-003": "no authored-origin code anywhere in the artifact (D-026)",
    "DT-004": "no baked rung-0 weights (and hence no engine) on any LM entry",
    "DT-005": "interpreter and tool entries are at receiver-bound endpoint "
              "rungs or absent (never in-process, never loader-materialized)",
}


def manifest_schema() -> dict:
    """The parsed manifest schema (cached)."""
    return _load_schema("manifest.schema.json")


def node_set_schema() -> dict:
    """The parsed node-set schema (cached)."""
    return _load_schema("node-set.schema.json")


def _load_schema(name: str) -> dict:
    if name not in _SCHEMA_CACHE:
        _SCHEMA_CACHE[name] = json.loads((SCHEMA_DIR / name).read_text())
    return _SCHEMA_CACHE[name]


# ─── Strict typed equality (the five-values discipline) ──────────────

def json_equal(a: Any, b: Any) -> bool:
    """Strict JSON-value equality: different JSON types are never equal.

    ``true != 1``, ``1 != 1.0``, ``"1" != 1``; arrays ordered; objects
    key-exact. Mirrors harness/check.py's comparator and PIR-004.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_number(a) or _is_number(b):
        return _is_number(a) and _is_number(b) and type(a) is type(b) and a == b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list) and isinstance(b, list)
            and len(a) == len(b)
            and all(json_equal(x, y) for x, y in zip(a, b, strict=False))
        )
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict) and isinstance(b, dict)
            and set(a) == set(b)
            and all(json_equal(a[k], b[k]) for k in a)
        )
    return False


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def json_type(value: Any) -> str:
    """The JSON type name of a Python value (bool is never integer)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


# ─── The subset evaluator ────────────────────────────────────────────

@dataclass(frozen=True)
class Violation:
    """One schema violation: instance path + the schema path that refused."""

    path: tuple  # instance path segments (str keys, int indexes)
    schema_path: str  # citation into the schema document, "#/$defs/..."
    constraint: str  # the keyword that failed ("type", "required", ...)
    message: str  # human text; NON-NORMATIVE (spec/errors.md)


def render_path(segs: tuple) -> str:
    """Render an instance path as ``$.key[0].key`` (check.py convention)."""
    out = "$"
    for seg in segs:
        out += f"[{seg}]" if isinstance(seg, int) else f".{seg}"
    return out


def check(instance: Any, schema: dict, *, root: dict | None = None,
          prefix: tuple = (), spath: str = "#") -> list[Violation]:
    """Validate `instance` against `schema`; return violations in walk order.

    Args:
        instance: The parsed JSON value to validate.
        schema: The (sub)schema to apply.
        root: The document ``$ref`` targets resolve against (defaults to
            `schema` itself).
        prefix: Instance-path segments to prepend to every violation
            (used when validating an extracted sub-block).
        spath: Schema-path string the walk starts at.

    Returns:
        Violations in deterministic walk order; empty when valid.
    """
    out: list[Violation] = []
    _check(instance, schema, root if root is not None else schema,
           prefix, spath, out)
    return out


def _resolve_ref(root: dict, ref: str, ipath: tuple, out: list[Violation]) -> Any:
    if not ref.startswith("#/"):
        out.append(Violation(ipath, ref, "$ref",
                             f"non-internal $ref {ref!r} is outside the "
                             "restricted subset (schema/README.md)"))
        return True  # permit-all: the schema itself is at fault, not the value
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            out.append(Violation(ipath, ref, "$ref", f"dangling $ref {ref!r}"))
            return True
        node = node[part]
    return node


def _deref(root: dict, schema: Any) -> Any:
    """Peek through a lone ``{"$ref": ...}`` wrapper (for introspection)."""
    if isinstance(schema, dict) and isinstance(schema.get("$ref"), str):
        scratch: list[Violation] = []
        return _resolve_ref(root, schema["$ref"], (), scratch)
    return schema


def _type_ok(value: Any, t: str) -> bool:
    if t == "object":
        return isinstance(value, dict)
    if t == "array":
        return isinstance(value, list)
    if t == "string":
        return isinstance(value, str)
    if t == "boolean":
        return isinstance(value, bool)
    if t == "null":
        return value is None
    if t == "integer":
        # Stricter than vanilla 2020-12 on purpose: 1.0 is NOT an integer
        # here (PIR-004 serde exactness; module docstring).
        return isinstance(value, int) and not isinstance(value, bool)
    if t == "number":
        return _is_number(value)
    return False


def _typed_key(v: Any) -> Any:
    """A hashable identity for typed uniqueness checks (uniqueItems)."""
    if isinstance(v, bool):
        return ("boolean", v)
    if isinstance(v, int):
        return ("integer", v)
    if isinstance(v, float):
        return ("float", v)
    if isinstance(v, str):
        return ("string", v)
    if v is None:
        return ("null",)
    if isinstance(v, list):
        return ("array", tuple(_typed_key(x) for x in v))
    if isinstance(v, dict):
        return ("object", frozenset((k, _typed_key(x)) for k, x in v.items()))
    return ("host", repr(v))


def _check(value: Any, schema: Any, root: dict, ipath: tuple, spath: str,
           out: list[Violation]) -> None:
    # Boolean schemas (2020-12 legal; used by embedded json_schema slots).
    if schema is True:
        return
    if schema is False:
        out.append(Violation(ipath, spath, "schema",
                             "false schema admits no value"))
        return
    if not isinstance(schema, dict):
        out.append(Violation(ipath, spath, "schema",
                             f"schema node is {json_type(schema)}, not a schema"))
        return

    if "$ref" in schema:
        target = _resolve_ref(root, schema["$ref"], ipath, out)
        _check(value, target, root, ipath, schema["$ref"], out)

    if "type" in schema:
        types = schema["type"]
        if isinstance(types, str):
            types = [types]
        if not any(_type_ok(value, t) for t in types):
            out.append(Violation(
                ipath, f"{spath}/type", "type",
                f"{json_type(value)} is not one of {types}"))
            return  # nothing below applies to the wrong type

    if "enum" in schema:
        if not any(json_equal(value, allowed) for allowed in schema["enum"]):
            out.append(Violation(
                ipath, f"{spath}/enum", "enum",
                f"value {value!r} is not in the closed vocabulary "
                f"{schema['enum']}"))
            return
    if "const" in schema:
        if not json_equal(value, schema["const"]):
            out.append(Violation(
                ipath, f"{spath}/const", "const",
                f"value {value!r} != required constant {schema['const']!r}"))
            return

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            out.append(Violation(
                ipath, f"{spath}/minLength", "minLength",
                f"length {len(value)} < {schema['minLength']}"))
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            out.append(Violation(
                ipath, f"{spath}/pattern", "pattern",
                f"{value!r} does not match {schema['pattern']!r}"))

    if _is_number(value) and "minimum" in schema and value < schema["minimum"]:
        out.append(Violation(
            ipath, f"{spath}/minimum", "minimum",
            f"{value} < minimum {schema['minimum']}"))

    if isinstance(value, dict):
        _check_object(value, schema, root, ipath, spath, out)

    if isinstance(value, list):
        _check_array(value, schema, root, ipath, spath, out)

    if "allOf" in schema:
        for i, sub in enumerate(schema["allOf"]):
            _check(value, sub, root, ipath, f"{spath}/allOf/{i}", out)
    if "anyOf" in schema:
        _check_branches(value, schema["anyOf"], root, ipath,
                        f"{spath}/anyOf", out, exactly_one=False)
    if "oneOf" in schema:
        _check_branches(value, schema["oneOf"], root, ipath,
                        f"{spath}/oneOf", out, exactly_one=True)
    if "not" in schema:
        inner: list[Violation] = []
        _check(value, schema["not"], root, ipath, f"{spath}/not", inner)
        if not inner:
            out.append(Violation(ipath, f"{spath}/not", "not",
                                 "value matches a schema it must not match"))
    if "if" in schema:
        cond: list[Violation] = []
        _check(value, schema["if"], root, ipath, f"{spath}/if", cond)
        branch = "then" if not cond else "else"
        if branch in schema:
            _check(value, schema[branch], root, ipath, f"{spath}/{branch}", out)


def _check_object(value: dict, schema: dict, root: dict, ipath: tuple,
                  spath: str, out: list[Violation]) -> None:
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in value:
            out.append(Violation(
                ipath + (key,), f"{spath}/required", "required",
                f"required key {key!r} is absent"))
    for key, sub in props.items():
        if key in value:
            _check(value[key], sub, root, ipath + (key,),
                   f"{spath}/properties/{key}", out)
    if "additionalProperties" in schema:
        extra_schema = schema["additionalProperties"]
        for key in value:
            if key in props:
                continue
            if extra_schema is False:
                out.append(Violation(
                    ipath + (key,), f"{spath}/additionalProperties",
                    "additionalProperties",
                    f"unexpected key {key!r} (closed object)"))
            else:
                _check(value[key], extra_schema, root, ipath + (key,),
                       f"{spath}/additionalProperties", out)
    if "propertyNames" in schema:
        for key in value:
            inner: list[Violation] = []
            _check(key, schema["propertyNames"], root, ipath + (key,),
                   f"{spath}/propertyNames", inner)
            if inner:
                out.append(Violation(
                    ipath + (key,), f"{spath}/propertyNames", "propertyNames",
                    f"key {key!r} violates the key schema "
                    f"({inner[0].schema_path})"))
    for trigger, needs in schema.get("dependentRequired", {}).items():
        if trigger in value:
            for req in needs:
                if req not in value:
                    out.append(Violation(
                        ipath + (req,), f"{spath}/dependentRequired",
                        "dependentRequired",
                        f"{req!r} is required when {trigger!r} is present"))


def _check_array(value: list, schema: dict, root: dict, ipath: tuple,
                 spath: str, out: list[Violation]) -> None:
    if "minItems" in schema and len(value) < schema["minItems"]:
        out.append(Violation(ipath, f"{spath}/minItems", "minItems",
                             f"{len(value)} item(s) < {schema['minItems']}"))
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        out.append(Violation(ipath, f"{spath}/maxItems", "maxItems",
                             f"{len(value)} item(s) > {schema['maxItems']}"))
    if schema.get("uniqueItems"):
        seen: set = set()
        for i, item in enumerate(value):
            key = _typed_key(item)
            if key in seen:
                out.append(Violation(
                    ipath + (i,), f"{spath}/uniqueItems", "uniqueItems",
                    f"duplicate item at index {i}"))
            seen.add(key)
    if "items" in schema:
        for i, item in enumerate(value):
            _check(item, schema["items"], root, ipath + (i,),
                   f"{spath}/items", out)


def _check_branches(value: Any, branches: list, root: dict, ipath: tuple,
                    spath: str, out: list[Violation], *,
                    exactly_one: bool) -> None:
    """oneOf/anyOf with two refinements over the naive rule.

    Discriminated unions (every branch pins ``properties.node.const``, the
    node-set statement/expression pattern) dispatch on the instance's
    ``node`` value: the matching branch's violations are THE violations,
    and no match is an "un-whitelisted node" — the error PIR-E-NODE-001
    names (spec/errors.md). Otherwise falls back to validate-all with a
    deepest-first-error heuristic for reporting the closest branch.
    """
    consts = []
    for branch in branches:
        resolved = _deref(root, branch)
        const = None
        if isinstance(resolved, dict):
            node_schema = resolved.get("properties", {}).get("node")
            if isinstance(node_schema, dict):
                const = node_schema.get("const")
        consts.append(const)

    if branches and all(isinstance(c, str) for c in consts):
        kind = value.get("node") if isinstance(value, dict) else None
        matching = [(branch, i) for i, (branch, const)
                    in enumerate(zip(branches, consts, strict=False)) if const == kind]
        if len(matching) == 1:
            branch, i = matching[0]
            sub_spath = (branch["$ref"]
                         if isinstance(branch, dict) and "$ref" in branch
                         else f"{spath}/{i}")
            _check(value, branch, root, ipath, sub_spath, out)
            return
        if matching:
            # Several branches share the const — the v0.3 Call forms
            # (leaf / builtin / method). Any passing branch accepts;
            # otherwise report the branch whose first error is deepest
            # (the closest-intent heuristic, same as the fallback path).
            errsets: list[list[Violation]] = []
            for branch, i in matching:
                sub_spath = (branch["$ref"]
                             if isinstance(branch, dict) and "$ref" in branch
                             else f"{spath}/{i}")
                errs: list[Violation] = []
                _check(value, branch, root, ipath, sub_spath, errs)
                if not errs:
                    return
                errsets.append(errs)
            best = max(range(len(errsets)),
                       key=lambda j: len(errsets[j][0].path)
                       if errsets[j] else -1)
            out.extend(errsets[best])
            return
        if isinstance(kind, str):
            message = f"node kind {kind!r} is not in the whitelist"
        else:
            message = "value is not a node object (missing string 'node')"
        out.append(Violation(ipath, spath, "node-whitelist", message))
        return

    passes = 0
    branch_errors: list[list[Violation]] = []
    for i, branch in enumerate(branches):
        errs: list[Violation] = []
        _check(value, branch, root, ipath, f"{spath}/{i}", errs)
        if not errs:
            passes += 1
        branch_errors.append(errs)
    if passes == 1 or (passes >= 1 and not exactly_one):
        return
    if passes > 1:
        out.append(Violation(ipath, spath, "oneOf",
                             f"matches {passes} branches; exactly one required"))
        return
    best = max(range(len(branch_errors)),
               key=lambda i: len(branch_errors[i][0].path)
               if branch_errors[i] else -1)
    out.extend(branch_errors[best])


# ─── Refusal builders (spec/errors.md shapes) ────────────────────────

def _component_of(segs: tuple) -> str:
    """Name the manifest component a violation falls under (PIR-002)."""
    if segs and segs[0] == "versions":
        return "versions"
    if len(segs) >= 2 and segs[0] == "components":
        return f"components.{segs[1]}"
    if segs and segs[0] == "components":
        return "components"
    return "manifest"


def _schema_refusal(code: str, violation: Violation) -> dict:
    """A MANIFEST-002-style refusal: named subjects at the error top level.

    The corpus convention (cases/, CASES-FORMAT.md "named-subject
    fields"): the failing ``path`` is top-level, plus ``entry`` when the
    violation falls inside the versions block; ``detail`` carries the
    schema citation.
    """
    component = _component_of(violation.path)
    refusal = {
        "code": code,
        "component": component,
        "path": render_path(violation.path),
        "detail": {
            "schema_path": violation.schema_path,
            "constraint": violation.constraint,
        },
        "message": violation.message,  # non-normative, never compared
    }
    if component == "versions" and len(violation.path) >= 2:
        refusal["entry"] = violation.path[1]
    return refusal


def versions_refusal(manifest: Any) -> dict | None:
    """PIR-001's malformed-block rules over the versions block alone.

    The corpus-pinned decomposition (cases/versions/,
    cases/manifest-errors/): an absent or non-object block is
    PIR-E-MANIFEST-003 (no unversioned grandfathering); a PRESENT block
    missing a required entry is PIR-E-MANIFEST-002 naming the entry (a
    schema violation naming the path, not a whole-block absence); an
    entry violating the version_string grammar is likewise -002. The
    required entry list is read from the schema, never restated.
    """
    if not isinstance(manifest, dict):
        return {
            "code": "PIR-E-MANIFEST-001",
            "component": "manifest",
            "detail": {"type": json_type(manifest),
                       "schema_path": "#"},
            "message": "manifest is not a JSON object",
        }
    schema = manifest_schema()
    required = schema["$defs"]["versions_block"]["required"]
    block = manifest.get("versions")
    if "versions" not in manifest or not isinstance(block, dict):
        return {
            "code": "PIR-E-MANIFEST-003",
            "component": "versions",
            "detail": {"missing": list(required),
                       "schema_path": "#/$defs/versions_block/required"},
            "message": "versions block absent or not an object (PIR-001: "
                       "no unversioned grandfathering)",
        }
    for entry in required:
        if entry not in block:
            return {
                "code": "PIR-E-MANIFEST-002",
                "component": "versions",
                "entry": entry,
                "path": render_path(("versions", entry)),
                "detail": {"schema_path": "#/$defs/versions_block/required",
                           "constraint": "required"},
                "message": f"required versions entry {entry!r} absent "
                           "(PIR-001)",
            }
    violations = check(block, {"$ref": "#/$defs/versions_block"},
                       root=schema, prefix=("versions",))
    if violations:
        return _schema_refusal("PIR-E-MANIFEST-002", violations[0])
    return None


def manifest_refusal(manifest: Any) -> dict | None:
    """Full structural validation of one manifest; None when it conforms.

    Order: not-an-object (-001), then the versions block's presence rules
    (-003, checked first so PIR-001 names the malformation precisely),
    then the whole-schema sweep (-002 naming the failing path).
    """
    early = versions_refusal(manifest)
    if early is not None:
        return early
    violations = check(manifest, manifest_schema())
    if violations:
        return _schema_refusal("PIR-E-MANIFEST-002", violations[0])
    return None


# ─── Tree walks and pool/binding cross-checks (the link step) ────────

def collect_predict_leaves(tree: dict) -> list[tuple[str, dict]]:
    """Every Predict leaf with its dotted predictor path, document order.

    The root's own name is conventionally 'self' and is excluded from
    child paths (the per-predictor map keying of spec/manifest.md); a
    bare-Predict root is keyed 'self'.
    """
    out: list[tuple[str, dict]] = []

    def walk(node: dict, prefix: str) -> None:
        if node.get("kind") == "Predict":
            out.append((prefix or "self", node))
            return
        for child in node.get("children", []):
            name = child.get("name", "?")
            walk(child, f"{prefix}.{name}" if prefix else name)

    walk(tree, "")
    return out


def collect_module_paths(tree: dict) -> list[str]:
    """Every module (non-Predict) node's dotted path, document order."""
    out: list[str] = []

    def walk(node: dict, prefix: str) -> None:
        if node.get("kind") == "Predict":
            return
        out.append(prefix or "self")
        for child in node.get("children", []):
            name = child.get("name", "?")
            walk(child, f"{prefix}.{name}" if prefix else name)

    walk(tree, "")
    return out


def resolve_bindings(manifest: dict) -> tuple[dict | None, dict | None]:
    """Resolve every tree binding against the pools (spec/linking.md 4).

    Assumes a schema-valid manifest. Returns ``(table, None)`` on success
    — the resolved-binding table keyed by predictor path (linking.md open
    item 1's provisional shape) — or ``(None, refusal)`` on the first
    dangling reference, walking the tree in document order.

    Refusals carry the named subjects at the error top level — the
    cases/link-errors/ convention: ``predictor``, ``binding``, ``pool``,
    ``entry`` (spec/linking.md: link errors name the binding, the pool,
    and the missing entry).

    A non-null ``delta`` always refuses with PIR-E-LINK-002: no component
    is ratified as the delta pool (schema/README.md ambiguity 5), so no
    delta name can resolve anywhere yet — refusing loudly beats inventing
    a resolution target (PIR-002, PIR-009).
    """
    components = manifest["components"]
    adapters = components["4_adapter"]
    lms = components["8_lm"]
    table: dict[str, dict] = {}
    for path, leaf in collect_predict_leaves(components["1_module_tree"]):
        bindings = leaf["bindings"]
        for binding, pool_key, pool in (("adapter", "4_adapter", adapters),
                                        ("lm", "8_lm", lms)):
            name = bindings[binding]
            if name not in pool:
                return None, {
                    "code": "PIR-E-LINK-001",
                    "component": "1_module_tree",
                    "predictor": path,
                    "binding": binding,
                    "pool": pool_key,
                    "entry": name,
                    "message": f"binding {binding!r} of predictor {path!r} "
                               f"names absent {pool_key} entry {name!r} "
                               "(PIR-009)",
                }
        delta = bindings["delta"]
        if delta is not None:
            return None, {
                "code": "PIR-E-LINK-002",
                "component": "1_module_tree",
                "predictor": path,
                "binding": "delta",
                "pool": None,
                "entry": delta,
                "message": f"delta {delta!r} of predictor {path!r} has no "
                           "ratified pool to resolve against "
                           "(schema/README.md ambiguity 5)",
            }
        table[path] = {"adapter": bindings["adapter"],
                       "lm": bindings["lm"],
                       "delta": delta}
    return table, None


# ─── Forward compile: node-set schema + leaf resolution ──────────────

def iter_calls(forward: dict) -> list[tuple[tuple, dict]]:
    """Every LEAF Call node in a schema-valid forward, with its path.

    Generic structural walk (v0.2/v0.3 made the expression grammar rich
    enough that per-node traversal invited drift): any dict with
    ``node == "Call"`` carrying a ``leaf`` object is a leaf call; the
    builtin/method call forms (v0.3, D-037) carry no ``leaf`` and are
    recursed through, never collected — they resolve against the closed
    tables at schema level, not against declared pools.
    """
    out: list[tuple[tuple, dict]] = []

    def walk(value: Any, path: tuple) -> None:
        if isinstance(value, dict):
            if value.get("node") == "Call" and isinstance(
                    value.get("leaf"), dict):
                out.append((path, value))
            for key, sub in value.items():
                walk(sub, path + (key,))
        elif isinstance(value, list):
            for i, sub in enumerate(value):
                walk(sub, path + (i,))

    walk(forward.get("body", []), ("body",))
    return out


def leaf_census(forward: dict) -> list[dict]:
    """The distinct leaves a forward calls, sorted (kind, then ref).

    A dynamic tool dispatch appears as ``{"kind": "tool", "ref": null}``
    (the model-dispatched call convention, spec/node-set.md leaf rule).
    """
    seen: set[tuple[str, str | None]] = set()
    for _path, call in iter_calls(forward):
        leaf = call["leaf"]
        seen.add((leaf["kind"], leaf.get("ref")))
    return [
        {"kind": kind, "ref": ref}
        for kind, ref in sorted(seen, key=lambda kr: (kr[0], kr[1] is not None,
                                                      kr[1] or ""))
    ]


def _nearest_node_kind(forward: Any, segs: tuple) -> str | None:
    """The deepest enclosing node kind along an instance path."""
    kind = None
    cursor = forward
    trail = [cursor]
    for seg in segs:
        try:
            cursor = cursor[seg]
        except (KeyError, IndexError, TypeError):
            break
        trail.append(cursor)
    for value in trail:
        if isinstance(value, dict) and isinstance(value.get("node"), str):
            kind = value["node"]
    return kind


def _call_tables() -> tuple[set, set]:
    """The closed v0.3 builtin and value-method tables, read from the
    schema enums (one source of truth; D-037)."""
    defs = node_set_schema()["$defs"]
    return (
        set(defs["expr_call_builtin"]["properties"]["builtin"]["enum"]),
        set(defs["expr_call_method"]["properties"]["method"]["enum"]),
    )


def _unknown_call_target(value: Any, path: tuple) -> dict | None:
    """An unknown builtin/method name is an UNRESOLVED CALL — the tables
    are closed (v0.3, D-037) — so it refuses PIR-E-NODE-002 naming the
    target, not NODE-001's malformed-node shape. Checked before schema
    validation so the enum violation never masks the better refusal."""
    builtins_, methods_ = _call_tables()
    if isinstance(value, dict):
        if value.get("node") == "Call":
            for field, table in (("builtin", builtins_),
                                 ("method", methods_)):
                name = value.get(field)
                if isinstance(name, str) and name not in table:
                    return {
                        "code": "PIR-E-NODE-002",
                        "component": "5_forward",
                        "target": name,
                        "location": render_path(path),
                        "message": f"unknown {field} {name!r} — the "
                                   "table is closed (v0.3, D-037); an "
                                   "unlisted name is an unresolved call",
                    }
        for key, sub in value.items():
            found = _unknown_call_target(sub, path + (key,))
            if found is not None:
                return found
    elif isinstance(value, list):
        for i, sub in enumerate(value):
            found = _unknown_call_target(sub, path + (i,))
            if found is not None:
                return found
    return None


def forward_refusal(forward: Any, declared: dict | None = None) -> dict | None:
    """Compile-check one forward object (the node_compile op's core).

    Refusal decomposition per the cases/node-set/compile/ corpus:

    - A leaf missing its required ``ref`` is an UNRESOLVED CALL —
      PIR-E-NODE-002 naming the leaf kind top-level (``leaf``) — the
      serialized-form analog of the undeclared-leaf refusal (D-029's
      required interpreter ref; the leaf rule).
    - Every other schema violation is PIR-E-NODE-001 naming the node
      kind top-level (``node``). ``location`` is an input-side JSON
      path, carried as a diagnostic — artifact form has no source spans
      (SEM-8), so fixtures do not pin it.

    When `declared` context is given ({"predict": [...], "module":
    [...], "tools": [...], "interpreters": [...]}; absent keys mean
    nothing declared), every static leaf ref must also resolve or the
    compile refuses PIR-E-NODE-002 naming ``leaf`` + ``entry``. Dynamic
    tool dispatch resolves at runtime by convention, never here.
    """
    unknown = _unknown_call_target(forward, ())
    if unknown is not None:
        return unknown
    violations = check(forward, node_set_schema())
    if violations:
        v = violations[0]
        if (v.constraint == "required" and len(v.path) >= 2
                and v.path[-2:] == ("leaf", "ref")):
            leaf_kind = None
            cursor: Any = forward
            for seg in v.path[:-1]:
                try:
                    cursor = cursor[seg]
                except (KeyError, IndexError, TypeError):
                    cursor = None
                    break
            if isinstance(cursor, dict):
                leaf_kind = cursor.get("kind")
            return {
                "code": "PIR-E-NODE-002",
                "component": "5_forward",
                "leaf": leaf_kind,
                "location": render_path(v.path[:-2]),
                "detail": {"schema_path": v.schema_path,
                           "constraint": v.constraint},
                "message": f"{leaf_kind!r} leaf without a ref names no "
                           "declared entry (the leaf rule; D-029)",
            }
        # The deepest node kind along the path names the offender; for a
        # node-whitelist violation the path ends AT the unknown node, whose
        # own kind string the walk picks up.
        kind = _nearest_node_kind(forward, v.path)
        return {
            "code": "PIR-E-NODE-001",
            "component": "5_forward",
            "node": kind,
            "location": render_path(v.path),
            "detail": {"schema_path": v.schema_path,
                       "constraint": v.constraint},
            "message": v.message,
        }
    if declared is None:
        return None
    pools = {
        "predict": set(declared.get("predict", [])),
        "module": set(declared.get("module", [])),
        "tool": set(declared.get("tools", [])),
        "interpreter": set(declared.get("interpreters", [])),
    }
    for path, call in iter_calls(forward):
        leaf = call["leaf"]
        kind, ref = leaf["kind"], leaf.get("ref")
        if ref is None:  # dynamic tool dispatch — runtime resolution
            continue
        if ref not in pools[kind]:
            return {
                "code": "PIR-E-NODE-002",
                "component": "5_forward",
                "leaf": kind,
                "entry": ref,
                "location": render_path(path),
                "message": f"call resolves to no declared {kind} leaf "
                           f"{ref!r} (the leaf rule, spec/node-set.md)",
            }
    return None


# ─── The declared-tier profile predicate (spec/placement.md D-023) ───

def _endpoint_bound(placement: Any) -> bool:
    """True when a placement is a receiver-bound endpoint rung."""
    if not isinstance(placement, dict):
        return False
    if placement.get("rung") == "in_process":
        return False
    ref = placement.get("endpoint_ref")
    return isinstance(ref, str) and ref != ""


def _has_authored_code(value: Any) -> bool:
    """Deep-scan for any code-identity block with origin 'authored'."""
    if isinstance(value, dict):
        if value.get("origin") == "authored":
            return True
        return any(_has_authored_code(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_authored_code(v) for v in value)
    return False


def declared_tier_violations(manifest: dict) -> list[dict]:
    """Evaluate the declared-tier clauses over a schema-valid manifest.

    Returns violation records ``{"clause", "subject"}`` sorted by
    (clause, subject) — the shape the cases/profile/ fixtures pin;
    ``subject`` is ``<component key>/<entry name>``. Empty means
    in-profile. Clause ids are `DECLARED_TIER_CLAUSES`. Schema-valid
    adapter entries satisfy DT-002 whether represented as builtin references
    or full canonical data; authored-origin references inside full entries
    are reported independently under DT-003.
    """
    components = manifest["components"]
    violations: list[dict] = []

    def hit(clause: str, subject: str) -> None:
        violations.append({"clause": clause, "subject": subject})

    for name in sorted(components["8_lm"]):
        entry = components["8_lm"][name]
        subject = f"8_lm/{name}"
        if not _endpoint_bound(entry.get("placement")):
            hit("DT-001", subject)
        if _has_authored_code(entry.get("class")):
            hit("DT-003", subject)
        if "weights" in entry or "engine" in entry:
            hit("DT-004", subject)

    for name in sorted(components["4_adapter"]):
        entry = components["4_adapter"][name]
        subject = f"4_adapter/{name}"
        # Schema-valid entries are either builtin references or full canonical
        # entries. Both are data-only under DT-002; authored-origin references
        # inside a full entry are the independent DT-003 concern.
        if _has_authored_code(entry):
            hit("DT-003", subject)

    for pool_key in ("6_tools", "7_interpreter"):
        for name in sorted(components[pool_key]):
            entry = components[pool_key][name]
            if not _endpoint_bound(entry.get("placement")):
                hit("DT-005", f"{pool_key}/{name}")

    violations.sort(key=lambda v: (v["clause"], v["subject"]))
    return violations
