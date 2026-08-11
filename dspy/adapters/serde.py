"""Entry serde: the extended (0.3.0-draft) adapter entry, dumped and
loaded exactly.

An adapter dumps to one JSON-able entry — template as raw message data,
parser as data (lens or pipeline), codec bindings, strategy bindings
(names or rule objects), config, and the declared requirement set.
Loading validates shape, versions, template, and every reference with
zero ambient reads; a dangling reference is a link error refused loudly
naming the reference, and unknown or missing versions refuse naming both
sides. Serde is exact: absent is not null, unknown keys refuse instead of
silently dropping, and `load_entry(dump_entry(x))` reproduces `x`.

The versions block is conditional (the drawn shape of the adapter-ir-stage
examples): `roles`, `strategies`, `codecs`, `template_language` always;
`parse_combinators`, `lm_capabilities`, `shapes` exactly when the entry
uses them — a used-but-missing vocabulary refuses, a stated-but-unknown
vocabulary refuses.
"""

import json
import re
from copy import deepcopy
from typing import Any

from dspy.adapters._engine.template.vocabulary import TEMPLATE_LANGUAGE_VERSION
from dspy.adapters.adapter import DERIVED, Adapter
from dspy.adapters.codecs import CODECS_VERSION, SHAPES_VERSION
from dspy.adapters.errors import AdapterError, EntryError
from dspy.adapters.parse import PARSE_COMBINATORS_VERSION
from dspy.adapters.strategies import (
    LM_CAPABILITIES_VERSION,
    LM_CAPABILITY_FACTS,
    STRATEGIES_RULES_VERSION,
    STRATEGIES_VERSION,
    predicate_capabilities,
)
from dspy.signatures.field import SEMANTIC_ROLES_VERSION

#: Version of the extended adapter IR entry shape.
ADAPTER_IR_VERSION = "0.3.0-draft"

#: Extended codecs-vocabulary version (per-field family codec objects).
CODECS_EXTENDED_VERSION = "1.1.0-draft"

#: The entry's key set, in canonical order. `requires` is the one
#: optional key (absent = the zero-requirement floor).
ENTRY_KEYS = (
    "name",
    "adapter_ir_version",
    "versions",
    "template",
    "parser",
    "codecs",
    "strategies",
    "config",
    "requires",
)
REQUIRED_ENTRY_KEYS = ENTRY_KEYS[:-1]

#: Vocabularies every entry names.
REQUIRED_VOCABULARIES = ("roles", "strategies", "codecs", "template_language")

#: The newest version of each vocabulary this engine speaks.
SPOKEN_VERSIONS = {
    "adapter_ir": ADAPTER_IR_VERSION,
    "roles": SEMANTIC_ROLES_VERSION,
    "strategies": STRATEGIES_RULES_VERSION,
    "codecs": CODECS_EXTENDED_VERSION,
    "template_language": TEMPLATE_LANGUAGE_VERSION,
    "parse_combinators": PARSE_COMBINATORS_VERSION,
    "lm_capabilities": LM_CAPABILITIES_VERSION,
    "shapes": SHAPES_VERSION,
}

_VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(?:-[0-9A-Za-z.]+)?$")


def parse_version(version: Any, *, what: str) -> tuple[int, int, int]:
    """Parse `major.minor.patch[-tag]`; refuse malformed naming `what`."""
    if isinstance(version, str):
        match = _VERSION_RE.match(version)
        if match:
            return tuple(int(part) for part in match.groups())  # type: ignore[return-value]
    raise EntryError(f"malformed {what} version {version!r} — expected 'major.minor.patch[-tag]'")


def check_version_compatible(kind: str, theirs: Any, ours: str) -> None:
    """Refuse loudly when `theirs` is not readable by `ours`.

    Different majors refuse; while a vocabulary is 0.x, different minors
    refuse too (the semver-0 convention); at 1.x+, an entry newer than the
    engine refuses.
    """
    their_major, their_minor, _ = parse_version(theirs, what=kind)
    our_major, our_minor, _ = parse_version(ours, what=kind)
    incompatible = (
        their_major != our_major
        or (our_major == 0 and their_minor != our_minor)
        or (our_major >= 1 and their_minor > our_minor)
    )
    if incompatible:
        raise EntryError(
            f"incompatible {kind} version: entry carries {theirs!r}, this engine speaks {ours!r} — "
            "refusing to load rather than misread"
        )


# ---------------------------------------------------------------------------
# Dump
# ---------------------------------------------------------------------------


def build_entry(adapter: Adapter, *, for_signature=None) -> dict[str, Any]:
    """One adapter as its canonical entry dict.

    Args:
        adapter: The adapter to dump.
        for_signature: Lower this signature's media fields into per-field
            shape codecs (per-field codecs only exist relative to a
            signature).
    """
    per_field = deepcopy(adapter.per_field_codecs)
    if for_signature is not None:
        per_field = {**_derived_shape_codecs(for_signature), **per_field}

    rules = [value for value in adapter.strategies.values() if isinstance(value, dict)]
    atoms: list[str] = []
    for rule in rules:
        for atom in predicate_capabilities(rule["predicate"]):
            if atom not in atoms:
                atoms.append(atom)

    stated_requires = adapter.requires if isinstance(adapter.requires, dict) else None
    uses_pipelines = adapter.parser_spec.get("kind") == "pipeline" or any(
        "text" in routing for rule in rules for routing in rule["routings"]
    )
    uses_capabilities = bool(atoms) or bool((stated_requires or {}).get("lm_capabilities"))
    uses_shapes = any(spec.get("kind") == "shape" for spec in per_field.values())
    uses_families = any(spec.get("kind") in ("family", "leaf") for spec in per_field.values())

    versions: dict[str, str] = {
        "roles": SEMANTIC_ROLES_VERSION,
        "strategies": STRATEGIES_RULES_VERSION if rules else STRATEGIES_VERSION,
        "codecs": CODECS_EXTENDED_VERSION if uses_families else CODECS_VERSION,
        "template_language": TEMPLATE_LANGUAGE_VERSION,
    }
    if uses_capabilities:
        versions["lm_capabilities"] = LM_CAPABILITIES_VERSION
    if uses_pipelines:
        versions["parse_combinators"] = PARSE_COMBINATORS_VERSION
    if uses_shapes:
        versions["shapes"] = SHAPES_VERSION

    codecs: dict[str, Any] = dict(adapter.codec_bindings)
    if per_field:
        codecs["per_field"] = per_field

    entry: dict[str, Any] = {
        "name": adapter.name,
        "adapter_ir_version": ADAPTER_IR_VERSION,
        "versions": versions,
        "template": deepcopy(adapter.template_raw),
        "parser": deepcopy(adapter.parser_spec),
        "codecs": codecs,
        "strategies": deepcopy(adapter.strategies),
        "config": deepcopy(adapter.config),
    }

    if stated_requires is not None:
        entry["requires"] = deepcopy(stated_requires)
    elif adapter.requires is DERIVED and atoms:
        entry["requires"] = {"lm_capabilities": atoms}
    return entry


def _derived_shape_codecs(signature) -> dict[str, dict]:
    """Media-role input fields lower to image shape codecs at dump."""
    from dspy.adapters.codecs import image_shape_codec_entry
    from dspy.adapters.strategies import fields_with_role
    from dspy.adapters.types.image import Image

    derived: dict[str, dict] = {}
    for name in fields_with_role(signature, "media"):
        if name not in signature.input_fields:
            continue
        annotation = signature.input_fields[name].annotation
        module = getattr(annotation, "__module__", "") or ""
        if annotation is Image or module.startswith("PIL."):
            frontend = f"{module}.{annotation.__qualname__}" if module else annotation.__name__
            derived[name] = image_shape_codec_entry(frontend)
    return derived


def dumps_entry(entry: dict) -> str:
    """The entry as canonical JSON: compact, unsorted (order is data)."""
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_entry(entry: dict) -> Adapter:
    """Validate and link one entry back into an executable Adapter.

    Zero ambient reads. Validation order: shape, versions (required plus
    used-conditional), then the adapter constructor's own eager checks
    (template, parser, codec refs, strategy bindings).
    """
    if not isinstance(entry, dict):
        raise EntryError(f"an adapter entry is a dict, got {type(entry).__name__}")
    unknown = set(entry) - set(ENTRY_KEYS)
    if unknown:
        raise EntryError(
            f"unknown entry keys {sorted(unknown)} — exact serde refuses keys it would drop; "
            f"valid keys: {', '.join(ENTRY_KEYS)}"
        )
    missing = [key for key in REQUIRED_ENTRY_KEYS if key not in entry]
    if missing:
        raise EntryError(
            f"adapter entry is missing {missing} — every entry carries all of: "
            f"{', '.join(REQUIRED_ENTRY_KEYS)} (a missing versions block is malformed, never grandfathered)"
        )

    name = entry["name"]
    if not isinstance(name, str) or not name:
        raise EntryError(f"adapter entry 'name' must be a non-empty string, got {name!r}")

    check_version_compatible("adapter_ir", entry["adapter_ir_version"], ADAPTER_IR_VERSION)
    _check_versions_block(entry, name)
    _check_requires_shape(entry.get("requires"), name)

    template = entry["template"]
    if not isinstance(template, list):
        raise EntryError(f"adapter entry {name!r}: 'template' is a list of message dicts")
    for key in ("codecs", "strategies", "config"):
        if not isinstance(entry[key], dict):
            raise EntryError(f"adapter entry {name!r}: {key!r} is a dict, got {type(entry[key]).__name__}")

    try:
        return Adapter(
            name=name,
            template=deepcopy(template),
            parser=deepcopy(entry["parser"]),
            codecs=deepcopy(entry["codecs"]),
            strategies=deepcopy(entry["strategies"]),
            config=deepcopy(entry["config"]),
            requires=deepcopy(entry["requires"]) if "requires" in entry else None,
        )
    except EntryError as error:
        raise EntryError(f"adapter entry {name!r}: {error}") from None
    except (AdapterError, ValueError) as error:
        raise EntryError(f"adapter entry {name!r}: {error}") from None


def _check_versions_block(entry: dict, name: str) -> None:
    versions = entry["versions"] if "versions" in entry else None
    if not isinstance(versions, dict):
        raise EntryError(f"adapter entry {name!r}: 'versions' is a dict, got {type(versions).__name__}")

    for kind in REQUIRED_VOCABULARIES:
        if kind not in versions:
            raise EntryError(
                f"adapter entry {name!r}: versions block is missing {kind!r} — every entry names the "
                f"version of each core vocabulary ({', '.join(REQUIRED_VOCABULARIES)})"
            )
    unknown = set(versions) - set(SPOKEN_VERSIONS)
    if unknown:
        raise EntryError(
            f"adapter entry {name!r}: versions block names unknown vocabularies {sorted(unknown)} — "
            f"this engine knows: {', '.join(k for k in SPOKEN_VERSIONS if k != 'adapter_ir')}"
        )
    for kind, theirs in versions.items():
        check_version_compatible(kind, theirs, SPOKEN_VERSIONS[kind])

    for kind, used, why in _used_vocabularies(entry):
        if used and kind not in versions:
            raise EntryError(
                f"adapter entry {name!r}: versions block is missing {kind!r} — the entry uses {why}, "
                "and a used vocabulary is always version-pinned"
            )


def _used_vocabularies(entry: dict):
    parser = entry.get("parser")
    rules = [value for value in entry.get("strategies", {}).values() if isinstance(value, dict)]
    uses_pipeline = isinstance(parser, dict) and parser.get("kind") == "pipeline"
    uses_routing_pipeline = any(
        isinstance(routing, dict) and "text" in routing
        for rule in rules
        for routing in (rule.get("routings") or [])
        if isinstance(rule.get("routings"), list)
    )
    requires = entry.get("requires") or {}
    per_field = entry.get("codecs", {}).get("per_field") or {}
    uses_shapes = any(isinstance(spec, dict) and spec.get("kind") == "shape" for spec in per_field.values())
    yield "parse_combinators", uses_pipeline or uses_routing_pipeline, "a combinator pipeline"
    yield (
        "lm_capabilities",
        bool(rules) or bool(requires.get("lm_capabilities") if isinstance(requires, dict) else False),
        "capability predicates or requirements",
    )
    yield "shapes", uses_shapes, "a shape codec"


def _check_requires_shape(requires: Any, name: str) -> None:
    if requires is None:
        return
    if not isinstance(requires, dict):
        raise EntryError(f"adapter entry {name!r}: 'requires' is a dict, got {type(requires).__name__}")
    unknown = set(requires) - {"lm_capabilities", "languages", "leaves"}
    if unknown:
        raise EntryError(
            f"adapter entry {name!r}: unknown requires keys {sorted(unknown)} — valid keys: "
            "lm_capabilities, languages, leaves"
        )
    capabilities = requires.get("lm_capabilities", [])
    if not isinstance(capabilities, list):
        raise EntryError(f"adapter entry {name!r}: requires.lm_capabilities is a list of capability names")
    for capability in capabilities:
        if capability not in LM_CAPABILITY_FACTS:
            raise EntryError(
                f"adapter entry {name!r}: unknown required capability {capability!r} — capability "
                f"vocabulary: {', '.join(LM_CAPABILITY_FACTS)}"
            )
    for key in ("languages", "leaves"):
        if key in requires and not isinstance(requires[key], list):
            raise EntryError(f"adapter entry {name!r}: requires.{key} is a list")
