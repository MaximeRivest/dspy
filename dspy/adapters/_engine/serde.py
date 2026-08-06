"""Preset serde: the component-4 entry as data, dumped and loaded exactly.

An adapter-as-preset dumps to one JSON-able entry — template as raw message
data, parser binding, codec bindings, strategy bindings, config — versioned
from its first shape (D-024): ``adapter_ir_version`` plus the closed-
vocabulary ``versions`` block. Loading validates shape, versions, template,
and every binding reference with ZERO ``dspy.settings`` reads; a dangling
codec/strategy/parser reference is a link error refused loudly naming the
reference (ADP-005/L5), and unknown or missing versions refuse naming both
sides. Serde is exact: absent is not null, unknown keys refuse instead of
silently dropping, and ``load(dump(x))`` reproduces ``x``.

The 7-key ``literal_table`` summary view derives from the template here —
never authored (spec section 5) — for explain/cross-language readers.
"""

import json
from copy import deepcopy
from typing import Any

from dspy.adapters._engine.presets import Preset, _make_preset
from dspy.adapters._engine.strategies.vocabulary import check_binding_name
from dspy.adapters._engine.template.parser import (
    AggregateSlot,
    Loop,
    LoopVarLiteral,
    LoopVarSlot,
    ParsedTemplate,
    Text,
    directive_pair,
)
from dspy.adapters._engine.template.vocabulary import VOCABULARY
from dspy.adapters._engine.versions import ADAPTER_IR_VERSION, check_version_compatible, versions_block

#: The entry's exact key set, in canonical order. Serde is exact: an entry
#: carrying any other key refuses (a key the loader would drop is silent
#: loss).
ENTRY_KEYS = ("name", "adapter_ir_version", "versions", "template", "parser", "codecs", "strategies", "config")

#: The 7-key summary-view vocabulary (spec section 5); keys are derived
#: from the template and absent when the template has no such literal.
LITERAL_TABLE_KEYS = (
    "input_field_render",
    "output_field_render",
    "field_separator",
    "output_structure",
    "completed_marker",
    "output_requirement",
    "parse_pattern",
)

_COMPLETED_MARKER = "[[ ## completed ## ]]"


class PresetSerdeError(ValueError):
    """A preset entry the loader refuses: malformed shape, incompatible
    version, or a dangling reference — always naming the offender."""


def build_entry(*, name, template_raw, parser, codecs, strategies, config) -> dict:
    """Assemble the canonical component-4 entry dict."""
    return {
        "name": name,
        "adapter_ir_version": ADAPTER_IR_VERSION,
        "versions": versions_block(),
        "template": deepcopy([dict(message) for message in template_raw]),
        "parser": parser,
        "codecs": dict(codecs),
        "strategies": dict(strategies),
        "config": deepcopy(dict(config)),
    }


def dump_preset(preset: Preset) -> dict:
    """One preset as its canonical component-4 entry."""
    return build_entry(
        name=preset.name,
        template_raw=preset.template.raw,
        parser=preset.parser,
        codecs=preset.codecs,
        strategies=preset.strategies,
        config=preset.config,
    )


def dumps_entry(entry: dict) -> str:
    """The entry as canonical JSON: compact, unsorted (order is data)."""
    return json.dumps(entry, ensure_ascii=False, separators=(",", ":"))


def load_preset(entry: dict) -> Preset:
    """Validate and link one component-4 entry back into a Preset.

    Zero ambient reads. Validation order: shape, versions, template (eager
    teaching errors), parser, codec refs, strategy bindings, config.
    """
    if not isinstance(entry, dict):
        raise PresetSerdeError(f"a preset entry is a dict, got {type(entry).__name__}")
    unknown = set(entry) - set(ENTRY_KEYS)
    if unknown:
        raise PresetSerdeError(
            f"unknown preset-entry keys {sorted(unknown)} — exact serde refuses keys it would drop; "
            f"valid keys: {', '.join(ENTRY_KEYS)}"
        )
    missing = [key for key in ENTRY_KEYS if key not in entry]
    if missing:
        raise PresetSerdeError(
            f"preset entry is missing {missing} — every entry carries all of: {', '.join(ENTRY_KEYS)} "
            "(a missing versions block is malformed, never grandfathered — D-024)"
        )

    name = entry["name"]
    if not isinstance(name, str) or not name:
        raise PresetSerdeError(f"preset entry 'name' must be a non-empty string, got {name!r}")

    check_version_compatible("adapter_ir", entry["adapter_ir_version"], ADAPTER_IR_VERSION)
    versions = entry["versions"]
    if not isinstance(versions, dict):
        raise PresetSerdeError(f"preset entry {name!r}: 'versions' must be a dict, got {type(versions).__name__}")
    ours = versions_block()
    for kind, our_version in ours.items():
        if kind not in versions:
            raise PresetSerdeError(
                f"preset entry {name!r}: versions block is missing {kind!r} — a serialized preset names the "
                f"version of every closed vocabulary it references ({', '.join(ours)}); malformed, refused (D-024)"
            )
        check_version_compatible(kind, versions[kind], our_version)
    unknown_versions = set(versions) - set(ours)
    if unknown_versions:
        raise PresetSerdeError(
            f"preset entry {name!r}: versions block names unknown vocabularies {sorted(unknown_versions)} — "
            f"this engine knows: {', '.join(ours)}"
        )

    codecs = entry["codecs"]
    if not isinstance(codecs, dict) or set(codecs) != {"input", "output"}:
        raise PresetSerdeError(
            f"preset entry {name!r}: 'codecs' must be a dict with exactly 'input' and 'output' bindings, "
            f"got {codecs!r}"
        )
    for direction, ref in codecs.items():
        _check_codec_ref(name, direction, ref)

    strategies = entry["strategies"]
    if not isinstance(strategies, dict):
        raise PresetSerdeError(f"preset entry {name!r}: 'strategies' must be a dict, got {type(strategies).__name__}")
    for role, strategy_name in strategies.items():
        # Load-time validation admits exactly what construction admits —
        # the shared check, with the entry named (L5 applied to loading).
        try:
            check_binding_name(role, strategy_name)
        except ValueError as error:
            raise PresetSerdeError(f"preset entry {name!r}: {error}") from None

    config = entry["config"]
    if not isinstance(config, dict):
        raise PresetSerdeError(f"preset entry {name!r}: 'config' must be a dict, got {type(config).__name__}")

    parser = entry["parser"]
    if parser not in VOCABULARY["parsers"]:
        raise PresetSerdeError(
            f"preset entry {name!r}: unknown parser {parser!r} — valid parsers: {', '.join(VOCABULARY['parsers'])}"
        )

    # Template parses eagerly with the language's own teaching errors;
    # _make_preset re-links parser and codec refs on the way through.
    return _make_preset(
        name=name,
        template_messages=deepcopy(entry["template"]),
        parser=parser,
        codecs=dict(codecs),
        strategies=dict(strategies),
        config=deepcopy(config),
    )


def _check_codec_ref(entry_name: str, direction: str, ref) -> None:
    """Link one codec reference eagerly.

    A string names a registered codec (a dangling ref is a link error,
    ADP-005). A dict is an origin-tagged code entry and materializes NOW
    through the three-origin loader (spec section 9) — import, identity
    verification, and the round-trip battery all refuse at link, never at
    render.
    """
    from dspy.adapters._engine.admission import materialize_codec_entry
    from dspy.adapters._engine.codecs import CODECS

    if isinstance(ref, str):
        if ref not in CODECS:
            raise PresetSerdeError(
                f"preset entry {entry_name!r}: dangling {direction} codec reference {ref!r} — "
                f"registered codecs: {', '.join(CODECS)} (ADP-005: a dangling ref is a link error)"
            )
        return
    if isinstance(ref, dict):
        try:
            materialize_codec_entry(ref)
        except ValueError as error:
            raise PresetSerdeError(f"preset entry {entry_name!r}: {error}") from None
        return
    raise PresetSerdeError(
        f"preset entry {entry_name!r}: codec reference for {direction!r} must be a string name or an "
        f"origin-tagged entry (origin: builtin, packaged, authored), got {ref!r}"
    )


# ---------------------------------------------------------------------------
# The derived 7-key summary view
# ---------------------------------------------------------------------------


def derive_literal_table(template: ParsedTemplate, parser: str) -> dict:
    """Derive the 7-key literal-table summary from a parsed template.

    Never authored (spec section 5): every value is computed from the
    template's own nodes and raw text, keys absent when the template has no
    such literal. The full form IS the template; this view exists for
    explain and cross-language readers.
    """
    table: dict = {}

    demos = template.demos_directive
    user_nodes, assistant_nodes = (
        directive_pair(demos, None, parser) if demos is not None else (None, None)
    )

    input_loop = _first_loop(user_nodes, "inputs")
    if input_loop is not None:
        table["input_field_render"] = _symbolic_pattern(input_loop.body)
        table["field_separator"] = input_loop.separator

    output_loop = _first_loop(assistant_nodes, "outputs")
    if output_loop is not None:
        table["output_field_render"] = _symbolic_pattern(output_loop.body)

    structure = _output_structure(assistant_nodes)
    if structure is not None:
        table["output_structure"] = structure

    if any(_COMPLETED_MARKER in raw for raw in _raw_strings(template)):
        table["completed_marker"] = _COMPLETED_MARKER

    requirement = _output_requirement(template)
    if requirement is not None:
        table["output_requirement"] = requirement

    pattern = _parse_pattern(parser)
    if pattern is not None:
        table["parse_pattern"] = pattern

    assert set(table) <= set(LITERAL_TABLE_KEYS)
    return table


def _first_loop(nodes, collection: str):
    for node in nodes or ():
        if isinstance(node, Loop) and node.collection == collection:
            return node
    return None


def _symbolic_pattern(body) -> str:
    """A loop body as the symbolic per-field pattern ({name}, {value}, ...)."""
    parts = []
    for node in body:
        if isinstance(node, Text):
            parts.append(node.text)
        elif isinstance(node, LoopVarSlot):
            parts.append("{" + node.attr + "}")
        elif isinstance(node, LoopVarLiteral):
            parts.append("{{" + node.attr + "}}")
        else:
            parts.append("{...}")
    return "".join(parts)


def _output_structure(assistant_nodes) -> str | None:
    """How the assistant turn structures outputs: markers, tags, or one
    JSON object — read from the demo assistant pattern's shape."""
    if assistant_nodes is None:
        return None
    for node in assistant_nodes:
        if isinstance(node, AggregateSlot) and node.kind == "outputs" and node.style == "json_object":
            return "json_object"
    loop = _first_loop(assistant_nodes, "outputs")
    if loop is None:
        return None
    pattern = _symbolic_pattern(loop.body)
    if "[[ ## " in pattern:
        return "markers"
    if "<{name}>" in pattern:
        return "tags"
    return "values"


def _raw_strings(template: ParsedTemplate):
    for message in template.raw:
        if not isinstance(message, dict):
            continue
        for key in ("content", "user", "assistant"):
            value = message.get(key)
            if isinstance(value, str):
                yield value


def _output_requirement(template: ParsedTemplate) -> str | None:
    """The main user message's trailing requirement literal, as the
    template spells it (after the user fragment slot)."""
    slot = "{fragments('user')}"
    for message in reversed(template.raw):
        if isinstance(message, dict) and message.get("role") == "user" and isinstance(message.get("content"), str):
            content = message["content"]
            if slot in content:
                return content.split(slot, 1)[1].lstrip("\n")
            return None
    return None


def _parse_pattern(parser: str) -> str | None:
    from dspy.adapters._engine.formats.chat import ChatFormat
    from dspy.adapters._engine.formats.xml import XMLFormat

    if parser == "chat":
        return ChatFormat.field_header_pattern.pattern
    if parser == "xml":
        return XMLFormat.field_pattern.pattern
    return None


def entry_summary(entry: dict) -> dict[str, Any]:
    """The literal table of a dumped entry (loads the template first)."""
    preset = load_preset(entry)
    return derive_literal_table(preset.template, preset.parser)
