"""D-5 gates: preset serde is exact, versioned, and loudly linked.

Dump/load round trips reproduce identical entries AND identical rendered
messages; every refusal names its offender: malformed shape, missing or
incompatible versions (D-024), dangling codec/strategy/parser references
(ADP-005), origin-tagged code entries without a language (D-025). The
7-key literal table derives from the template, never authored.
"""

import ast
import json
from pathlib import Path

import pytest
from golden.cases import history_payload, multifield_payload, qa_payload

import dspy
from dspy.adapters._engine.serde import (
    LITERAL_TABLE_KEYS,
    PresetSerdeError,
    dumps_entry,
    load_preset,
)
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.serde import PresetAdapter, load_entry

ADAPTERS = {
    "chat": lambda: dspy.ChatAdapter(),
    "json": lambda: dspy.JSONAdapter(),
    "xml": lambda: dspy.XMLAdapter(),
    "baml": lambda: BAMLAdapter(),
}

PAYLOADS = {
    "qa-none": lambda: qa_payload("none"),
    "qa-complete": lambda: qa_payload("complete"),
    "qa-incomplete": lambda: qa_payload("incomplete"),
    "report-complete": lambda: multifield_payload("complete"),
    "history": history_payload,
}


# ---------------------------------------------------------------------------
# Round trips: identical entries, identical bytes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_dump_load_dump_is_identity(name):
    adapter = ADAPTERS[name]()
    entry = adapter.dump_entry()
    assert entry["name"] == name
    assert isinstance(entry["adapter_ir_version"], str)
    assert set(entry["versions"]) == {"roles", "strategies", "codecs", "template_language"}

    loaded = load_entry(entry)
    assert isinstance(loaded, PresetAdapter)
    assert loaded.dump_entry() == entry
    # Canonical JSON round trip: pure data, order preserved.
    assert json.loads(dumps_entry(entry)) == entry


@pytest.mark.parametrize("name", sorted(ADAPTERS))
@pytest.mark.parametrize("payload_name", sorted(PAYLOADS))
def test_loaded_entry_renders_byte_identical_messages(name, payload_name):
    adapter = ADAPTERS[name]()
    loaded = load_entry(adapter.dump_entry())
    payload = PAYLOADS[payload_name]()
    assert loaded.format(payload["signature"], list(payload["demos"]), dict(payload["inputs"])) == adapter.format(
        payload["signature"], list(payload["demos"]), dict(payload["inputs"])
    )


@pytest.mark.parametrize(
    "name,completion",
    [
        ("chat", "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"),
        ("json", '{"answer": "Paris"}'),
        ("xml", "<answer>\nParis\n</answer>"),
        ("baml", '{"answer": "Paris"}'),
    ],
)
def test_loaded_entry_parses_identically(name, completion):
    adapter = ADAPTERS[name]()
    loaded = load_entry(adapter.dump_entry())
    signature = dspy.Signature("question -> answer")
    assert loaded.parse(signature, completion) == adapter.parse(signature, completion) == {"answer": "Paris"}


def test_baml_entry_carries_the_pairing():
    entry = BAMLAdapter().dump_entry()
    assert entry["parser"] == "json"
    assert entry["codecs"] == {"input": "baml", "output": "baml"}
    assert "Output field" not in entry["template"][1].get("content", "")  # demos ride the json patterns
    assert "{f.typed_placeholder}" in entry["template"][0]["content"]  # the schema-prose arrangement


def test_strategies_bindings_ride_the_entry():
    adapter = dspy.ChatAdapter(strategies={"reasoning": "textual_field"})
    entry = adapter.dump_entry()
    assert entry["strategies"]["reasoning"] == "textual_field"
    assert entry["strategies"]["citations"] == "auto"


def test_loaded_entry_strategy_bindings_govern_the_loaded_adapter():
    """The entry's strategies block binds the loaded adapter exactly as the
    constructor kwarg bound the source: a textual reasoning binding must
    not silently re-enable the native channel under a live call."""
    from golden.harness import Recorder, StubLM

    from dspy.adapters._engine.builder import build_plan

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    source = dspy.ChatAdapter(strategies={"reasoning": "textual_field"})
    loaded = load_entry(source.dump_entry())
    assert loaded.strategies["reasoning"] == "textual_field"

    source_kwargs, loaded_kwargs = {}, {}
    source_built = build_plan(source, StubLM(Recorder(), supports_reasoning=True), source_kwargs, Sig, {"question": "Q?"})
    loaded_built = build_plan(loaded, StubLM(Recorder(), supports_reasoning=True), loaded_kwargs, Sig, {"question": "Q?"})

    assert source_kwargs == loaded_kwargs == {}  # the native channel stood down on BOTH
    assert loaded_built.plan.find_field("output", "reasoning").hidden is False
    assert list(loaded_built.render_signature.output_fields) == list(source_built.render_signature.output_fields)
    assert loaded.format(loaded_built.render_signature, [], {"question": "Q?"}) == source.format(
        source_built.render_signature, [], {"question": "Q?"}
    )


def test_loaded_entry_auto_bindings_still_resolve_native():
    """`auto` stays auto through the round trip: an unbound entry on a
    reasoning-capable LM serves the channel natively, like its source."""
    from golden.harness import Recorder, StubLM

    from dspy.adapters._engine.builder import build_plan

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    source = dspy.ChatAdapter()
    loaded = load_entry(source.dump_entry())

    source_kwargs, loaded_kwargs = {}, {}
    source_built = build_plan(source, StubLM(Recorder(), supports_reasoning=True), source_kwargs, Sig, {"question": "Q?"})
    loaded_built = build_plan(loaded, StubLM(Recorder(), supports_reasoning=True), loaded_kwargs, Sig, {"question": "Q?"})

    assert source_kwargs == loaded_kwargs == {"reasoning_effort": "low"}
    assert loaded_built.plan.find_field("output", "reasoning").hidden is True
    assert list(loaded_built.render_signature.output_fields) == list(source_built.render_signature.output_fields)


def test_loaded_tools_native_binding_enables_native_function_calling():
    """A tools=native_fc entry loads with native function calling on — the
    binding is the declaration, and the two surfaces must agree (L5)."""
    source = dspy.JSONAdapter(use_native_function_calling=True, strategies={"tools": "native_fc"})
    entry = source.dump_entry()
    assert entry["strategies"]["tools"] == "native_fc"
    loaded = load_entry(entry)
    assert loaded.use_native_function_calling is True
    assert loaded.strategies["tools"] == "native_fc"
    assert loaded.dump_entry() == entry


def test_declared_but_unimplemented_strategy_refuses_at_load():
    """Load-time validation admits exactly what construction admits: a
    vocabulary name with no implementation refuses with the same teaching
    error the constructor gives."""
    entry = _entry()
    entry["strategies"]["reasoning"] = "prefill"
    with pytest.raises(PresetSerdeError, match="not implemented.*implemented reasoning strategies"):
        load_preset(entry)


def test_full_text_entry_loads_and_round_trips():
    base = dspy.ChatAdapter().dump_entry()
    base["parser"] = "full_text"
    loaded = load_entry(base)
    signature = dspy.Signature("question -> answer")
    assert loaded.parse(signature, "anything at all") == {"answer": "anything at all"}
    with pytest.raises(ValueError, match="exactly one output field"):
        loaded.parse(dspy.Signature("question -> a, b"), "text")
    assert loaded.dump_entry() == base


# ---------------------------------------------------------------------------
# Loud refusals: shape, versions, links
# ---------------------------------------------------------------------------


def _entry():
    return dspy.ChatAdapter().dump_entry()


def test_missing_versions_block_is_malformed():
    entry = _entry()
    del entry["versions"]
    with pytest.raises(PresetSerdeError, match="missing \\['versions'\\]"):
        load_preset(entry)


def test_missing_vocabulary_version_is_malformed():
    entry = _entry()
    del entry["versions"]["codecs"]
    with pytest.raises(PresetSerdeError, match="missing 'codecs'"):
        load_preset(entry)


def test_unknown_major_refuses_naming_both_versions():
    entry = _entry()
    entry["versions"]["roles"] = "99.0.0"
    with pytest.raises(ValueError, match="99.0.0") as excinfo:
        load_preset(entry)
    assert "1.0.0" in str(excinfo.value)  # ours is named too


def test_zero_major_minor_bump_refuses():
    """While a version is 0.x the minor is breaking (semver-0)."""
    entry = _entry()
    entry["adapter_ir_version"] = "0.99.0"
    with pytest.raises(ValueError, match="0.99.0"):
        load_preset(entry)


def test_unknown_entry_key_refuses_instead_of_dropping():
    entry = _entry()
    entry["extra"] = True
    with pytest.raises(PresetSerdeError, match="unknown preset-entry keys \\['extra'\\]"):
        load_preset(entry)


def test_dangling_codec_reference_is_a_link_error():
    entry = _entry()
    entry["codecs"]["input"] = "morse"
    with pytest.raises(PresetSerdeError, match="dangling input codec reference 'morse'"):
        load_preset(entry)


def test_unknown_strategy_binding_refuses():
    entry = _entry()
    entry["strategies"]["reasoning"] = "quantum"
    with pytest.raises(PresetSerdeError, match="unknown reasoning strategy 'quantum'"):
        load_preset(entry)


def test_unknown_parser_refuses_naming_the_valid_set():
    entry = _entry()
    entry["parser"] = "yaml"
    with pytest.raises(PresetSerdeError, match="valid parsers: chat, json, xml, full_text"):
        load_preset(entry)


def test_origin_tagged_codec_entries_carry_language():
    entry = _entry()
    entry["codecs"]["input"] = {"origin": "authored", "source": "..."}
    with pytest.raises(PresetSerdeError, match="carries no 'language'"):
        load_preset(entry)
    entry["codecs"]["input"] = {"origin": "authored", "source": "...", "language": "python"}
    with pytest.raises(PresetSerdeError, match="registration API"):
        load_preset(entry)


def test_template_errors_surface_eagerly_at_load():
    entry = _entry()
    entry["template"][0]["content"] = "{outpts()}"
    with pytest.raises(Exception, match="unknown slot function"):
        load_preset(entry)


def test_dump_refuses_for_override_routed_and_presetless_adapters():
    class Custom(dspy.ChatAdapter):
        def format_field_description(self, signature):
            return "custom"

    from dspy.utils.dummies import DummyLM

    with pytest.raises(ValueError, match="non-engine surface"):
        Custom().dump_entry()
    with pytest.raises(ValueError, match="declares no preset"):
        dspy.TwoStepAdapter(extraction_model=DummyLM([])).dump_entry()


def test_serde_surface_reads_no_ambient_settings():
    """Load is the link step: zero dspy.settings reads (L5). Mechanical:
    neither serde module imports settings."""
    for module in ("dspy/adapters/serde.py", "dspy/adapters/_engine/serde.py"):
        tree = ast.parse((Path(__file__).resolve().parents[3] / module).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "settings" in node.module:
                raise AssertionError(f"{module} imports {node.module}")
            if isinstance(node, ast.Import) and any("settings" in alias.name for alias in node.names):
                raise AssertionError(f"{module} imports settings")


# ---------------------------------------------------------------------------
# The derived 7-key summary view
# ---------------------------------------------------------------------------


def test_literal_table_is_derived_per_preset():
    chat = dspy.ChatAdapter().literal_table()
    assert set(chat) <= set(LITERAL_TABLE_KEYS)
    assert chat["input_field_render"] == "[[ ## {name} ## ]]\n{value}"
    assert chat["output_field_render"] == "[[ ## {name} ## ]]\n{value}"
    assert chat["field_separator"] == "\n\n"
    assert chat["output_structure"] == "markers"
    assert chat["completed_marker"] == "[[ ## completed ## ]]"
    assert chat["output_requirement"].startswith("Respond with the corresponding output fields")
    assert chat["parse_pattern"] == r"\[\[ ## (\w+) ## \]\]"

    json_table = dspy.JSONAdapter().literal_table()
    assert json_table["output_structure"] == "json_object"
    assert "output_field_render" not in json_table  # whole-object assistant: no per-field pattern
    assert "parse_pattern" not in json_table  # structural parser: no regex literal

    xml = dspy.XMLAdapter().literal_table()
    assert xml["output_structure"] == "tags"
    assert xml["input_field_render"] == "<{name}>\n{value}\n</{name}>"
    assert "completed_marker" not in xml
    assert xml["parse_pattern"] == r"<(?P<name>\w+)>((?P<content>.*?))</\1>"

    baml = BAMLAdapter().literal_table()
    assert baml["output_structure"] == "json_object"
    assert baml["completed_marker"] == "[[ ## completed ## ]]"


def test_loaded_entry_literal_table_matches_the_source_adapter():
    for factory in ADAPTERS.values():
        adapter = factory()
        assert load_entry(adapter.dump_entry()).literal_table() == adapter.literal_table()
