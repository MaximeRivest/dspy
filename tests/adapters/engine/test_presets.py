"""D-2 gates: presets defined as templates ARE the render path, byte-identical.

Three layers of proof, from narrow to wide:

- construction: the built-in presets parse eagerly, declare the classic
  message shape, and their codec bindings resolve to the same codec objects
  the Format classes expose;
- method parity: every delegated Format surface equals the corresponding
  byte-untouched legacy adapter method, across the golden payload shapes;
- walker parity: rendering the preset template directly (the pure template
  walker, no Format involved) reproduces the forced-legacy adapter's full
  ``format()`` message list.

The golden corpus remains the outer gate; these tests point at the exact
seam when it trips.
"""

import pytest
from golden.cases import history_payload, multifield_payload, qa_payload

from dspy.adapters._engine.codecs import resolve_codec
from dspy.adapters._engine.formats.chat import ChatFormat
from dspy.adapters._engine.formats.json import JSONFormat
from dspy.adapters._engine.formats.xml import XMLFormat
from dspy.adapters._engine.presets import PRESETS, get_preset
from dspy.adapters._engine.template import (
    ContentMessage,
    DemosDirective,
    HistoryDirective,
    declared_capacity,
    preview,
    render_template_messages,
)
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.xml_adapter import XMLAdapter


class _LegacyChat(ChatAdapter):
    """Pass-through format override: routes the instance byte-untouched legacy."""

    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


class _LegacyJSON(JSONAdapter):
    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


class _LegacyXML(XMLAdapter):
    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


FORMATS = {"chat": ChatFormat(), "json": JSONFormat(), "xml": XMLFormat()}
LEGACY_ADAPTERS = {"chat": _LegacyChat(), "json": _LegacyJSON(), "xml": _LegacyXML()}

PAYLOADS = {
    "qa-none": lambda: qa_payload("none"),
    "qa-complete": lambda: qa_payload("complete"),
    "qa-incomplete": lambda: qa_payload("incomplete"),
    "report-complete": lambda: multifield_payload("complete"),
    "report-incomplete": lambda: multifield_payload("incomplete"),
    "history": history_payload,
}


# ---------------------------------------------------------------------------
# Construction and binding consistency
# ---------------------------------------------------------------------------


def test_builtin_presets_declare_the_classic_message_shape():
    for preset in PRESETS.values():
        shapes = [type(message) for message in preset.template.messages]
        assert shapes == [ContentMessage, DemosDirective, HistoryDirective, ContentMessage]
        assert preset.template.messages[0].role == "system"
        assert preset.template.messages[-1].role == "user"
        assert preset.template.demos_directive.user is not None


def test_builtin_presets_parser_and_strategy_bindings():
    assert get_preset("chat").parser == "chat"
    assert get_preset("json").parser == "json"
    assert get_preset("xml").parser == "xml"
    for preset in PRESETS.values():
        assert preset.strategies["history"] == "directive_turns"
        assert preset.strategies["reasoning"] == "auto"


def test_preset_codec_bindings_match_format_objects():
    """Preset data and Format properties must name the same codecs until
    D-5 flips authority to the preset."""
    for name, fmt in FORMATS.items():
        preset = get_preset(name)
        assert resolve_codec(preset.codecs["input"]) is fmt.input_codec
        assert resolve_codec(preset.codecs["output"]) is fmt.output_codec


def test_format_classes_name_their_presets():
    assert ChatFormat.preset_name == "chat"
    assert JSONFormat.preset_name == "json"
    assert XMLFormat.preset_name == "xml"


def test_unknown_preset_and_codec_refuse_loudly():
    with pytest.raises(KeyError, match="available presets: chat, json, xml"):
        get_preset("yaml")
    with pytest.raises(KeyError, match="registered codecs"):
        resolve_codec("morse")


def test_builtin_preset_capacity_hosts_the_textual_roles():
    for preset in PRESETS.values():
        capacity = declared_capacity(preset.template)
        assert capacity.iterates_inputs and capacity.iterates_outputs
        assert capacity.fragment_targets == {"system", "user"}
        assert capacity.hosts_demos and capacity.hosts_history
        assert capacity.hosts_role_textually("reasoning")


# ---------------------------------------------------------------------------
# Method-level parity: delegated Format surface vs legacy adapter bodies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt_name", sorted(FORMATS))
@pytest.mark.parametrize("payload_name", sorted(PAYLOADS))
def test_delegated_methods_match_legacy_adapter(fmt_name, payload_name):
    fmt = FORMATS[fmt_name]
    legacy = LEGACY_ADAPTERS[fmt_name]
    payload = PAYLOADS[payload_name]()
    signature, inputs = payload["signature"], payload["inputs"]

    assert fmt.render_system(signature) == legacy.format_system_message(signature)
    assert fmt.render_user_content(signature, inputs, main_request=True) == legacy.format_user_message_content(
        signature, inputs, main_request=True
    )
    assert fmt.render_user_content(signature, inputs, prefix="P ", suffix=" S") == legacy.format_user_message_content(
        signature, inputs, prefix="P ", suffix=" S"
    )

    outputs = payload.get("surface_outputs") or {}
    assert fmt.render_assistant_content(signature, outputs) == legacy.format_assistant_message_content(
        signature, outputs
    )
    missing = "Not supplied for this particular example. "
    assert fmt.render_assistant_content(signature, {}, missing_field_message=missing) == (
        legacy.format_assistant_message_content(signature, {}, missing_field_message=missing)
    )


# ---------------------------------------------------------------------------
# Walker parity: the template alone reproduces the legacy pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt_name", sorted(FORMATS))
@pytest.mark.parametrize("payload_name", sorted(PAYLOADS))
def test_template_walker_matches_legacy_adapter_format(fmt_name, payload_name):
    """The preset template rendered by the pure walker — no Format, no
    adapter — equals the byte-untouched legacy pipeline's message list."""
    fmt = FORMATS[fmt_name]
    legacy = LEGACY_ADAPTERS[fmt_name]
    payload = PAYLOADS[payload_name]()

    walked = render_template_messages(
        get_preset(fmt_name).template,
        payload["signature"],
        demos=payload["demos"],
        inputs=dict(payload["inputs"]),
        input_codec=fmt.input_codec,
        output_codec=fmt.output_codec,
    )
    assert walked == legacy.format(payload["signature"], list(payload["demos"]), dict(payload["inputs"]))


ENGINE_ADAPTERS = {"chat": ChatAdapter(), "json": JSONAdapter(), "xml": XMLAdapter()}


@pytest.mark.parametrize("fmt_name", sorted(FORMATS))
@pytest.mark.parametrize("payload_name", sorted(PAYLOADS))
def test_preview_matches_the_engine_adapter_format(fmt_name, payload_name):
    """preview() is contract surface: its message list must be byte-equal to
    what the engine-backed adapter sends for the same preset and inputs."""
    fmt = FORMATS[fmt_name]
    payload = PAYLOADS[payload_name]()

    previewed = preview(
        get_preset(fmt_name).template,
        payload["signature"],
        demos=payload["demos"],
        inputs=dict(payload["inputs"]),
        input_codec=fmt.input_codec,
        output_codec=fmt.output_codec,
    )
    engine = ENGINE_ADAPTERS[fmt_name].format(payload["signature"], list(payload["demos"]), dict(payload["inputs"]))
    assert previewed == engine


@pytest.mark.parametrize("fmt_name", sorted(FORMATS))
def test_empty_field_sets_render_byte_identical_to_forced_legacy(fmt_name):
    """Zero-output and zero-field signatures: legacy join-then-strip
    collapses the trailing empty structure sections; the templates must
    reproduce that byte-for-byte (the xml preset's section block exists
    for exactly this)."""
    import dspy

    zero_outputs = dspy.make_signature({"question": (str, dspy.InputField())}, "No outputs.")
    zero_fields = dspy.make_signature({}, "Nothing.")
    zero_inputs = dspy.make_signature({"answer": (str, dspy.OutputField())}, "Only outputs.")
    for signature, inputs in ((zero_outputs, {"question": "hm?"}), (zero_fields, {}), (zero_inputs, {})):
        engine = ENGINE_ADAPTERS[fmt_name].format(signature, [], dict(inputs))
        legacy = LEGACY_ADAPTERS[fmt_name].format(signature, [], dict(inputs))
        assert engine == legacy


def test_xml_native_function_calling_toolcalls_only_output_matches_forced_legacy():
    """The production-reachable zero-visible-outputs shape: native FC hides
    the ToolCalls-only output set from the render signature."""
    import dspy

    def add(a: int, b: int) -> int:
        return a + b

    signature = dspy.make_signature(
        {
            "question": (str, dspy.InputField()),
            "tools": (list[dspy.Tool], dspy.InputField()),
            "tool_calls": (dspy.ToolCalls, dspy.OutputField()),
        },
        "Use the tools.",
    )
    inputs = {"question": "2+2?", "tools": [dspy.Tool(add)]}
    engine = XMLAdapter(use_native_function_calling=True).format(signature, [], dict(inputs))
    legacy = _LegacyXML(use_native_function_calling=True).format(signature, [], dict(inputs))
    assert engine == legacy


@pytest.mark.parametrize("fmt_name", sorted(FORMATS))
def test_reserved_field_names_render_byte_identical_to_forced_legacy(fmt_name):
    """The preset templates spell the instructions as {instruction(...)} —
    the call form — so a signature field named 'instruction' (or any other
    reserved name) renders exactly as legacy rendered it, no refusal."""
    import dspy

    signature = dspy.Signature("instruction, outputs -> response", "Follow the instruction.")
    inputs = {"instruction": "Make it formal.", "outputs": "one paragraph"}
    engine = ENGINE_ADAPTERS[fmt_name].format(signature, [], dict(inputs))
    legacy = LEGACY_ADAPTERS[fmt_name].format(signature, [], dict(inputs))
    assert engine == legacy
    assert "Make it formal." in engine[-1]["content"]


# ---------------------------------------------------------------------------
# The BAML pairing: preset json + baml codecs + the schema-prose arrangement
# ---------------------------------------------------------------------------

from dspy.adapters._engine.formats import Format
from dspy.adapters._engine.formats.baml import BAMLFormat
from dspy.adapters.baml_adapter import BAMLAdapter


class _LegacyBAML(BAMLAdapter):
    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


@pytest.mark.parametrize("payload_name", sorted(PAYLOADS))
def test_baml_system_template_matches_the_composed_legacy_body(payload_name):
    """The pairing's system arrangement (template data + codec schema
    spelling) diffs against the frozen composed path so the two sources
    cannot drift."""
    fmt = BAMLFormat()
    signature = PAYLOADS[payload_name]()["signature"]
    assert fmt.render_system(signature) == Format.render_system(fmt, signature)


@pytest.mark.parametrize("payload_name", sorted(PAYLOADS))
def test_baml_delegated_surfaces_match_the_legacy_adapter(payload_name):
    fmt = BAMLFormat()
    legacy = _LegacyBAML()
    payload = PAYLOADS[payload_name]()
    signature, inputs = payload["signature"], payload["inputs"]

    assert fmt.render_system(signature) == legacy.format_system_message(signature)
    assert fmt.render_user_content(signature, inputs, main_request=True) == legacy.format_user_message_content(
        signature, inputs, main_request=True
    )
    outputs = payload.get("surface_outputs") or {}
    assert fmt.render_assistant_content(signature, outputs) == legacy.format_assistant_message_content(
        signature, outputs
    )


def test_baml_empty_field_sets_render_byte_identical_to_forced_legacy():
    """The legacy \\n-join collapses empty input/output sections without
    stray blank lines — a shape the chat preset's structure region does NOT
    have, which is why the pairing carries its own arrangement."""
    import dspy

    zero_outputs = dspy.make_signature({"question": (str, dspy.InputField())}, "No outputs.")
    zero_fields = dspy.make_signature({}, "Nothing.")
    zero_inputs = dspy.make_signature({"answer": (str, dspy.OutputField())}, "Only outputs.")
    for signature, inputs in ((zero_outputs, {"question": "hm?"}), (zero_fields, {}), (zero_inputs, {})):
        engine = BAMLAdapter().format(signature, [], dict(inputs))
        legacy = _LegacyBAML().format(signature, [], dict(inputs))
        assert engine == legacy


def test_baml_schema_prose_renders_through_the_output_codec_binding():
    import pydantic

    import dspy

    class Fact(pydantic.BaseModel):
        claim: str
        confidence: float

    signature = dspy.make_signature(
        {"note": (str, dspy.InputField()), "facts": (list[Fact], dspy.OutputField())}, "Extract."
    )
    system = BAMLFormat().render_system(signature)
    assert "[[ ## facts ## ]]\nOutput field `facts` should be of type: [\n" in system
    assert "claim: string," in system
    assert "[[ ## completed ## ]]\nIn adhering to this structure" in system


# ---------------------------------------------------------------------------
# The named byte traps, pinned close to the seam
# ---------------------------------------------------------------------------


def test_chat_demo_assistant_keeps_trailing_newline_after_completed_marker():
    payload = qa_payload("complete")
    content = FORMATS["chat"].render_assistant_content(payload["signature"], {"answer": "Berlin"})
    assert content.endswith("\n\n[[ ## completed ## ]]\n")


def test_chat_system_keeps_single_newline_before_objective_and_indent():
    payload = qa_payload("none")
    system = FORMATS["chat"].render_system(payload["signature"])
    assert "[[ ## completed ## ]]\nIn adhering to this structure, your objective is: \n        " in system


def test_empty_body_between_prefix_and_suffix_joins_like_legacy_chat():
    """Pipeline-unreachable corner of the delegated surface: an empty body
    contributes no join element, the legacy ChatAdapter/JSONAdapter shape.
    Legacy XMLAdapter accidentally kept the empty block ('P\\n\\n\\n\\nS');
    the engine unifies all three formats on the chat shape."""
    import dspy

    signature = dspy.make_signature(
        {"q": (str, dspy.InputField()), "r": (str, dspy.OutputField())}, "x"
    )
    for fmt_name in ("chat", "json"):
        got = FORMATS[fmt_name].render_user_content(signature, {}, prefix="P", suffix="S")
        want = LEGACY_ADAPTERS[fmt_name].format_user_message_content(signature, {}, prefix="P", suffix="S")
        assert got == want == "P\n\nS"
    assert FORMATS["xml"].render_user_content(signature, {}, prefix="P", suffix="S") == "P\n\nS"


def test_incomplete_demo_missing_message_trailing_space_strips_at_block_end():
    payload = qa_payload("incomplete")
    content = FORMATS["chat"].render_assistant_content(
        payload["signature"], {}, missing_field_message="Not supplied for this particular example. "
    )
    assert content == "[[ ## answer ## ]]\nNot supplied for this particular example.\n\n[[ ## completed ## ]]\n"
