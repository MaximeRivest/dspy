"""The section-9 registration APIs and their admission gates (Epic D, D-7).

Registration is the extension API and every door carries a gate, run
eagerly: codecs face the round-trip probe battery (ADP-003), strategies
must self-declare capabilities and carry their parser (L9), presets
validate their templates with the language's teaching errors. Shipped code
entries (packaged/authored codecs) load through the same gates at link
time — import, identity verification, battery — never lazily at render.
"""

import hashlib
import importlib.metadata

import pytest

import dspy
from dspy.adapters._engine.admission import run_codec_battery
from dspy.adapters._engine.codecs import BAML, CODECS, PYDANTIC_JSON, TEXT_PYTHONISH, TextPythonishCodec
from dspy.adapters._engine.presets import PRESETS, get_preset
from dspy.adapters.serde import PresetAdapter, load_entry


class _RegisteredCodec(TextPythonishCodec):
    """A registrable codec: inherits a working triple under a new name."""

    name = "registered_text"


class _LossyCodec(TextPythonishCodec):
    """Round-trip breaker: renders lowercased, parses verbatim."""

    name = "lossy"

    def render_value(self, value, field_info):
        return super().render_value(value, field_info).lower()


# ---------------------------------------------------------------------------
# Discoverability: the language vocabulary is public data
# ---------------------------------------------------------------------------


def test_describe_template_language_is_public_vocabulary_data():
    vocabulary = dspy.adapters.describe_template_language()
    assert set(vocabulary["aggregate_slots"]) == {"inputs", "outputs", "demos", "history"}
    assert "chat" in vocabulary["parsers"]
    # A deep copy: mutating the caller's view never touches the language.
    vocabulary["parsers"] = ()
    assert "chat" in dspy.adapters.describe_template_language()["parsers"]


# ---------------------------------------------------------------------------
# Codec registration: the probe battery is the door
# ---------------------------------------------------------------------------


def test_builtin_codecs_pass_the_battery():
    """The gate is calibrated to admit the builtins — a third party
    registering an equivalent codec is never refused."""
    for codec in (TEXT_PYTHONISH, PYDANTIC_JSON, BAML):
        run_codec_battery(codec, name=codec.name)


def test_register_codec_round_trip():
    dspy.adapters.register_codec("registered_text", _RegisteredCodec())
    try:
        adapter = dspy.TemplateAdapter(
            messages=[{"role": "user", "content": "{question}"}],
            parse_mode="full_text",
            codecs={"input": "registered_text"},
        )
        assert adapter.format(dspy.Signature("question -> answer"), [], {"question": "Q?"}) == [
            {"role": "user", "content": "Q?"}
        ]
        assert adapter.dump_entry()["codecs"]["input"] == "registered_text"
    finally:
        dspy.adapters.unregister_codec("registered_text")
    assert "registered_text" not in CODECS


def test_register_codec_refuses_round_trip_failures_naming_the_probe():
    with pytest.raises(ValueError, match="round trip broke.*ADP-003"):
        dspy.adapters.register_codec("lossy", _LossyCodec())
    assert "lossy" not in CODECS


def test_register_codec_refuses_a_non_triple():
    class NotACodec:
        def render_value(self, value, field_info):
            return str(value)

    with pytest.raises(ValueError, match="render/parse/schema triple"):
        dspy.adapters.register_codec("half", NotACodec())


def test_register_codec_refuses_builtin_shadowing_and_duplicates():
    with pytest.raises(ValueError, match="builtin"):
        dspy.adapters.register_codec("text_pythonish", _RegisteredCodec())
    dspy.adapters.register_codec("dupe_check", _RegisteredCodec())
    try:
        with pytest.raises(ValueError, match="already registered"):
            dspy.adapters.register_codec("dupe_check", _RegisteredCodec())
    finally:
        dspy.adapters.unregister_codec("dupe_check")


def test_unregister_codec_refuses_builtins_and_unknowns():
    with pytest.raises(ValueError, match="builtin"):
        dspy.adapters.unregister_codec("baml")
    with pytest.raises(KeyError, match="no registered codec"):
        dspy.adapters.unregister_codec("never_registered")


# ---------------------------------------------------------------------------
# Strategy registration: declaration is the door
# ---------------------------------------------------------------------------


class _WellDeclaredStrategy:
    name = "test.well_declared"
    priority = 500
    exclusive_group = None
    capability_requirements = ("supports_reasoning",)
    parser = None  # request-only, declared explicitly

    def applies(self, ctx):
        return False

    def contribute(self, ctx):
        raise AssertionError("never applies in this test")


def test_register_strategy_by_role_and_by_annotation():
    from dspy.adapters._engine.strategies import strategy_for

    strategy = _WellDeclaredStrategy()
    dspy.adapters.register_strategy(strategy, role="reasoning")
    try:
        resolved, resolved_by = strategy_for("reasoning", str)
        assert resolved is strategy
        assert resolved_by == "role"
    finally:
        dspy.adapters.unregister_strategy(role="reasoning")

    dspy.adapters.register_strategy(strategy, annotation=_WellDeclaredStrategy)
    try:
        resolved, resolved_by = strategy_for("plain", _WellDeclaredStrategy)
        assert resolved is strategy
        assert resolved_by == "annotation"
    finally:
        dspy.adapters.unregister_strategy(annotation=_WellDeclaredStrategy)


def test_register_strategy_requires_exactly_one_key():
    strategy = _WellDeclaredStrategy()
    with pytest.raises(ValueError, match="exactly one"):
        dspy.adapters.register_strategy(strategy)
    with pytest.raises(ValueError, match="exactly one"):
        dspy.adapters.register_strategy(strategy, role="reasoning", annotation=str)


def test_register_strategy_requires_capability_self_declaration():
    class Undeclared:
        name = "test.undeclared"
        parser = None

        def applies(self, ctx):
            return False

        def contribute(self, ctx):
            return None

    with pytest.raises(ValueError, match="capability_requirements"):
        dspy.adapters.register_strategy(Undeclared(), role="reasoning")


def test_register_strategy_requires_the_carried_parser():
    class NoParser:
        name = "test.no_parser"
        capability_requirements = ()

        def applies(self, ctx):
            return False

        def contribute(self, ctx):
            return None

    with pytest.raises(ValueError, match="ADP-003"):
        dspy.adapters.register_strategy(NoParser(), role="reasoning")

    class BadParser(NoParser):
        name = "test.bad_parser"
        parser = object()  # no .parse

    with pytest.raises(ValueError, match="parse"):
        dspy.adapters.register_strategy(BadParser(), role="reasoning")


def test_register_strategy_refuses_unknown_roles():
    with pytest.raises(ValueError, match="valid roles"):
        dspy.adapters.register_strategy(_WellDeclaredStrategy(), role="vibes")


# ---------------------------------------------------------------------------
# Preset registration: the template mint is the door
# ---------------------------------------------------------------------------

_TRIAGE_MESSAGES = [
    {"role": "system", "content": "Triage tickets. {instruction}"},
    {"role": "demos"},
    {"role": "user", "content": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}"},
]


def test_register_preset_and_use_it_by_name():
    dspy.adapters.register_preset("triage", _TRIAGE_MESSAGES, parser="json")
    try:
        preset = get_preset("triage")
        assert preset.parser == "json"
        adapter = PresetAdapter("triage")
        signature = dspy.Signature("ticket -> category")
        messages = adapter.format(signature, [], {"ticket": "broken checkout"})
        assert messages[-1] == {"role": "user", "content": "ticket: broken checkout"}
        entry = adapter.dump_entry()
        assert entry["name"] == "triage"
        reloaded = load_entry(entry)
        assert reloaded.format(signature, [], {"ticket": "x"}) == adapter.format(signature, [], {"ticket": "x"})
    finally:
        dspy.adapters.unregister_preset("triage")
    assert "triage" not in PRESETS


def test_register_preset_refuses_builtin_shadowing_and_duplicates():
    with pytest.raises(ValueError, match="builtin"):
        dspy.adapters.register_preset("chat", _TRIAGE_MESSAGES, parser="chat")
    dspy.adapters.register_preset("dupe_preset", _TRIAGE_MESSAGES, parser="json")
    try:
        with pytest.raises(ValueError, match="already registered"):
            dspy.adapters.register_preset("dupe_preset", _TRIAGE_MESSAGES, parser="json")
    finally:
        dspy.adapters.unregister_preset("dupe_preset")


def test_register_preset_validates_the_template_eagerly():
    with pytest.raises(Exception, match="unknown slot function"):
        dspy.adapters.register_preset("broken", [{"role": "user", "content": "{outpts()}"}], parser="json")
    assert "broken" not in PRESETS


def test_register_preset_validates_strategy_bindings():
    with pytest.raises(ValueError, match="bindable roles"):
        dspy.adapters.register_preset(
            "bad_bindings", _TRIAGE_MESSAGES, parser="json", strategies={"vibes": "auto"}
        )


def test_unregister_preset_refuses_builtins_and_unknowns():
    with pytest.raises(ValueError, match="builtin"):
        dspy.adapters.unregister_preset("chat")
    with pytest.raises(KeyError, match="no registered preset"):
        dspy.adapters.unregister_preset("never_registered")


# ---------------------------------------------------------------------------
# The three-origin codec loader (shipped code entries)
# ---------------------------------------------------------------------------

_AUTHORED_SOURCE = """\
from dspy.adapters._engine.codecs import TextPythonishCodec


class ShippedCodec(TextPythonishCodec):
    name = "shipped"
"""


def _authored_entry(**overrides):
    entry = {
        "origin": "authored",
        "name": "shipped",
        "language": "python",
        "source": _AUTHORED_SOURCE,
        "attr": "ShippedCodec",
        "authored_by": "test-suite",
        "identity": "sha256:" + hashlib.sha256(_AUTHORED_SOURCE.encode("utf-8")).hexdigest(),
    }
    entry.update(overrides)
    return entry


def _entry_with_input_codec(ref):
    entry = dspy.ChatAdapter().dump_entry()
    entry["codecs"]["input"] = ref
    return entry


def test_authored_codec_entry_loads_verified_and_gated():
    entry = _entry_with_input_codec(_authored_entry())
    loaded = load_entry(entry)
    signature = dspy.Signature("question -> answer")
    assert loaded.format(signature, [], {"question": "Q?"}) == dspy.ChatAdapter().format(
        signature, [], {"question": "Q?"}
    )
    # The origin-tagged ref survives the round trip exactly.
    assert loaded.dump_entry()["codecs"]["input"] == _authored_entry()


def test_authored_codec_identity_mismatch_refuses():
    from dspy.adapters._engine.serde import PresetSerdeError

    entry = _entry_with_input_codec(_authored_entry(identity="sha256:" + "0" * 64))
    with pytest.raises(PresetSerdeError, match="identity mismatch.*ADP-011"):
        load_entry(entry)


def test_authored_codec_missing_provenance_refuses():
    from dspy.adapters._engine.serde import PresetSerdeError

    bad = _authored_entry()
    del bad["authored_by"]
    with pytest.raises(PresetSerdeError, match="authored_by"):
        load_entry(_entry_with_input_codec(bad))


def test_authored_codec_failing_the_battery_refuses():
    from dspy.adapters._engine.serde import PresetSerdeError

    source = _AUTHORED_SOURCE + "\n\nclass Broken(ShippedCodec):\n    def parse_value(self, text, annotation):\n        return None\n"
    bad = _authored_entry(
        source=source,
        attr="Broken",
        identity="sha256:" + hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    with pytest.raises(PresetSerdeError, match="round trip broke"):
        load_entry(_entry_with_input_codec(bad))


def test_packaged_codec_entry_loads_at_the_declared_version():
    entry = _entry_with_input_codec(
        {
            "origin": "packaged",
            "name": "pkg_codec",
            "language": "python",
            "import": "dspy.adapters._engine.codecs:PydanticJSONCodec",
            "dist": "dspy",
            "version": importlib.metadata.version("dspy"),
        }
    )
    loaded = load_entry(entry)
    assert loaded.preset.codecs["input"]["origin"] == "packaged"


def test_packaged_codec_version_mismatch_refuses_naming_both():
    from dspy.adapters._engine.serde import PresetSerdeError

    entry = _entry_with_input_codec(
        {
            "origin": "packaged",
            "name": "pkg_codec",
            "language": "python",
            "import": "dspy.adapters._engine.codecs:PydanticJSONCodec",
            "dist": "dspy",
            "version": "0.0.1-nope",
        }
    )
    with pytest.raises(PresetSerdeError, match="0.0.1-nope"):
        load_entry(entry)


def test_packaged_codec_missing_dist_refuses():
    from dspy.adapters._engine.serde import PresetSerdeError

    entry = _entry_with_input_codec(
        {
            "origin": "packaged",
            "name": "pkg_codec",
            "language": "python",
            "import": "nowhere.at.all:Nothing",
            "dist": "not-a-real-distribution",
            "version": "1.0.0",
        }
    )
    with pytest.raises(PresetSerdeError, match="not installed"):
        load_entry(entry)


def test_builtin_origin_resolves_internally():
    entry = _entry_with_input_codec({"origin": "builtin", "name": "baml"})
    loaded = load_entry(entry)
    assert loaded.preset.codecs["input"] == {"origin": "builtin", "name": "baml"}


def test_unknown_origin_refuses_naming_the_valid_set():
    from dspy.adapters._engine.serde import PresetSerdeError

    entry = _entry_with_input_codec({"origin": "cloud", "name": "x"})
    with pytest.raises(PresetSerdeError, match="builtin, packaged, authored"):
        load_entry(entry)


def test_foreign_language_refuses_at_the_profile_level():
    from dspy.adapters._engine.serde import PresetSerdeError

    entry = _entry_with_input_codec(_authored_entry(language="go"))
    with pytest.raises(PresetSerdeError, match="python only"):
        load_entry(entry)
