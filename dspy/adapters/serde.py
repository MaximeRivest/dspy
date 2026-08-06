"""Adapter ⇄ component-4 entry: the D-5 dump/load surface.

``Adapter.dump_entry()`` (defined on the base class) serializes an
engine-backed adapter's effective preset — template as data, parser
binding, codec bindings, strategy bindings, config, versions — and
``load_entry`` links it back into a :class:`PresetAdapter` that renders
through the pure template walker and parses through the entry's parser
binding, with zero ``dspy.settings`` reads anywhere on the path. Round
trip is exact: the loaded adapter renders byte-identical messages and
re-dumps the identical entry.
"""

from typing import Any

from dspy.adapters.base import Adapter


def load_entry(entry: dict) -> "PresetAdapter":
    """Link a component-4 preset entry back into a working adapter.

    Validation is eager and loud: malformed shape, missing or incompatible
    versions, template errors, and dangling codec/strategy/parser
    references all refuse naming the offender (ADP-005/L5). No ambient
    state is consulted.
    """
    from dspy.adapters._engine.serde import load_preset

    return PresetAdapter(load_preset(entry))


class PresetAdapter(Adapter):
    """An adapter reconstructed from a component-4 preset entry.

    Rendering walks the entry's template (``render_template_messages``)
    with the entry's codec bindings and parser-keyed directive defaults;
    parsing dispatches on the entry's parser binding, and the entry's
    strategy bindings feed the same surface the constructor kwarg feeds —
    the plan builder consults a loaded adapter and its source identically.
    Everything the adapter does is stated by the entry — nothing resolves
    ambiently.
    """

    def __init__(self, preset, **kwargs):
        # Non-"auto" bindings from the entry ARE constructor bindings: a
        # dumped `reasoning: textual_field` must stand the native channel
        # down under a live call exactly as it did on the source adapter.
        bindings = {role: name for role, name in preset.strategies.items() if name != "auto"}
        if bindings.get("tools") == "native_fc":
            # The binding is the declaration; the legacy kwarg must agree.
            kwargs.setdefault("use_native_function_calling", True)
        super().__init__(strategies=bindings or None, **kwargs)
        self.preset = preset
        self._parser_impl = _parser_for(preset)

    def format(self, signature, demos, inputs) -> list[dict[str, Any]]:
        from dspy.adapters._engine.codecs import resolve_codec
        from dspy.adapters._engine.render import active_plan_fragments
        from dspy.adapters._engine.template import render_template_messages
        from dspy.adapters.base import _expand_legacy_custom_type_markers_in_chat_message

        messages = render_template_messages(
            self.preset.template,
            signature,
            demos=list(demos),
            inputs=dict(inputs),
            input_codec=resolve_codec(self.preset.codecs["input"]),
            output_codec=resolve_codec(self.preset.codecs["output"]),
            fragments=active_plan_fragments(),
            parser=self.preset.parser,
        )
        return [_expand_legacy_custom_type_markers_in_chat_message(message) for message in messages]

    def parse(self, signature, completion: str) -> dict[str, Any]:
        return self._parser_impl.parse(signature, completion)

    def dump_entry(self) -> dict:
        from dspy.adapters._engine.serde import dump_preset

        return dump_preset(self.preset)

    def literal_table(self) -> dict:
        from dspy.adapters._engine.serde import derive_literal_table

        return derive_literal_table(self.preset.template, self.preset.parser)


class _FullTextParser:
    """The full_text parser binding: the whole completion into exactly one
    output field (spec section 4)."""

    def __init__(self, codec):
        self._codec = codec

    def parse(self, signature, completion: str) -> dict[str, Any]:
        fields = signature.output_fields
        if len(fields) != 1:
            raise ValueError(
                f"the full_text parser requires exactly one output field; this signature declares "
                f"{sorted(fields) or '(none)'}"
            )
        name, info = next(iter(fields.items()))
        return {name: self._codec.parse_value(completion, info.annotation)}


def _parser_for(preset):
    """The parse implementation for one entry's parser binding, bound to
    the entry's output codec."""
    from dspy.adapters._engine.codecs import resolve_codec
    from dspy.adapters._engine.formats.chat import ChatFormat
    from dspy.adapters._engine.formats.json import JSONFormat
    from dspy.adapters._engine.formats.xml import XMLFormat

    if preset.parser == "full_text":
        return _FullTextParser(resolve_codec(preset.codecs["output"]))
    fmt = {"chat": ChatFormat, "json": JSONFormat, "xml": XMLFormat}[preset.parser]()
    fmt.codec_binding_overrides = dict(preset.codecs)
    return fmt
