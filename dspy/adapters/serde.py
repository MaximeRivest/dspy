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

    #: The template lane runs the plan's strategy-contributed channel
    #: parsers in postprocess (D-δ): a strategy-hidden field must never
    #: silently come back None.
    _runs_plan_channel_hooks = True

    def __init__(self, preset, **kwargs):
        if isinstance(preset, str):
            # A name resolves through the preset pool — builtin or
            # registered (dspy.adapters.register_preset); dangling refuses.
            from dspy.adapters._engine.presets import get_preset

            preset = get_preset(preset)
        # Non-"auto" bindings from the entry ARE constructor bindings: a
        # dumped `reasoning: textual_field` must stand the native channel
        # down under a live call exactly as it did on the source adapter.
        bindings = {role: name for role, name in preset.strategies.items() if name != "auto"}
        if bindings.get("tools") == "native_fc":
            # The binding is the declaration; the legacy kwarg must agree.
            kwargs.setdefault("use_native_function_calling", True)
        # Behavior-bearing constructor flags recorded in the entry's config
        # are honored on load (D-δ): the entry states them, nothing ambient.
        if "use_native_function_calling" in preset.config:
            kwargs.setdefault("use_native_function_calling", preset.config["use_native_function_calling"])
        super().__init__(strategies=bindings or None, **kwargs)
        if "use_json_adapter_fallback" in preset.config:
            self.use_json_adapter_fallback = preset.config["use_json_adapter_fallback"]
        self.preset = preset
        self._parser_impl = _parser_for(preset, error_name=f"{type(self).__name__}({preset.name!r})")

    @property
    def parse_mode(self) -> str:
        """The parser binding, readable back (the constructor spelling).

        The serialized entry carries the same value under the ``parser``
        key — the IR vocabulary's name for the binding (deliberate split,
        recorded in the epic doc).
        """
        return self.preset.parser

    def format(self, signature, demos, inputs) -> list[dict[str, Any]]:
        from dspy.adapters._engine.codecs import resolve_codec
        from dspy.adapters._engine.render import active_plan_fragments
        from dspy.adapters._engine.template import render_template_messages
        from dspy.adapters.base import _expand_legacy_custom_type_markers_in_chat_message

        if isinstance(signature, str):
            # format() accepts string signatures wherever preview() does (D-δ).
            from dspy.signatures.signature import ensure_signature

            signature = ensure_signature(signature)
        self._check_hosting(signature, demos, inputs)
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

    def _check_hosting(self, signature, demos, inputs) -> None:
        """Refuse loudly when call data has nowhere to render (L5).

        A template states everything it shows; demos with no demos hosting
        (directive or ``{demos()}`` slot) or history turns with no history
        hosting would silently vanish from the prompt — an optimizer's
        examples dropping without a sound. The builtin presets host both,
        so entries dumped from the class adapters never refuse here.
        Statically-unservable signatures refuse here too: a second History
        field (one history host exists) and a ``full_text`` binding whose
        surplus output fields no strategy could ever hide.
        """
        from dspy.adapters._engine.template.preview import _history_field_name

        capacity = self.preset.capacity
        self._check_statically_serviceable(signature)
        if demos and not capacity.hosts_demos:
            raise ValueError(
                f"adapter {self.preset.name!r} received {len(demos)} demo(s), but its template hosts "
                'none — add a {"role": "demos"} directive (or a {demos()} slot); dropping examples '
                "silently would vanish from the prompt"
            )
        history_field = _history_field_name(signature)
        if history_field is not None:
            value = inputs.get(history_field)
            turns = list(getattr(value, "messages", value) or [])
            if turns and not capacity.hosts_history:
                raise ValueError(
                    f"adapter {self.preset.name!r} received {len(turns)} history turn(s) in "
                    f"{history_field!r}, but its template hosts none — add a "
                    '{"role": "history"} directive (or a {history()} slot); dropping turns silently '
                    "would lose conversation state"
                )

    def _check_statically_serviceable(self, signature) -> None:
        """Refusals that need no LM: statically decidable at format/preview
        time (D-δ), so a doomed adapter/signature pair never costs a token."""
        from dspy.adapters.types import Type
        from dspy.adapters.types.history import History

        history_fields = [name for name, info in signature.input_fields.items() if info.annotation == History]
        if len(history_fields) > 1:
            raise ValueError(
                f"signature declares {len(history_fields)} History fields ({', '.join(history_fields)}) but a "
                f"template hosts exactly one conversation history — the turns of {', '.join(history_fields[1:])} "
                "would silently vanish; merge the histories into one field"
            )

        if self.preset.parser == "full_text" and len(signature.output_fields) != 1:
            from dspy.signatures.roles import resolve_semantic_role

            def could_hide(name, info):
                if isinstance(info.annotation, type) and info.annotation in self.native_response_types:
                    return issubclass(info.annotation, Type)
                return resolve_semantic_role(info, field_name=name) != "plain"

            if not any(could_hide(name, info) for name, info in signature.output_fields.items()):
                raise ValueError(
                    f"the full_text parser requires exactly one output field; this signature declares "
                    f"{sorted(signature.output_fields)} and none of them can leave the token stream — "
                    "refusing before any LM call"
                )

    def dump_entry(self) -> dict:
        from dspy.adapters._engine.serde import build_entry, dump_preset

        entry = dump_preset(self.preset)
        entry_config = self._entry_config(self.preset.config)
        if entry_config != entry["config"]:
            entry = build_entry(
                name=entry["name"],
                template_raw=self.preset.template.raw,
                parser=entry["parser"],
                codecs=entry["codecs"],
                strategies=entry["strategies"],
                config=entry_config,
            )
        return entry

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


def _parser_for(preset, error_name: str | None = None):
    """The parse implementation for one entry's parser binding, bound to
    the entry's output codec.

    ``error_name`` self-identifies the template-lane adapter in parse
    errors (D-δ) — e.g. ``"TemplateAdapter('my_analyst')"`` — instead of
    the borrowed parser's historical class name. Legacy class adapters
    keep their pinned error identities untouched.
    """
    from dspy.adapters._engine.codecs import resolve_codec
    from dspy.adapters._engine.formats.chat import ChatFormat
    from dspy.adapters._engine.formats.json import JSONFormat
    from dspy.adapters._engine.formats.xml import XMLFormat

    if preset.parser == "full_text":
        return _FullTextParser(resolve_codec(preset.codecs["output"]))
    fmt = {"chat": ChatFormat, "json": JSONFormat, "xml": XMLFormat}[preset.parser]()
    fmt.codec_binding_overrides = dict(preset.codecs)
    if error_name is not None:
        fmt.parse_error_adapter_name = error_name
    return fmt
