"""Format objects: the literal-string layer of the engine.

A Format owns EVERY literal string and section-rendering rule for one wire
format (field markers, completed markers, schema sentences, indentation,
demo prefixes). The renderer (``render.py``) is pure message assembly and
must contain zero format-specific literals — that split is what lets a new
wire format land as a Format object without touching the renderer (gated:
the XML/BAML migration PR must show a zero diff on render.py).

``resolve_format`` walks the adapter's MRO against a private registry —
this mapping is deliberately the single attachment point where a future
preset-string vocabulary would plug in, per the epic's deferred-doors list.

The granular method surface (per-section renderers rather than one
render_system/render_user pair) exists because demos and conversation
history render PER-MESSAGE content; coarser surfaces cannot reproduce the
legacy pipeline byte-for-byte.
"""

from typing import Any, Mapping

from dspy.adapters._engine.template.renderer import (
    COMPLETE_DEMO_MISSING_FIELD_MESSAGE,
    INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE,
    INCOMPLETE_DEMO_PREFIX,
)


class Format:
    """Base class for wire formats. Subclasses own all literals.

    The pipeline-level demo strings live here (shared by every format
    today, single-sourced with the template walker) so the renderer holds
    no literals at all; a format may override them.

    Value syntax is delegated to codec bindings (``codecs.py``): a format
    owns the structure of the exchange, its codecs own the wire syntax of
    each value. Since D-3 codec AUTHORITY lives in the preset's binding
    data: ``codec_bindings()`` reads the preset's named codecs, layers the
    class's declared overrides, and the registry resolves the names — the
    Format object holds no codec objects of its own.
    """

    #: Preset whose template this format's rendered content delegates to;
    #: None for formats not defined as presets (TwoStep).
    preset_name: str | None = None

    #: Name this format's serialized component-4 entry carries; defaults to
    #: the preset name. The BAML pairing dumps as "baml" — an entry name,
    #: not a preset name (spec section 4: no baml preset exists).
    entry_name: str | None = None

    #: Named codec-binding overrides layered over the preset's bindings
    #: (the BAML pairing overrides both directions to the ``baml`` codec);
    #: None keeps the preset bindings as-is.
    codec_binding_overrides: Mapping[str, str] | None = None

    #: Parsed system-message template replacing the preset's system message
    #: (a ``template.ContentMessage``); None renders the preset's own. The
    #: BAML pairing carries its schema-prose arrangement here (spec
    #: section 4).
    system_template_message = None

    incomplete_demo_prefix = INCOMPLETE_DEMO_PREFIX
    incomplete_demo_missing_field_message = INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE
    complete_demo_missing_field_message = COMPLETE_DEMO_MISSING_FIELD_MESSAGE

    def codec_bindings(self) -> dict[str, str]:
        """The named codec bindings this format renders and parses through.

        The preset names the codecs, the class (or an instance, for loaded
        entries) may override the names, and the registry resolves them.
        Formats with no preset bind the shared text codec both ways.
        """
        if self.preset_name:
            from dspy.adapters._engine.presets import get_preset

            bindings = dict(get_preset(self.preset_name).codecs)
        else:
            bindings = {"input": "text_pythonish", "output": "text_pythonish"}
        if self.codec_binding_overrides:
            bindings.update(self.codec_binding_overrides)
        return bindings

    @property
    def input_codec(self):
        from dspy.adapters._engine.codecs import resolve_codec

        return resolve_codec(self.codec_bindings()["input"])

    @property
    def output_codec(self):
        from dspy.adapters._engine.codecs import resolve_codec

        return resolve_codec(self.codec_bindings()["output"])

    def render_field_description(self, signature) -> str:
        raise NotImplementedError

    def render_field_structure(self, signature) -> str:
        raise NotImplementedError

    def render_task_description(self, signature) -> str:
        raise NotImplementedError

    def render_system(self, signature) -> str:
        return (
            f"{self.render_field_description(signature)}\n"
            f"{self.render_field_structure(signature)}\n"
            f"{self.render_task_description(signature)}"
        )

    def render_user_content(
        self,
        signature,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        raise NotImplementedError

    def render_assistant_content(self, signature, outputs: dict[str, Any], missing_field_message=None) -> str:
        raise NotImplementedError

    def output_requirements(self, signature) -> str | None:
        raise NotImplementedError

    def make_parser_hook(self, adapter):
        """The plan-carried parser for this format. Formats whose parsing
        needs per-adapter state (TwoStep's extraction model) override this."""
        from dspy.adapters._engine.parse import FormatParserHook

        return FormatParserHook(self)


_FORMATS: dict[type, Format] = {}


def register_format(adapter_cls: type, fmt: Format) -> None:
    """Pair an engine-backed adapter class with its Format (migration PRs only)."""
    _FORMATS[adapter_cls] = fmt


def resolve_format(adapter) -> Format | None:
    """Find the Format for an adapter instance via its MRO, so engine-eligible
    no-op subclasses resolve their parent's format."""
    for klass in type(adapter).__mro__:
        fmt = _FORMATS.get(klass)
        if fmt is not None:
            return fmt
    return None


def _reset_formats_for_tests() -> None:
    _FORMATS.clear()
