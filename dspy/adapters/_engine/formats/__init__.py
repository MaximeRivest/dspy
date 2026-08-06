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

from typing import Any

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
    each value. The bindings are directional and independent — BAML proves
    it by overriding only ``input_codec``.
    """

    #: Preset whose template this format's rendered content delegates to;
    #: None for formats not yet defined as presets (TwoStep, and BAML's
    #: system section until D-3 reclassifies BAML as a codec binding).
    preset_name: str | None = None

    incomplete_demo_prefix = INCOMPLETE_DEMO_PREFIX
    incomplete_demo_missing_field_message = INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE
    complete_demo_missing_field_message = COMPLETE_DEMO_MISSING_FIELD_MESSAGE

    @property
    def input_codec(self):
        from dspy.adapters._engine.codecs import TEXT_PYTHONISH

        return TEXT_PYTHONISH

    @property
    def output_codec(self):
        from dspy.adapters._engine.codecs import TEXT_PYTHONISH

        return TEXT_PYTHONISH

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
