"""Value codecs: shape-generic render/parse pairs for wire syntax.

A codec decides how a *value* appears on the wire — how an object is shown
to the model, and what syntax the model's emission is parsed from. Codecs
are the layer below formats in the mechanism taxonomy (see
``roadmap/IR-program-spec.md`` §Adapter-semantics): a format owns the
structure of one exchange (markers, sections, schema sentences); a codec
owns the syntax of one value inside it. Codecs are shape-generic by law —
a codec works for any schema; anything that only works for one field is
literal-table text and belongs to the format.

Codec bindings are directional and independent: a format binds an
``input_codec`` (rendering values into the prompt) and an ``output_codec``
(parsing values back out), and the two need not match.

The two codecs here reproduce today's behavior byte-for-byte; they are the
extracted single sources of the value-syntax rules that previously lived
inline in the formats. The golden corpus adjudicates.
"""

from typing import Any

from pydantic import BaseModel
from pydantic.fields import FieldInfo


class ValueCodec:
    """One wire syntax for values: a render/parse pair.

    ``render_value`` turns a Python value into the string shown to the
    model; ``parse_value`` turns emitted text back into a Python value of
    the requested annotation. Subclasses own all value-syntax literals.
    """

    name: str

    def render_value(self, value: Any, field_info: FieldInfo) -> str:
        raise NotImplementedError

    def parse_value(self, text: Any, annotation: Any) -> Any:
        raise NotImplementedError


class TextPythonishCodec(ValueCodec):
    """Today's shared value syntax: JSON-ish rendering, tolerant parsing.

    Render: guillemet blobs for plain string lists, compact
    ``json.dumps(ensure_ascii=False)`` for structured values, bare ``str``
    otherwise. Parse: the ``json_repair`` → ``ast.literal_eval`` → pydantic
    coercion chain. Exactly the legacy ``format_field_value`` /
    ``parse_value`` pair, now named as the codec every format inherited
    implicitly.
    """

    name = "text_pythonish"

    def render_value(self, value: Any, field_info: FieldInfo) -> str:
        from dspy.adapters.utils import format_field_value

        return format_field_value(field_info=field_info, value=value)

    def parse_value(self, text: Any, annotation: Any) -> Any:
        from dspy.adapters.utils import parse_value

        return parse_value(text, annotation)


class PydanticJSONCodec(TextPythonishCodec):
    """BAML's input preference: pydantic models as indented JSON.

    A ``BaseModel`` value renders as ``model_dump_json(indent=2,
    by_alias=True)``; everything else falls back to the shared text codec.
    Parsing is inherited unchanged — the preference is input-side only,
    which is exactly why codec bindings are directional.
    """

    name = "pydantic_json"

    def render_value(self, value: Any, field_info: FieldInfo) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json(indent=2, by_alias=True)
        return super().render_value(value, field_info)


TEXT_PYTHONISH = TextPythonishCodec()
PYDANTIC_JSON = PydanticJSONCodec()

#: Named codec entries preset bindings resolve against (spec section 5's
#: initial vocabulary, populated as codecs land). Engine-private until the
#: registration API ships with its round-trip admission gate (spec
#: section 9).
CODECS: dict[str, ValueCodec] = {
    TEXT_PYTHONISH.name: TEXT_PYTHONISH,
    PYDANTIC_JSON.name: PYDANTIC_JSON,
}


def resolve_codec(name: str) -> ValueCodec:
    """A codec by registry name; a dangling ref refuses naming itself."""
    try:
        return CODECS[name]
    except KeyError:
        raise KeyError(f"unknown codec {name!r} — registered codecs: {', '.join(CODECS)}") from None
