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

The codecs here reproduce today's behavior byte-for-byte; they are the
extracted single sources of the value-syntax rules that previously lived
inline in the formats. The golden corpus adjudicates.
"""

import inspect
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


class ValueCodec:
    """One wire syntax for values: a render/parse/schema triple.

    ``render_value`` turns a Python value into the string shown to the
    model; ``parse_value`` turns emitted text back into a Python value of
    the requested annotation; ``render_typed_placeholder`` spells the
    field's *schema* — how the expected shape is described in schema
    positions (spec section 3: ``{f.typed_placeholder}`` routes here).
    Subclasses own all value-syntax literals.
    """

    name: str

    def render_value(self, value: Any, field_info: FieldInfo) -> str:
        raise NotImplementedError

    def parse_value(self, text: Any, annotation: Any) -> Any:
        raise NotImplementedError

    def render_typed_placeholder(self, name: str, field_info: FieldInfo) -> str:
        """The schema spelling: the historical placeholder-plus-type-note."""
        from dspy.adapters.utils import translate_field_type

        return translate_field_type(name, field_info)


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


# ---------------------------------------------------------------------------
# Schema prose: the baml codec's schema spelling
# ---------------------------------------------------------------------------

# The comment symbol is Python's # rather than other languages' //, which
# was observed to help models follow the schema.
COMMENT_SYMBOL = "#"
INDENTATION = "  "


def render_schema_prose(
    annotation: Any,
    depth: int = 0,
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Render a type annotation as simplified, human-readable schema prose.

    The BAML-style spelling: primitives by lowercase name, pydantic models
    as brace blocks with field comments, lists with ``[]`` or bracket
    notation, unions with ``or``. Shape-generic over any annotation.

    Args:
        annotation: The type annotation to render.
        depth: Current recursion depth (prevents infinite recursion).
        indent: Current indentation level for nested structures.
        seen_models: Visited pydantic models (prevents infinite recursion).
    """
    # Non-nested types
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_simplified_schema(annotation, indent, seen_models)

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    # Optional[T] or T | None
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        # Render the non-None part of the union
        type_render = " or ".join([render_schema_prose(arg, depth + 1, indent, seen_models) for arg in non_none_args])
        # Add "or null" if None was part of the union
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    # Literal[T1, T2, ...]
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    # list[T]
    if origin is list:
        # For Pydantic models in lists, use bracket notation
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            # Build inner schema - the Pydantic model inside should use indent level for array contents
            inner_schema = _build_simplified_schema(inner_type, indent + 1, seen_models)
            # Format with proper bracket notation and indentation
            current_indent = INDENTATION * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        else:
            return f"{render_schema_prose(inner_type, depth + 1, indent, seen_models)}[]"

    # dict[T1, T2]
    if origin is dict:
        return f"dict[{render_schema_prose(args[0], depth + 1, indent, seen_models)}, {render_schema_prose(args[1], depth + 1, indent, seen_models)}]"

    # fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _build_simplified_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Build the brace-block schema prose for one pydantic model."""
    seen_models = seen_models or set()

    if pydantic_model in seen_models:
        # Error identity is pinned wire behavior: the message predates the
        # codec factoring and keeps naming the adapter.
        raise ValueError("BAMLAdapter cannot handle recursive pydantic models, please use a different adapter.")

    # Add `pydantic_model` to `seen_models` with a placeholder value to avoid infinite recursion.
    seen_models.add(pydantic_model)

    lines = []
    current_indent = INDENTATION * indent
    next_indent = INDENTATION * (indent + 1)

    # Add model docstring as a comment above the object if it exists
    # Only do this for top-level schemas (indent=0), since nested field docstrings
    # are already added before the field name in the parent schema
    if indent == 0 and pydantic_model.__doc__:
        docstring = pydantic_model.__doc__.strip()
        # Handle multiline docstrings by prefixing each line with the comment symbol
        for line in docstring.split("\n"):
            line = line.strip()
            if line:
                lines.append(f"{current_indent}{COMMENT_SYMBOL} {line}")

    lines.append(f"{current_indent}{{")

    fields = pydantic_model.model_fields
    if not fields:
        lines.append(f"{next_indent}{COMMENT_SYMBOL} No fields defined")
    for name, field in fields.items():
        if field.description:
            lines.append(f"{next_indent}{COMMENT_SYMBOL} {field.description}")
        elif field.alias and field.alias != name:
            # If there's an alias but no description, show the alias as a comment
            lines.append(f"{next_indent}{COMMENT_SYMBOL} alias: {field.alias}")

        # If the field type is a BaseModel, add its docstring as a comment before the field
        field_annotation = field.annotation
        # Handle Optional types
        origin = get_origin(field_annotation)
        if origin in (types.UnionType, Union):
            args = get_args(field_annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_annotation = non_none_args[0]

        if inspect.isclass(field_annotation) and issubclass(field_annotation, BaseModel):
            if field_annotation.__doc__:
                docstring = field_annotation.__doc__.strip()
                for line in docstring.split("\n"):
                    line = line.strip()
                    if line:
                        lines.append(f"{next_indent}{COMMENT_SYMBOL} {line}")

        rendered_type = render_schema_prose(field.annotation, indent=indent + 1, seen_models=seen_models)
        line = f"{next_indent}{name}: {rendered_type},"

        lines.append(line)

    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


class BAMLCodec(PydanticJSONCodec):
    """The BAML pairing's codec: indented-pydantic values, schema-prose schemas.

    Value rendering and parsing inherit the pydantic-JSON codec unchanged;
    the schema spelling replaces the historical placeholder-plus-type-note
    with the simplified schema prose (``Output field `name` should be of
    type: …``). Spec section 5's ``baml`` entry.
    """

    name = "baml"

    def render_typed_placeholder(self, name: str, field_info: FieldInfo) -> str:
        return f"Output field `{name}` should be of type: {render_schema_prose(field_info.annotation, indent=0)}"


TEXT_PYTHONISH = TextPythonishCodec()
PYDANTIC_JSON = PydanticJSONCodec()
BAML = BAMLCodec()

#: Version of the codec vocabulary; carried in every serialized preset's
#: versions block (D-024). Adding or changing a codec entry is a versioned
#: act.
CODECS_VERSION = "1.0.0"

#: Named codec entries preset bindings resolve against (spec section 5's
#: initial vocabulary). ``dspy.adapters.register_codec`` is the public door
#: and gates admission with the round-trip probe battery (spec section 9).
CODECS: dict[str, ValueCodec] = {
    TEXT_PYTHONISH.name: TEXT_PYTHONISH,
    PYDANTIC_JSON.name: PYDANTIC_JSON,
    BAML.name: BAML,
}

#: The names no registration may shadow and no unregistration may remove.
BUILTIN_CODEC_NAMES = frozenset(CODECS)


def resolve_codec(ref) -> ValueCodec:
    """A codec by registry name or origin-tagged entry.

    A string names a registered codec; a dangling ref refuses naming
    itself. A dict is an origin-tagged code entry (spec section 9) and
    materializes through the three-origin loader, admission gate included.
    """
    if isinstance(ref, dict):
        from dspy.adapters._engine.admission import materialize_codec_entry

        return materialize_codec_entry(ref)
    try:
        return CODECS[ref]
    except KeyError:
        raise KeyError(f"unknown codec {ref!r} — registered codecs: {', '.join(CODECS)}") from None
