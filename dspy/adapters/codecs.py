"""Codecs: the type boundary, as named families plus shape-level codecs.

A codec decides how a *value* appears on the wire — how an object is shown
to the model, and what syntax the model's emission is parsed from. The
builtin families port the legacy codec bodies: `text_pythonish` (the shared
JSON-ish render / tolerant parse pair), `pydantic_json` (models as indented
JSON), `schema_prose` (the BAML-style schema spelling), and `json`
(canonical JSON both ways). Directional bindings are independent: an entry
binds an `input` and an `output` codec by name, and the two need not match.

Shape-level codecs are the two-layer rule (adapter-ir-stage example 09):
the IR carries only `(shape, wire)` against the neutral shapes vocabulary,
and the host type (`PIL.Image.Image`) is a per-frontend binding annotation,
never IR content.

`coerce_shape` is the one coercion door parse routings and combinators
share: shape names (`str`, `int`, `ToolCalls`, `Citations`, ...) are
vocabulary words with pinned semantics, never Python classes in the entry.
"""

import inspect
import json
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo

from dspy.adapters.errors import AdapterError

#: Version of the codec vocabulary; carried in every entry's versions block.
CODECS_VERSION = "1.0.0"

#: Version of the neutral shapes vocabulary (wire encodings included);
#: carried only by entries that use a shape codec or a `coerce` target
#: beyond the primitives.
SHAPES_VERSION = "0.1.0"


class ValueCodec:
    """One wire syntax for values: a render/parse/schema triple.

    `render_value` turns a Python value into the string shown to the model;
    `parse_value` turns emitted text back into a value of the requested
    annotation; `render_typed_placeholder` spells the field's *schema* in
    schema positions (`{f.typed_placeholder}` routes here).
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
    """The shared value syntax: JSON-ish rendering, tolerant parsing.

    Render: guillemet blobs for plain string lists, compact
    `json.dumps(ensure_ascii=False)` for structured values, bare `str`
    otherwise. Parse: the `json_repair` -> `ast.literal_eval` -> pydantic
    coercion chain.
    """

    name = "text_pythonish"

    def render_value(self, value: Any, field_info: FieldInfo) -> str:
        from dspy.adapters.utils import format_field_value

        return format_field_value(field_info=field_info, value=value)

    def parse_value(self, text: Any, annotation: Any) -> Any:
        from dspy.adapters.utils import parse_value

        return parse_value(text, annotation)


class PydanticJSONCodec(TextPythonishCodec):
    """Pydantic models as indented JSON on the input side.

    A `BaseModel` value renders as `model_dump_json(indent=2,
    by_alias=True)`; everything else falls back to the shared text codec.
    Parsing is inherited unchanged — the preference is input-side only,
    which is exactly why codec bindings are directional.
    """

    name = "pydantic_json"

    def render_value(self, value: Any, field_info: FieldInfo) -> str:
        if isinstance(value, BaseModel):
            return value.model_dump_json(indent=2, by_alias=True)
        return super().render_value(value, field_info)


class JSONCodec(TextPythonishCodec):
    """Canonical JSON both ways: values render as `json.dumps`, parse
    through the tolerant chain then pydantic coercion."""

    name = "json"

    def render_value(self, value: Any, field_info: FieldInfo) -> str:
        from dspy.adapters.utils import serialize_for_json

        return json.dumps(serialize_for_json(value), ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schema prose: the simplified-schema spelling (the historical BAML codec)
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

    Primitives by lowercase name, pydantic models as brace blocks with
    field comments, lists with `[]` or bracket notation, unions with `or`.
    Shape-generic over any annotation.

    Args:
        annotation: The type annotation to render.
        depth: Current recursion depth (prevents infinite recursion).
        indent: Current indentation level for nested structures.
        seen_models: Visited pydantic models (prevents infinite recursion).
    """
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
        type_render = " or ".join([render_schema_prose(arg, depth + 1, indent, seen_models) for arg in non_none_args])
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    if origin is list:
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            inner_schema = _build_simplified_schema(inner_type, indent + 1, seen_models)
            current_indent = INDENTATION * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        else:
            return f"{render_schema_prose(inner_type, depth + 1, indent, seen_models)}[]"

    if origin is dict:
        return (
            f"dict[{render_schema_prose(args[0], depth + 1, indent, seen_models)}, "
            f"{render_schema_prose(args[1], depth + 1, indent, seen_models)}]"
        )

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
        raise AdapterError("the schema_prose codec cannot render recursive pydantic models — bind a different codec")

    seen_models.add(pydantic_model)

    lines = []
    current_indent = INDENTATION * indent
    next_indent = INDENTATION * (indent + 1)

    if indent == 0 and pydantic_model.__doc__:
        docstring = pydantic_model.__doc__.strip()
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
            lines.append(f"{next_indent}{COMMENT_SYMBOL} alias: {field.alias}")

        field_annotation = field.annotation
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
        lines.append(f"{next_indent}{name}: {rendered_type},")

    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


class SchemaProseCodec(PydanticJSONCodec):
    """Indented-pydantic values, schema-prose schemas.

    Value rendering and parsing inherit the pydantic-JSON codec unchanged;
    the schema spelling replaces the placeholder-plus-type-note with the
    simplified schema prose.
    """

    name = "schema_prose"

    def render_typed_placeholder(self, name: str, field_info: FieldInfo) -> str:
        return f"Output field `{name}` should be of type: {render_schema_prose(field_info.annotation, indent=0)}"


TEXT_PYTHONISH = TextPythonishCodec()
PYDANTIC_JSON = PydanticJSONCodec()
JSON_CODEC = JSONCodec()
SCHEMA_PROSE = SchemaProseCodec()

#: Named codec-family entries codec bindings resolve against.
CODECS: dict[str, ValueCodec] = {
    TEXT_PYTHONISH.name: TEXT_PYTHONISH,
    PYDANTIC_JSON.name: PYDANTIC_JSON,
    JSON_CODEC.name: JSON_CODEC,
    SCHEMA_PROSE.name: SCHEMA_PROSE,
}

#: The names no registration may shadow and no unregistration may remove.
BUILTIN_CODEC_NAMES = frozenset(CODECS)


def resolve_codec(ref: str) -> ValueCodec:
    """A codec family by registry name; a dangling ref refuses naming itself."""
    try:
        return CODECS[ref]
    except KeyError:
        raise AdapterError(
            f"unknown codec {ref!r} — registered codecs: {', '.join(CODECS)} "
            "(a dangling ref is a link error, refused at bind)"
        ) from None


def register_codec(codec: ValueCodec) -> None:
    """Register a codec family under its declared `name`.

    Builtin names may not be shadowed; a duplicate registered name refuses.
    """
    name = getattr(codec, "name", None)
    if not isinstance(name, str) or not name:
        raise AdapterError("a codec declares a non-empty string `name`")
    if name in CODECS:
        raise AdapterError(f"a codec named {name!r} already exists — pick another name")
    CODECS[name] = codec


# ---------------------------------------------------------------------------
# Shape codecs: the two-layer rule
# ---------------------------------------------------------------------------

#: The neutral shapes an entry may name, with their legal wire encodings.
SHAPE_WIRES: dict[str, tuple[str, ...]] = {
    "image": ("base64",),
    "text": ("utf8",),
}


def validate_shape_codec(spec: dict, *, where: str) -> None:
    """Refuse a malformed `{"kind": "shape"}` codec entry, naming the slot."""
    unknown = set(spec) - {"kind", "shape", "wire", "frontend_bindings"}
    if unknown:
        raise AdapterError(f"{where}: unknown shape-codec keys {sorted(unknown)}")
    shape = spec.get("shape")
    if shape not in SHAPE_WIRES:
        raise AdapterError(f"{where}: unknown shape {shape!r} — shapes vocabulary: {', '.join(SHAPE_WIRES)}")
    wire = spec.get("wire")
    if not isinstance(wire, dict) or "encoding" not in wire:
        raise AdapterError(f"{where}: shape codec needs a wire dict with an 'encoding'")
    if wire["encoding"] not in SHAPE_WIRES[shape]:
        raise AdapterError(
            f"{where}: unknown wire encoding {wire['encoding']!r} for shape {shape!r} — "
            f"valid encodings: {', '.join(SHAPE_WIRES[shape])}"
        )
    bindings = spec.get("frontend_bindings", {})
    if not isinstance(bindings, dict):
        raise AdapterError(f"{where}: frontend_bindings must be a dict of frontend -> host type name")


def encode_shape_value(spec: dict, value: Any, *, field_name: str) -> str:
    """Lower one host value through a shape codec into its marker string.

    The marker string carries the content parts; message assembly splits it
    into real parts (`split_message_content_for_custom_types`). The host
    type is the frontend's business: for `image`, a PIL image, raw bytes, a
    data URI, or a `dspy.Image` all lower to base64 parts. PIL itself is an
    optional import — values that need it refuse with a teaching error when
    it is absent.
    """
    if spec["shape"] == "image":
        from dspy.adapters.types.image import Image

        try:
            image = value if isinstance(value, Image) else Image(value)
        except ImportError as error:
            raise AdapterError(
                f"field {field_name!r} carries an image shape but Pillow is not installed — "
                "install pillow to lower host image objects to the wire"
            ) from error
        return image.serialize_model()
    # shape == "text": the wire is the string itself.
    return str(value)


def image_shape_codec_entry(frontend_type: str = "PIL.Image.Image") -> dict:
    """The canonical image shape-codec entry (example 09's spelling)."""
    return {
        "kind": "shape",
        "shape": "image",
        "wire": {"encoding": "base64", "media_type": "image/png"},
        "frontend_bindings": {"python": frontend_type},
    }


# ---------------------------------------------------------------------------
# Shape coercion: the one door routings and combinators share
# ---------------------------------------------------------------------------


def coerce_shape(value: Any, shape: str) -> Any:
    """Coerce a parsed value into a named shape from the closed vocabulary.

    Args:
        value: The raw parsed value (string, list, or object).
        shape: A shapes-vocabulary name: `str`, `int`, `float`, `bool`,
            `json`, `ToolCalls`, or `Citations`.

    Raises:
        AdapterError: On an unknown shape name.
    """
    if shape == "str":
        return value if isinstance(value, str) else str(value)
    if shape == "int":
        return int(value)
    if shape == "float":
        return float(value)
    if shape == "bool":
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in ("true", "yes", "1"):
                return True
            if lowered in ("false", "no", "0"):
                return False
            raise ValueError(f"cannot coerce {value!r} to bool")
        return bool(value)
    if shape == "json":
        return json.loads(value) if isinstance(value, str) else value
    if shape == "ToolCalls":
        from dspy.adapters.types.tool import ToolCalls

        return TypeAdapter(ToolCalls).validate_python(value)
    if shape == "Citations":
        from dspy.adapters.types.citation import Citations

        return TypeAdapter(Citations).validate_python(value)
    raise AdapterError(
        f"unknown coercion shape {shape!r} — shapes vocabulary: str, int, float, bool, json, ToolCalls, Citations"
    )


#: Shape names `coerce_shape` accepts (validation reads this set).
COERCE_SHAPES = ("str", "int", "float", "bool", "json", "ToolCalls", "Citations")
