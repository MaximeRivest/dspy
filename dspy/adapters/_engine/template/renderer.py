"""Pure rendering of parsed template nodes against a signature and values.

Rendering is a pure function of (nodes, context): no ambient reads, no
clock, no network, no LM (ADP-001/ADP-002). The context carries the three
things the closed language may consult — the signature, the call values,
and the codec bindings — plus the strategy-fragment lists that positional
``{fragments(...)}`` slots interpolate.

Value-presence semantics, byte-pinned by the golden corpus:

- ``schema`` mode (system messages): loops iterate every field, and the
  message renders without call values — ``f.value`` refuses in this mode
  regardless of what the context carries, so every surface (the engine
  delegation path and the preview walker alike) shows the same bytes.
- ``user_values`` / ``assistant_values`` modes (user and assistant turns):
  ``inputs`` loops iterate only fields present in the values dict — absent
  fields contribute no block (the historical ``if k in inputs`` skip) —
  while ``outputs`` loops iterate every field: in assistant position absent
  values render the context's missing-field message through the codec (the
  historical ``outputs.get(k, missing_field_message)``), and in user
  position they render schema-side (the output-requirements enumeration
  names every field regardless of the call's input values).
"""

import json
import textwrap
from dataclasses import dataclass, field, replace
from typing import Any

from dspy.adapters._engine.template.parser import (
    AggregateSlot,
    FragmentsSlot,
    InstructionSlot,
    Loop,
    LoopVarLiteral,
    LoopVarSlot,
    Section,
    Text,
    ValueSlot,
)
from dspy.adapters._engine.template.vocabulary import spell_out
from dspy.adapters.utils import (
    get_annotation_name,
    get_field_description_string,
    get_field_description_suffix,
    serialize_for_json,
    translate_field_type,
)

#: The pipeline-level demo strings, single-sourced here for the template
#: walker and referenced by the Format base class. Trailing spaces are
#: byte-pinned wire behavior.
INCOMPLETE_DEMO_PREFIX = "This is an example of the task, though some input or output fields are not supplied."
INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE = "Not supplied for this particular example. "
COMPLETE_DEMO_MISSING_FIELD_MESSAGE = "Not supplied for this conversation history message. "

_MODES = ("schema", "user_values", "assistant_values")


class TemplateRenderError(ValueError):
    """A render-time refusal: a construct the signature or context cannot serve."""


@dataclass
class RenderContext:
    """Everything one template render may consult."""

    signature: Any
    mode: str
    input_codec: Any
    output_codec: Any
    values: dict[str, Any] | None = None
    missing_field_message: Any = None
    fragments: dict[str, list[str]] = field(default_factory=dict)
    demos: tuple = ()
    history: tuple = ()
    _loop_state: tuple | None = None  # (position, name, field_info, collection)

    def __post_init__(self):
        if self.mode not in _MODES:
            raise TemplateRenderError(f"unknown render mode {self.mode!r} — valid modes: {spell_out(_MODES)}")


def render_nodes(nodes, ctx: RenderContext) -> str:
    parts: list[str] = []
    swallow_newline = False
    for node in nodes:
        if isinstance(node, FragmentsSlot):
            content = "\n".join(ctx.fragments.get(node.target, ()))
            if content == "" and node.owns_line:
                swallow_newline = True
                continue
            rendered = content
        else:
            rendered = _render_node(node, ctx)
        if swallow_newline:
            rendered = rendered.removeprefix("\n")
            swallow_newline = False
        parts.append(rendered)
    return "".join(parts)


def render_user_content(nodes, ctx: RenderContext, prefix: str = "", suffix: str = "") -> str:
    """User-turn assembly: the historical join-then-strip shape.

    Reproduces legacy ``"\\n\\n".join([prefix, *blocks, suffix]).strip()``
    where only existing blocks join: parts that render empty contribute no
    join element (an empty body between a nonempty prefix and suffix costs
    zero separators, the ChatAdapter/JSONAdapter shape), and the joined
    result strips. This assembly — and the walker's rule that a user
    message assembling to nothing is omitted entirely — is declared
    semantics (spec section 3, user-turn assembly), not renderer accident.
    """
    parts = [part for part in (prefix, render_nodes(nodes, ctx), suffix) if part]
    return "\n\n".join(parts).strip()


def _render_node(node, ctx: RenderContext) -> str:
    if isinstance(node, Text):
        return node.text
    if isinstance(node, InstructionSlot):
        return _render_instruction(node, ctx)
    if isinstance(node, ValueSlot):
        return _render_value_slot(node.name, ctx)
    if isinstance(node, Loop):
        return _render_loop(node, ctx)
    if isinstance(node, Section):
        return render_nodes(node.body, ctx).strip()
    if isinstance(node, LoopVarSlot):
        return _loop_attr(node.attr, ctx)
    if isinstance(node, LoopVarLiteral):
        return "{" + _loop_attr(node.attr, ctx) + "}"
    if isinstance(node, AggregateSlot):
        return _render_aggregate(node, ctx)
    raise TemplateRenderError(f"unrenderable node {node!r}")  # pragma: no cover - parser prevents


def _render_instruction(node: InstructionSlot, ctx: RenderContext) -> str:
    if not node.explicit and (
        "instruction" in ctx.signature.input_fields or "instruction" in ctx.signature.output_fields
    ):
        from dspy.adapters._engine.template.vocabulary import RESERVED_SLOT_NAMES

        raise TemplateRenderError(
            "{instruction} is reserved for the signature instructions, but this signature also declares "
            "a field named 'instruction' — write {instruction(style='raw')} for the instructions and "
            "{field('instruction')} for the field's value; a field carrying a reserved name has no bare "
            f"value-slot spelling (reserved names: {spell_out(RESERVED_SLOT_NAMES)})"
        )
    instructions = ctx.signature.instructions or ""
    if node.style == "raw":
        return instructions
    # style == "indented": the historical ChatAdapter objective block.
    dedented = textwrap.dedent(instructions)
    return ("\n" + " " * 8).join([""] + dedented.splitlines())


def _field_map(ctx: RenderContext, collection: str) -> dict:
    return ctx.signature.input_fields if collection == "inputs" else ctx.signature.output_fields


def _codec_for(ctx: RenderContext, collection: str):
    return ctx.input_codec if collection == "inputs" else ctx.output_codec


def _render_value_slot(name: str, ctx: RenderContext) -> str:
    for collection in ("inputs", "outputs"):
        fields = _field_map(ctx, collection)
        if name in fields:
            if ctx.values is None or name not in ctx.values:
                return ""
            return _codec_for(ctx, collection).render_value(ctx.values[name], fields[name])
    available = [*ctx.signature.input_fields, *ctx.signature.output_fields]
    raise TemplateRenderError(
        f"unknown value slot {{{name}}} — fields available here: {spell_out(available) or '(none)'}"
    )


def _render_loop(node: Loop, ctx: RenderContext) -> str:
    fields = _field_map(ctx, node.collection)
    items = list(fields.items())
    if node.collection == "inputs" and ctx.mode != "schema":
        values = ctx.values or {}
        items = [(name, info) for name, info in items if name in values]

    chunks = []
    for position, (name, info) in enumerate(items, start=1):
        item_ctx = replace(ctx, _loop_state=(position, name, info, node.collection))
        chunks.append(render_nodes(node.body, item_ctx))
    rendered = node.separator.join(chunks)
    return rendered.strip() if node.strip else rendered


def _loop_attr(attr: str, ctx: RenderContext) -> str:
    if ctx._loop_state is None:  # pragma: no cover - parser prevents
        raise TemplateRenderError("loop variable rendered outside a loop")
    position, name, info, collection = ctx._loop_state

    if attr in ("i", "index"):
        return str(position)
    if attr == "name":
        return name
    if attr == "type":
        return get_annotation_name(info.annotation)
    if attr == "desc":
        desc = (getattr(info, "json_schema_extra", None) or {}).get("desc", "")
        return "" if desc == f"${{{name}}}" else str(desc)
    if attr == "desc_suffix":
        return get_field_description_suffix(name, info)
    if attr == "value":
        return _loop_value(ctx, name, info, collection)
    if attr == "placeholder":
        return "{" + name + "}"
    if attr == "typed_placeholder":
        # The schema spelling is the codec's (spec section 3): the shared
        # text codec renders the historical placeholder-plus-type-note.
        return _codec_for(ctx, collection).render_typed_placeholder(name, info)
    if attr == "marker":
        return f"[[ ## {name} ## ]]"
    if attr == "chat_type_hint":
        return _chat_type_hint(info.annotation)
    raise TemplateRenderError(f"unknown loop attribute {attr!r}")  # pragma: no cover - parser prevents


def _loop_value(ctx: RenderContext, name: str, info, collection: str) -> str:
    if ctx.mode == "schema":
        raise TemplateRenderError(
            "{f.value} needs call values, and schema positions render without them — "
            "use {f.placeholder} or {f.typed_placeholder} in schema positions"
        )
    codec = _codec_for(ctx, collection)
    values = ctx.values or {}
    if ctx.mode == "assistant_values" and collection == "outputs":
        return codec.render_value(values.get(name, ctx.missing_field_message), info)
    if name in values:
        return codec.render_value(values[name], info)
    return ""


def _chat_type_hint(annotation) -> str:
    from dspy.adapters.types.tool import ToolCalls

    if annotation == ToolCalls:
        return ' (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})'
    if annotation is not str:
        return f" (must be formatted as a valid Python {get_annotation_name(annotation)})"
    return ""


# ---------------------------------------------------------------------------
# Aggregate slots
# ---------------------------------------------------------------------------


def _render_aggregate(node: AggregateSlot, ctx: RenderContext) -> str:
    if node.kind == "inputs":
        return _render_inputs_aggregate(node.style, ctx)
    if node.kind == "outputs":
        return _render_outputs_aggregate(node.style, node.wrap, ctx)
    if node.kind == "demos":
        return _render_demos_aggregate(node.style, ctx)
    return _render_history_aggregate(node.style, ctx)


def _unknown_style(kind: str, style: str) -> TemplateRenderError:
    """A style the renderer does not cover refuses instead of silently
    rendering default-style bytes — the vocabulary and the renderer cannot
    drift apart without a teaching error surfacing it."""
    from dspy.adapters._engine.template.vocabulary import VOCABULARY

    return TemplateRenderError(
        f"unknown {kind} style {style!r} — valid styles: {spell_out(VOCABULARY['aggregate_styles'][kind])}"
    )


def _aggregate_input_items(ctx: RenderContext):
    """Input fields for aggregate rendering; History fields render through
    the history directive or ``{history(...)}``, never ``{inputs(...)}``."""
    from dspy.adapters.types.history import History

    values = ctx.values or {}
    for name, info in ctx.signature.input_fields.items():
        if info.annotation == History:
            continue
        rendered = ctx.input_codec.render_value(values[name], info) if name in values else ""
        yield name, rendered


def _render_inputs_aggregate(style: str, ctx: RenderContext) -> str:
    items = list(_aggregate_input_items(ctx))
    if style == "json":
        lines = [f'"{name}": {json.dumps(value, ensure_ascii=False)}' for name, value in items]
        return "{\n  " + ",\n  ".join(lines) + "\n}"
    if style == "xml":
        return "\n".join(f"<{name}>{value}</{name}>" for name, value in items)
    if style == "chat":
        return "\n\n".join(f"[[ ## {name} ## ]]\n{value}" for name, value in items)
    if style in ("default", "yaml"):
        return "\n".join(f"{name}: {value}" for name, value in items)
    raise _unknown_style("inputs", style)


def _render_outputs_aggregate(style: str, wrap: str | None, ctx: RenderContext) -> str:
    import pydantic

    output_fields = ctx.signature.output_fields
    if style == "schema":
        schema: dict[str, Any] = {}
        for name, info in output_fields.items():
            try:
                schema[name] = pydantic.TypeAdapter(info.annotation).json_schema()
            except Exception:
                schema[name] = {"type": get_annotation_name(info.annotation)}
        return json.dumps(schema, indent=2, ensure_ascii=False)
    if style == "xml":
        indent = "  " if wrap else ""
        lines = []
        for name, info in output_fields.items():
            desc = (getattr(info, "json_schema_extra", None) or {}).get("desc")
            if not desc or desc == f"${{{name}}}":
                desc = name
            lines.append(f"{indent}<{name}>{desc}</{name}>")
        body = "\n".join(lines)
        return f"<{wrap}>\n{body}\n</{wrap}>" if wrap else body
    if style == "chat":
        return "\n\n".join(f"[[ ## {name} ## ]]\n{translate_field_type(name, info)}" for name, info in output_fields.items())
    if style == "json_object":
        if ctx.mode == "assistant_values":
            values = ctx.values or {}
            obj = {name: values.get(name, ctx.missing_field_message) for name in output_fields}
        else:
            obj = {name: translate_field_type(name, info) for name, info in output_fields.items()}
        return json.dumps(serialize_for_json(obj), indent=2, ensure_ascii=False)
    if style == "default":
        return get_field_description_string(output_fields)
    raise _unknown_style("outputs", style)


def _render_demos_aggregate(style: str, ctx: RenderContext) -> str:
    if style not in ("default", "json", "yaml", "xml", "chat"):
        raise _unknown_style("demos", style)
    if not ctx.demos:
        return ""
    signature = ctx.signature
    ordered = [*signature.input_fields, *signature.output_fields]

    def rendered_value(demo, name):
        info = signature.input_fields.get(name) or signature.output_fields[name]
        codec = ctx.input_codec if name in signature.input_fields else ctx.output_codec
        return codec.render_value(demo[name], info)

    blocks = []
    for index, demo in enumerate(ctx.demos, start=1):
        present = [name for name in ordered if name in demo]
        if style == "json":
            blocks.append(
                json.dumps(serialize_for_json({name: demo[name] for name in present}), indent=2, ensure_ascii=False)
            )
        elif style == "xml":
            blocks.append("\n".join(f"<{name}>{rendered_value(demo, name)}</{name}>" for name in present))
        elif style == "chat":
            blocks.append("\n\n".join(f"[[ ## {name} ## ]]\n{rendered_value(demo, name)}" for name in present))
        elif style == "yaml":
            blocks.append("\n".join(f"{name}: {rendered_value(demo, name)}" for name in present))
        else:
            lines = [f"Example {index}:"]
            lines.extend(f"  {name}: {rendered_value(demo, name)}" for name in present)
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_history_aggregate(style: str, ctx: RenderContext) -> str:
    if style not in ("default", "json", "yaml", "xml", "chat"):
        raise _unknown_style("history", style)
    turns = list(ctx.history or ())
    if not turns:
        return ""
    if style == "json":
        return json.dumps(serialize_for_json(turns), indent=2, ensure_ascii=False)
    if style == "yaml":
        blocks = []
        for index, turn in enumerate(turns, start=1):
            lines = [f"- turn: {index}"]
            lines.extend(f"  {key}: {value}" for key, value in turn.items())
            blocks.append("\n".join(lines))
        return "\n".join(blocks)
    if style == "xml":
        blocks = []
        for turn in turns:
            fields = "\n".join(f"  <{key}>{value}</{key}>" for key, value in turn.items())
            blocks.append(f"<turn>\n{fields}\n</turn>")
        return "\n".join(blocks)
    if style == "chat":
        blocks = []
        for turn in turns:
            blocks.append("\n\n".join(f"[[ ## {key} ## ]]\n{value}" for key, value in turn.items()))
        return "\n\n".join(blocks)
    # default
    blocks = []
    for index, turn in enumerate(turns, start=1):
        lines = [f"Turn {index}:"]
        lines.extend(f"  {key}: {value}" for key, value in turn.items())
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Demo classification (shared with the engine renderer)
# ---------------------------------------------------------------------------


def classify_demos(signature, demos) -> tuple[list, list]:
    """Split demos into (incomplete, complete), the historical rules.

    Complete demos carry every signature field with a non-None value;
    incomplete demos carry at least one input and one output; anything
    else is dropped.
    """
    incomplete: list = []
    complete: list = []
    for demo in demos:
        is_complete = all(k in demo and demo[k] is not None for k in signature.fields)
        has_input = any(k in demo for k in signature.input_fields)
        has_output = any(k in demo for k in signature.output_fields)
        if is_complete:
            complete.append(demo)
        elif has_input and has_output:
            incomplete.append(demo)
    return incomplete, complete
