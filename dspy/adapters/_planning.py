"""Private adapter-call planning for normalized LM requests."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Protocol, get_args, get_origin

from dspy.adapters.types import Audio, File, History, Image
from dspy.adapters.types.document import Document
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMImagePart,
    LMMessage,
    LMPart,
    LMRequestPatch,
    LMTextPart,
    LMToolSpec,
)
from dspy.experimental import Citations
from dspy.signatures.signature import Signature


@dataclass
class _AdapterPlan:
    """Private adapter-call plan for prompt rendering and native LM features."""

    adapter: Any
    original_signature: type[Signature]
    render_signature: type[Signature]
    inputs: dict[str, Any]
    lm_kwargs: dict[str, Any]
    messages: list[LMMessage] = dataclass_field(default_factory=list)
    user_parts: list[LMPart] = dataclass_field(default_factory=list)
    user_part_segments: list[tuple[str, list[LMPart]]] = dataclass_field(default_factory=list)
    tools: list[LMToolSpec] = dataclass_field(default_factory=list)


@dataclass
class _AdapterRenderContext:
    """Mutable private context used while renderers build an adapter plan."""

    adapter: Any
    lm: BaseLM
    original_signature: type[Signature]
    render_signature: type[Signature]
    inputs: dict[str, Any]
    lm_kwargs: dict[str, Any]
    patch: LMRequestPatch = dataclass_field(default_factory=LMRequestPatch)
    user_part_segments: list[tuple[str, list[LMPart]]] = dataclass_field(default_factory=list)

    def apply_patch(self, patch: LMRequestPatch) -> None:
        self.patch = self.patch.merge(patch)
        self.render_signature = _apply_patch_field_deletions(self.render_signature, patch)
        if patch.config is not None:
            patch_kwargs = patch.as_lm_kwargs()
            patch_kwargs.pop("tools", None)
            self.lm_kwargs.update(patch_kwargs)


class _FieldRenderer(Protocol):
    """Private adapter-owned field renderer/planner interface."""

    def render_input(
        self,
        field_name: str,
        field: Any,
        value: Any,
        context: _AdapterRenderContext,
    ) -> LMRequestPatch | None:
        return None

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        return None


class _AdapterRendererPipeline:
    def __init__(self, renderers: list[_FieldRenderer] | None = None):
        self.renderers = renderers or [
            _HistoryRenderer(),
            _NativeInputTypeRenderer(),
            _NativeToolRenderer(),
            _NativeReasoningRenderer(),
            _NativeCitationsRenderer(),
            _LegacyNativeTypeRenderer(),
        ]

    def plan(
        self,
        adapter: Any,
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> _AdapterPlan:
        context = _AdapterRenderContext(
            adapter=adapter,
            lm=lm,
            original_signature=signature,
            render_signature=signature,
            inputs=dict(inputs),
            lm_kwargs=dict(lm_kwargs),
        )

        for name, field in list(context.render_signature.input_fields.items()):
            if name not in context.inputs:
                continue
            value = context.inputs[name]
            for renderer in self.renderers:
                patch = renderer.render_input(name, field, value, context)
                if patch is not None:
                    context.apply_patch(patch)
                    context.inputs.pop(name, None)
                    break

        for renderer in self.renderers:
            renderer.render_outputs(context)

        return _AdapterPlan(
            adapter=context.adapter,
            original_signature=context.original_signature,
            render_signature=context.render_signature,
            inputs=context.inputs,
            lm_kwargs=context.lm_kwargs,
            messages=context.patch.messages,
            user_parts=context.patch.user_parts,
            user_part_segments=context.user_part_segments,
            tools=context.patch.tools,
        )


class _HistoryRenderer:
    def render_input(
        self,
        field_name: str,
        field: Any,
        value: Any,
        context: _AdapterRenderContext,
    ) -> LMRequestPatch | None:
        if field.annotation != History:
            return None
        history = value if isinstance(value, History) else History.model_validate(value)
        signature_without_history = context.render_signature.delete(field_name)
        return LMRequestPatch(messages=_history_to_lm_messages(context.adapter, signature_without_history, history))

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        return None


class _NativeInputTypeRenderer:
    def render_input(
        self,
        field_name: str,
        field: Any,
        value: Any,
        context: _AdapterRenderContext,
    ) -> LMRequestPatch | None:
        parts = _native_input_parts(field_name, field.annotation, value, context.adapter)
        if parts is None:
            return None
        context.user_part_segments.append((field_name, parts))
        return LMRequestPatch()

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        return None


class _NativeToolRenderer:
    def render_input(self, field_name: str, field: Any, value: Any, context: _AdapterRenderContext) -> LMRequestPatch | None:
        return None

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        adapter = context.adapter
        if not adapter.use_native_function_calling:
            return

        tool_call_input_field_name = _get_tool_call_input_field_name(context.render_signature)
        tool_call_output_field_name = _get_tool_call_output_field_name(context.render_signature)

        if tool_call_output_field_name and tool_call_input_field_name is None:
            raise ValueError(
                f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                "input field with type `list[dspy.Tool]`."
            )

        if not (tool_call_output_field_name and context.lm.supports_function_calling):
            return

        tool_values = context.inputs[tool_call_input_field_name]
        tool_values = tool_values if isinstance(tool_values, list) else [tool_values]
        context.apply_patch(
            LMRequestPatch(
                tools=[_tool_to_lm_tool_spec(tool) for tool in tool_values],
                delete_input_fields=(tool_call_input_field_name,),
                delete_output_fields=(tool_call_output_field_name,),
            )
        )
        context.inputs.pop(tool_call_input_field_name, None)


class _NativeReasoningRenderer:
    def render_input(self, field_name: str, field: Any, value: Any, context: _AdapterRenderContext) -> LMRequestPatch | None:
        return None

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        for name, field in list(context.render_signature.output_fields.items()):
            if field.annotation != Reasoning or Reasoning not in context.adapter.native_response_types:
                continue
            reasoning_effort = context.lm_kwargs.get("reasoning_effort", context.lm.kwargs.get("reasoning_effort", "low"))
            if reasoning_effort is None or not context.lm.supports_reasoning:
                continue
            if "gpt-5" in context.lm.model and getattr(context.lm, "model_type", None) == "chat":
                continue
            context.lm_kwargs["reasoning_effort"] = reasoning_effort
            context.apply_patch(LMRequestPatch(delete_output_fields=(name,)))


class _LegacyNativeTypeRenderer:
    """Compatibility renderer for custom `Type.adapt_to_native_lm_feature()` hooks."""

    def render_input(self, field_name: str, field: Any, value: Any, context: _AdapterRenderContext) -> LMRequestPatch | None:
        return None

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        from dspy.adapters.types import Type

        for name, field in list(context.render_signature.output_fields.items()):
            annotation = field.annotation
            if not (isinstance(annotation, type) and annotation in context.adapter.native_response_types and issubclass(annotation, Type)):
                continue
            if annotation in (Reasoning, Citations):
                continue
            adapted_signature = annotation.adapt_to_native_lm_feature(
                context.render_signature,
                name,
                context.lm,
                context.lm_kwargs,
            )
            if adapted_signature is not context.render_signature:
                deleted = tuple(
                    field_name for field_name in context.render_signature.output_fields if field_name not in adapted_signature.output_fields
                )
                context.apply_patch(LMRequestPatch(delete_output_fields=deleted))


class _NativeCitationsRenderer:
    def render_input(self, field_name: str, field: Any, value: Any, context: _AdapterRenderContext) -> LMRequestPatch | None:
        return None

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        for name, field in list(context.render_signature.output_fields.items()):
            if field.annotation in context.adapter.native_response_types and field.annotation is Citations:
                if getattr(context.lm, "model", "").startswith("anthropic/"):
                    context.apply_patch(LMRequestPatch(delete_output_fields=(name,)))


def _plan_fields(
    adapter: Any,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    inputs: dict[str, Any],
) -> _AdapterPlan:
    """Plan which fields are rendered by the prompt adapter and which use native LM features."""
    return _AdapterRendererPipeline().plan(adapter, lm, lm_kwargs, signature, inputs)


def _apply_planned_messages(adapter: Any, messages: list[LMMessage], plan: _AdapterPlan) -> list[LMMessage]:
    planned_messages = adapter._coerce_lm_messages(list(plan.messages))
    user_parts = list(plan.user_parts)
    if planned_messages:
        insert_at = _last_user_message_index(messages)
        if insert_at is None:
            insert_at = len(messages)
        messages[insert_at:insert_at] = planned_messages
    if plan.user_part_segments:
        _insert_user_part_segments(messages, plan)
    if user_parts:
        user_index = _last_user_message_index(messages)
        if user_index is None:
            messages.append(LMMessage(role="user", parts=user_parts))
        else:
            messages[user_index].parts.extend(user_parts)
    return messages


def _insert_user_part_segments(messages: list[LMMessage], plan: _AdapterPlan) -> None:
    user_index = _last_user_message_index(messages)
    if user_index is None:
        segments = [part for _, parts in plan.user_part_segments for part in parts]
        messages.append(LMMessage(role="user", parts=segments))
        return

    grouped: dict[str, list[LMPart]] = {}
    order: list[str] = []
    for field_name, parts in plan.user_part_segments:
        anchor = _next_rendered_input_field(field_name, plan) or "__output_requirements__"
        if anchor not in grouped:
            grouped[anchor] = []
            order.append(anchor)
        grouped[anchor].extend(parts)

    message = messages[user_index]
    for anchor in order:
        _insert_parts_before_anchor(message, anchor, grouped[anchor], plan)


def _insert_parts_before_anchor(message: LMMessage, anchor: str, parts: list[LMPart], plan: _AdapterPlan) -> None:
    header = _anchor_header(anchor, plan)
    if header is None:
        message.parts.extend(parts)
        return

    for index, part in enumerate(message.parts):
        if not isinstance(part, LMTextPart):
            continue
        position = part.text.find(header)
        if position == -1:
            continue
        before = part.text[:position]
        after = part.text[position:]
        inserted = _trim_leading_newlines(parts) if index == 0 and not before else parts
        replacement = []
        if before:
            replacement.append(LMTextPart(text=before, metadata=part.metadata))
        replacement.extend(inserted)
        if after:
            replacement.append(LMTextPart(text="\n\n" + after, metadata=part.metadata))
        message.parts[index : index + 1] = replacement
        message.parts = _merge_adjacent_text_parts(message.parts)
        return

    message.parts.extend(parts)
    message.parts = _merge_adjacent_text_parts(message.parts)


def _merge_adjacent_text_parts(parts: list[LMPart]) -> list[LMPart]:
    merged: list[LMPart] = []
    for part in parts:
        if merged and isinstance(merged[-1], LMTextPart) and isinstance(part, LMTextPart) and not merged[-1].metadata and not part.metadata:
            merged[-1] = LMTextPart(text=merged[-1].text + part.text)
        else:
            merged.append(part)
    return merged


def _anchor_header(anchor: str, plan: _AdapterPlan) -> str | None:
    if anchor == "__output_requirements__":
        return "Respond with"
    if type(plan.adapter).__name__ == "XMLAdapter":
        return f"<{anchor}>"
    return f"[[ ## {anchor} ## ]]"


def _next_rendered_input_field(field_name: str, plan: _AdapterPlan) -> str | None:
    seen = False
    for name in plan.original_signature.input_fields:
        if name == field_name:
            seen = True
            continue
        if seen and name in plan.inputs:
            return name
    return None


def _trim_leading_newlines(parts: list[LMPart]) -> list[LMPart]:
    if not parts or not isinstance(parts[0], LMTextPart):
        return parts
    return [parts[0].model_copy(update={"text": parts[0].text.lstrip()}) , *parts[1:]]


def _native_input_parts(field_name: str, annotation: Any, value: Any, adapter: Any) -> list[LMPart] | None:
    if value is None:
        return None

    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        item_annotation = args[0] if args else Any
        if not isinstance(value, (list, tuple, set, frozenset)):
            return None
        parts: list[LMPart] = []
        for index, item in enumerate(value):
            item_parts = _native_input_parts(field_name, item_annotation, item, adapter)
            if item_parts is None:
                return None
            parts.extend(item_parts if index == 0 else _drop_leading_text_part(item_parts))
        return parts

    part = _native_input_part(annotation, value)
    if part is None:
        return None
    parts: list[LMPart] = [LMTextPart(text=_native_input_header(field_name, adapter)), part]
    if type(adapter).__name__ == "XMLAdapter":
        parts.append(LMTextPart(text=f"\n</{field_name}>"))
    return parts


def _native_input_header(field_name: str, adapter: Any) -> str:
    if type(adapter).__name__ == "XMLAdapter":
        return f"\n\n<{field_name}>\n"
    return f"\n\n[[ ## {field_name} ## ]]\n"


def _drop_leading_text_part(parts: list[LMPart]) -> list[LMPart]:
    if parts and isinstance(parts[0], LMTextPart):
        return parts[1:]
    return parts


def _native_input_part(annotation: Any, value: Any) -> LMPart | None:
    if _annotation_matches(annotation, Image):
        image = value if isinstance(value, Image) else Image(value)
        return _image_to_lm_part(image)
    if _annotation_matches(annotation, Audio):
        audio = value if isinstance(value, Audio) else Audio.model_validate(value)
        return LMAudioPart(data=audio.data, media_type=f"audio/{audio.audio_format}")
    if _annotation_matches(annotation, File):
        file = value if isinstance(value, File) else File.model_validate(value)
        return _file_to_lm_part(file)
    if _annotation_matches(annotation, Document):
        document = value if isinstance(value, Document) else Document.model_validate(value)
        return LMDocumentPart(
            media_type=document.media_type,
            source={"type": "text", "media_type": document.media_type, "data": document.data},
            citations={"enabled": True},
            title=document.title,
            context=document.context,
        )
    return None


def _image_to_lm_part(image: Image) -> LMImagePart:
    source = image.url
    if source.startswith("data:") and "," in source:
        header, data = source.split(",", 1)
        media_type = header.removeprefix("data:").split(";", 1)[0]
        return LMImagePart(data=data, media_type=media_type)
    return LMImagePart(url=source)


def _file_to_lm_part(file: File) -> LMBinaryPart:
    if file.file_data is not None:
        media_type = "application/octet-stream"
        data = file.file_data
        if file.file_data.startswith("data:") and "," in file.file_data:
            header, data = file.file_data.split(",", 1)
            media_type = header.removeprefix("data:").split(";", 1)[0]
        return LMBinaryPart(data=data, media_type=media_type, filename=file.filename)
    if file.file_id is not None:
        return LMBinaryPart(file_id=file.file_id, filename=file.filename)
    raise ValueError("File must have file_data or file_id.")


def _history_to_lm_messages(adapter: Any, signature: type[Signature], history: History) -> list[LMMessage]:
    messages: list[LMMessage] = []
    for turn in history.messages:
        messages.append(LMMessage(role="user", parts=[LMTextPart(text=adapter.format_user_message_content(signature, turn))]))
        messages.append(
            LMMessage(role="assistant", parts=[LMTextPart(text=adapter.format_assistant_message_content(signature, turn))])
        )
    return messages


def _last_user_message_index(messages: list[LMMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return None


def _apply_patch_field_deletions(signature: type[Signature], patch: LMRequestPatch) -> type[Signature]:
    for field_name in patch.delete_input_fields:
        if field_name in signature.input_fields:
            signature = signature.delete(field_name)
    for field_name in patch.delete_output_fields:
        if field_name in signature.output_fields:
            signature = signature.delete(field_name)
    return signature


def _annotation_matches(annotation: Any, expected: type) -> bool:
    try:
        return isinstance(annotation, type) and issubclass(annotation, expected)
    except TypeError:
        return False


def _tool_to_lm_tool_spec(tool: Tool) -> LMToolSpec:
    args = tool.args or {}
    return LMToolSpec(
        name=tool.name or "",
        description=tool.desc,
        parameters={"type": "object", "properties": args, "required": list(args.keys())},
    )


def _get_tool_call_input_field_name(signature: type[Signature]) -> str | None:
    for name, field in signature.input_fields.items():
        origin = get_origin(field.annotation)
        if origin is list and field.annotation.__args__[0] == Tool:
            return name
        if field.annotation == Tool:
            return name
    return None


def _get_tool_call_output_field_name(signature: type[Signature]) -> str | None:
    for name, field in signature.output_fields.items():
        if field.annotation == ToolCalls:
            return name
    return None
