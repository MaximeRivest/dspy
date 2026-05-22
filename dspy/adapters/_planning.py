"""Private adapter-call rendering pipeline for native LM features and type patches."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, get_args, get_origin

from dspy.adapters.types import Audio, File, History, Image, Type
from dspy.adapters.types.base_type import _AdapterTypeContext
from dspy.adapters.types.document import Document
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.core.types import LMMessage, LMPart, LMRequestPatch, LMTextPart, LMToolSpec
from dspy.experimental import Citations
from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)


@dataclass
class _AdapterPlan:
    """Private adapter-call plan for prompt rendering and native LM features."""

    original_signature: type[Signature]
    render_signature: type[Signature]
    inputs: dict[str, Any]
    lm_kwargs: dict[str, Any]
    messages: list[LMMessage] = field(default_factory=list)
    user_parts: list[LMPart] = field(default_factory=list)
    tools: list[LMToolSpec] = field(default_factory=list)


@dataclass
class _AdapterRenderContext:
    """Mutable private context used while renderers build an adapter plan."""

    adapter: Any
    lm: BaseLM
    original_signature: type[Signature]
    render_signature: type[Signature]
    inputs: dict[str, Any]
    lm_kwargs: dict[str, Any]
    patch: LMRequestPatch = field(default_factory=LMRequestPatch)

    def apply_patch(self, patch: LMRequestPatch) -> None:
        self.patch = self.patch.merge(patch)
        self.render_signature = _apply_patch_field_deletions(self.render_signature, patch)
        if patch.config is not None:
            patch_kwargs = patch.as_lm_kwargs()
            patch_kwargs.pop("tools", None)
            self.lm_kwargs.update(patch_kwargs)


class _FieldRenderer(Protocol):
    """Private strategy interface for adapter field rendering/planning."""

    def render_input(self, field_name: str, field: Any, value: Any, context: _AdapterRenderContext) -> LMRequestPatch | None:
        return None

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        return None


class _AdapterRendererPipeline:
    def __init__(self, renderers: list[_FieldRenderer] | None = None):
        self.renderers = renderers or [
            _HistoryRenderer(),
            _TypeGeneratePatchRenderer(),
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
            original_signature=context.original_signature,
            render_signature=context.render_signature,
            inputs=context.inputs,
            lm_kwargs=context.lm_kwargs,
            messages=context.patch.messages,
            user_parts=context.patch.user_parts,
            tools=context.patch.tools,
        )


class _HistoryRenderer:
    def render_input(self, field_name: str, field: Any, value: Any, context: _AdapterRenderContext) -> LMRequestPatch | None:
        if field.annotation != History:
            return None
        signature_without_history = context.render_signature.delete(field_name)
        return LMRequestPatch(
            messages=_history_to_lm_messages(context.adapter, signature_without_history, value),
            delete_input_fields=(field_name,),
        )

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        return None


class _TypeGeneratePatchRenderer:
    """Renderer that delegates local adapter-type value rendering to `Type.generate_patch()`."""

    def render_input(self, field_name: str, field: Any, value: Any, context: _AdapterRenderContext) -> LMRequestPatch | None:
        try:
            return self._render_input_unchecked(field_name, field, value, context)
        except Exception:
            logger.debug("Failed to normalize %r as native LM patch; falling back to text rendering.", field.annotation, exc_info=True)
            return None

    def render_outputs(self, context: _AdapterRenderContext) -> None:
        return None

    def _render_input_unchecked(
        self,
        field_name: str,
        field: Any,
        value: Any,
        context: _AdapterRenderContext,
    ) -> LMRequestPatch | None:
        if value is None:
            return None

        annotation = field.annotation
        origin = get_origin(annotation)
        if origin in (list, tuple, set, frozenset):
            args = get_args(annotation)
            item_annotation = args[0] if args else Any
            if not isinstance(value, (list, tuple, set, frozenset)):
                return None
            merged = LMRequestPatch(delete_input_fields=(field_name,))
            first_item = True
            for item in value:
                item_patch = self._render_value_for_annotation(field_name, field, item_annotation, item, context)
                if item_patch is None:
                    return None
                if not first_item and item_patch.user_parts and isinstance(item_patch.user_parts[0], LMTextPart):
                    item_patch = _drop_first_user_part(item_patch)
                merged = merged.merge(item_patch)
                first_item = False
            return merged

        return self._render_value_for_annotation(field_name, field, annotation, value, context)

    def _render_value_for_annotation(
        self,
        field_name: str,
        field: Any,
        annotation: Any,
        value: Any,
        context: _AdapterRenderContext,
    ) -> LMRequestPatch | None:
        typed_value = _coerce_adapter_type_value(annotation, value)
        if not isinstance(typed_value, Type):
            return None
        type_context = _AdapterTypeContext(
            field_name=field_name,
            field_info=field,
            signature=context.render_signature,
            lm=context.lm,
            lm_kwargs=context.lm_kwargs,
            adapter=context.adapter,
            role="input",
        )
        return typed_value.generate_patch(type_context)


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
    if user_parts:
        user_index = _last_user_message_index(messages)
        if user_index is None:
            messages.append(LMMessage(role="user", parts=user_parts))
        else:
            messages[user_index].parts.extend(user_parts)
    return messages


def _drop_first_user_part(patch: LMRequestPatch) -> LMRequestPatch:
    return LMRequestPatch(
        messages=patch.messages,
        system_parts=patch.system_parts,
        user_parts=patch.user_parts[1:],
        assistant_parts=patch.assistant_parts,
        tools=patch.tools,
        config=patch.config,
        delete_input_fields=patch.delete_input_fields,
        delete_output_fields=patch.delete_output_fields,
        metadata=patch.metadata,
    )


def _last_user_message_index(messages: list[LMMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == "user":
            return index
    return None


def _history_to_lm_messages(adapter: Any, signature: type[Signature], history: History) -> list[LMMessage]:
    messages: list[LMMessage] = []
    for turn in history.messages:
        messages.append(LMMessage(role="user", parts=[LMTextPart(text=adapter.format_user_message_content(signature, turn))]))
        messages.append(
            LMMessage(role="assistant", parts=[LMTextPart(text=adapter.format_assistant_message_content(signature, turn))])
        )
    return messages


def _coerce_adapter_type_value(annotation: Any, value: Any) -> Type | None:
    if isinstance(value, Type):
        return value
    if _annotation_matches(annotation, Image):
        return Image(value)
    if _annotation_matches(annotation, Audio):
        return Audio.model_validate(value)
    if _annotation_matches(annotation, File):
        return File.model_validate(value)
    if _annotation_matches(annotation, Document):
        return Document.model_validate(value)
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
