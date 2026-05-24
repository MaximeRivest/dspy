from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, get_args

from dspy.adapters._type_runtime import _AdapterCallPlan, _CallContext, _TypeFeatureHandler
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.core.types import (
    LMCitationPart,
    LMConfig,
    LMMessage,
    LMOutput,
    LMPart,
    LMReasoningConfig,
    LMThinkingPart,
    LMToolChoice,
)
from dspy.experimental import Citations


@dataclass
class _RenderedTypeOutput:
    parts: list[LMPart] = field(default_factory=list)
    messages: list[LMMessage] = field(default_factory=list)

    @property
    def consumed(self) -> bool:
        return bool(self.parts or self.messages)


class _ToolTypeHandler(_TypeFeatureHandler):
    def prepare(self, call: _AdapterCallPlan, ctx: _CallContext) -> None:
        tool_call_output_field_name = _tool_call_output_field_name(call.source_signature)
        tool_input_field_name = _tool_input_field_name(call.source_signature)
        native = _native_tool_calling_enabled(call, ctx)

        if ctx.use_native_function_calling and tool_call_output_field_name is not None and tool_input_field_name is None:
            raise ValueError(
                f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                "input field with type `list[dspy.Tool]`."
            )

        if not native:
            self._apply_text_parallel_policy(call, ctx)
            return

        if ctx.allow_parallel_tool_calls is not None:
            call.merge_config(LMConfig(tool_choice=LMToolChoice(parallel=ctx.allow_parallel_tool_calls)))

        if tool_input_field_name is not None:
            value = call.inputs.get(tool_input_field_name)
            if value is not None:
                for tool in value if isinstance(value, list) else [value]:
                    tool = tool if isinstance(tool, Tool) else Tool.model_validate(tool)
                    call.tools.append(tool.to_lm_tool_spec())
            call.inputs.pop(tool_input_field_name, None)
            call.delete_field(tool_input_field_name)

        if tool_call_output_field_name is not None:
            call.delete_field(tool_call_output_field_name)

        for field_name, field_info in call.source_signature.input_fields.items():
            if not _annotation_includes(field_info.annotation, ToolCallResults):
                continue
            value = call.inputs.get(field_name)
            if value is None:
                continue
            results = value if isinstance(value, ToolCallResults) else ToolCallResults.model_validate(value)
            call.messages.extend(results.to_lm_messages())
            call.inputs.pop(field_name, None)
            call.delete_field(field_name)

    def parse(
        self,
        values: dict[str, object],
        output: LMOutput,
        call: _AdapterCallPlan,
        ctx: _CallContext,
    ) -> None:
        if not _native_tool_calling_enabled(call, ctx):
            return
        field_name = _tool_call_output_field_name(call.source_signature)
        if field_name is None or field_name in values or not output.tool_calls:
            return
        values[field_name] = ToolCalls.from_dict_list(
            [{"name": tool_call.name, "args": tool_call.args, "id": tool_call.id} for tool_call in output.tool_calls]
        )

    def format_output(
        self,
        field_name: str,
        value: object,
        call: _AdapterCallPlan,
        ctx: _CallContext,
    ) -> _RenderedTypeOutput | None:
        if not _native_tool_calling_enabled(call, ctx):
            return None
        if field_name != _tool_call_output_field_name(call.source_signature):
            return None
        tool_calls = value if isinstance(value, ToolCalls) else ToolCalls.model_validate(value)
        return _RenderedTypeOutput(parts=list(tool_calls.to_lm_parts()))

    def _apply_text_parallel_policy(self, call: _AdapterCallPlan, ctx: _CallContext) -> None:
        if ctx.allow_parallel_tool_calls is not False:
            return
        field_name = _tool_call_output_field_name(call.render_signature)
        if field_name is None:
            return
        call.render_signature = call.render_signature.with_updated_fields(
            field_name,
            **ToolCalls.json_schema_extra_for_max_items(1),
        )


class _ReasoningTypeHandler(_TypeFeatureHandler):
    def prepare(self, call: _AdapterCallPlan, ctx: _CallContext) -> None:
        reasoning_effort = self._native_reasoning_effort(call, ctx)
        if reasoning_effort is None:
            return
        for field_name, field_info in call.source_signature.output_fields.items():
            if field_info.annotation is Reasoning:
                call.merge_config(LMConfig(reasoning=LMReasoningConfig(effort=reasoning_effort)))
                call.delete_field(field_name)

    def parse(
        self,
        values: dict[str, object],
        output: LMOutput,
        call: _AdapterCallPlan,
        ctx: _CallContext,
    ) -> None:
        if self._native_reasoning_effort(call, ctx) is None or not output.reasoning_content:
            return
        for field_name, field_info in call.source_signature.output_fields.items():
            if field_info.annotation is Reasoning and field_name not in values:
                values[field_name] = Reasoning(content=output.reasoning_content)

    def format_output(
        self,
        field_name: str,
        value: object,
        call: _AdapterCallPlan,
        ctx: _CallContext,
    ) -> _RenderedTypeOutput | None:
        if self._native_reasoning_effort(call, ctx) is None:
            return None
        field_info = call.source_signature.output_fields.get(field_name)
        if field_info is None or field_info.annotation is not Reasoning:
            return None
        reasoning = value if isinstance(value, Reasoning) else Reasoning.model_validate(value)
        return _RenderedTypeOutput(parts=[LMThinkingPart(text=reasoning.content)])

    @staticmethod
    def _native_reasoning_effort(call: _AdapterCallPlan, ctx: _CallContext) -> str | None:
        if Reasoning not in ctx.native_response_types:
            return None
        if not ctx.lm.supports_reasoning:
            return None
        if "gpt-5" in ctx.lm.model and ctx.lm.model_type == "chat":
            return None

        if "reasoning_effort" in call.lm_kwargs:
            reasoning_effort = call.lm_kwargs["reasoning_effort"]
        elif "reasoning_effort" in ctx.lm_default_kwargs:
            reasoning_effort = ctx.lm_default_kwargs["reasoning_effort"]
        else:
            reasoning_effort = "low"
        return reasoning_effort


class _CitationsTypeHandler(_TypeFeatureHandler):
    def prepare(self, call: _AdapterCallPlan, ctx: _CallContext) -> None:
        if not self._native_citations_enabled(ctx):
            return
        for field_name, field_info in call.source_signature.output_fields.items():
            if field_info.annotation is Citations:
                call.delete_field(field_name)

    def parse(
        self,
        values: dict[str, object],
        output: LMOutput,
        call: _AdapterCallPlan,
        ctx: _CallContext,
    ) -> None:
        if not self._native_citations_enabled(ctx) or not output.citations:
            return
        for field_name, field_info in call.source_signature.output_fields.items():
            if field_info.annotation is Citations and field_name not in values:
                values[field_name] = Citations.from_dict_list(
                    [_lm_part_to_citation_dict(citation) for citation in output.citations]
                )

    def format_output(
        self,
        field_name: str,
        value: object,
        call: _AdapterCallPlan,
        ctx: _CallContext,
    ) -> _RenderedTypeOutput | None:
        if not self._native_citations_enabled(ctx):
            return None
        field_info = call.source_signature.output_fields.get(field_name)
        if field_info is None or field_info.annotation is not Citations:
            return None
        citations = value if isinstance(value, Citations) else Citations.model_validate(value)
        return _RenderedTypeOutput(parts=[_citation_to_lm_part(citation) for citation in citations.citations])

    @staticmethod
    def _native_citations_enabled(ctx: _CallContext) -> bool:
        return Citations in ctx.native_response_types and ctx.lm.model.startswith("anthropic/")


def _native_tool_calling_enabled(call: _AdapterCallPlan, ctx: _CallContext) -> bool:
    return bool(
        ctx.use_native_function_calling
        and ctx.lm.supports_function_calling
        and _tool_call_output_field_name(call.source_signature) is not None
    )


def _tool_input_field_name(signature: type[Any]) -> str | None:
    for name, field in signature.input_fields.items():
        if _annotation_includes(field.annotation, Tool):
            return name
    return None


def _tool_call_output_field_name(signature: type[Any]) -> str | None:
    for name, field in signature.output_fields.items():
        if _annotation_includes(field.annotation, ToolCalls):
            return name
    return None


def _annotation_includes(annotation: Any, target: type) -> bool:
    if annotation is target:
        return True
    return any(_annotation_includes(arg, target) for arg in get_args(annotation))


def _citation_to_lm_part(citation: Citations.Citation) -> LMCitationPart:
    return LMCitationPart(
        text=citation.cited_text,
        title=citation.document_title,
        metadata={
            "type": citation.type,
            "document_index": citation.document_index,
            "start_char_index": citation.start_char_index,
            "end_char_index": citation.end_char_index,
            **({"supported_text": citation.supported_text} if citation.supported_text is not None else {}),
        },
    )


def _lm_part_to_citation_dict(citation: LMCitationPart) -> dict[str, Any]:
    data = dict(citation.metadata)
    if citation.text is not None:
        data["cited_text"] = citation.text
    if citation.title is not None:
        data["document_title"] = citation.title
    data.setdefault("document_index", 0)
    data.setdefault("start_char_index", 0)
    data.setdefault("end_char_index", len(data.get("cited_text", "")))
    return data
