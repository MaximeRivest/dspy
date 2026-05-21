"""Rendering strategies for `dspy.Tool` and `dspy.ToolCalls` fields."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, get_origin

from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMOutput, LMRequestPatch, LMToolChoice


@dataclass(frozen=True)
class NativeToolCalls(TypeStrategy[ToolCalls]):
    """Expose tools through the normalized LM tool interface and parse native tool calls."""

    marker_type: type[ToolCalls] = ToolCalls
    tool_choice: LMToolChoice | str | dict[str, Any] | None = None

    def prepare(
        self,
        *,
        signature: Any,
        lm: Any,
        lm_kwargs: dict[str, Any],
        inputs: dict[str, Any],
        adapter: Any | None = None,
    ) -> LMRequestPatch:
        tool_call_output_field_name = _tool_call_output_field_name(signature)
        if tool_call_output_field_name is None:
            return LMRequestPatch()

        tool_input_field_name = _tool_input_field_name(signature)
        if tool_input_field_name is None:
            raise ValueError(
                f"You provided an output field {tool_call_output_field_name} to receive tool calls, "
                "but did not provide an input field with type `dspy.Tool` or `list[dspy.Tool]`."
            )
        if tool_input_field_name not in inputs:
            return LMRequestPatch(delete_output_fields=(tool_call_output_field_name,))

        tools = inputs[tool_input_field_name]
        tools = tools if isinstance(tools, list) else [tools]
        return LMRequestPatch(
            delete_input_fields=(tool_input_field_name,),
            delete_output_fields=(tool_call_output_field_name,),
            tools=[tool.to_lm_tool_spec() if hasattr(tool, "to_lm_tool_spec") else tool for tool in tools],
            config=_tool_choice_patch(self.tool_choice),
        )

    def parse_output(
        self,
        *,
        field_name: str,
        output: LMOutput | dict[str, Any] | str,
        field: Any | None = None,
        adapter: Any | None = None,
    ) -> ToolCalls | None:
        if isinstance(output, LMOutput):
            if not output.tool_calls:
                return None
            return ToolCalls.from_dict_list([{"name": call.name, "args": call.args} for call in output.tool_calls])
        if isinstance(output, dict):
            calls = output.get("tool_calls") or output.get(field_name)
            if isinstance(calls, ToolCalls):
                return calls
            if isinstance(calls, list):
                normalized = []
                for call in calls:
                    if isinstance(call, dict) and "function" in call:
                        arguments = call["function"].get("arguments", {})
                        if isinstance(arguments, str):
                            try:
                                arguments = json.loads(arguments)
                            except json.JSONDecodeError:
                                arguments = {}
                        normalized.append({"name": call["function"].get("name"), "args": arguments})
                    else:
                        normalized.append(call)
                return ToolCalls.from_dict_list(normalized)
        return None


@dataclass(frozen=True)
class TextToolCalls(TypeStrategy[ToolCalls]):
    """Keep tool calls in the adapter's ordinary text/JSON/XML field format."""

    marker_type: type[ToolCalls] = ToolCalls


def _tool_choice_patch(choice: LMToolChoice | str | dict[str, Any] | None):
    if choice is None:
        return None
    from dspy.clients.language_models.types import LMConfig

    if isinstance(choice, LMToolChoice):
        tool_choice = choice
    elif isinstance(choice, str):
        tool_choice = LMToolChoice(mode=choice)
    else:
        tool_choice = LMToolChoice(**choice)
    return LMConfig(tool_choice=tool_choice)


def _tool_input_field_name(signature: Any) -> str | None:
    for name, field in signature.input_fields.items():
        origin = get_origin(field.annotation)
        if origin is list and field.annotation.__args__[0] == Tool:
            return name
        if field.annotation == Tool:
            return name
    return None


def _tool_call_output_field_name(signature: Any) -> str | None:
    for name, field in signature.output_fields.items():
        if field.annotation == ToolCalls:
            return name
    return None
