"""Pure message assembly: the engine-side replacement for ``Adapter.format``.

This module reproduces the legacy rendering pipeline (``Adapter.format`` +
``format_demos`` + ``format_conversation_history``) structurally — which
messages exist, in what order, with what roles — while delegating EVERY
content string to the resolved :class:`Format` object. It must contain zero
format-specific literals (a hard review gate, proven mechanically when the
next wire format lands with no diff here).

Structural behaviors preserved verbatim from base.py: the inputs dict is
copied before history removal; demo classification (complete vs incomplete
vs dropped); native tool-call history replay with the call-id consistency
guard; the textual tool-result fallback through the synthetic
``tool_call_results`` signature; and legacy custom-type marker expansion at
the same single point, behind one displaceable function.
"""

from typing import Any

from dspy.adapters._engine.formats import Format


def render_messages(adapter, fmt: Format, signature, demos: list[dict[str, Any]], inputs: dict[str, Any]):
    """Render the full message list for one adapter call."""
    inputs_copy = dict(inputs)

    history_field_name = adapter._get_history_field_name(signature)
    if history_field_name:
        signature_without_history = signature.delete(history_field_name)
        conversation_history = _render_history(adapter, fmt, signature_without_history, history_field_name, inputs_copy)

    messages = []
    messages.append({"role": "system", "content": fmt.render_system(signature)})
    messages.extend(_render_demos(fmt, signature, demos))
    if history_field_name:
        content = fmt.render_user_content(signature_without_history, inputs_copy, main_request=True)
        messages.extend(conversation_history)
        if content:
            messages.append({"role": "user", "content": content})
    else:
        content = fmt.render_user_content(signature, inputs_copy, main_request=True)
        if content:
            messages.append({"role": "user", "content": content})

    return [_expand_markers(message) for message in messages]


def _expand_markers(message):
    """Legacy custom-type marker expansion, at the same pipeline point as the
    legacy renderer. One displaceable function so a future media strategy can
    take over part construction without hunting call sites."""
    from dspy.adapters.base import _expand_legacy_custom_type_markers_in_chat_message

    return _expand_legacy_custom_type_markers_in_chat_message(message)


def _render_demos(fmt: Format, signature, demos: list[dict[str, Any]]):
    """Structural twin of ``Adapter.format_demos``."""
    from dspy.adapters._engine.template.renderer import classify_demos

    incomplete_demos, complete_demos = classify_demos(signature, demos)

    messages = []
    for demo in incomplete_demos:
        messages.append(
            {
                "role": "user",
                "content": fmt.render_user_content(signature, demo, prefix=fmt.incomplete_demo_prefix),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": fmt.render_assistant_content(
                    signature, demo, missing_field_message=fmt.incomplete_demo_missing_field_message
                ),
            }
        )

    for demo in complete_demos:
        messages.append({"role": "user", "content": fmt.render_user_content(signature, demo)})
        messages.append(
            {
                "role": "assistant",
                "content": fmt.render_assistant_content(
                    signature, demo, missing_field_message=fmt.complete_demo_missing_field_message
                ),
            }
        )

    return messages


def _render_history(adapter, fmt: Format, signature, history_field_name: str, inputs: dict[str, Any]):
    """Structural twin of ``Adapter.format_conversation_history``, including
    its mutation contract (removes the history key from ``inputs``, which the
    caller passed as its own copy)."""
    from dspy.adapters.base import (
        _TOOL_CALL_RESULTS_SIGNATURE,
        _tool_call_as_openai_message_tool_call,
        _tool_calls_from_message,
        _tool_result_content,
    )
    from dspy.adapters.types.tool import ToolCallResults, ToolCalls

    conversation_history = inputs[history_field_name].messages if history_field_name in inputs else None

    if conversation_history is None:
        return []

    messages = []
    for message in conversation_history:
        tool_call_field_name, tool_calls = _tool_calls_from_message(message)
        tool_call_results = (
            ToolCallResults.model_validate(tool_calls.tool_call_results)
            if tool_calls is not None and tool_calls.tool_call_results is not None
            else None
        )

        user_content = fmt.render_user_content(signature, message)
        if user_content:
            messages.append({"role": "user", "content": user_content})

        if adapter.use_native_function_calling and tool_calls is not None:
            content_signature = signature
            for name, field in signature.output_fields.items():
                if field.annotation == ToolCalls or message.get(name) is None:
                    content_signature = content_signature.delete(name)

            content = (
                fmt.render_assistant_content(content_signature, message) if content_signature.output_fields else ""
            )

            if tool_call_results is not None:
                tool_call_ids = [tool_call.id for tool_call in tool_calls.tool_calls]
                result_ids = [result.call_id for result in tool_call_results.tool_call_results]
                if tool_call_ids != result_ids or not all(tool_call_ids):
                    tool_call_results = None

            if content or tool_call_results is not None:
                assistant_message: dict[str, Any] = {"role": "assistant", "content": content or None}
                if tool_call_results is not None:
                    assistant_message["tool_calls"] = [
                        _tool_call_as_openai_message_tool_call(tool_call) for tool_call in tool_calls.tool_calls
                    ]
                messages.append(assistant_message)

            if tool_call_results is not None:
                for result in tool_call_results.tool_call_results:
                    content = _tool_result_content(result.value)
                    messages.append(
                        {"role": "tool", "tool_call_id": result.call_id, "name": result.name, "content": content}
                    )
            continue

        assistant_values = message
        if tool_call_field_name is not None and tool_call_results is not None:
            assistant_values = dict(message)
            assistant_values[tool_call_field_name] = tool_calls.model_copy(update={"tool_call_results": None})

        assistant_content = fmt.render_assistant_content(signature, assistant_values)
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})
        if tool_call_results is not None:
            result_input = {"tool_call_results": tool_call_results}
            content = fmt.render_user_content(_TOOL_CALL_RESULTS_SIGNATURE, result_input)
            messages.append({"role": "user", "content": content})

    del inputs[history_field_name]

    return messages
