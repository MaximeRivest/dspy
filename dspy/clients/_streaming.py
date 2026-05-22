"""Private helpers for normalizing provider stream chunks."""

from __future__ import annotations

from typing import Any

from dspy.clients.openai_format import citation_to_part
from dspy.core.types import (
    LMCitationDelta,
    LMStreamDeltaEvent,
    LMStreamEvent,
    LMTextDelta,
    LMThinkingDelta,
    LMToolCallDelta,
)


def is_litellm_model_response_stream(value: Any) -> bool:
    cls = type(value)
    return cls.__name__ == "ModelResponseStream" and cls.__module__.startswith("litellm")


def stream_event_text(event: LMStreamEvent) -> str | None:
    if isinstance(event, LMStreamDeltaEvent) and isinstance(event.delta, LMTextDelta):
        return event.delta.text
    return None


def litellm_chunk_text(chunk: Any) -> str | None:
    try:
        return chunk.choices[0].delta.content
    except Exception:
        return None


def litellm_chunk_to_lm_stream_event(chunk: Any) -> LMStreamEvent | None:
    """Return the first normalized stream event represented by a LiteLLM chunk."""
    events = litellm_chunk_to_lm_stream_events(chunk)
    return events[0] if events else None


def litellm_chunk_to_lm_stream_events(chunk: Any) -> list[LMStreamEvent]:
    """Convert a LiteLLM stream chunk into zero or more normalized stream events.

    A single provider chunk can contain multiple tool-call deltas, so callers
    that can emit events directly should use this plural form.
    """
    try:
        delta = chunk.choices[0].delta
    except Exception:
        return []

    events: list[LMStreamEvent] = []

    reasoning = getattr(delta, "reasoning_content", None)
    if reasoning:
        events.append(LMStreamDeltaEvent(output_index=0, part_index=0, delta=LMThinkingDelta(text=str(reasoning))))

    provider_specific_fields = getattr(delta, "provider_specific_fields", None) or {}
    citation = provider_specific_fields.get("citation")
    if citation:
        events.append(LMStreamDeltaEvent(output_index=0, part_index=0, delta=LMCitationDelta(citation=citation_to_part(citation))))

    for index, tool_call in enumerate(getattr(delta, "tool_calls", None) or []):
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", None) if function is not None else getattr(tool_call, "name", None)
        args_delta = getattr(function, "arguments", None) if function is not None else getattr(tool_call, "arguments", None)
        call_id = getattr(tool_call, "id", None) or getattr(tool_call, "call_id", None)
        part_index = getattr(tool_call, "index", None)
        if part_index is None:
            part_index = index
        events.append(
            LMStreamDeltaEvent(
                output_index=0,
                part_index=part_index,
                delta=LMToolCallDelta(id=call_id, name=name, args_delta=args_delta),
            )
        )

    content = getattr(delta, "content", None)
    if content:
        events.append(LMStreamDeltaEvent(output_index=0, part_index=0, delta=LMTextDelta(text=content)))

    return events


def with_stream_metadata(event: LMStreamEvent, **metadata: Any) -> LMStreamEvent:
    current = dict(getattr(event, "metadata", {}) or {})
    current.update(metadata)
    return event.model_copy(update={"metadata": current})
