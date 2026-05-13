"""Call LiteLLM and translate its streams into DSPy stream events.

This module is the transport sibling of `openai_format.py`. The OpenAI-format
module maps DSPy requests and responses. This module sends those mapped requests
through LiteLLM and handles LiteLLM's streaming shapes.

Read the file in this order when learning it:

1. `litellm_*_completion()` and `alitellm_*_completion()` make provider calls.
2. `prepare_litellm_request()` strips DSPy-only fields and adds DSPy headers.
3. `completion_stream_to_events()` maps Chat/text completion chunks.
4. `responses_stream_to_events()` maps Responses API stream events.

LiteLLM's own cache is explicitly disabled here. DSPy request caching happens
outside this module in `LanguageModel` and the LM factories.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

import litellm

from dspy.__metadata__ import __version__
from dspy.clients.language_models.openai_format import cost_from_response as openai_cost_from_response
from dspy.clients.language_models.openai_format import responses_to_lm_response, usage_from_response
from dspy.clients.language_models.types import (
    LMRequest,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamErrorEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMThinkingDelta,
    LMToolCallDelta,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LiteLLM calls
#
# These functions receive already-mapped OpenAI-shaped kwargs from the factory
# layer. They only prepare LiteLLM-specific transport details and call LiteLLM.
# ---------------------------------------------------------------------------


def litellm_chat_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Call LiteLLM's synchronous Chat Completions endpoint."""
    request, headers = prepare_litellm_request(request)
    return litellm.completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


async def alitellm_chat_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Call LiteLLM's asynchronous Chat Completions endpoint."""
    request, headers = prepare_litellm_request(request)
    return await litellm.acompletion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


def litellm_chat_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Open a synchronous Chat Completions stream through LiteLLM."""
    request, headers = prepare_litellm_request(request)
    return litellm.completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


async def alitellm_chat_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Open an asynchronous Chat Completions stream through LiteLLM."""
    request, headers = prepare_litellm_request(request)
    return await litellm.acompletion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


def litellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Call LiteLLM's synchronous text-completion endpoint."""
    request, headers = prepare_litellm_text_request(request)
    return litellm.text_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


async def alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Call LiteLLM's asynchronous text-completion endpoint."""
    request, headers = prepare_litellm_text_request(request)
    return await litellm.atext_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


def litellm_text_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Open a synchronous text-completion stream through LiteLLM."""
    request, headers = prepare_litellm_text_request(request)
    return litellm.text_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


async def alitellm_text_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Open an asynchronous text-completion stream through LiteLLM."""
    request, headers = prepare_litellm_text_request(request)
    return await litellm.atext_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


def litellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Call LiteLLM's synchronous Responses endpoint."""
    request, headers = prepare_litellm_request(request)
    return litellm.responses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


async def alitellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Call LiteLLM's asynchronous Responses endpoint."""
    request, headers = prepare_litellm_request(request)
    return await litellm.aresponses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


def litellm_responses_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Open a synchronous Responses stream through LiteLLM."""
    request, headers = prepare_litellm_request(request)
    return litellm.responses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, **request)


async def alitellm_responses_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    """Open an asynchronous Responses stream through LiteLLM."""
    request, headers = prepare_litellm_request(request)
    return await litellm.aresponses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, **request)


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------


def prepare_litellm_request(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare OpenAI-shaped kwargs for a LiteLLM call.

    `rollout_id` is a DSPy cache key, not a provider argument. Headers are
    returned separately because LiteLLM accepts them as a transport parameter.
    """
    request = dict(request)
    request.pop("rollout_id", None)
    headers = add_dspy_identifier_to_headers(request.pop("headers", None))
    return request, headers


def prepare_litellm_text_request(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare a LiteLLM text-completion request using LiteLLM's text routing convention."""
    import os

    request, headers = prepare_litellm_request(request)
    model = str(request.pop("model"))
    parts = model.split("/", 1)
    provider, model_name = (parts[0], parts[1]) if len(parts) == 2 else ("openai", parts[0])
    request["model"] = f"text-completion-openai/{model_name}"
    request["api_key"] = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    request["api_base"] = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")
    return request, headers


# ---------------------------------------------------------------------------
# Chat/text completion streams -> DSPy stream events
# ---------------------------------------------------------------------------


def completion_stream_to_events(stream: Iterator[Any], *, model: str) -> Iterator[LMStreamEvent]:
    """Convert a LiteLLM Chat/text stream into normalized DSPy events."""
    yield LMStreamStartEvent(model=model)
    state = _CompletionStreamState()
    usage = None
    cost = None
    for chunk in stream:
        usage = usage_from_response(chunk) or usage
        cost = cost_from_response(chunk) if cost_from_response(chunk) is not None else cost
        yield from state.chunk_to_events(chunk)
    yield from state.missing_output_end_events()
    yield LMStreamEndEvent(usage=usage, cost=cost)


async def acompletion_stream_to_events(stream: AsyncIterator[Any], *, model: str) -> AsyncIterator[LMStreamEvent]:
    """Convert an async LiteLLM Chat/text stream into DSPy events."""
    yield LMStreamStartEvent(model=model)
    state = _CompletionStreamState()
    usage = None
    cost = None
    async for chunk in stream:
        usage = usage_from_response(chunk) or usage
        cost = cost_from_response(chunk) if cost_from_response(chunk) is not None else cost
        for event in state.chunk_to_events(chunk):
            yield event
    for event in state.missing_output_end_events():
        yield event
    yield LMStreamEndEvent(usage=usage, cost=cost)


class _CompletionStreamState:
    """Build DSPy events from Chat/text completion chunks.

    OpenAI-compatible streams interleave deltas for text, reasoning, and tool
    calls. DSPy needs stable `(output_index, part_index)` pairs so
    `LMOutputBuilder` can assemble one final `LMResponse`.
    """

    _REASONING_PART_INDEX = 0
    _TEXT_PART_INDEX = 1
    _TOOL_PART_INDEX_OFFSET = 2

    def __init__(self):
        self._ended_outputs: set[int] = set()
        self._seen_outputs: set[int] = {0}

    def chunk_to_events(self, chunk: Any) -> list[LMStreamEvent]:
        events: list[LMStreamEvent] = []
        for choice in get_value(chunk, "choices", []) or []:
            output_index = get_value(choice, "index", 0) or 0
            self._seen_outputs.add(output_index)
            delta = get_value(choice, "delta") or get_value(choice, "message")
            if delta is not None:
                events.extend(self._delta_to_events(delta, output_index=output_index))
            finish_reason = get_value(choice, "finish_reason")
            if finish_reason is not None:
                self._ended_outputs.add(output_index)
                events.append(
                    LMStreamOutputEndEvent(
                        output_index=output_index,
                        finish_reason=finish_reason,
                        truncated=finish_reason == "length",
                    )
                )
        return events

    def missing_output_end_events(self) -> list[LMStreamOutputEndEvent]:
        return [
            LMStreamOutputEndEvent(output_index=output_index)
            for output_index in sorted(self._seen_outputs - self._ended_outputs)
        ]

    def _delta_to_events(self, delta: Any, *, output_index: int) -> list[LMStreamEvent]:
        events: list[LMStreamEvent] = []
        reasoning = get_value(delta, "reasoning_content")
        if reasoning:
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._REASONING_PART_INDEX,
                    delta=LMThinkingDelta(text=str(reasoning)),
                )
            )
        content = get_value(delta, "content")
        if content:
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._TEXT_PART_INDEX,
                    delta=LMTextDelta(text=str(content)),
                )
            )
        for fallback_index, tool_call in enumerate(get_value(delta, "tool_calls") or []):
            tool_index = get_value(tool_call, "index", fallback_index) or 0
            function = get_value(tool_call, "function", {})
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._TOOL_PART_INDEX_OFFSET + int(tool_index),
                    delta=LMToolCallDelta(
                        id=get_value(tool_call, "id"),
                        name=get_value(function, "name") or get_value(tool_call, "name"),
                        args_delta=get_value(function, "arguments") or get_value(tool_call, "arguments") or "",
                    ),
                )
            )
        return events


def _completion_chunk_to_delta_events(chunk: Any) -> list[LMStreamEvent]:
    return _CompletionStreamState().chunk_to_events(chunk)


# ---------------------------------------------------------------------------
# Responses streams -> DSPy stream events
# ---------------------------------------------------------------------------


def responses_stream_to_events(stream: Iterator[Any], *, model: str) -> Iterator[LMStreamEvent]:
    """Convert a LiteLLM Responses stream into normalized DSPy events."""
    yield LMStreamStartEvent(model=model)
    state = _ResponsesStreamState()
    for event in stream:
        yield from state.event_to_events(event)
    yield from state.finish_events()


async def aresponses_stream_to_events(stream: AsyncIterator[Any], *, model: str) -> AsyncIterator[LMStreamEvent]:
    """Convert an async LiteLLM Responses stream into DSPy events."""
    yield LMStreamStartEvent(model=model)
    state = _ResponsesStreamState()
    async for event in stream:
        for mapped in state.event_to_events(event):
            yield mapped
    for event in state.finish_events():
        yield event


class _ResponsesStreamState:
    """Build DSPy events from Responses API stream events.

    Responses streams are event-based rather than choice-based. The state keeps
    a stable part index for each function-call item and, when available, stores
    the completed response object so the final `LMStreamEndEvent` can carry a
    fully mapped `LMResponse`.
    """

    _REASONING_PART_INDEX = 0
    _TEXT_PART_INDEX = 1
    _TOOL_PART_INDEX_OFFSET = 2

    def __init__(self):
        self._ended = False
        self._usage = None
        self._cost = None
        self._response = None
        self._tool_part_by_item: dict[str, int] = {}
        self._next_tool_part_index = self._TOOL_PART_INDEX_OFFSET

    def event_to_events(self, event: Any) -> list[LMStreamEvent]:
        event_type = get_value(event, "type")
        if event_type in {"response.output_text.delta", "output_text.delta"}:
            return [
                LMStreamDeltaEvent(
                    output_index=0,
                    part_index=self._TEXT_PART_INDEX,
                    delta=LMTextDelta(text=str(get_value(event, "delta", ""))),
                )
            ]
        if event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            return [
                LMStreamDeltaEvent(
                    output_index=0,
                    part_index=self._REASONING_PART_INDEX,
                    delta=LMThinkingDelta(text=str(get_value(event, "delta", ""))),
                )
            ]
        if event_type in {"response.output_item.added", "response.output_item.done"}:
            return self._record_output_item(event)
        if event_type in {"response.function_call_arguments.delta", "function_call_arguments.delta"}:
            return [
                LMStreamDeltaEvent(
                    output_index=0,
                    part_index=self._tool_part_index(event),
                    delta=LMToolCallDelta(
                        id=get_value(event, "call_id"),
                        name=get_value(event, "name"),
                        args_delta=str(get_value(event, "delta", "")),
                    ),
                )
            ]
        if event_type == "response.completed":
            self._ended = True
            response = get_value(event, "response") or event
            self._response = response if get_value(response, "output") is not None else None
            self._usage = usage_from_response(response) or self._usage
            cost = cost_from_response(response)
            self._cost = cost if cost is not None else self._cost
            return [LMStreamOutputEndEvent(output_index=0)]
        if event_type in {"response.failed", "error"}:
            return [LMStreamErrorEvent(error=RuntimeError(str(get_value(event, "error", event))))]
        return []

    def finish_events(self) -> list[LMStreamEvent]:
        events: list[LMStreamEvent] = []
        if not self._ended:
            events.append(LMStreamOutputEndEvent(output_index=0))
        if self._response is not None:
            events.append(LMStreamEndEvent(response=responses_to_lm_response(self._response, _request_for_response(self._response))))
        else:
            events.append(LMStreamEndEvent(usage=self._usage, cost=self._cost))
        return events

    def _record_output_item(self, event: Any) -> list[LMStreamEvent]:
        item = get_value(event, "item") or event
        if get_value(item, "type") != "function_call":
            return []
        return [
            LMStreamDeltaEvent(
                output_index=0,
                part_index=self._tool_part_index(item),
                delta=LMToolCallDelta(
                    id=get_value(item, "call_id") or get_value(item, "id"),
                    name=get_value(item, "name"),
                    args_delta="",
                ),
            )
        ]

    def _tool_part_index(self, event: Any) -> int:
        key = str(get_value(event, "item_id") or get_value(event, "id") or get_value(event, "call_id") or get_value(event, "output_index", ""))
        if not key:
            key = str(get_value(event, "output_index", 0) or 0)
        if key not in self._tool_part_by_item:
            self._tool_part_by_item[key] = self._next_tool_part_index
            self._next_tool_part_index += 1
        return self._tool_part_by_item[key]


def responses_stream_event_to_events(event: Any) -> list[LMStreamEvent]:
    return _ResponsesStreamState().event_to_events(event)


def _request_for_response(response: Any) -> LMRequest:
    return LMRequest.from_call(model=get_value(response, "model") or "", prompt="")


def cost_from_response(response: Any) -> float | None:
    return openai_cost_from_response(response)


def add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add DSPy's user agent without dropping caller-provided headers."""
    return {"User-Agent": f"DSPy/{__version__}", **(headers or {})}


def get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
