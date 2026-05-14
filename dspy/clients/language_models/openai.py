"""OpenAI language model backends for normalized DSPy requests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import anyio

from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.openai_format import (
    completion_to_lm_response,
    cost_from_response,
    responses_to_lm_response,
    to_openai_chat_request,
    to_openai_responses_request,
    to_openai_text_request,
    usage_from_response,
)
from dspy.clients.language_models.types import (
    LMRequest,
    LMResponse,
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

CompletionProtocol = Literal["chat", "text"]

__all__ = [
    "OpenAICompletionsLM",
    "OpenAIResponsesLM",
    "completion_stream_to_events",
    "responses_stream_to_events",
]


class OpenAIResponsesLM(LanguageModel):
    """Call OpenAI's Responses API with DSPy's normalized LM types.

    Requests are normalized by `LanguageModel`, translated to Responses API
    kwargs, sent directly to an OpenAI-compatible `/responses` endpoint, and
    returned as `LMResponse` objects.

    Args:
        model: OpenAI model name. A leading `openai/` provider prefix is removed
            before the provider call.
        responses: Optional callable or object with `.create(**kwargs)` used for
            the Responses API call. This is useful in tests and custom clients.
        client: Optional OpenAI client with a `.responses.create(...)` method.
        api_key: Optional OpenAI API key used when creating a client lazily.
        base_url: Optional OpenAI-compatible base URL.
        cache: Whether DSPy request memoization is enabled by default.
        callbacks: Optional DSPy callbacks.
        **kwargs: Default LM config values, such as `temperature` or
            `max_tokens`.
    """

    model_type = "responses"

    def __init__(
        self,
        model: str,
        *,
        responses: Any | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self._responses = responses
        self._client = client
        self.api_key = api_key
        self.base_url = base_url

    def forward(self, request: LMRequest) -> LMResponse:
        data = self._request_kwargs(request)
        response = self._call_responses(data)
        return responses_to_lm_response(response, request)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        data = self._request_kwargs(request)
        data["stream"] = True
        yield from responses_stream_to_events(self._call_responses(data), model=request.model)

    async def aforward(self, request: LMRequest) -> LMResponse:
        return await anyio.to_thread.run_sync(self.forward, request)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        async for event in _async_iter_stream(self.forward_stream(request)):
            yield event

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        if isinstance(error, urllib.error.HTTPError):
            body = error.read().decode("utf-8", errors="replace")
            return _openai_error(error.code, body, request.model)
        return error

    def dump_state(self) -> dict[str, Any]:
        return super().dump_state()

    def _request_kwargs(self, request: LMRequest) -> dict[str, Any]:
        data = to_openai_responses_request(request)
        data["model"] = _strip_openai_provider_prefix(str(data["model"]))
        return data

    def _call_responses(self, data: dict[str, Any]) -> Any:
        if self._responses is not None:
            return _call_create_target(self._responses, data)
        if self._client is not None:
            return _call_create_target(self._client.responses, data)
        return _direct_openai_call(
            base_url=self.base_url,
            api_key=self.api_key,
            endpoint="responses",
            data=data,
            stream=bool(data.get("stream")),
        )


class OpenAICompletionsLM(LanguageModel):
    """Call OpenAI Chat or text completions with DSPy's normalized LM types.

    The default `protocol="chat"` calls an OpenAI-compatible
    `/chat/completions` endpoint. Set `protocol="text"` to call `/completions`
    with a text-completion prompt.

    Args:
        model: OpenAI model name. A leading `openai/` provider prefix is removed
            before the provider call.
        completions: Optional callable or object with `.create(**kwargs)` used
            for the completion call. This is useful in tests and custom clients.
        client: Optional OpenAI client.
        protocol: Whether to call the Chat Completions or legacy text
            completions API.
        api_key: Optional OpenAI API key used when creating a client lazily.
        base_url: Optional OpenAI-compatible base URL.
        cache: Whether DSPy request memoization is enabled by default.
        callbacks: Optional DSPy callbacks.
        **kwargs: Default LM config values, such as `temperature` or
            `max_tokens`.
    """

    def __init__(
        self,
        model: str,
        *,
        completions: Any | None = None,
        client: Any | None = None,
        protocol: CompletionProtocol = "chat",
        api_key: str | None = None,
        base_url: str | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        if protocol not in {"chat", "text"}:
            raise ValueError("protocol must be 'chat' or 'text'.")
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self.protocol = protocol
        self.model_type = protocol
        self._completions = completions
        self._client = client
        self.api_key = api_key
        self.base_url = base_url

    def forward(self, request: LMRequest) -> LMResponse:
        data = self._request_kwargs(request)
        response = self._call_completions(data)
        return completion_to_lm_response(response, request)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        data = self._request_kwargs(request)
        data["stream"] = True
        if self.protocol == "chat":
            data.setdefault("stream_options", {"include_usage": True})
        yield from completion_stream_to_events(self._call_completions(data), model=request.model)

    async def aforward(self, request: LMRequest) -> LMResponse:
        return await anyio.to_thread.run_sync(self.forward, request)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        async for event in _async_iter_stream(self.forward_stream(request)):
            yield event

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        if isinstance(error, urllib.error.HTTPError):
            body = error.read().decode("utf-8", errors="replace")
            return _openai_error(error.code, body, request.model)
        return error

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        state["protocol"] = self.protocol
        return state

    def _request_kwargs(self, request: LMRequest) -> dict[str, Any]:
        data = to_openai_text_request(request) if self.protocol == "text" else to_openai_chat_request(request)
        data["model"] = _strip_openai_provider_prefix(str(data["model"]))
        return data

    def _call_completions(self, data: dict[str, Any]) -> Any:
        if self._completions is not None:
            return _call_create_target(self._completions, data)
        if self._client is not None:
            target = self._client.completions if self.protocol == "text" else self._client.chat.completions
            return _call_create_target(target, data)
        return _direct_openai_call(
            base_url=self.base_url,
            api_key=self.api_key,
            endpoint="completions" if self.protocol == "text" else "chat/completions",
            data=data,
            stream=bool(data.get("stream")),
        )


# ---------------------------------------------------------------------------
# OpenAI stream events -> DSPy stream events
# ---------------------------------------------------------------------------


def completion_stream_to_events(stream: Iterator[Any], *, model: str) -> Iterator[LMStreamEvent]:
    """Convert an OpenAI Chat/text stream into normalized DSPy events."""
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


class _CompletionStreamState:
    """Build DSPy events from Chat/text completion chunks."""

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


def responses_stream_to_events(stream: Iterator[Any], *, model: str) -> Iterator[LMStreamEvent]:
    """Convert an OpenAI Responses stream into normalized DSPy events."""
    yield LMStreamStartEvent(model=model)
    state = _ResponsesStreamState()
    for event in stream:
        yield from state.event_to_events(event)
    yield from state.finish_events()


class _ResponsesStreamState:
    """Build DSPy events from Responses API stream events."""

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
        key = str(
            get_value(event, "item_id")
            or get_value(event, "id")
            or get_value(event, "call_id")
            or get_value(event, "output_index", "")
        )
        if not key:
            key = str(get_value(event, "output_index", 0) or 0)
        if key not in self._tool_part_by_item:
            self._tool_part_by_item[key] = self._next_tool_part_index
            self._next_tool_part_index += 1
        return self._tool_part_by_item[key]


def _request_for_response(response: Any) -> LMRequest:
    return LMRequest.from_call(model=get_value(response, "model") or "", prompt="")


async def _async_iter_stream(events: Iterator[LMStreamEvent]) -> AsyncIterator[LMStreamEvent]:
    sentinel = object()
    iterator = iter(events)
    while True:
        event = await anyio.to_thread.run_sync(_next_or_sentinel, iterator, sentinel)
        if event is sentinel:
            break
        yield event


def _next_or_sentinel(iterator: Iterator[Any], sentinel: object) -> Any:
    try:
        return next(iterator)
    except StopIteration:
        return sentinel


def _call_create_target(target: Any, data: dict[str, Any]) -> Any:
    if callable(target):
        return target(**data)
    if hasattr(target, "create"):
        return target.create(**data)
    raise TypeError("OpenAI target must be callable or expose create(**kwargs).")


def _direct_openai_call(
    *,
    base_url: str | None,
    api_key: str | None,
    endpoint: str,
    data: dict[str, Any],
    stream: bool,
) -> Any:
    request = urllib.request.Request(
        _openai_url(base_url=base_url, endpoint=endpoint),
        data=json.dumps(data).encode("utf-8"),
        headers=_openai_headers(api_key),
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=120 if stream else 60)
    if stream:
        return _iter_sse_payloads(response)
    return json.loads(response.read().decode("utf-8"))


def _openai_url(*, base_url: str | None, endpoint: str) -> str:
    return f"{(base_url or 'https://api.openai.com/v1').rstrip('/')}/{endpoint.lstrip('/')}"


def _openai_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _iter_sse_payloads(stream: Any) -> Iterator[dict[str, Any]]:
    lines: list[str] = []
    for raw_line in stream:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\n")
        if not line:
            for payload in _parse_sse_event(lines):
                yield payload
            lines = []
            continue
        lines.append(line)
    for payload in _parse_sse_event(lines):
        yield payload


def _parse_sse_event(lines: list[str]) -> Iterator[dict[str, Any]]:
    data = "\n".join(line.removeprefix("data:").strip() for line in lines if line.startswith("data:"))
    if not data or data == "[DONE]":
        return
    yield json.loads(data)


def _openai_error(status: int, body: str, model: str) -> Exception:
    from dspy.utils.exceptions import ContextWindowExceededError, LMAuthError, LMProviderError, LMRateLimitError

    try:
        data = json.loads(body)
        error = data.get("error", {}) if isinstance(data, dict) else {}
        message = str(error.get("message") or body)
    except Exception:
        message = body
    lowered = message.lower()
    if "context" in lowered or ("token" in lowered and ("limit" in lowered or "exceed" in lowered)):
        return ContextWindowExceededError(model=model, provider="openai", message=message, status=status)
    if status in {401, 403}:
        return LMAuthError(model=model, provider="openai", message=message, status=status)
    if status == 429:
        return LMRateLimitError(model=model, provider="openai", message=message, status=status)
    return LMProviderError(model=model, provider="openai", message=message, status=status)


def _strip_openai_provider_prefix(model: str) -> str:
    return model.removeprefix("openai/")


def get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
