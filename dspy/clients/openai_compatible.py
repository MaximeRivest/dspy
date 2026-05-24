"""OpenAI-compatible wire-format LM backends for normalized DSPy requests."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import anyio

from dspy.clients.base_lm import BaseLM, LMCapabilities
from dspy.clients.openai_format import (
    completion_to_lm_response,
    cost_from_response,
    responses_to_lm_response,
    to_openai_chat_request,
    to_openai_responses_request,
    to_openai_text_request,
    usage_from_response,
)
from dspy.core.types import (
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

CompletionModelType = Literal["chat", "text"]

_KNOWN_OPENAI_COMPATIBLE_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {"api_base": "https://api.openai.com/v1", "api_key_env": "OPENAI_API_KEY"},
    "groq": {"api_base": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"},
    "ollama": {"api_base": "http://localhost:11434/v1", "api_key_env": "", "default_api_key": "ollama"},
    "openrouter": {"api_base": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "fireworks": {"api_base": "https://api.fireworks.ai/inference/v1", "api_key_env": "FIREWORKS_API_KEY"},
    "fireworks_ai": {"api_base": "https://api.fireworks.ai/inference/v1", "api_key_env": "FIREWORKS_API_KEY"},
    "together": {"api_base": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "together_ai": {"api_base": "https://api.together.xyz/v1", "api_key_env": "TOGETHER_API_KEY"},
    "deepinfra": {"api_base": "https://api.deepinfra.com/v1/openai", "api_key_env": "DEEPINFRA_API_KEY"},
}

__all__ = [
    "OpenAIChatLM",
    "OpenAITextLM",
    "OpenAIResponsesLM",
    "OpenAICompatibleChatLM",
    "OpenAICompatibleTextLM",
    "OpenAICompatibleResponsesLM",
    "completion_stream_to_events",
    "responses_stream_to_events",
]


class OpenAIResponsesLM(BaseLM):
    """Call an OpenAI-compatible Responses API with DSPy's normalized LM types."""

    model_type = "responses"

    def __init__(
        self,
        model: str,
        *,
        responses: Any | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        base_url: str | None = None,
        endpoint_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        provider = _infer_openai_compatible_provider(model)
        api_base = _resolve_api_base(provider=provider, api_base=api_base, base_url=base_url)
        super().__init__(
            model=model,
            model_type="responses",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )
        self._responses = responses
        self._client = client
        self.api_key = api_key
        self.api_base = api_base
        self.endpoint_url = endpoint_url
        self.provider = provider

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(
            function_calling=True,
            reasoning=True,
            response_schema=True,
            streaming=True,
            input_image=True,
            input_audio=True,
            input_file=True,
            output_image=True,
            output_audio=True,
            tool_results=True,
        )

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
            return _openai_error(error.code, body, request.model, provider=self.provider or "openai")
        return error

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        if self.api_base is not None:
            state["api_base"] = self.api_base
        if self.endpoint_url is not None:
            state["endpoint_url"] = self.endpoint_url
        return state

    def _request_kwargs(self, request: LMRequest) -> dict[str, Any]:
        data = to_openai_responses_request(request)
        data["model"] = _provider_wire_model(str(data["model"]), self.provider)
        return data

    def _call_responses(self, data: dict[str, Any]) -> Any:
        if self._responses is not None:
            return _call_create_target(self._responses, data)
        if self._client is not None:
            return _call_create_target(self._client.responses, data)
        return _direct_openai_call(
            api_base=self.api_base,
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            provider=self.provider,
            endpoint="responses",
            data=data,
            stream=bool(data.get("stream")),
        )


class _OpenAICompletionsBase(BaseLM):
    """Shared implementation for OpenAI-compatible completion endpoints."""

    model_type: CompletionModelType

    def __init__(
        self,
        model: str,
        *,
        model_type: CompletionModelType,
        completions: Any | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        base_url: str | None = None,
        endpoint_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        if model_type not in {"chat", "text"}:
            raise ValueError("model_type must be 'chat' or 'text'.")
        provider = _infer_openai_compatible_provider(model)
        api_base = _resolve_api_base(provider=provider, api_base=api_base, base_url=base_url)
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )
        self._completions = completions
        self._client = client
        self.api_key = api_key
        self.api_base = api_base
        self.endpoint_url = endpoint_url
        self.provider = provider

    def get_capabilities(self) -> LMCapabilities:
        if self.model_type == "text":
            return LMCapabilities(streaming=True)
        return LMCapabilities(
            function_calling=True,
            reasoning=True,
            response_schema=True,
            streaming=True,
            input_image=True,
            input_audio=True,
            input_file=True,
            tool_results=True,
        )

    def forward(self, request: LMRequest) -> LMResponse:
        data = self._request_kwargs(request)
        response = self._call_completions(data)
        return completion_to_lm_response(response, request)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        data = self._request_kwargs(request)
        data["stream"] = True
        if self.model_type == "chat":
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
            return _openai_error(error.code, body, request.model, provider=self.provider or "openai")
        return error

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        if self.api_base is not None:
            state["api_base"] = self.api_base
        if self.endpoint_url is not None:
            state["endpoint_url"] = self.endpoint_url
        return state

    def _request_kwargs(self, request: LMRequest) -> dict[str, Any]:
        data = to_openai_text_request(request) if self.model_type == "text" else to_openai_chat_request(request)
        data["model"] = _provider_wire_model(str(data["model"]), self.provider)
        return data

    def _call_completions(self, data: dict[str, Any]) -> Any:
        if self._completions is not None:
            return _call_create_target(self._completions, data)
        if self._client is not None:
            target = self._client.completions if self.model_type == "text" else self._client.chat.completions
            return _call_create_target(target, data)
        return _direct_openai_call(
            api_base=self.api_base,
            endpoint_url=self.endpoint_url,
            api_key=self.api_key,
            provider=self.provider,
            endpoint="completions" if self.model_type == "text" else "chat/completions",
            data=data,
            stream=bool(data.get("stream")),
        )


class OpenAIChatLM(_OpenAICompletionsBase):
    """Call an OpenAI-compatible Chat Completions endpoint with normalized LM types."""

    def __init__(
        self,
        model: str,
        *,
        completions: Any | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        base_url: str | None = None,
        endpoint_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            model_type="chat",
            completions=completions,
            client=client,
            api_key=api_key,
            api_base=api_base,
            base_url=base_url,
            endpoint_url=endpoint_url,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )


class OpenAITextLM(_OpenAICompletionsBase):
    """Call an OpenAI-compatible legacy text Completions endpoint with normalized LM types."""

    def __init__(
        self,
        model: str,
        *,
        completions: Any | None = None,
        client: Any | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        base_url: str | None = None,
        endpoint_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 0,
        **kwargs: Any,
    ):
        super().__init__(
            model=model,
            model_type="text",
            completions=completions,
            client=client,
            api_key=api_key,
            api_base=api_base,
            base_url=base_url,
            endpoint_url=endpoint_url,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            **kwargs,
        )


OpenAICompatibleChatLM = OpenAIChatLM
OpenAICompatibleTextLM = OpenAITextLM
OpenAICompatibleResponsesLM = OpenAIResponsesLM


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
        chunk_cost = cost_from_response(chunk)
        cost = chunk_cost if chunk_cost is not None else cost
        yield from state.chunk_to_events(chunk)
    yield from state.missing_output_end_events()
    yield LMStreamEndEvent(usage=usage, cost=cost)


class _CompletionStreamState:
    """Build DSPy events from Chat/text completion chunks."""

    def __init__(self):
        self._ended_outputs: set[int] = set()
        self._seen_outputs: set[int] = {0}
        self._part_indices: dict[tuple[int, str], int] = {}
        self._next_part_index_by_output: dict[int, int] = {}

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
                    part_index=self._part_index(output_index, "reasoning"),
                    delta=LMThinkingDelta(text=str(reasoning)),
                )
            )
        content = get_value(delta, "content")
        if content:
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._part_index(output_index, "text"),
                    delta=LMTextDelta(text=str(content)),
                )
            )
        for fallback_index, tool_call in enumerate(get_value(delta, "tool_calls") or []):
            tool_index = get_value(tool_call, "index", fallback_index) or 0
            function = get_value(tool_call, "function", {})
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._part_index(output_index, f"tool:{int(tool_index)}"),
                    delta=LMToolCallDelta(
                        id=get_value(tool_call, "id"),
                        name=get_value(function, "name") or get_value(tool_call, "name"),
                        args_delta=get_value(function, "arguments") or get_value(tool_call, "arguments") or "",
                    ),
                )
            )
        return events

    def _part_index(self, output_index: int, key: str) -> int:
        compound_key = (output_index, key)
        if compound_key not in self._part_indices:
            next_index = self._next_part_index_by_output.get(output_index, 0)
            self._part_indices[compound_key] = next_index
            self._next_part_index_by_output[output_index] = next_index + 1
        return self._part_indices[compound_key]


def responses_stream_to_events(stream: Iterator[Any], *, model: str) -> Iterator[LMStreamEvent]:
    """Convert an OpenAI Responses stream into normalized DSPy events."""
    yield LMStreamStartEvent(model=model)
    state = _ResponsesStreamState()
    for event in stream:
        yield from state.event_to_events(event)
    yield from state.finish_events()


class _ResponsesStreamState:
    """Build DSPy events from Responses API stream events."""

    def __init__(self):
        self._ended = False
        self._usage = None
        self._cost = None
        self._response = None
        self._part_indices: dict[str, int] = {}
        self._next_part_index = 0

    def event_to_events(self, event: Any) -> list[LMStreamEvent]:
        event_type = get_value(event, "type")
        if event_type in {"response.output_text.delta", "output_text.delta"}:
            return [
                LMStreamDeltaEvent(
                    output_index=0,
                    part_index=self._part_index("text"),
                    delta=LMTextDelta(text=str(get_value(event, "delta", ""))),
                )
            ]
        if event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            return [
                LMStreamDeltaEvent(
                    output_index=0,
                    part_index=self._part_index("reasoning"),
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
            event_cost = cost_from_response(response)
            self._cost = event_cost if event_cost is not None else self._cost
            return [LMStreamOutputEndEvent(output_index=0)]
        if event_type in {"response.failed", "error"}:
            return [LMStreamErrorEvent(error=RuntimeError(str(get_value(event, "error", event))))]
        return []

    def finish_events(self) -> list[LMStreamEvent]:
        events: list[LMStreamEvent] = []
        if not self._ended:
            events.append(LMStreamOutputEndEvent(output_index=0))
        if self._response is not None:
            events.append(
                LMStreamEndEvent(response=responses_to_lm_response(self._response, _request_for_response(self._response)))
            )
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
        return self._part_index(f"tool:{key}")

    def _part_index(self, key: str) -> int:
        if key not in self._part_indices:
            self._part_indices[key] = self._next_part_index
            self._next_part_index += 1
        return self._part_indices[key]


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
    raise TypeError("OpenAI-compatible target must be callable or expose create(**kwargs).")


def _direct_openai_call(
    *,
    api_base: str | None,
    api_key: str | None,
    provider: str | None,
    endpoint: str,
    data: dict[str, Any],
    stream: bool,
    endpoint_url: str | None = None,
) -> Any:
    request = urllib.request.Request(
        endpoint_url or _openai_url(api_base=api_base, endpoint=endpoint),
        data=json.dumps(data).encode("utf-8"),
        headers=_openai_headers(api_key=api_key, provider=provider),
        method="POST",
    )
    response = urllib.request.urlopen(request, timeout=120 if stream else 60)
    if stream:
        return _iter_sse_payloads(response)
    return json.loads(response.read().decode("utf-8"))


def _openai_url(*, api_base: str | None, endpoint: str) -> str:
    return f"{(api_base or 'https://api.openai.com/v1').rstrip('/')}/{endpoint.lstrip('/')}"


def _infer_openai_compatible_provider(model: str) -> str | None:
    if "/" not in model:
        return "openai"
    provider = model.split("/", 1)[0]
    return provider if provider in _KNOWN_OPENAI_COMPATIBLE_PROVIDERS else None


def _resolve_api_base(*, provider: str | None, api_base: str | None, base_url: str | None) -> str | None:
    if base_url is not None:
        if api_base is not None and api_base != base_url:
            raise ValueError("Pass only one of `api_base` or deprecated `base_url`, not both.")
        warnings.warn(
            "`base_url` is deprecated for normalized OpenAI-compatible LMs; use `api_base` instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        api_base = base_url
    if api_base is not None:
        return api_base
    if provider is None:
        return None
    return _KNOWN_OPENAI_COMPATIBLE_PROVIDERS.get(provider, {}).get("api_base")


def _provider_wire_model(model: str, provider: str | None) -> str:
    if provider is not None and model.startswith(f"{provider}/"):
        return model.split("/", 1)[1]
    if provider == "openai":
        return model.removeprefix("openai/")
    return model


def _openai_headers(*, api_key: str | None, provider: str | None) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "User-Agent": "DSPy"}
    key = api_key if api_key is not None else _provider_api_key(provider)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _provider_api_key(provider: str | None) -> str | None:
    config = _KNOWN_OPENAI_COMPATIBLE_PROVIDERS.get(provider or "openai", {})
    env_name = config.get("api_key_env")
    if env_name:
        value = os.environ.get(env_name)
        if value:
            return value
    return config.get("default_api_key")


def _iter_sse_payloads(stream: Any) -> Iterator[dict[str, Any]]:
    lines: list[str] = []
    for raw_line in stream:
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else str(raw_line)
        line = line.rstrip("\n")
        if not line:
            yield from _parse_sse_event(lines)
            lines = []
            continue
        lines.append(line)
    yield from _parse_sse_event(lines)


def _parse_sse_event(lines: list[str]) -> Iterator[dict[str, Any]]:
    data = "\n".join(line.removeprefix("data:").strip() for line in lines if line.startswith("data:"))
    if not data or data == "[DONE]":
        return
    yield json.loads(data)


def _openai_error(status: int, body: str, model: str, *, provider: str) -> Exception:
    from dspy.utils.exceptions import ContextWindowExceededError, LMAuthError, LMProviderError, LMRateLimitError

    try:
        data = json.loads(body)
        error = data.get("error", {}) if isinstance(data, dict) else {}
        message = str(error.get("message") or body)
    except Exception:
        message = body
    lowered = message.lower()
    if "context" in lowered or ("token" in lowered and ("limit" in lowered or "exceed" in lowered)):
        return ContextWindowExceededError(model=model, provider=provider, message=message, status=status)
    if status in {401, 403}:
        return LMAuthError(model=model, provider=provider, message=message, status=status)
    if status == 429:
        return LMRateLimitError(model=model, provider=provider, message=message, status=status)
    return LMProviderError(model=model, provider=provider, message=message, status=status)


def get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
