"""LiteLLM-backed implementations of the normalized DSPy LM contract."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, Literal

import litellm
from litellm import ContextWindowExceededError as LitellmContextWindowExceededError

from dspy.__metadata__ import __version__
from dspy.clients.language_models.base import LMCapabilities
from dspy.clients.language_models.openai_format import (
    OpenAIChatLM,
    OpenAIResponsesLM,
    OpenAITextLM,
    _responses_api_to_lm_response,
    _usage_from_response,
)
from dspy.clients.language_models.openai_format import (
    _cost_from_response as _openai_cost_from_response,
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
from dspy.utils.exceptions import ContextWindowExceededError

logger = logging.getLogger(__name__)


class _LiteLLMTransportMixin:
    # This mixin owns LiteLLM behavior only: provider routing, retries, headers,
    # LiteLLM cache suppression, cost extraction, and LiteLLM error mapping.
    # OpenAI protocol classes still own request and response shape.
    model_type: Literal["chat", "text", "responses"]

    def __init__(
        self,
        model: str,
        *,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        num_retries: int = 3,
        **kwargs: Any,
    ):
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self.num_retries = num_retries

    @property
    def _provider_name(self) -> str:
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return "openai"

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        if isinstance(error, LitellmContextWindowExceededError):
            return ContextWindowExceededError(model=request.model, provider="litellm")
        return error

    def _litellm_cache_args(self) -> dict[str, Any]:
        return {"no-cache": True, "no-store": True}

    def _common_dump_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "cache": self.cache,
            "num_retries": self.num_retries,
            **{k: v for k, v in self.kwargs.items() if not k.startswith("api_") and k != "api_key"},
        }

    def _warn_if_truncated(self, response: LMResponse) -> None:
        if any(output.truncated for output in response.outputs):
            logger.warning(
                "LM response was truncated. Inspect the latest LM interactions with `dspy.inspect_history()` "
                "or increase `max_tokens`."
            )

    def dump_state(self) -> dict[str, Any]:
        return self._common_dump_state()

    def copy(self, **overrides: Any):
        state = self.dump_state()
        state.update(overrides)
        new_instance = type(self).load_state(state)
        new_instance.history = []
        return new_instance


class LiteLLMChatLM(_LiteLLMTransportMixin, OpenAIChatLM):
    """Call LiteLLM's chat-completions API with normalized DSPy requests."""

    # Inheritance is deliberate: OpenAIChatLM maps the protocol; this class only
    # chooses LiteLLM as the transport and asks LiteLLM for model capabilities.

    model_type: Literal["chat"] = "chat"

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(
            function_calling=_litellm_supports("supports_function_calling", self.model, provider=self._provider_name),
            reasoning=_litellm_supports("supports_reasoning", self.model, provider=self._provider_name),
            response_schema=_litellm_supports("supports_response_schema", self.model, provider=self._provider_name),
            streaming=True,
            input_image=_litellm_supports("supports_vision", self.model, provider=self._provider_name),
            input_audio=_litellm_supports("supports_audio_input", self.model, provider=self._provider_name),
            input_file=_litellm_supports("supports_pdf_input", self.model, provider=self._provider_name),
            tool_results=True,
        )

    def _to_chat_request(self, request: LMRequest) -> dict[str, Any]:
        return self._to_completion_request(request)

    def completion(self, request: dict[str, Any]) -> Any:
        return _litellm_chat_completion(request=request, num_retries=self.num_retries, cache=self._litellm_cache_args())

    async def acompletion(self, request: dict[str, Any]) -> Any:
        return await _alitellm_chat_completion(request=request, num_retries=self.num_retries, cache=self._litellm_cache_args())

    def forward(self, request: LMRequest) -> LMResponse:
        response = super().forward(request)
        response.cost = _cost_from_response(response.provider_response)
        self._warn_if_truncated(response)
        return response

    async def aforward(self, request: LMRequest) -> LMResponse:
        response = await super().aforward(request)
        response.cost = _cost_from_response(response.provider_response)
        self._warn_if_truncated(response)
        return response

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        stream = _litellm_chat_stream(request=self._to_completion_request(request), num_retries=self.num_retries, cache=self._litellm_cache_args())
        yield from _completion_stream_to_events(stream, model=request.model)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        stream = await _alitellm_chat_stream(request=self._to_completion_request(request), num_retries=self.num_retries, cache=self._litellm_cache_args())
        async for event in _acompletion_stream_to_events(stream, model=request.model):
            yield event


class LiteLLMResponsesLM(_LiteLLMTransportMixin, OpenAIResponsesLM):
    """Call LiteLLM's Responses API with normalized DSPy requests."""

    # Inheritance is deliberate: OpenAIResponsesLM maps the protocol; this class
    # only chooses LiteLLM as the transport and asks LiteLLM for capabilities.

    model_type: Literal["responses"] = "responses"

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(
            function_calling=_litellm_supports("supports_function_calling", self.model, provider=self._provider_name),
            reasoning=_litellm_supports("supports_reasoning", self.model, provider=self._provider_name),
            response_schema=_litellm_supports("supports_response_schema", self.model, provider=self._provider_name),
            streaming=True,
            input_image=_litellm_supports("supports_vision", self.model, provider=self._provider_name),
            input_file=_litellm_supports("supports_pdf_input", self.model, provider=self._provider_name),
            tool_results=True,
        )

    def responses(self, request: dict[str, Any]) -> Any:
        return _litellm_responses_completion(request=request, num_retries=self.num_retries, cache=self._litellm_cache_args())

    async def aresponses(self, request: dict[str, Any]) -> Any:
        return await _alitellm_responses_completion(request=request, num_retries=self.num_retries, cache=self._litellm_cache_args())

    def forward(self, request: LMRequest) -> LMResponse:
        response = super().forward(request)
        response.cost = _cost_from_response(response.provider_response)
        self._warn_if_truncated(response)
        return response

    async def aforward(self, request: LMRequest) -> LMResponse:
        response = await super().aforward(request)
        response.cost = _cost_from_response(response.provider_response)
        self._warn_if_truncated(response)
        return response

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        stream = _litellm_responses_stream(request=self._to_responses_request(request), num_retries=self.num_retries, cache=self._litellm_cache_args())
        yield from _responses_stream_to_events(stream, model=request.model)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        stream = await _alitellm_responses_stream(request=self._to_responses_request(request), num_retries=self.num_retries, cache=self._litellm_cache_args())
        async for event in _aresponses_stream_to_events(stream, model=request.model):
            yield event


class LiteLLMTextLM(_LiteLLMTransportMixin, OpenAITextLM):
    """Call LiteLLM's text-completion API with text-only normalized requests."""

    # Inheritance is deliberate: OpenAITextLM keeps this path text-only; this
    # class only chooses LiteLLM as the transport.

    model_type: Literal["text"] = "text"

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(streaming=True)

    def text_completion(self, request: dict[str, Any]) -> Any:
        return _litellm_text_completion(request=request, num_retries=self.num_retries, cache=self._litellm_cache_args())

    async def atext_completion(self, request: dict[str, Any]) -> Any:
        return await _alitellm_text_completion(request=request, num_retries=self.num_retries, cache=self._litellm_cache_args())

    def forward(self, request: LMRequest) -> LMResponse:
        response = super().forward(request)
        response.cost = _cost_from_response(response.provider_response)
        self._warn_if_truncated(response)
        return response

    async def aforward(self, request: LMRequest) -> LMResponse:
        response = await super().aforward(request)
        response.cost = _cost_from_response(response.provider_response)
        self._warn_if_truncated(response)
        return response

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        stream = _litellm_text_stream(request=self._to_text_request(request), num_retries=self.num_retries, cache=self._litellm_cache_args())
        yield from _completion_stream_to_events(stream, model=request.model)

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        stream = await _alitellm_text_stream(request=self._to_text_request(request), num_retries=self.num_retries, cache=self._litellm_cache_args())
        async for event in _acompletion_stream_to_events(stream, model=request.model):
            yield event


def _litellm_chat_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return litellm.completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


async def _alitellm_chat_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return await litellm.acompletion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


def _litellm_chat_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return litellm.completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


async def _alitellm_chat_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return await litellm.acompletion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


def _litellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_text_request(request)
    return litellm.text_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


async def _alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_text_request(request)
    return await litellm.atext_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


def _litellm_text_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_text_request(request)
    return litellm.text_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


async def _alitellm_text_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_text_request(request)
    return await litellm.atext_completion(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, stream_options={"include_usage": True}, **request)


def _litellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return litellm.responses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


async def _alitellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return await litellm.aresponses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, **request)


def _litellm_responses_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return litellm.responses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, **request)


async def _alitellm_responses_stream(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    request, headers = _prepare_litellm_request(request)
    return await litellm.aresponses(cache=cache or {"no-cache": True, "no-store": True}, num_retries=num_retries, retry_strategy="exponential_backoff_retry", headers=headers, stream=True, **request)


def _prepare_litellm_request(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    return request, headers


def _prepare_litellm_text_request(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare a LiteLLM text-completion request using LiteLLM's text routing convention."""
    import os

    request, headers = _prepare_litellm_request(request)
    model = str(request.pop("model"))
    parts = model.split("/", 1)
    provider, model_name = (parts[0], parts[1]) if len(parts) == 2 else ("openai", parts[0])
    request["model"] = f"text-completion-openai/{model_name}"
    request["api_key"] = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    request["api_base"] = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")
    return request, headers


def _completion_stream_to_events(stream: Iterator[Any], *, model: str) -> Iterator[LMStreamEvent]:
    yield LMStreamStartEvent(model=model)
    state = _CompletionStreamState()
    usage = None
    cost = None
    for chunk in stream:
        usage = _usage_from_response(chunk) or usage
        cost = _cost_from_response(chunk) if _cost_from_response(chunk) is not None else cost
        yield from state.chunk_to_events(chunk)
    yield from state.missing_output_end_events()
    yield LMStreamEndEvent(usage=usage, cost=cost)


async def _acompletion_stream_to_events(stream: AsyncIterator[Any], *, model: str) -> AsyncIterator[LMStreamEvent]:
    yield LMStreamStartEvent(model=model)
    state = _CompletionStreamState()
    usage = None
    cost = None
    async for chunk in stream:
        usage = _usage_from_response(chunk) or usage
        cost = _cost_from_response(chunk) if _cost_from_response(chunk) is not None else cost
        for event in state.chunk_to_events(chunk):
            yield event
    for event in state.missing_output_end_events():
        yield event
    yield LMStreamEndEvent(usage=usage, cost=cost)


class _CompletionStreamState:
    """Map OpenAI/LiteLLM completion chunks to stable normalized stream part indices."""

    _REASONING_PART_INDEX = 0
    _TEXT_PART_INDEX = 1
    _TOOL_PART_INDEX_OFFSET = 2

    def __init__(self):
        self._ended_outputs: set[int] = set()
        self._seen_outputs: set[int] = {0}

    def chunk_to_events(self, chunk: Any) -> list[LMStreamEvent]:
        events: list[LMStreamEvent] = []
        for choice in _get_value(chunk, "choices", []) or []:
            output_index = _get_value(choice, "index", 0) or 0
            self._seen_outputs.add(output_index)
            delta = _get_value(choice, "delta") or _get_value(choice, "message")
            if delta is not None:
                events.extend(self._delta_to_events(delta, output_index=output_index))
            finish_reason = _get_value(choice, "finish_reason")
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
        reasoning = _get_value(delta, "reasoning_content")
        if reasoning:
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._REASONING_PART_INDEX,
                    delta=LMThinkingDelta(text=str(reasoning)),
                )
            )
        content = _get_value(delta, "content")
        if content:
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._TEXT_PART_INDEX,
                    delta=LMTextDelta(text=str(content)),
                )
            )
        for fallback_index, tool_call in enumerate(_get_value(delta, "tool_calls") or []):
            tool_index = _get_value(tool_call, "index", fallback_index) or 0
            function = _get_value(tool_call, "function", {})
            events.append(
                LMStreamDeltaEvent(
                    output_index=output_index,
                    part_index=self._TOOL_PART_INDEX_OFFSET + int(tool_index),
                    delta=LMToolCallDelta(
                        id=_get_value(tool_call, "id"),
                        name=_get_value(function, "name") or _get_value(tool_call, "name"),
                        args_delta=_get_value(function, "arguments") or _get_value(tool_call, "arguments") or "",
                    ),
                )
            )
        return events


def _completion_chunk_to_delta_events(chunk: Any) -> list[LMStreamEvent]:
    return _CompletionStreamState().chunk_to_events(chunk)


def _responses_stream_to_events(stream: Iterator[Any], *, model: str) -> Iterator[LMStreamEvent]:
    yield LMStreamStartEvent(model=model)
    state = _ResponsesStreamState()
    for event in stream:
        yield from state.event_to_events(event)
    yield from state.finish_events()


async def _aresponses_stream_to_events(stream: AsyncIterator[Any], *, model: str) -> AsyncIterator[LMStreamEvent]:
    yield LMStreamStartEvent(model=model)
    state = _ResponsesStreamState()
    async for event in stream:
        for mapped in state.event_to_events(event):
            yield mapped
    for event in state.finish_events():
        yield event


class _ResponsesStreamState:
    """Map Responses API events to one coherent normalized output stream."""

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
        event_type = _get_value(event, "type")
        if event_type in {"response.output_text.delta", "output_text.delta"}:
            return [
                LMStreamDeltaEvent(
                    output_index=0,
                    part_index=self._TEXT_PART_INDEX,
                    delta=LMTextDelta(text=str(_get_value(event, "delta", ""))),
                )
            ]
        if event_type in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            return [
                LMStreamDeltaEvent(
                    output_index=0,
                    part_index=self._REASONING_PART_INDEX,
                    delta=LMThinkingDelta(text=str(_get_value(event, "delta", ""))),
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
                        id=_get_value(event, "call_id"),
                        name=_get_value(event, "name"),
                        args_delta=str(_get_value(event, "delta", "")),
                    ),
                )
            ]
        if event_type == "response.completed":
            self._ended = True
            response = _get_value(event, "response") or event
            self._response = response if _get_value(response, "output") is not None else None
            self._usage = _usage_from_response(response) or self._usage
            cost = _cost_from_response(response)
            self._cost = cost if cost is not None else self._cost
            return [LMStreamOutputEndEvent(output_index=0)]
        if event_type in {"response.failed", "error"}:
            return [LMStreamErrorEvent(error=RuntimeError(str(_get_value(event, "error", event))))]
        return []

    def finish_events(self) -> list[LMStreamEvent]:
        events: list[LMStreamEvent] = []
        if not self._ended:
            events.append(LMStreamOutputEndEvent(output_index=0))
        if self._response is not None:
            events.append(LMStreamEndEvent(response=_responses_api_to_lm_response(self._response, _request_for_response(self._response))))
        else:
            events.append(LMStreamEndEvent(usage=self._usage, cost=self._cost))
        return events

    def _record_output_item(self, event: Any) -> list[LMStreamEvent]:
        item = _get_value(event, "item") or event
        if _get_value(item, "type") != "function_call":
            return []
        return [
            LMStreamDeltaEvent(
                output_index=0,
                part_index=self._tool_part_index(item),
                delta=LMToolCallDelta(
                    id=_get_value(item, "call_id") or _get_value(item, "id"),
                    name=_get_value(item, "name"),
                    args_delta="",
                ),
            )
        ]

    def _tool_part_index(self, event: Any) -> int:
        key = str(_get_value(event, "item_id") or _get_value(event, "id") or _get_value(event, "call_id") or _get_value(event, "output_index", ""))
        if not key:
            key = str(_get_value(event, "output_index", 0) or 0)
        if key not in self._tool_part_by_item:
            self._tool_part_by_item[key] = self._next_tool_part_index
            self._next_tool_part_index += 1
        return self._tool_part_by_item[key]


def _responses_stream_event_to_events(event: Any) -> list[LMStreamEvent]:
    return _ResponsesStreamState().event_to_events(event)


def _request_for_response(response: Any) -> LMRequest:
    return LMRequest.from_call(model=_get_value(response, "model") or "", prompt="")


def _litellm_supports(name: str, model: str, *, provider: str) -> bool:
    checker = getattr(litellm, name, None)
    if checker is None:
        return False
    try:
        return bool(checker(model=model, custom_llm_provider=provider))
    except TypeError:
        try:
            return bool(checker(model))
        except Exception:
            return False
    except Exception:
        return False


def _cost_from_response(response: Any) -> float | None:
    return _openai_cost_from_response(response)


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"User-Agent": f"DSPy/{__version__}", **(headers or {})}


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
