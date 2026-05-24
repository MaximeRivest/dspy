from __future__ import annotations

import copy as copy_module
import datetime
import inspect
import json
import logging
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, TextIO

import anyio
from typing_extensions import Self

from dspy.core.types import (
    AsyncLMStream,
    LMAudioDelta,
    LMAudioPart,
    LMCitationDelta,
    LMCitationPart,
    LMHistoryEntry,
    LMImageDelta,
    LMImagePart,
    LMOutputBuilder,
    LMRequest,
    LMResponse,
    LMStream,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMTextPart,
    LMThinkingDelta,
    LMThinkingPart,
    LMToolCallDelta,
    LMToolCallPart,
)
from dspy.dsp.utils import settings
from dspy.utils.callback import ACTIVE_CALL_ID, with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []

logger = logging.getLogger(__name__)
_LM_MIGRATION_URL = "https://dspy.ai/migration/baselm"


@dataclass(frozen=True)
class LMCapabilities:
    """Optional model and deployment metadata for an LM backend.

    Capabilities are descriptive hints. Adapters can use them to select native
    paths, but concrete LM implementations still decide how to handle requests.
    """

    function_calling: bool = False
    reasoning: bool = False
    response_schema: bool = False
    streaming: bool = False
    input_image: bool = False
    input_audio: bool = False
    input_file: bool = False
    output_image: bool = False
    output_audio: bool = False
    tool_results: bool = False
    extensions: dict[str, Any] = field(default_factory=dict)


def _detect_contract_version(cls: type) -> int:
    """Return 1 for legacy forward(prompt, messages) and 2 for typed forward(request)."""
    fwd = None
    for klass in cls.__mro__:
        if klass is BaseLM:
            break
        if "forward" in klass.__dict__:
            fwd = klass.__dict__["forward"]
            break
    if fwd is None:
        return 2
    try:
        sig = inspect.signature(fwd)
    except (TypeError, ValueError):
        return 1

    params = [p for p in sig.parameters.values() if p.name != "self"]
    names = {p.name for p in params}
    if "prompt" in names or "messages" in names:
        return 1

    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 1:
        return 2
    return 1


class BaseLM:
    """Base class for both legacy and normalized DSPy language models.

    New implementations should override ``forward(self, request: LMRequest) ->
    LMResponse``. Existing subclasses that override ``forward(self, prompt,
    messages=None, **kwargs)`` continue to work through a compatibility path and
    keep returning legacy ``list[str | dict]`` outputs.
    """

    _lm_contract_version: int = 2
    _V1_DEFAULT_TEMPERATURE = 0.0
    _V1_DEFAULT_MAX_TOKENS = 1000
    _UNSET: Any = object()

    def __init_subclass__(cls, *, _internal: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._lm_contract_version = _detect_contract_version(cls)
        if _internal or cls.__module__.startswith("dspy."):
            return
        if cls._lm_contract_version == 1:
            warnings.warn(
                "Subclassing dspy.BaseLM with `forward(self, prompt, messages, ...)` is the legacy LM contract. "
                "The legacy signature is deprecated and will be removed in DSPy 4.0. Override "
                "`forward(self, request: LMRequest) -> LMResponse` instead. "
                f"See {_LM_MIGRATION_URL}.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __init__(
        self,
        model,
        model_type="chat",
        temperature: Any = _UNSET,
        max_tokens: Any = _UNSET,
        cache=True,
        callbacks=None,
        num_retries=0,
        **kwargs,
    ):
        is_v1 = type(self)._lm_contract_version == 1
        if temperature is BaseLM._UNSET:
            temperature = self._V1_DEFAULT_TEMPERATURE if is_v1 else None
        if max_tokens is BaseLM._UNSET:
            max_tokens = self._V1_DEFAULT_MAX_TOKENS if is_v1 else None

        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.callbacks = callbacks or []
        self.num_retries = num_retries
        self.kwargs = _default_lm_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self._warned_zero_temp_rollout = False

    @property
    def capabilities(self) -> LMCapabilities:
        """Native metadata available for this model instance."""
        return self.get_capabilities()

    def get_capabilities(self) -> LMCapabilities:
        """Return optional native model and deployment hints."""
        return LMCapabilities()

    @property
    def supports_function_calling(self) -> bool:
        """Whether the model supports function calling (tool use)."""
        return self.capabilities.function_calling

    @property
    def supports_reasoning(self) -> bool:
        """Whether the model supports native reasoning (extended thinking)."""
        return self.capabilities.reasoning

    @property
    def supports_response_schema(self) -> bool:
        """Whether the model supports structured output via response schema."""
        return self.capabilities.response_schema

    @property
    def supported_params(self) -> set[str]:
        """Set of supported OpenAI-style parameter names for the model."""
        supported = self.capabilities.extensions.get("supported_params", set())
        return set(supported) if supported else set()

    def __call__(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs,
    ):
        if request is not None or self._lm_contract_version == 2:
            return self._v2_call(*items, prompt=prompt, messages=messages, request=request, **kwargs)

        if items and len(items) == 1 and isinstance(items[0], str) and prompt is None:
            prompt = items[0]
            items = ()
        if items:
            raise TypeError(
                f"{type(self).__name__} uses the legacy v1 LM contract; positional content items require a v2 "
                "subclass. Use prompt= or messages=."
            )
        return self._v1_call(prompt=prompt, messages=messages, **kwargs)

    async def acall(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs,
    ):
        if request is not None or self._lm_contract_version == 2:
            return await self._v2_acall(*items, prompt=prompt, messages=messages, request=request, **kwargs)

        if items and len(items) == 1 and isinstance(items[0], str) and prompt is None:
            prompt = items[0]
            items = ()
        if items:
            raise TypeError(
                f"{type(self).__name__} uses the legacy v1 LM contract; positional content items require a v2 "
                "subclass. Use prompt= or messages=."
            )
        return await self._v1_acall(prompt=prompt, messages=messages, **kwargs)

    @with_callbacks
    def _v1_call(self, prompt=None, messages=None, **kwargs):
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process_lm_response(response, prompt, messages, **kwargs)

    @with_callbacks
    async def _v1_acall(self, prompt=None, messages=None, **kwargs):
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process_lm_response(response, prompt, messages, **kwargs)

    def _v2_call(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs,
    ) -> LMResponse:
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        call_id = self._start_lm_callbacks(
            callbacks,
            request=normalized_request,
            raw_inputs=self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs),
        )
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        result = None
        exception = None
        try:
            response = self._forward_with_retry(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def _v2_acall(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs,
    ) -> LMResponse:
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        call_id = self._start_lm_callbacks(
            callbacks,
            request=normalized_request,
            raw_inputs=self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs),
        )
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        result = None
        exception = None
        try:
            response = await self._aforward_with_retry(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    def forward(self, request: LMRequest) -> LMResponse:
        """Run one normalized language model request."""
        raise NotImplementedError(f"{type(self).__name__} must implement forward(request).")

    async def aforward(self, request: LMRequest) -> LMResponse:
        """Run one normalized language model request asynchronously."""
        raise NotImplementedError("Subclasses must implement aforward(request) for async calls.")

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        """Run one normalized language model request as a stream of events."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming.")

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        """Run one normalized language model request as an async stream of events."""
        raise NotImplementedError(f"{type(self).__name__} does not support async streaming.")

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        """Map a provider exception to a DSPy exception."""
        return error

    def normalize_request(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMRequest:
        if request is None and items and isinstance(items[0], LMRequest):
            request = items[0]
            items = items[1:]

        if request is not None:
            if prompt is not None or messages is not None or items:
                raise ValueError(
                    "Pass either an LMRequest or direct-call inputs, not both. Use call kwargs to override request "
                    "config."
                )
            normalized = self._override_request(request, **kwargs)
            self._warn_zero_temp_rollout(normalized)
            return normalized

        merged_kwargs = {**self.kwargs, **kwargs}
        merged_kwargs.setdefault("cache", self.cache)
        normalized = LMRequest.from_call(
            model=self.model,
            items=items,
            prompt=prompt,
            messages=messages,
            **merged_kwargs,
        )
        self._warn_zero_temp_rollout(normalized)
        return normalized

    def stream(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMStream:
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        events = self._cached_stream_events(normalized_request, mode="stream")
        if events is None:
            try:
                self._require_stream_support(async_=False)
            except Exception as error:
                self._observe_failed_stream_construction(
                    normalized_request,
                    error,
                    callbacks=callbacks,
                    raw_inputs=raw_inputs,
                )
                raise
            events = self._cache_wrapped_stream_events(
                normalized_request,
                self.forward_stream(normalized_request),
                mode="stream",
            )
        return LMStream(
            request=normalized_request,
            events=self._callback_wrapped_stream_events(
                normalized_request,
                events,
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    def astream(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> AsyncLMStream:
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        events = self._cached_astream_events(normalized_request, mode="astream")
        if events is None:
            try:
                self._require_stream_support(async_=True)
            except Exception as error:
                self._observe_failed_stream_construction(
                    normalized_request,
                    error,
                    callbacks=callbacks,
                    raw_inputs=raw_inputs,
                )
                raise
            events = self._cache_wrapped_astream_events(
                normalized_request,
                self.aforward_stream(normalized_request),
                mode="astream",
            )
        return AsyncLMStream(
            request=normalized_request,
            events=self._callback_wrapped_astream_events(
                normalized_request,
                events,
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    def dump_state(self) -> dict[str, Any]:
        """Return a sanitized reconstruction state for this LM."""
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key != "api_key"}
        state = {
            "model": self.model,
            "cache": self.cache,
            "num_retries": self.num_retries,
            **filtered_kwargs,
        }
        if self._lm_contract_version == 1:
            state["model_type"] = self.model_type
        return state

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Self:
        """Reconstruct this LM from `dump_state()` output."""
        return cls(**state)

    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters."""
        if self._lm_contract_version == 1:
            new_instance = copy_module.deepcopy(self)
            new_instance.history = []
            for key, value in kwargs.items():
                if hasattr(self, key):
                    setattr(new_instance, key, value)
                if (key in self.kwargs) or (not hasattr(self, key)):
                    if value is None:
                        new_instance.kwargs.pop(key, None)
                    else:
                        new_instance.kwargs[key] = value
            if hasattr(new_instance, "_warned_zero_temp_rollout"):
                new_instance._warned_zero_temp_rollout = False
            return new_instance

        new_instance = copy_module.copy(self)
        new_instance.history = []
        new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
        new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})
        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if key in new_instance.kwargs or not hasattr(self, key):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        if hasattr(new_instance, "_warned_zero_temp_rollout"):
            new_instance._warned_zero_temp_rollout = False
        return new_instance

    def inspect_history(self, n: int = 1, file: TextIO | None = None) -> None:
        pretty_print_history(self.history, n, file=file)

    def update_history(self, entry):
        if settings.disable_history:
            return

        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)
        GLOBAL_HISTORY.append(entry)

        if settings.max_history_size != 0:
            if len(self.history) >= settings.max_history_size:
                self.history.pop(0)
            self.history.append(entry)

        for module in settings.caller_modules or []:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)

    def _get_active_callbacks(self) -> list[Any]:
        return list(settings.get("callbacks", []) or []) + list(getattr(self, "callbacks", []) or [])

    def _raw_callback_inputs(
        self,
        *,
        items: tuple[Any, ...],
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        return _sanitize_callback_value({"items": items, "prompt": prompt, "messages": messages, "kwargs": kwargs})

    def _start_lm_callbacks(
        self,
        callbacks: list[Any],
        *,
        request: LMRequest,
        raw_inputs: dict[str, Any],
    ) -> str | None:
        if not callbacks:
            return None

        call_id = uuid.uuid4().hex
        inputs = {"request": _sanitize_lm_request_for_callbacks(request), "raw": raw_inputs}
        for callback in callbacks:
            try:
                callback.on_lm_start(call_id=call_id, instance=self, inputs=inputs)
            except Exception as error:
                logger.warning("Error when calling callback %s: %s", callback, error)
        return call_id

    def _end_lm_callbacks(
        self,
        callbacks: list[Any],
        *,
        call_id: str | None,
        outputs: LMResponse | None,
        exception: Exception | None,
    ) -> None:
        if not callbacks or call_id is None:
            return
        for callback in callbacks:
            try:
                callback.on_lm_end(call_id=call_id, outputs=outputs, exception=exception)
            except Exception as error:
                logger.warning("Error when applying callback %s's LM end handler: %s", callback, error)

    def _observe_failed_stream_construction(
        self,
        request: LMRequest,
        error: Exception,
        *,
        callbacks: list[Any],
        raw_inputs: dict[str, Any],
    ) -> None:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        try:
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=None, exception=error)
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)

    def _callback_wrapped_stream_events(
        self,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        *,
        callbacks: list[Any],
        raw_inputs: dict[str, Any],
    ) -> Iterator[LMStreamEvent]:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        builder = LMOutputBuilder()
        result = None
        exception = None
        try:
            for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def _callback_wrapped_astream_events(
        self,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        *,
        callbacks: list[Any],
        raw_inputs: dict[str, Any],
    ) -> AsyncIterator[LMStreamEvent]:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        builder = LMOutputBuilder()
        result = None
        exception = None
        try:
            async for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    def _forward_with_retry(self, request: LMRequest) -> LMResponse:
        attempts = max(0, int(getattr(self, "num_retries", 0) or 0)) + 1
        for attempt in range(attempts):
            try:
                return self._forward_with_cache(request)
            except Exception as error:
                normalized_error = self.normalize_error(error, request)
                if attempt >= attempts - 1 or not _is_retryable_lm_error(normalized_error):
                    if normalized_error is error:
                        raise
                    raise normalized_error from error
                _sleep_before_retry(attempt)
        raise RuntimeError("unreachable")

    async def _aforward_with_retry(self, request: LMRequest) -> LMResponse:
        attempts = max(0, int(getattr(self, "num_retries", 0) or 0)) + 1
        for attempt in range(attempts):
            try:
                return await self._aforward_with_cache(request)
            except Exception as error:
                normalized_error = self.normalize_error(error, request)
                if attempt >= attempts - 1 or not _is_retryable_lm_error(normalized_error):
                    if normalized_error is error:
                        raise
                    raise normalized_error from error
                await _asleep_before_retry(attempt)
        raise RuntimeError("unreachable")

    def _forward_with_cache(self, request: LMRequest) -> LMResponse:
        if not _request_cache_enabled(request, self.cache):
            return self.forward(request)
        response = _cached_base_lm_forward(
            cache_request=self._cache_request_for_mode(request, mode="sync"),
            lm=self,
            request=request,
        )
        return _prepare_cached_lm_response(response)

    async def _aforward_with_cache(self, request: LMRequest) -> LMResponse:
        if not _request_cache_enabled(request, self.cache):
            return await self.aforward(request)
        response = await _cached_base_lm_aforward(
            cache_request=self._cache_request_for_mode(request, mode="async"),
            lm=self,
            request=request,
        )
        return _prepare_cached_lm_response(response)

    def _cache_request(self, request: LMRequest) -> dict[str, Any]:
        return {
            "lm_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "lm_state": _sanitize_cache_value(self.dump_state()),
            "request": _sanitize_cache_value(_model_dump_for_cache(request)),
        }

    def _cache_request_for_mode(self, request: LMRequest, *, mode: str) -> dict[str, Any]:
        cache_request = self._cache_request(request)
        cache_request["execution_mode"] = mode
        return cache_request

    def _cached_stream_events(self, request: LMRequest, *, mode: str) -> Iterator[LMStreamEvent] | None:
        if not _request_cache_enabled(request, self.cache):
            return None
        cached = _get_cached_lm_response(self._cache_request_for_mode(request, mode=mode))
        if cached is None:
            return None
        return _response_to_stream_events(_prepare_cached_lm_response(cached), model=request.model)

    def _cache_wrapped_stream_events(
        self,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        *,
        mode: str,
    ) -> Iterator[LMStreamEvent]:
        if not _request_cache_enabled(request, self.cache):
            yield from events
            return

        builder = LMOutputBuilder()
        for event in events:
            built = builder.apply(event)
            if built is not None:
                _put_cached_lm_response(self._cache_request_for_mode(request, mode=mode), built)
            yield event

    def _cached_astream_events(self, request: LMRequest, *, mode: str) -> AsyncIterator[LMStreamEvent] | None:
        if not _request_cache_enabled(request, self.cache):
            return None
        cached = _get_cached_lm_response(self._cache_request_for_mode(request, mode=mode))
        if cached is None:
            return None
        return _async_iter(_response_to_stream_events(_prepare_cached_lm_response(cached), model=request.model))

    async def _cache_wrapped_astream_events(
        self,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        *,
        mode: str,
    ) -> AsyncIterator[LMStreamEvent]:
        if not _request_cache_enabled(request, self.cache):
            async for event in events:
                yield event
            return

        builder = LMOutputBuilder()
        async for event in events:
            built = builder.apply(event)
            if built is not None:
                _put_cached_lm_response(self._cache_request_for_mode(request, mode=mode), built)
            yield event

    def _warn_zero_temp_rollout(self, request: LMRequest) -> None:
        cache = getattr(getattr(request, "config", None), "cache", None)
        rollout_id = getattr(cache, "rollout_id", None)
        temperature = getattr(getattr(request, "config", None), "temperature", None)
        if self._warned_zero_temp_rollout or rollout_id is None or temperature != 0:
            return
        warnings.warn(
            "rollout_id only affects DSPy's request cache when temperature=0; set temperature>0 to request a "
            "potentially different provider output.",
            UserWarning,
            stacklevel=3,
        )
        self._warned_zero_temp_rollout = True

    def _override_request(self, request: LMRequest, **kwargs: Any) -> LMRequest:
        if not kwargs:
            return request
        return request.with_config_overrides(**kwargs)

    def _require_stream_support(self, *, async_: bool) -> None:
        method_name = "aforward_stream" if async_ else "forward_stream"
        if self._method_overridden(method_name):
            return
        name = "async streaming" if async_ else "streaming"
        raise NotImplementedError(f"{type(self).__name__} does not support {name}; {method_name}() is not overridden.")

    def _finalize_response(self, request: LMRequest, response: LMResponse) -> LMResponse:
        self._track_usage(response)

        if not settings.disable_history:
            entry = LMHistoryEntry(
                request=request,
                response=response,
                timestamp=datetime.datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                model_type=getattr(self, "model_type", None),
            )
            self.update_history(entry)
        return response

    def _normalize_and_observe_error(self, error: Exception, request: LMRequest) -> Exception:
        return self.normalize_error(error, request)

    def _track_usage(self, response: LMResponse) -> None:
        if getattr(response, "cache_hit", False):
            return
        if not settings.usage_tracker:
            return
        usage = _response_usage_as_dict(response)
        if usage:
            settings.usage_tracker.add_usage(self.model, usage)

    def _method_overridden(self, method_name: str) -> bool:
        method = getattr(type(self), method_name, None)
        base_method = getattr(BaseLM, method_name, None)
        return method is not None and base_method is not None and method is not base_method

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "responses":
            outputs = self._process_response(response)
        else:
            outputs = self._process_completion(response, merged_kwargs)

        if settings.disable_history:
            return outputs

        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": response,
            "outputs": outputs,
            "usage": dict(getattr(response, "usage", {}) or {}),
            "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "response_model": getattr(response, "model", None),
            "model_type": self.model_type,
        }
        self.update_history(entry)
        return outputs

    def _process_completion(self, response, merged_kwargs):
        outputs = []
        for c in response.choices:
            output = {}
            output["text"] = c.message.content if hasattr(c, "message") else c["text"]

            if hasattr(c, "message") and hasattr(c.message, "reasoning_content") and c.message.reasoning_content:
                output["reasoning_content"] = c.message.reasoning_content

            if merged_kwargs.get("logprobs"):
                output["logprobs"] = c.logprobs if hasattr(c, "logprobs") else c["logprobs"]
            if hasattr(c, "message") and getattr(c.message, "tool_calls", None):
                output["tool_calls"] = c.message.tool_calls

            citations = self._extract_citations_from_response(c)
            if citations:
                output["citations"] = citations
            outputs.append(output)

        if all(len(output) == 1 for output in outputs):
            outputs = [output["text"] for output in outputs]
        return outputs

    def _extract_citations_from_response(self, choice):
        try:
            citations_data = choice.message.provider_specific_fields.get("citations")
            if isinstance(citations_data, list):
                return [citation for citations in citations_data for citation in citations]
        except Exception:
            return None

    def _process_response(self, response):
        text_outputs = []
        tool_calls = []
        reasoning_contents = []

        for output_item in response.output:
            output_item_type = output_item.type
            if output_item_type == "message":
                for content_item in output_item.content:
                    text_outputs.append(content_item.text)
            elif output_item_type == "function_call":
                tool_calls.append(output_item.model_dump())
            elif output_item_type == "reasoning":
                if getattr(output_item, "content", None) and len(output_item.content) > 0:
                    for content_item in output_item.content:
                        reasoning_contents.append(content_item.text)
                elif getattr(output_item, "summary", None) and len(output_item.summary) > 0:
                    for summary_item in output_item.summary:
                        reasoning_contents.append(summary_item.text)

        result = {}
        if len(text_outputs) > 0:
            result["text"] = "".join(text_outputs)
        if len(tool_calls) > 0:
            result["tool_calls"] = tool_calls
        if len(reasoning_contents) > 0:
            result["reasoning_content"] = "".join(reasoning_contents)
        return [result]


def inspect_history(n: int = 1, file: TextIO | None = None) -> None:
    """The global history shared across all LMs."""
    pretty_print_history(GLOBAL_HISTORY, n, file=file)


def _default_lm_kwargs(
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    defaults = dict(kwargs)
    if temperature is not None:
        defaults["temperature"] = temperature
    if max_tokens is not None:
        defaults["max_tokens"] = max_tokens
    return defaults


def _prepare_cached_lm_response(response: Any) -> Any:
    if getattr(response, "cache_hit", False) and hasattr(response, "cost"):
        response.cost = None
    return response


def _is_retryable_lm_error(error: Exception) -> bool:
    try:
        from dspy.utils.exceptions import RETRYABLE_LM_ERRORS, LMProviderError
    except Exception:
        return False
    if isinstance(error, RETRYABLE_LM_ERRORS):
        return True
    if isinstance(error, LMProviderError):
        status = getattr(error, "status", None)
        return status is None or int(status) >= 500
    return False


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(2**attempt, 8))


async def _asleep_before_retry(attempt: int) -> None:
    await anyio.sleep(min(2**attempt, 8))


def _cached_base_lm_forward(cache_request: dict[str, Any], lm: BaseLM, request: LMRequest) -> LMResponse:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    def run(cache_request: dict[str, Any], lm: BaseLM, request: LMRequest) -> LMResponse:
        return lm.forward(request)

    return run(cache_request=cache_request, lm=lm, request=request)


async def _cached_base_lm_aforward(cache_request: dict[str, Any], lm: BaseLM, request: LMRequest) -> LMResponse:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    async def run(cache_request: dict[str, Any], lm: BaseLM, request: LMRequest) -> LMResponse:
        return await lm.aforward(request)

    return await run(cache_request=cache_request, lm=lm, request=request)


def _get_cached_lm_response(cache_request: dict[str, Any]) -> Any:
    import dspy

    return dspy.cache.get(_stream_cache_key(cache_request))


def _put_cached_lm_response(cache_request: dict[str, Any], response: Any) -> None:
    import dspy

    dspy.cache.put(_stream_cache_key(cache_request), response)


def _stream_cache_key(cache_request: dict[str, Any]) -> dict[str, Any]:
    return {**cache_request, "_fn_identifier": "dspy.BaseLM.stream"}


def _response_to_stream_events(response: LMResponse, *, model: str | None = None) -> Iterator[LMStreamEvent]:
    yield LMStreamStartEvent(model=response.model or model)
    for output_index, output in enumerate(response.outputs):
        for part_index, part in enumerate(output.parts):
            delta = _part_to_stream_delta(part)
            if delta is not None:
                yield LMStreamDeltaEvent(output_index=output_index, part_index=part_index, delta=delta)
        yield LMStreamOutputEndEvent(
            output_index=output_index,
            finish_reason=output.finish_reason,
            truncated=output.truncated,
        )
    yield LMStreamEndEvent(response=response)


def _part_to_stream_delta(part: Any) -> Any | None:
    if isinstance(part, LMTextPart):
        return LMTextDelta(text=part.text)
    if isinstance(part, LMThinkingPart):
        return LMThinkingDelta(text=part.text)
    if isinstance(part, LMToolCallPart):
        return LMToolCallDelta(id=part.id, name=part.name, args_delta=json.dumps(part.args))
    if isinstance(part, LMCitationPart):
        return LMCitationDelta(citation=part)
    if isinstance(part, LMImagePart):
        return LMImageDelta(image=part)
    if isinstance(part, LMAudioPart):
        return LMAudioDelta(audio=part)
    return None


async def _async_iter(events: Iterator[Any]) -> AsyncIterator[Any]:
    for event in events:
        yield event


def _request_cache_enabled(request: LMRequest, default: bool) -> bool:
    cache = getattr(request.config, "cache", None)
    enabled = getattr(cache, "enabled", None)
    return default if enabled is None else bool(enabled)


def _model_dump_for_cache(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _sanitize_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if key_text in {"api_key", "authorization", "x_api_key"}:
                continue
            sanitized[key] = _sanitize_cache_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_cache_value(item) for item in value]
    return value


def _response_usage_as_dict(response: LMResponse) -> dict[str, Any]:
    if hasattr(response, "usage_as_dict"):
        return response.usage_as_dict()
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage)


def _sanitize_lm_request_for_callbacks(request: LMRequest) -> Any:
    config = request.config.model_copy(
        update={"extensions": _sanitize_callback_value(getattr(request.config, "extensions", {}) or {})},
        deep=True,
    )
    return request.model_copy(
        update={
            "config": config,
            "metadata": _sanitize_callback_value(getattr(request, "metadata", {}) or {}),
        },
        deep=True,
    )


def _sanitize_callback_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if key_text == "api_key" or key_text in {"authorization", "x_api_key"}:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_callback_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_callback_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_callback_value(item) for item in value]
    return value
