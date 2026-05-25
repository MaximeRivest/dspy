from __future__ import annotations

import copy as copy_module
import datetime
import importlib
import inspect
import logging
import time
import uuid
import warnings
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, TextIO

import anyio

from dspy.dsp.utils import settings
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback
from dspy.utils.exceptions import RETRYABLE_LM_ERRORS, LMError, LMProviderError
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []
LM_CLASS_STATE_KEY = "_dspy_lm_class"
_BUILTIN_LM_CLASS_PATH = "dspy.clients.lm.LM"
_LM_MIGRATION_URL = "https://dspy.ai/migration/baselm"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LMCapabilities:
    """Capabilities that DSPy adapters use when formatting LM requests.

    Custom `BaseLM` subclasses can return this from `get_capabilities()`.
    The legacy `supports_function_calling`, `supports_reasoning`, and
    `supports_response_schema` properties read from these fields.
    """

    function_calling: bool = False
    reasoning: bool = False
    response_schema: bool = False


def _import_lm_class(class_path: str) -> type:
    parts = class_path.split(".")
    last_error = None

    for split_index in range(len(parts) - 1, 0, -1):
        module_name = ".".join(parts[:split_index])
        try:
            obj = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name or module_name.startswith(f"{exc.name}."):
                last_error = exc
                continue
            raise

        try:
            for attr in parts[split_index:]:
                obj = getattr(obj, attr)
        except AttributeError as exc:
            last_error = exc
            continue

        if not isinstance(obj, type):
            raise TypeError(f"Serialized LM class `{class_path}` did not resolve to a class.")
        return obj

    raise ImportError(f"Could not import serialized LM class `{class_path}`.") from last_error


def _detect_contract_version(cls: type) -> int:
    """Return 1 for legacy forward(prompt, messages), 2 for forward(request)."""
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
    """Base class for DSPy language model backends.

    DSPy 3.3 supports two custom-LM contracts during the migration to the typed
    LM boundary.

    New subclasses should implement::

        forward(self, request: dspy.LMRequest) -> dspy.LMResponse

    In this contract, `BaseLM` normalizes direct-call inputs into an
    `LMRequest`, handles retries, non-streaming request caching, callbacks,
    history, usage tracking, and secret redaction, and validates that
    `forward()` returns an `LMResponse`.

    Legacy subclasses may still implement::

        forward(self, prompt=None, messages=None, **kwargs)

    and return an OpenAI-shaped provider response. Legacy direct calls keep
    returning `list[str | dict]`; normalized calls with `request=` return an
    `LMResponse` for both legacy and new subclasses. DSPy 3.3 does not warn for
    legacy subclasses yet, but they should migrate before the legacy contract is
    removed in a later release.
    """

    _lm_contract_version: int = 2

    def __init_subclass__(cls, *, _internal: bool = False, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._lm_contract_version = _detect_contract_version(cls)
        # DSPy 3.3 recognizes legacy subclasses but does not warn yet. The
        # migration warning is planned for DSPy 3.4; keep detection centralized
        # here so that release can add the warning without changing call paths.

    def __init__(
        self,
        model,
        model_type="chat",
        temperature=None,
        max_tokens=None,
        cache=True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        **kwargs,
    ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.callbacks = list(callbacks or [])
        self.num_retries = num_retries
        self.kwargs = self._get_initial_kwargs(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []
        self._warned_zero_temp_rollout = False

    def _get_initial_kwargs(self, *, temperature, max_tokens, **kwargs) -> dict[str, Any]:
        return dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

    # ------------------------------------------------------------------
    # Capabilities and compatibility properties
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> LMCapabilities:
        return self.get_capabilities()

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities()

    @property
    def supports_function_calling(self) -> bool:
        caps = getattr(self, "capabilities", None)
        return bool(caps and caps.function_calling)

    @property
    def supports_reasoning(self) -> bool:
        caps = getattr(self, "capabilities", None)
        return bool(caps and caps.reasoning)

    @property
    def supports_response_schema(self) -> bool:
        caps = getattr(self, "capabilities", None)
        return bool(caps and caps.response_schema)

    @property
    def supports_streaming(self) -> bool:
        return self._method_overridden("forward_stream")

    @property
    def supports_async(self) -> bool:
        return self._method_overridden("aforward")

    @property
    def supported_params(self) -> set[str]:
        return set()

    # ------------------------------------------------------------------
    # Core hooks for normalized subclasses
    # ------------------------------------------------------------------

    def forward(self, request: Any):
        raise NotImplementedError(f"{type(self).__name__} must implement forward(request).")

    async def aforward(self, request: Any):
        raise NotImplementedError(f"{type(self).__name__} must implement aforward(request) for async calls.")

    def forward_stream(self, request: Any) -> Iterator[Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support streaming.")

    async def aforward_stream(self, request: Any) -> AsyncIterator[Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support async streaming.")

    def normalize_error(self, error: Exception, request: Any) -> Exception:
        return error

    # ------------------------------------------------------------------
    # Public call API
    # ------------------------------------------------------------------

    def __call__(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: Any = None,
        **kwargs,
    ):
        if request is None and items and _is_lm_request(items[0]):
            request = items[0]
            items = items[1:]

        if settings.experimental:
            if request is not None or self._lm_contract_version == 2:
                return self._normalized_call(*items, prompt=prompt, messages=messages, request=request, **kwargs)

        elif request is not None or self._lm_contract_version == 2:
            warnings.warn(
                "The typed BaseLM path is experimental in DSPy 3.3. Use dspy.configure(experimental=True) "
                "or dspy.context(experimental=True) to opt in.",
                UserWarning,
                stacklevel=2,
            )
            raise ValueError("Typed BaseLM calls require experimental=True in DSPy 3.3.")

        # v1 backward compatibility: `lm("text")` historically meant prompt="text".
        if items and len(items) == 1 and isinstance(items[0], str) and prompt is None:
            prompt = items[0]
            items = ()
        if items:
            raise TypeError(
                f"{type(self).__name__} uses the legacy v1 LM contract; positional content items require "
                "the experimental typed LM path. Use prompt= or messages=, or opt in with experimental=True."
            )
        return self._legacy_callback_call(prompt=prompt, messages=messages, **kwargs)

    async def acall(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: Any = None,
        **kwargs,
    ):
        if request is None and items and _is_lm_request(items[0]):
            request = items[0]
            items = items[1:]

        if settings.experimental:
            if request is not None or self._lm_contract_version == 2:
                return await self._normalized_acall(*items, prompt=prompt, messages=messages, request=request, **kwargs)

        elif request is not None or self._lm_contract_version == 2:
            warnings.warn(
                "The typed BaseLM path is experimental in DSPy 3.3. Use dspy.configure(experimental=True) "
                "or dspy.context(experimental=True) to opt in.",
                UserWarning,
                stacklevel=2,
            )
            raise ValueError("Typed BaseLM calls require experimental=True in DSPy 3.3.")

        if items and len(items) == 1 and isinstance(items[0], str) and prompt is None:
            prompt = items[0]
            items = ()
        if items:
            raise TypeError(
                f"{type(self).__name__} uses the legacy v1 LM contract; positional content items require "
                "the experimental typed LM path. Use prompt= or messages=, or opt in with experimental=True."
            )
        return await self._legacy_callback_acall(prompt=prompt, messages=messages, **kwargs)

    def _normalized_call(self, *items: Any, prompt=None, messages=None, request=None, **kwargs: Any):
        normalized_request = self.normalize_request(*items, prompt=prompt, messages=messages, request=request, **kwargs)
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
            if self._lm_contract_version == 1:
                response = self._legacy_request_forward(normalized_request)
            else:
                response = self._forward_with_retry(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def _normalized_acall(self, *items: Any, prompt=None, messages=None, request=None, **kwargs: Any):
        normalized_request = self.normalize_request(*items, prompt=prompt, messages=messages, request=request, **kwargs)
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
            if self._lm_contract_version == 1:
                response = await self._legacy_request_aforward(normalized_request)
            else:
                response = await self._aforward_with_retry(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    def normalize_request(self, *items: Any, prompt=None, messages=None, request=None, **kwargs: Any):
        lm_types = _import_lm_types()
        request_cls = lm_types.LMRequest

        if request is None and items and isinstance(items[0], request_cls):
            request = items[0]
            items = items[1:]

        if request is not None:
            if prompt is not None or messages is not None or items:
                raise ValueError("Pass either an LMRequest or direct-call inputs, not both. Use call kwargs to override request config.")
            normalized = self._override_request(request, **kwargs)
            self._warn_zero_temp_rollout_for_request(normalized)
            return normalized

        merged_kwargs = {**self.kwargs, **kwargs}
        merged_kwargs.setdefault("cache", self.cache)
        normalized = request_cls.from_call(model=self.model, items=items, prompt=prompt, messages=messages, **merged_kwargs)
        self._warn_zero_temp_rollout_for_request(normalized)
        return normalized

    def stream(self, *items: Any, prompt=None, messages=None, request=None, **kwargs: Any):
        if not settings.experimental:
            warnings.warn(
                "BaseLM.stream() is experimental in DSPy 3.3. Use dspy.configure(experimental=True) "
                "or dspy.context(experimental=True) to opt in.",
                UserWarning,
                stacklevel=2,
            )
            raise ValueError("BaseLM.stream() requires experimental=True in DSPy 3.3.")
        normalized_request = self.normalize_request(*items, prompt=prompt, messages=messages, request=request, **kwargs)
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        try:
            self._require_stream_support(async_=False)
        except Exception as error:
            self._observe_failed_stream_construction(normalized_request, error, callbacks=callbacks, raw_inputs=raw_inputs)
            raise
        return _import_lm_types().LMStream(
            request=normalized_request,
            events=self._callback_wrapped_stream_events(
                normalized_request,
                self.forward_stream(normalized_request),
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    def astream(self, *items: Any, prompt=None, messages=None, request=None, **kwargs: Any):
        if not settings.experimental:
            warnings.warn(
                "BaseLM.astream() is experimental in DSPy 3.3. Use dspy.configure(experimental=True) "
                "or dspy.context(experimental=True) to opt in.",
                UserWarning,
                stacklevel=2,
            )
            raise ValueError("BaseLM.astream() requires experimental=True in DSPy 3.3.")
        normalized_request = self.normalize_request(*items, prompt=prompt, messages=messages, request=request, **kwargs)
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        try:
            self._require_stream_support(async_=True)
        except Exception as error:
            self._observe_failed_stream_construction(normalized_request, error, callbacks=callbacks, raw_inputs=raw_inputs)
            raise
        return _import_lm_types().AsyncLMStream(
            request=normalized_request,
            events=self._callback_wrapped_astream_events(
                normalized_request,
                self.aforward_stream(normalized_request),
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    # ------------------------------------------------------------------
    # Legacy v1 machinery
    # ------------------------------------------------------------------

    def _legacy_callback_call(self, prompt=None, messages=None, **kwargs):
        callbacks = self._get_active_callbacks()
        call_id = self._start_legacy_callbacks(callbacks, prompt=prompt, messages=messages, kwargs=kwargs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        result = None
        exception = None
        try:
            response = self.forward(prompt=prompt, messages=messages, **kwargs)
            result = self._process_lm_response(response, prompt, messages, **kwargs)
            return result
        except Exception as error:
            exception = error
            raise
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def _legacy_callback_acall(self, prompt=None, messages=None, **kwargs):
        callbacks = self._get_active_callbacks()
        call_id = self._start_legacy_callbacks(callbacks, prompt=prompt, messages=messages, kwargs=kwargs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        result = None
        exception = None
        try:
            response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
            result = self._process_lm_response(response, prompt, messages, **kwargs)
            return result
        except Exception as error:
            exception = error
            raise
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    def _legacy_request_forward(self, request: Any):
        data = self._legacy_call_kwargs(request)
        prompt = data.pop("prompt", None)
        messages = data.pop("messages", None)
        response = self.forward(prompt=prompt, messages=messages, **data)
        return self._legacy_provider_response_to_lm_response(response, request)

    async def _legacy_request_aforward(self, request: Any):
        data = self._legacy_call_kwargs(request)
        prompt = data.pop("prompt", None)
        messages = data.pop("messages", None)
        response = await self.aforward(prompt=prompt, messages=messages, **data)
        return self._legacy_provider_response_to_lm_response(response, request)

    def _legacy_call_kwargs(self, request: Any) -> dict[str, Any]:
        from dspy.clients.openai_format import to_openai_chat_request, to_openai_text_request

        if self.model_type == "responses":
            # Legacy LM.forward expects chat-shaped messages and performs the final
            # chat->Responses provider conversion itself.
            from dspy.core.types import _history_request_messages_as_openai  # pyright: ignore[reportPrivateUsage]

            data = {**_history_request_kwargs(request), "messages": _history_request_messages_as_openai(request)}
        elif self.model_type == "text":
            data = to_openai_text_request(request)
            data["prompt"] = data.pop("prompt", None)
        else:
            data = to_openai_chat_request(request)
        data.pop("model", None)
        cache = getattr(getattr(request, "config", None), "cache", None)
        if cache is not None:
            if getattr(cache, "enabled", None) is not None:
                data["cache"] = cache.enabled
            if getattr(cache, "rollout_id", None) is not None:
                data["rollout_id"] = cache.rollout_id
        return data

    def _legacy_provider_response_to_lm_response(self, response: Any, request: Any):
        from dspy.clients.openai_format import completion_to_lm_response, responses_to_lm_response

        if _is_lm_response(response):
            return response
        if self.model_type == "responses" and getattr(response, "output", None) is not None:
            return responses_to_lm_response(response, request)
        if isinstance(response, dict) and "output" in response:
            return responses_to_lm_response(response, request)
        return completion_to_lm_response(response, request)

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "responses":
            outputs = self._process_response(response)
        else:
            outputs = self._process_completion(response, merged_kwargs)

        if not getattr(response, "cache_hit", False) and settings.usage_tracker:
            settings.usage_tracker.add_usage(self.model, dict(getattr(response, "usage", {}) or {}))

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

    # ------------------------------------------------------------------
    # State, copying, history
    # ------------------------------------------------------------------

    def dump_state(self) -> dict[str, Any]:
        filtered_kwargs = {key: value for key, value in self.kwargs.items() if key not in ("api_key", LM_CLASS_STATE_KEY)}
        return {
            LM_CLASS_STATE_KEY: f"{type(self).__module__}.{type(self).__qualname__}",
            "model": self.model,
            "model_type": self.model_type,
            "cache": self.cache,
            "num_retries": getattr(self, "num_retries", 3),
            **filtered_kwargs,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> BaseLM:
        state = dict(state)
        class_path = state.pop(LM_CLASS_STATE_KEY, None)

        if cls is BaseLM:
            if class_path is None:
                from dspy.clients.lm import LM

                return LM(**state)

            if class_path != _BUILTIN_LM_CLASS_PATH and not allow_custom_lm_class:
                raise ValueError(
                    f"Refusing to import custom serialized LM class `{class_path}`. "
                    "Pass allow_unsafe_lm_state=True when loading trusted files to enable custom LM classes."
                )

            lm_cls = _import_lm_class(class_path)
            if not issubclass(lm_cls, BaseLM):
                raise TypeError(f"Serialized LM class `{class_path}` must be a subclass of dspy.BaseLM.")
            if "allow_custom_lm_class" in inspect.signature(lm_cls.load_state).parameters:
                return lm_cls.load_state(state, allow_custom_lm_class=allow_custom_lm_class)
            return lm_cls.load_state(state)

        return cls(**state)

    def copy(self, **kwargs):
        new_instance = copy_module.copy(self)
        new_instance.history = []
        new_instance.callbacks = list(getattr(self, "callbacks", []) or [])
        new_instance.kwargs = dict(getattr(self, "kwargs", {}) or {})

        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if (key in new_instance.kwargs) or (not hasattr(self, key)):
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

        if settings.max_history_size == 0:
            return

        if len(self.history) >= settings.max_history_size:
            self.history.pop(0)
        self.history.append(entry)

        for module in settings.caller_modules or []:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)

    # ------------------------------------------------------------------
    # Normalized execution internals
    # ------------------------------------------------------------------

    def _forward_with_retry(self, request: Any):
        attempts = max(0, int(getattr(self, "num_retries", 0) or 0)) + 1
        for attempt in range(attempts):
            try:
                return self._forward_with_cache(request)
            except Exception as error:
                normalized_error = self._normalize_error(error, request)
                if attempt >= attempts - 1 or not _is_retryable_lm_error(normalized_error):
                    if normalized_error is error:
                        raise
                    raise normalized_error from error
                _sleep_before_retry(attempt)
        raise RuntimeError("unreachable")

    async def _aforward_with_retry(self, request: Any):
        attempts = max(0, int(getattr(self, "num_retries", 0) or 0)) + 1
        for attempt in range(attempts):
            try:
                return await self._aforward_with_cache(request)
            except Exception as error:
                normalized_error = self._normalize_error(error, request)
                if attempt >= attempts - 1 or not _is_retryable_lm_error(normalized_error):
                    if normalized_error is error:
                        raise
                    raise normalized_error from error
                await _asleep_before_retry(attempt)
        raise RuntimeError("unreachable")

    def _forward_with_cache(self, request: Any):
        if not _request_cache_enabled(request, self.cache):
            return self.forward(request)
        response = _cached_baselm_forward(cache_request=self._cache_request_for_mode(request, mode="sync"), lm=self, request=request)
        return _prepare_cached_lm_response(response)

    async def _aforward_with_cache(self, request: Any):
        if not _request_cache_enabled(request, self.cache):
            return await self.aforward(request)
        response = await _cached_baselm_aforward(cache_request=self._cache_request_for_mode(request, mode="async"), lm=self, request=request)
        return _prepare_cached_lm_response(response)

    def _cache_request(self, request: Any) -> dict[str, Any]:
        return {
            "lm_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "lm_state": _sanitize_cache_value(self.dump_state()),
            "request": _sanitize_cache_value(_model_dump_for_cache(request)),
        }

    def _cache_request_for_mode(self, request: Any, *, mode: str) -> dict[str, Any]:
        cache_request = self._cache_request(request)
        cache_request["execution_mode"] = mode
        return cache_request

    def _finalize_response(self, request: Any, response: Any):
        if not _is_lm_response(response):
            raise TypeError(
                f"{type(self).__name__}.forward(request) must return an LMResponse, got {type(response).__name__}. "
                "Legacy provider-shaped responses are only supported by subclasses using "
                "forward(self, prompt=None, messages=None, **kwargs)."
            )

        self._track_usage(response)

        if not settings.disable_history:
            entry = _import_lm_types().LMHistoryEntry(
                request=_sanitize_lm_request_for_history(request),
                response=response,
                timestamp=datetime.datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                model_type=getattr(self, "model_type", None),
            )
            self.update_history(entry)

        return response

    def _normalize_error(self, error: Exception, request: Any) -> Exception:
        if isinstance(error, LMError):
            return error
        return self.normalize_error(error, request)

    def _track_usage(self, response: Any) -> None:
        if getattr(response, "cache_hit", False) or not settings.usage_tracker:
            return
        usage = _response_usage_as_dict(response)
        if usage:
            settings.usage_tracker.add_usage(self.model, usage)

    def _warn_zero_temp_rollout_for_request(self, request: Any) -> None:
        cache = getattr(getattr(request, "config", None), "cache", None)
        rollout_id = getattr(cache, "rollout_id", None)
        temperature = getattr(getattr(request, "config", None), "temperature", None)
        if self._warned_zero_temp_rollout or rollout_id is None or temperature != 0:
            return
        warnings.warn(
            "rollout_id only affects DSPy's request cache when temperature=0; set temperature>0 "
            "to request a potentially different provider output.",
            UserWarning,
            stacklevel=3,
        )
        self._warned_zero_temp_rollout = True

    def _override_request(self, request: Any, **kwargs: Any):
        if not kwargs:
            return request
        if hasattr(request, "with_config_overrides"):
            return request.with_config_overrides(**kwargs)
        if hasattr(request, "model_copy"):
            return request.model_copy(update=_request_config_update(request, kwargs), deep=True)
        raise TypeError("LMRequest overrides require with_config_overrides() or Pydantic model_copy().")

    def _require_stream_support(self, *, async_: bool) -> None:
        method_name = "aforward_stream" if async_ else "forward_stream"
        if self._method_overridden(method_name):
            return
        name = "async streaming" if async_ else "streaming"
        raise NotImplementedError(f"{type(self).__name__} does not support {name}; {method_name}() is not overridden.")

    def _method_overridden(self, method_name: str) -> bool:
        method = getattr(type(self), method_name, None)
        base_method = getattr(BaseLM, method_name, None)
        return method is not None and base_method is not None and method is not base_method

    # ------------------------------------------------------------------
    # Callback helpers
    # ------------------------------------------------------------------

    def _get_active_callbacks(self) -> list[BaseCallback]:
        return list(settings.get("callbacks", []) or []) + list(getattr(self, "callbacks", []) or [])

    def _raw_callback_inputs(self, *, items: tuple[Any, ...], prompt: str | None, messages: list[dict[str, Any]] | None, kwargs: dict[str, Any]):
        return _sanitize_callback_value({"items": items, "prompt": prompt, "messages": messages, "kwargs": kwargs})

    def _start_legacy_callbacks(self, callbacks: list[BaseCallback], *, prompt, messages, kwargs):
        if not callbacks:
            return None
        call_id = uuid.uuid4().hex
        inputs = _sanitize_callback_value({"prompt": prompt, "messages": messages, **kwargs})
        for callback in callbacks:
            try:
                callback.on_lm_start(call_id=call_id, instance=self, inputs=inputs)
            except Exception as error:
                logger.warning("Error when calling callback %s: %s", callback, error)
        return call_id

    def _start_lm_callbacks(self, callbacks: list[BaseCallback], *, request: Any, raw_inputs: dict[str, Any]):
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

    def _end_lm_callbacks(self, callbacks: list[BaseCallback], *, call_id: str | None, outputs: Any, exception: Exception | None) -> None:
        if not callbacks or call_id is None:
            return
        for callback in callbacks:
            try:
                callback.on_lm_end(call_id=call_id, outputs=outputs, exception=exception)
            except Exception as error:
                logger.warning("Error when applying callback %s's LM end handler: %s", callback, error)

    def _observe_failed_stream_construction(self, request: Any, error: Exception, *, callbacks: list[BaseCallback], raw_inputs: dict[str, Any]) -> None:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        try:
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=None, exception=error)
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)

    def _callback_wrapped_stream_events(self, request: Any, events: Iterator[Any], *, callbacks: list[BaseCallback], raw_inputs: dict[str, Any]):
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        builder = _import_lm_types().LMOutputBuilder()
        result = None
        exception = None
        try:
            for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def _callback_wrapped_astream_events(self, request: Any, events: AsyncIterator[Any], *, callbacks: list[BaseCallback], raw_inputs: dict[str, Any]):
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        builder = _import_lm_types().LMOutputBuilder()
        result = None
        exception = None
        try:
            async for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    # ------------------------------------------------------------------
    # Legacy provider response processors
    # ------------------------------------------------------------------

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
    pretty_print_history(GLOBAL_HISTORY, n, file=file)


def _import_lm_types():
    from dspy.core import types as lm_types

    return lm_types


def _is_lm_request(value: Any) -> bool:
    return isinstance(value, _import_lm_types().LMRequest)


def _is_lm_response(value: Any) -> bool:
    return isinstance(value, _import_lm_types().LMResponse)


def _prepare_cached_lm_response(response: Any) -> Any:
    if getattr(response, "cache_hit", False) and hasattr(response, "cost"):
        response.cost = None
    return response


def _is_retryable_lm_error(error: Exception) -> bool:
    if isinstance(error, RETRYABLE_LM_ERRORS):
        return True
    if isinstance(error, LMProviderError):
        status = getattr(error, "status", None)
        return status is not None and int(status) >= 500
    return False


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(2**attempt, 8))


async def _asleep_before_retry(attempt: int) -> None:
    await anyio.sleep(min(2**attempt, 8))


def _cached_baselm_forward(cache_request: dict[str, Any], lm: BaseLM, request: Any) -> Any:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    def run(cache_request: dict[str, Any], lm: BaseLM, request: Any) -> Any:
        return lm.forward(request)

    return run(cache_request=cache_request, lm=lm, request=request)


async def _cached_baselm_aforward(cache_request: dict[str, Any], lm: BaseLM, request: Any) -> Any:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    async def run(cache_request: dict[str, Any], lm: BaseLM, request: Any) -> Any:
        return await lm.aforward(request)

    return await run(cache_request=cache_request, lm=lm, request=request)


def _request_cache_enabled(request: Any, default: bool) -> bool:
    cache = getattr(getattr(request, "config", None), "cache", None)
    enabled = getattr(cache, "enabled", None)
    return default if enabled is None else bool(enabled)


def _model_dump_for_cache(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _is_secret_key(key: Any) -> bool:
    key_text = str(key).lower().replace("-", "_")
    return (
        key_text in {"api_key", "authorization", "x_api_key", "token", "access_token", "refresh_token"}
        or key_text.endswith("_api_key")
        or key_text.endswith("_token")
        or "secret" in key_text
    )


def _sanitize_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            if _is_secret_key(key):
                continue
            sanitized[key] = _sanitize_cache_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_cache_value(item) for item in value]
    return value


def _request_config_update(request: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    config = getattr(request, "config", None)
    if config is None:
        return {"config": kwargs}
    if hasattr(config, "model_copy"):
        return {"config": config.model_copy(update=kwargs, deep=True)}
    if isinstance(config, dict):
        return {"config": {**config, **kwargs}}
    raise TypeError("Cannot override config on this LMRequest object.")


def _response_usage_as_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "usage_as_dict"):
        return response.usage_as_dict()
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage)


def _history_request_kwargs(request: Any) -> dict[str, Any]:
    data = request.config.model_dump(exclude_none=True)
    extensions = data.pop("extensions", {}) or {}
    return {**extensions, **data}


def _sanitize_lm_request_for_callbacks(request: Any) -> Any:
    return _sanitize_lm_request(request, redact=True)


def _sanitize_lm_request_for_history(request: Any) -> Any:
    return _sanitize_lm_request(request, redact=True)


def _sanitize_lm_request(request: Any, *, redact: bool) -> Any:
    config = getattr(request, "config", None)
    if config is not None and hasattr(config, "model_copy") and hasattr(request, "model_copy"):
        config = config.model_copy(update={"extensions": _sanitize_callback_value(getattr(config, "extensions", {}) or {})}, deep=True)
        return request.model_copy(update={"config": config, "metadata": _sanitize_callback_value(getattr(request, "metadata", {}) or {})}, deep=True)
    return _sanitize_callback_value(request) if redact else _sanitize_cache_value(request)


def _sanitize_callback_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            if _is_secret_key(key):
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_callback_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_callback_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_callback_value(item) for item in value]
    return value
