"""Typed LM for OpenAI Chat Completions-compatible HTTP endpoints.

``OpenAICompatLM`` uses direct ``requests`` transport and DSPy's typed
``LMRequest`` / ``LMResponse`` boundary. Async calls run the same synchronous
transport in an AnyIO worker thread. Streaming is intentionally out of scope.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import anyio
import requests

import dspy
from dspy.clients.base_lm import LM_CLASS_STATE_KEY, BaseLM
from dspy.clients.openai_format import completion_to_lm_response, to_openai_chat_request
from dspy.core.types import LMRequest, LMResponse
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import (
    ContextWindowExceededError,
    LMAuthError,
    LMBillingError,
    LMConfigurationError,
    LMError,
    LMInvalidRequestError,
    LMNotConfiguredError,
    LMProviderError,
    LMRateLimitError,
    LMServerError,
    LMTimeoutError,
    LMTransportError,
    LMUnsupportedModelError,
    is_retryable_lm_error,
)

__all__ = ["OpenAICompatLM"]

logger = logging.getLogger(__name__)

_PROVIDER_CODE_MAP: dict[str, type[LMError]] = {
    "context_length_exceeded": ContextWindowExceededError,
    "model_not_found": LMUnsupportedModelError,
    "model_not_available": LMUnsupportedModelError,
    "unsupported_model": LMUnsupportedModelError,
    "insufficient_quota": LMBillingError,
    "invalid_api_key": LMAuthError,
    "authentication_error": LMAuthError,
    "rate_limit_exceeded": LMRateLimitError,
    "rate_limit_error": LMRateLimitError,
}
_MODEL_ERROR_CODES = frozenset({"model_not_found", "model_not_available", "unsupported_model"})
_SENSITIVE_HEADER_NAMES = frozenset(
    {"authorization", "proxy-authorization", "api-key", "x-api-key", "cookie", "set-cookie"}
)


@dataclass(frozen=True)
class _RawResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes


_Post = Callable[[str, dict[str, Any], dict[str, str], float], _RawResponse]


def _requests_post(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> _RawResponse:
    response = requests.post(url, json=body, headers=headers, timeout=timeout)
    return _RawResponse(status=response.status_code, headers=dict(response.headers), body=response.content)


def _header(headers: Mapping[str, str] | None, *names: str) -> str | None:
    if not headers:
        return None
    lowered = {str(key).lower(): str(value) for key, value in headers.items()}
    return next((lowered[name.lower()] for name in names if name.lower() in lowered), None)


def _retry_after(headers: Mapping[str, str] | None) -> float | None:
    value = _header(headers, "retry-after")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _is_model_error(message: str, *codes: str) -> bool:
    lowered = " ".join(str(value) for value in (message, *codes) if value).lower()
    return "model" in lowered and any(
        marker in lowered
        for marker in (
            "not found",
            "does not exist",
            "not exist",
            "not supported",
            "unsupported",
            "not available",
            "unknown",
        )
    )


def _normalize_error(
    status: int,
    body: str,
    *,
    headers: Mapping[str, str] | None = None,
    model: str | None = None,
    provider: str = "openai_compat",
) -> LMError:
    """Map an OpenAI-shaped HTTP error response to a structured DSPy error."""
    message: str | None = None
    provider_code: str | None = None
    error_type: str | None = None

    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        data = None

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            raw_message = error.get("message")
            message = str(raw_message) if raw_message is not None else None
            raw_code = error.get("code")
            raw_type = error.get("type")
            provider_code = str(raw_code) if raw_code not in (None, "") else None
            error_type = str(raw_type) if raw_type not in (None, "") else None
        elif error is not None:
            message = str(error)

    if message is None:
        message = body.strip()[:500] if body and body.strip() else f"HTTP {status}"

    metadata = {
        "model": model,
        "provider": provider,
        "provider_code": provider_code or error_type,
        "status": status,
        "request_id": _header(headers, "x-request-id", "request-id", "x-amzn-requestid", "x-ms-request-id"),
        "retry_after": _retry_after(headers),
    }

    error_class = next(
        (_PROVIDER_CODE_MAP[value] for value in (provider_code, error_type) if value in _PROVIDER_CODE_MAP),
        None,
    )
    if error_class is not None:
        return error_class(message=message, **metadata)

    if status == 404 and _is_model_error(message, provider_code or "", error_type or ""):
        return LMUnsupportedModelError(message=message, **metadata)

    return _error_class_from_status(status)(message=message, **metadata)


def _error_class_from_status(status: int) -> type[LMError]:
    if status in (401, 403):
        return LMAuthError
    if status == 402:
        return LMBillingError
    if status in (408, 504):
        return LMTimeoutError
    if status == 429:
        return LMRateLimitError
    if 400 <= status < 500:
        return LMInvalidRequestError
    if status >= 500:
        return LMServerError
    return LMProviderError


class OpenAICompatLM(BaseLM):
    """A typed LM for an OpenAI Chat Completions-compatible endpoint.

    Args:
        model: Model identifier accepted by the endpoint.
        base_url: API base URL, such as ``http://localhost:8000/v1``.
        api_key: Optional explicit bearer token, or a zero-argument callable
            returning one. A callable is invoked on every request, so vaults,
            OAuth refreshers, and rotating credentials plug in without the LM
            knowing which; the resolved token is only ever placed in the
            ``Authorization`` header.
        api_key_env: Optional environment variable checked when ``api_key`` is absent.
        use_openai_api_key_env: Whether to fall back to ``OPENAI_API_KEY``. Off by
            default so a key meant for OpenAI is never sent to another endpoint
            without an explicit opt-in.
        require_auth: When True, raise ``dspy.LMNotConfiguredError`` locally if
            no credential resolves, instead of sending an unauthenticated
            request and waiting for the endpoint to reject it. Leave False for
            local endpoints that need no key.
        timeout: Request timeout in seconds.
        extra_headers: Additional HTTP headers. Explicit values override defaults.
        supports_function_calling: Opt into native tool calls.
        supports_reasoning: Opt into native reasoning.
        supports_response_schema: Opt into structured response schemas.
        **kwargs: Default LM request parameters.

    API keys are never serialized. If an explicit key was used, serialized state
    disables ambient ``OPENAI_API_KEY`` fallback so loading cannot silently
    switch to another account.
    """

    forward_contract = "typed_lm"

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | Callable[[], str] | None = None,
        api_key_env: str | None = None,
        use_openai_api_key_env: bool = False,
        require_auth: bool = False,
        timeout: float = 60.0,
        extra_headers: dict[str, str] | None = None,
        supports_function_calling: bool = False,
        supports_reasoning: bool = False,
        supports_response_schema: bool = False,
        model_type: str = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        _post: _Post | None = None,
        **kwargs: Any,
    ):
        if model_type != "chat":
            raise LMConfigurationError(
                f"OpenAICompatLM only supports model_type='chat', but got {model_type!r}.",
                model=model,
                provider="openai_compat",
            )
        if not isinstance(base_url, str) or not base_url.strip():
            raise LMConfigurationError(
                "OpenAICompatLM requires a non-empty base_url.", model=model, provider="openai_compat"
            )
        if timeout <= 0:
            raise LMConfigurationError(
                "OpenAICompatLM timeout must be greater than zero.", model=model, provider="openai_compat"
            )
        if num_retries < 0:
            raise LMConfigurationError(
                "OpenAICompatLM num_retries cannot be negative.", model=model, provider="openai_compat"
            )

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
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.use_openai_api_key_env = bool(use_openai_api_key_env)
        self.require_auth = bool(require_auth)
        self.timeout = float(timeout)
        self.extra_headers = {str(key): str(value) for key, value in (extra_headers or {}).items()}
        self._supports_function_calling = bool(supports_function_calling)
        self._supports_reasoning = bool(supports_reasoning)
        self._supports_response_schema = bool(supports_response_schema)
        self._api_key = api_key
        self._post = _post or _requests_post

    def _resolved_api_key(self) -> str | None:
        if self._api_key is not None:
            return self._api_key() if callable(self._api_key) else self._api_key
        if self.api_key_env is not None:
            value = os.environ.get(self.api_key_env)
            if value:
                return value
        if self.use_openai_api_key_env:
            return os.environ.get("OPENAI_API_KEY")
        return None

    def _request_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "User-Agent": f"DSPy/{dspy.__version__}"}
        api_key = self._resolved_api_key()
        if api_key is None and self.require_auth:
            exits = [
                "pass api_key= to OpenAICompatLM",
                f"set {self.api_key_env}" if self.api_key_env else None,
                "set OPENAI_API_KEY with use_openai_api_key_env=True" if self.use_openai_api_key_env else None,
            ]
            raise LMNotConfiguredError(
                "OpenAICompatLM requires a credential (require_auth=True) but none resolved. "
                f"To fix: {'; or '.join(exit for exit in exits if exit)}.",
                model=self.model,
                provider="openai_compat",
            )
        if api_key is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(self.extra_headers)
        return headers

    def _cache_request(self, request: LMRequest) -> dict[str, Any]:
        # Key on the OpenAI-shaped wire payload rather than the LMRequest model:
        # config values such as a pydantic `response_format` class are only
        # JSON-serializable after `to_openai_chat_request()` maps them, and the
        # payload naturally excludes DSPy-only cache config.
        api_key = self._resolved_api_key()
        header_fingerprints = {
            name.lower(): hashlib.sha256(value.encode()).hexdigest()
            for name, value in sorted(self.extra_headers.items(), key=lambda item: item[0].lower())
        }
        cache_config = request.config.cache
        return {
            "_fn_identifier": "dspy.clients.openai_compat_lm.OpenAICompatLM.forward",
            "base_url": self.base_url,
            "request": to_openai_chat_request(request),
            "rollout_id": cache_config.rollout_id if cache_config is not None else None,
            "credential_fingerprint": hashlib.sha256(api_key.encode()).hexdigest() if api_key else None,
            "header_fingerprints": header_fingerprints,
        }

    def _cache_enabled(self, request: LMRequest) -> bool:
        config = request.config.cache
        if config is None or config.enabled is None:
            return self.cache
        return config.enabled

    def _request_once(self, request: LMRequest) -> LMResponse:
        payload = to_openai_chat_request(request)
        try:
            raw = self._post(f"{self.base_url}/chat/completions", payload, self._request_headers(), self.timeout)
        except requests.Timeout as error:
            raise LMTimeoutError(
                str(error) or "LM request timed out", model=self.model, provider="openai_compat"
            ) from error
        except requests.RequestException as error:
            raise LMTransportError(
                str(error) or "LM transport failed", model=self.model, provider="openai_compat"
            ) from error
        except TimeoutError as error:
            raise LMTimeoutError(
                str(error) or "LM request timed out", model=self.model, provider="openai_compat"
            ) from error
        except OSError as error:
            raise LMTransportError(
                str(error) or "LM transport failed", model=self.model, provider="openai_compat"
            ) from error

        body = raw.body.decode("utf-8", errors="replace")
        if raw.status >= 400:
            raise _normalize_error(raw.status, body, headers=raw.headers, model=self.model)
        try:
            provider_response = json.loads(body)
        except (TypeError, ValueError) as error:
            raise LMProviderError(
                "OpenAI-compatible endpoint returned a non-JSON success response.",
                model=self.model,
                provider="openai_compat",
                status=raw.status,
                request_id=_header(raw.headers, "x-request-id", "request-id"),
            ) from error
        if not isinstance(provider_response, dict):
            raise LMProviderError(
                "OpenAI-compatible endpoint returned a JSON response that was not an object.",
                model=self.model,
                provider="openai_compat",
                status=raw.status,
            )

        try:
            response = completion_to_lm_response(provider_response, request)
        except Exception as error:
            raise LMProviderError(
                "OpenAI-compatible endpoint returned an invalid Chat Completions response.",
                model=self.model,
                provider="openai_compat",
                status=raw.status,
                request_id=_header(raw.headers, "x-request-id", "request-id"),
            ) from error
        self._warn_on_truncation(response, request)
        return response

    def _warn_on_truncation(self, response: LMResponse, request: LMRequest) -> None:
        if any(output.truncated for output in response.outputs):
            logger.warning(
                "OpenAICompatLM response was truncated (finish_reason='length', max_tokens=%s). "
                "Inspect recent LM calls with `dspy.inspect_history()`, or pass a larger max_tokens.",
                request.config.max_tokens,
            )

    def _request_with_retries(self, request: LMRequest) -> LMResponse:
        for attempt in range(self.num_retries + 1):
            try:
                return self._request_once(request)
            except LMError as error:
                if attempt >= self.num_retries or not is_retryable_lm_error(error):
                    raise
                delay = error.retry_after if error.retry_after is not None else min(2**attempt, 60)
                time.sleep(max(delay, 0.0))
        raise AssertionError("retry loop did not return or raise")

    def forward(self, request: LMRequest) -> LMResponse:
        """Call the endpoint synchronously and return a normalized response."""
        if not self._cache_enabled(request):
            return self._request_with_retries(request)

        cache_request = self._cache_request(request)
        cached = dspy.cache.get(cache_request)
        if cached is not None:
            return cached
        response = self._request_with_retries(request)
        dspy.cache.put(cache_request, response)
        return response

    async def aforward(self, request: LMRequest) -> LMResponse:
        """Run the synchronous non-streaming transport in an AnyIO worker."""
        return await anyio.to_thread.run_sync(self.forward, request)

    @property
    def supports_function_calling(self) -> bool:
        return self._supports_function_calling

    @property
    def supports_reasoning(self) -> bool:
        return self._supports_reasoning

    @property
    def supports_response_schema(self) -> bool:
        return self._supports_response_schema

    @property
    def supported_params(self) -> set[str]:
        params = {"temperature", "max_tokens", "top_p", "stop", "n", "logprobs"}
        if self.supports_function_calling:
            params.update({"tools", "tool_choice", "parallel_tool_calls"})
        if self.supports_reasoning:
            params.add("reasoning_effort")
        if self.supports_response_schema:
            params.add("response_format")
        return params

    def dump_state(self) -> dict[str, Any]:
        state = super().dump_state()
        safe_headers = {
            key: value for key, value in self.extra_headers.items() if key.lower() not in _SENSITIVE_HEADER_NAMES
        }
        state.update(
            {
                "base_url": self.base_url,
                "api_key_env": self.api_key_env,
                "use_openai_api_key_env": self.use_openai_api_key_env if self._api_key is None else False,
                "require_auth": self.require_auth,
                "timeout": self.timeout,
                "extra_headers": safe_headers,
                "supports_function_calling": self._supports_function_calling,
                "supports_reasoning": self._supports_reasoning,
                "supports_response_schema": self._supports_response_schema,
            }
        )
        return state

    @classmethod
    def load_state(cls, state: dict[str, Any], *, allow_custom_lm_class: bool = False) -> OpenAICompatLM:
        state = dict(state)
        state.pop(LM_CLASS_STATE_KEY, None)
        return cls(**state)
