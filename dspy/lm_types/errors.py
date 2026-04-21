"""Typed error hierarchy for DSPy LM and Adapter interactions.

Every provider maps its idiosyncratic errors onto this hierarchy. DSPy
code catches these — not litellm errors, not HTTP codes.

Hierarchy::

    DSPyError
    ├── LMError                     (any LM-level failure)
    │   ├── AuthError               (bad/missing API key)
    │   ├── RateLimitError          (429)
    │   ├── ContextLengthError      (input too long for model)
    │   ├── InvalidRequestError     (bad request shape)
    │   ├── TimeoutError            (request timed out)
    │   ├── ServerError             (provider 5xx)
    │   └── TransportError          (network/connection failure)
    └── AdapterError                (adapter-level failure)
        └── AdapterParseError       (can't parse LM output)
"""

from __future__ import annotations

from typing import Literal


# ─────────────────────────────────────────────────────────────
# Base hierarchy
# ─────────────────────────────────────────────────────────────

class DSPyError(Exception):
    """Base for all DSPy errors."""


class LMError(DSPyError):
    """Base for all LM-level errors.

    Every ``BaseLMv2`` subclass should raise ``LMError`` subclasses — never
    provider-specific exceptions.  This is the contract.

    Attributes:
        model: The model identifier that produced the error.
        provider_code: The provider's original error code, if any.
    """

    def __init__(
        self,
        message: str = "",
        *,
        model: str | None = None,
        provider_code: str | None = None,
    ):
        self.model = model
        self.provider_code = provider_code
        prefix = f"[{model}] " if model else ""
        super().__init__(f"{prefix}{message}")


class AuthError(LMError):
    """Authentication failed — invalid, expired, or missing API key."""


class RateLimitError(LMError):
    """Rate limited by the provider (HTTP 429)."""


class ContextLengthError(LMError):
    """The input exceeds the model's context window.

    Adapters and modules catch this to trigger truncation/fallback.
    """


class InvalidRequestError(LMError):
    """Bad request shape (400/422) — e.g. unsupported parameter."""


class TimeoutError(LMError):
    """Request timed out."""


class ServerError(LMError):
    """Provider-side failure (5xx)."""


class TransportError(LMError):
    """Network or connection failure."""


# ─────────────────────────────────────────────────────────────
# Retry classification
# ─────────────────────────────────────────────────────────────

RETRYABLE_ERRORS: tuple[type[LMError], ...] = (
    RateLimitError,
    TimeoutError,
    ServerError,
    TransportError,
)
"""LM errors that are safe to retry with backoff.

Used by ``BaseLMv2.__call__`` for automatic retries.  Custom LM subclasses
should raise these when appropriate so retry logic works generically.
"""


# ─────────────────────────────────────────────────────────────
# Adapter errors
# ─────────────────────────────────────────────────────────────

class AdapterError(DSPyError):
    """Base for adapter-level errors."""


class AdapterParseError(AdapterError):
    """Adapter cannot parse the LM response into output fields.

    Attributes:
        adapter_name: Name of the adapter that failed.
        lm_response: The raw LM response text.
        expected_fields: Output field names the adapter expected.
        parsed_fields: Field names successfully parsed (if partial).
    """

    def __init__(
        self,
        message: str = "",
        *,
        adapter_name: str | None = None,
        lm_response: str | None = None,
        expected_fields: list[str] | None = None,
        parsed_fields: list[str] | None = None,
    ):
        self.adapter_name = adapter_name
        self.lm_response = lm_response
        self.expected_fields = expected_fields
        self.parsed_fields = parsed_fields

        parts: list[str] = []
        if message:
            parts.append(message)
        if adapter_name:
            parts.append(f"Adapter: {adapter_name}")
        if expected_fields:
            parts.append(f"Expected fields: {expected_fields}")
        if parsed_fields is not None:
            parts.append(f"Parsed fields: {parsed_fields}")
        if lm_response is not None:
            preview = lm_response[:500] + ("..." if len(lm_response) > 500 else "")
            parts.append(f"LM response: {preview}")
        super().__init__("\n".join(parts))


# ─────────────────────────────────────────────────────────────
# Canonical error codes
# ─────────────────────────────────────────────────────────────

ErrorCode = Literal[
    "auth", "rate_limit", "context_length", "invalid_request",
    "timeout", "server", "transport", "parse", "unknown",
]

_CLASS_TO_CODE: dict[type[DSPyError], ErrorCode] = {
    AuthError: "auth",
    RateLimitError: "rate_limit",
    ContextLengthError: "context_length",
    InvalidRequestError: "invalid_request",
    TimeoutError: "timeout",
    ServerError: "server",
    TransportError: "transport",
    AdapterParseError: "parse",
}

_CODE_TO_CLASS: dict[str, type[DSPyError]] = {v: k for k, v in _CLASS_TO_CODE.items()}


def error_code(error: DSPyError) -> ErrorCode:
    """Return the canonical string code for an error instance."""
    # Check most specific first
    order = [
        ContextLengthError, AuthError, RateLimitError,
        InvalidRequestError, TimeoutError, ServerError, TransportError,
        AdapterParseError,
    ]
    for cls in order:
        if isinstance(error, cls):
            return _CLASS_TO_CODE[cls]
    return "unknown"


def error_class(code: str) -> type[DSPyError]:
    """Return the error class for a canonical code."""
    return _CODE_TO_CLASS.get(code, DSPyError)


def map_http_status(status: int, message: str, *, model: str | None = None) -> LMError:
    """Map an HTTP status code to the appropriate ``LMError`` subclass.

    Used by LM implementations that talk HTTP directly (not via litellm).
    Includes a small heuristic to detect context-length-style errors that
    share the generic 400 status code.
    """
    if status in (401, 403):
        return AuthError(message, model=model)
    if status == 429:
        return RateLimitError(message, model=model)
    if status in (408, 504):
        return TimeoutError(message, model=model)
    if status in (400, 404, 422):
        lower = message.lower()
        # Heuristic: catch context length errors masquerading as 400s
        if (
            "context" in lower and ("length" in lower or "window" in lower)
            or ("token" in lower and ("limit" in lower or "exceed" in lower))
            or "too long" in lower
        ):
            return ContextLengthError(message, model=model)
        return InvalidRequestError(message, model=model)
    if 500 <= status <= 599:
        return ServerError(message, model=model)
    return LMError(message, model=model)


__all__ = [
    "DSPyError",
    "LMError",
    "AuthError",
    "RateLimitError",
    "ContextLengthError",
    "InvalidRequestError",
    "TimeoutError",
    "ServerError",
    "TransportError",
    "RETRYABLE_ERRORS",
    "AdapterError",
    "AdapterParseError",
    "ErrorCode",
    "error_code",
    "error_class",
    "map_http_status",
]
