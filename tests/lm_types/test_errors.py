"""Tests for the typed error hierarchy."""


import pytest

from dspy.lm_types import (
    AdapterError,
    AdapterParseError,
    AuthError,
    ContextLengthError,
    DSPyError,
    InvalidRequestError,
    LMError,
    RateLimitError,
    RETRYABLE_ERRORS,
    ServerError,
    TimeoutError as LMTimeoutError,
    TransportError,
    error_class,
    error_code,
    map_http_status,
)


class TestErrorHierarchy:
    def test_all_lm_errors_subclass_lm_error(self):
        for cls in [AuthError, RateLimitError, ContextLengthError,
                    InvalidRequestError, LMTimeoutError, ServerError,
                    TransportError]:
            assert issubclass(cls, LMError)
            assert issubclass(cls, DSPyError)

    def test_adapter_error_hierarchy(self):
        assert issubclass(AdapterParseError, AdapterError)
        assert issubclass(AdapterError, DSPyError)

    def test_context_length_is_lm_error(self):
        assert issubclass(ContextLengthError, LMError)


class TestLMError:
    def test_basic_construction(self):
        err = LMError("something broke", model="gpt-4")
        assert err.model == "gpt-4"
        assert "gpt-4" in str(err)
        assert "something broke" in str(err)

    def test_without_model(self):
        err = LMError("broke")
        assert err.model is None

    def test_provider_code(self):
        err = RateLimitError("too many", model="gpt-4", provider_code="rate_limit_exceeded")
        assert err.provider_code == "rate_limit_exceeded"


class TestRetryableErrors:
    def test_rate_limit_is_retryable(self):
        assert RateLimitError in RETRYABLE_ERRORS

    def test_timeout_is_retryable(self):
        assert LMTimeoutError in RETRYABLE_ERRORS

    def test_server_is_retryable(self):
        assert ServerError in RETRYABLE_ERRORS

    def test_transport_is_retryable(self):
        assert TransportError in RETRYABLE_ERRORS

    def test_auth_not_retryable(self):
        assert AuthError not in RETRYABLE_ERRORS

    def test_context_length_not_retryable(self):
        assert ContextLengthError not in RETRYABLE_ERRORS

    def test_invalid_request_not_retryable(self):
        assert InvalidRequestError not in RETRYABLE_ERRORS


class TestErrorCodes:
    def test_error_code_for_rate_limit(self):
        assert error_code(RateLimitError("x")) == "rate_limit"

    def test_error_code_for_auth(self):
        assert error_code(AuthError("x")) == "auth"

    def test_error_code_for_context_length(self):
        assert error_code(ContextLengthError("x")) == "context_length"

    def test_error_class_roundtrip(self):
        assert error_class("rate_limit") == RateLimitError
        assert error_class("auth") == AuthError
        assert error_class("context_length") == ContextLengthError

    def test_adapter_parse_error_code(self):
        assert error_code(AdapterParseError("x")) == "parse"


class TestMapHttpStatus:
    def test_401_to_auth(self):
        err = map_http_status(401, "unauthorized")
        assert isinstance(err, AuthError)

    def test_403_to_auth(self):
        err = map_http_status(403, "forbidden")
        assert isinstance(err, AuthError)

    def test_429_to_rate_limit(self):
        err = map_http_status(429, "rate limited")
        assert isinstance(err, RateLimitError)

    def test_408_to_timeout(self):
        err = map_http_status(408, "timeout")
        assert isinstance(err, LMTimeoutError)

    def test_504_to_timeout(self):
        err = map_http_status(504, "gateway timeout")
        assert isinstance(err, LMTimeoutError)

    def test_500_to_server(self):
        err = map_http_status(500, "server error")
        assert isinstance(err, ServerError)

    def test_503_to_server(self):
        err = map_http_status(503, "unavailable")
        assert isinstance(err, ServerError)

    def test_400_to_invalid_request(self):
        err = map_http_status(400, "bad request")
        assert isinstance(err, InvalidRequestError)

    def test_context_length_heuristic(self):
        err = map_http_status(400, "context length exceeded for model")
        assert isinstance(err, ContextLengthError)

    def test_token_limit_heuristic(self):
        err = map_http_status(400, "max token limit exceeded")
        assert isinstance(err, ContextLengthError)

    def test_model_propagated(self):
        err = map_http_status(429, "rate limited", model="gpt-4")
        assert err.model == "gpt-4"


class TestAdapterParseError:
    def test_basic(self):
        err = AdapterParseError(
            "cannot parse",
            adapter_name="ChatAdapter",
            lm_response="garbage",
            expected_fields=["answer"],
            parsed_fields=[],
        )
        assert err.adapter_name == "ChatAdapter"
        assert err.lm_response == "garbage"
        assert err.expected_fields == ["answer"]

    def test_message_composition(self):
        err = AdapterParseError(
            "cannot parse",
            adapter_name="Test",
            lm_response="x",
        )
        msg = str(err)
        assert "cannot parse" in msg
        assert "Test" in msg

    def test_truncates_long_response(self):
        long_resp = "x" * 2000
        err = AdapterParseError("fail", lm_response=long_resp)
        assert "..." in str(err)
