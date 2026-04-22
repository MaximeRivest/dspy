"""Tests for the new BaseLM contract."""


import asyncio

import pytest

import dspy

from dspy.lm_types import (
    AuthError,
    BaseLMv2,
    ContextLengthError,
    LMCompletion,
    LMConfig,
    LMMessage,
    LMResponse,
    LMStreamError,
    LMStreamEvent,
    LMUsage,
    PartDelta,
    RateLimitError,
    ServerError,
    TextPart,
    ThinkingPart,
    TransportError,
)


class _FixedLM(BaseLMv2):
    """Test LM that returns a fixed response."""

    def __init__(self, text="hello", model="fixed", **kw):
        super().__init__(model=model, **kw)
        self._text = text
        self.calls = []

    def forward(self, messages, config):
        self.calls.append((messages, config))
        return LMResponse(
            completions=[LMCompletion(
                parts=[TextPart(text=self._text)],
                finish_reason="stop",
            )],
            usage=LMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            model=self.model,
        )

    async def aforward(self, messages, config):
        return self.forward(messages, config)


class _FlakyLM(BaseLMv2):
    """Fails the first `fail_count` times with `error`, then succeeds."""

    def __init__(self, fail_count=2, error_cls=RateLimitError, **kw):
        super().__init__(model="flaky", num_retries=3, **kw)
        self.fail_count = fail_count
        self.error_cls = error_cls
        self.attempts = 0

    def forward(self, messages, config):
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise self.error_cls("transient", model=self.model)
        return LMResponse(
            completions=[LMCompletion(parts=[TextPart(text="ok")])],
            model=self.model,
        )

    async def aforward(self, messages, config):
        self.attempts += 1
        if self.attempts <= self.fail_count:
            raise self.error_cls("transient", model=self.model)
        return LMResponse(
            completions=[LMCompletion(parts=[TextPart(text="ok")])],
            model=self.model,
        )


class TestBaseLMContract:
    def test_simple_call(self):
        lm = _FixedLM(text="hello")
        response = lm([LMMessage.user("hi")])
        assert response.completions[0].text == "hello"

    def test_config_passed(self):
        lm = _FixedLM()
        lm([LMMessage.user("hi")], LMConfig(temperature=0.9))
        _, cfg = lm.calls[0]
        assert cfg.temperature == 0.9

    def test_default_config_merged(self):
        lm = _FixedLM(temperature=0.5)
        lm([LMMessage.user("hi")])
        _, cfg = lm.calls[0]
        assert cfg.temperature == 0.5

    def test_config_override_wins(self):
        lm = _FixedLM(temperature=0.5)
        lm([LMMessage.user("hi")], LMConfig(temperature=0.9))
        _, cfg = lm.calls[0]
        assert cfg.temperature == 0.9

    def test_copy_keeps_default_config_in_sync(self):
        lm = _FixedLM(temperature=0.5)
        copied = lm.copy(temperature=0.9, top_p=0.7, custom_flag=True)

        copied([LMMessage.user("hi")])
        _, cfg = copied.calls[0]
        assert cfg.temperature == 0.9
        assert cfg.top_p == 0.7
        assert cfg.extensions["custom_flag"] is True

    def test_direct_v2_calls_record_history(self):
        lm = _FixedLM(text="hello")
        lm([LMMessage.user("hi")])
        assert len(lm.history) == 1
        assert lm.history[0]["outputs"] == ["hello"]

    def test_direct_v2_calls_track_usage(self):
        lm = _FixedLM(text="hello")
        with dspy.track_usage() as tracker:
            lm([LMMessage.user("hi")])
        totals = tracker.get_total_tokens()
        assert totals[lm.model]["total_tokens"] == 15

    def test_raises_if_forward_not_implemented(self):
        class Incomplete(BaseLMv2):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete(model="x").forward([LMMessage.user("x")], LMConfig())


class TestRetryBehavior:
    def test_retries_on_rate_limit(self):
        lm = _FlakyLM(fail_count=2, error_cls=RateLimitError)
        # Patch sleep so tests run fast
        import dspy.lm_types.base_lm_v2 as _bl
        original_sleep = _bl.time.sleep
        _bl.time.sleep = lambda _: None
        try:
            response = lm([LMMessage.user("hi")])
        finally:
            _bl.time.sleep = original_sleep
        assert response.completions[0].text == "ok"
        assert lm.attempts == 3  # 2 failures + 1 success

    def test_retries_on_server_error(self):
        lm = _FlakyLM(fail_count=1, error_cls=ServerError)
        import dspy.lm_types.base_lm_v2 as _bl
        original_sleep = _bl.time.sleep
        _bl.time.sleep = lambda _: None
        try:
            response = lm([LMMessage.user("hi")])
        finally:
            _bl.time.sleep = original_sleep
        assert response.completions[0].text == "ok"

    def test_retries_on_transport(self):
        lm = _FlakyLM(fail_count=1, error_cls=TransportError)
        import dspy.lm_types.base_lm_v2 as _bl
        original_sleep = _bl.time.sleep
        _bl.time.sleep = lambda _: None
        try:
            response = lm([LMMessage.user("hi")])
        finally:
            _bl.time.sleep = original_sleep
        assert response.completions[0].text == "ok"

    def test_no_retry_on_auth(self):
        lm = _FlakyLM(fail_count=1, error_cls=AuthError)
        with pytest.raises(AuthError):
            lm([LMMessage.user("hi")])
        assert lm.attempts == 1

    def test_no_retry_on_context_length(self):
        lm = _FlakyLM(fail_count=1, error_cls=ContextLengthError)
        with pytest.raises(ContextLengthError):
            lm([LMMessage.user("hi")])
        assert lm.attempts == 1

    def test_exhausts_retries(self):
        lm = _FlakyLM(fail_count=10, error_cls=RateLimitError)  # always fails
        import dspy.lm_types.base_lm_v2 as _bl
        original_sleep = _bl.time.sleep
        _bl.time.sleep = lambda _: None
        try:
            with pytest.raises(RateLimitError):
                lm([LMMessage.user("hi")])
        finally:
            _bl.time.sleep = original_sleep
        assert lm.attempts == 4  # 1 + 3 retries


class TestAsync:
    def test_acall(self):
        lm = _FixedLM(text="async ok")
        response = asyncio.run(lm.acall([LMMessage.user("hi")]))
        assert response.completions[0].text == "async ok"

    def test_acall_retries(self):
        lm = _FlakyLM(fail_count=1, error_cls=RateLimitError)
        import dspy.lm_types.base_lm_v2 as _bl
        original_sleep = _bl.asyncio.sleep
        async def fast(_):
            return None
        _bl.asyncio.sleep = fast
        try:
            response = asyncio.run(lm.acall([LMMessage.user("hi")]))
        finally:
            _bl.asyncio.sleep = original_sleep
        assert response.completions[0].text == "ok"


class TestDefaultStreaming:
    def test_fake_stream_from_forward(self):
        """If subclass doesn't override stream(), fake-stream from forward()."""
        lm = _FixedLM(text="hello")
        events = []

        async def collect():
            async for event in lm.stream([LMMessage.user("hi")]):
                events.append(event)

        asyncio.run(collect())

        types = [e.type for e in events]
        assert types[0] == "start"
        assert types[-1] == "end"
        assert any(e.type == "delta" for e in events)

        delta_events = [e for e in events if e.type == "delta"]
        assert any(e.delta.type == "text" for e in delta_events)

    def test_fake_stream_thinking(self):
        class _LM(BaseLMv2):
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[
                    ThinkingPart(text="thinking..."),
                    TextPart(text="answer"),
                ])])
            async def aforward(self, messages, config):
                return self.forward(messages, config)

        lm = _LM(model="x")
        events = []

        async def collect():
            async for event in lm.stream([LMMessage.user("hi")]):
                events.append(event)

        asyncio.run(collect())
        delta_types = [e.delta.type for e in events if e.type == "delta"]
        assert "thinking" in delta_types
        assert "text" in delta_types


class TestCustomStreaming:
    def test_custom_stream_implementation(self):
        class _StreamingLM(BaseLMv2):
            def forward(self, messages, config):
                raise NotImplementedError

            async def aforward(self, messages, config):
                raise NotImplementedError

            async def stream(self, messages, config=None):
                yield LMStreamEvent(type="start", model=self.model)
                for chunk in ["Par", "is"]:
                    yield LMStreamEvent(
                        type="delta",
                        delta=PartDelta(type="text", text=chunk),
                    )
                yield LMStreamEvent(type="end", finish_reason="stop")

        lm = _StreamingLM(model="stream")
        events = []

        async def collect():
            async for e in lm.stream([LMMessage.user("hi")]):
                events.append(e)

        asyncio.run(collect())
        deltas = [e.delta.text for e in events if e.type == "delta"]
        assert "".join(deltas) == "Paris"

    def test_mid_stream_error(self):
        class _ErrorLM(BaseLMv2):
            def forward(self, messages, config):
                raise NotImplementedError
            async def aforward(self, messages, config):
                raise NotImplementedError
            async def stream(self, messages, config=None):
                yield LMStreamEvent(type="start", model=self.model)
                yield LMStreamEvent(
                    type="error",
                    error=LMStreamError(code="rate_limit", message="slow down"),
                )

        lm = _ErrorLM(model="err")
        events = []

        async def collect():
            async for e in lm.stream([LMMessage.user("hi")]):
                events.append(e)

        asyncio.run(collect())
        assert any(e.type == "error" for e in events)
        error_event = [e for e in events if e.type == "error"][0]
        assert error_event.error.code == "rate_limit"
