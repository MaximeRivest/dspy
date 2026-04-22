"""Tests for streaming types: PartDelta, LMStreamEvent, LMStreamError, PartAccumulator."""


import pytest

from dspy.lm_types import (
    AudioPart,
    ImagePart,
    LMStreamError,
    LMStreamEvent,
    LMUsage,
    PartAccumulator,
    PartDelta,
    RateLimitError,
    ServerError,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)


class TestPartDelta:
    def test_text_delta(self):
        d = PartDelta(type="text", text="hello")
        assert d.type == "text"
        assert d.text == "hello"
        assert d.part_index == 0

    def test_thinking_delta(self):
        d = PartDelta(type="thinking", text="thinking...")
        assert d.type == "thinking"

    def test_tool_call_delta_first_chunk(self):
        d = PartDelta(
            type="tool_call",
            part_index=1,
            id="call_1",
            name="search",
            input='{"q":',
        )
        assert d.id == "call_1"
        assert d.name == "search"
        assert d.input == '{"q":'

    def test_tool_call_delta_continuation(self):
        d = PartDelta(type="tool_call", part_index=1, input='"test"}')
        assert d.input == '"test"}'
        assert d.id is None  # only first chunk has identity

    def test_audio_delta(self):
        d = PartDelta(type="audio", data="base64chunk", media_type="audio/wav")
        assert d.data == "base64chunk"

    def test_image_delta(self):
        d = PartDelta(type="image", data="pngbase64", media_type="image/png")
        assert d.data == "pngbase64"

    def test_citation_delta(self):
        d = PartDelta(
            type="citation",
            cited_text="hello",
            source_index=0,
            title="Source",
        )
        assert d.cited_text == "hello"
        assert d.source_index == 0


class TestLMStreamError:
    def test_basic(self):
        e = LMStreamError(code="rate_limit", message="too many requests")
        assert e.code == "rate_limit"
        assert e.message == "too many requests"

    def test_to_exception_rate_limit(self):
        e = LMStreamError(code="rate_limit", message="too many")
        exc = e.to_exception(model="gpt-4")
        assert isinstance(exc, RateLimitError)
        assert exc.model == "gpt-4"

    def test_to_exception_server(self):
        e = LMStreamError(code="server", message="down")
        exc = e.to_exception()
        assert isinstance(exc, ServerError)

    def test_retryable_rate_limit(self):
        e = LMStreamError(code="rate_limit", message="x")
        assert e.retryable is True

    def test_retryable_server(self):
        e = LMStreamError(code="server", message="x")
        assert e.retryable is True

    def test_not_retryable_auth(self):
        e = LMStreamError(code="auth", message="x")
        assert e.retryable is False

    def test_not_retryable_context_length(self):
        e = LMStreamError(code="context_length", message="x")
        assert e.retryable is False

    def test_provider_code(self):
        e = LMStreamError(
            code="rate_limit",
            message="too many",
            provider_code="rate_limit_exceeded",
        )
        assert e.provider_code == "rate_limit_exceeded"


class TestLMStreamEvent:
    def test_start(self):
        e = LMStreamEvent(type="start", model="gpt-4", id="resp-1")
        assert e.type == "start"
        assert e.model == "gpt-4"

    def test_delta_event(self):
        e = LMStreamEvent(
            type="delta",
            delta=PartDelta(type="text", text="hi"),
        )
        assert e.type == "delta"
        assert e.delta.text == "hi"

    def test_error_event(self):
        e = LMStreamEvent(
            type="error",
            error=LMStreamError(code="rate_limit", message="x"),
        )
        assert e.type == "error"
        assert e.error.code == "rate_limit"

    def test_end_event(self):
        e = LMStreamEvent(
            type="end",
            finish_reason="stop",
            usage=LMUsage(total_tokens=100),
        )
        assert e.type == "end"
        assert e.finish_reason == "stop"
        assert e.usage.total_tokens == 100

    def test_completion_on_end(self):
        from dspy.lm_types import LMCompletion
        completion = LMCompletion(parts=[TextPart(text="done")])
        e = LMStreamEvent(type="end", completion=completion)
        assert e.completion.text == "done"


class TestPartAccumulator:
    def test_accumulate_text(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="text", text="hel"))
        acc.feed(PartDelta(type="text", text="lo "))
        acc.feed(PartDelta(type="text", text="world"))
        parts = acc.build()
        assert len(parts) == 1
        assert isinstance(parts[0], TextPart)
        assert parts[0].text == "hello world"

    def test_accumulate_thinking(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="thinking", text="let me "))
        acc.feed(PartDelta(type="thinking", text="think"))
        parts = acc.build()
        assert len(parts) == 1
        assert isinstance(parts[0], ThinkingPart)
        assert parts[0].text == "let me think"

    def test_accumulate_tool_call(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="tool_call", part_index=0, id="c1", name="search", input='{"q":'))
        acc.feed(PartDelta(type="tool_call", part_index=0, input=' "hello"'))
        acc.feed(PartDelta(type="tool_call", part_index=0, input="}"))
        parts = acc.build()
        assert len(parts) == 1
        assert isinstance(parts[0], ToolCallPart)
        assert parts[0].id == "c1"
        assert parts[0].name == "search"
        assert parts[0].input == {"q": "hello"}

    def test_accumulate_multiple_tool_calls(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="tool_call", part_index=0, id="a", name="fn1", input='{}'))
        acc.feed(PartDelta(type="tool_call", part_index=1, id="b", name="fn2", input='{"x": 1}'))
        parts = acc.build()
        assert len(parts) == 2
        assert parts[0].id == "a"
        assert parts[1].id == "b"
        assert parts[1].input == {"x": 1}

    def test_accumulate_audio(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="audio", data="abc", media_type="audio/wav"))
        acc.feed(PartDelta(type="audio", data="def"))
        parts = acc.build()
        assert len(parts) == 1
        assert isinstance(parts[0], AudioPart)
        assert parts[0].data == "abcdef"

    def test_accumulate_mixed_text_and_tool(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="text", part_index=0, text="calling "))
        acc.feed(PartDelta(type="tool_call", part_index=1, id="c", name="fn", input="{}"))
        acc.feed(PartDelta(type="text", part_index=0, text="tool"))
        parts = acc.build()
        # Parts are ordered by part_index
        assert len(parts) == 2
        assert isinstance(parts[0], TextPart)
        assert parts[0].text == "calling tool"
        assert isinstance(parts[1], ToolCallPart)

    def test_tool_call_with_malformed_json_uses_json_repair(self):
        acc = PartAccumulator()
        # Intentionally malformed trailing
        acc.feed(PartDelta(type="tool_call", part_index=0, id="c", name="fn", input='{"x": 1,'))
        parts = acc.build()
        assert len(parts) == 1
        assert parts[0].input == {"x": 1}

    def test_reset(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="text", text="hi"))
        acc.reset()
        assert acc.build() == []

    def test_citation_part(self):
        acc = PartAccumulator()
        acc.feed(PartDelta(type="citation", cited_text="hello", source_index=0, title="S"))
        parts = acc.build()
        assert len(parts) == 1
        from dspy.lm_types import CitationPart
        assert isinstance(parts[0], CitationPart)
        assert parts[0].cited_text == "hello"
