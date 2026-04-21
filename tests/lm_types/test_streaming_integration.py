"""End-to-end streaming tests: LM stream → Adapter → field-level StreamResponse."""


import asyncio

import pytest

import dspy
from dspy.lm_types import (
    AdapterV2,
    BaseLMv2,
    ChatStyle,
    CSVStyle,
    LMCompletion,
    LMConfig,
    LMMessage,
    LMResponse,
    LMStreamEvent,
    LMUsage,
    PartDelta,
    TextPart,
    ThinkingPart,
    from_preset,
)


class _StreamingLM(BaseLMv2):
    """LM whose stream() yields a predefined sequence of events."""

    def __init__(self, events, model="stream"):
        super().__init__(model=model)
        self._events = events

    def forward(self, messages, config):
        # Derive a response from the streamed events for fallback purposes
        text = "".join(
            e.delta.text for e in self._events
            if e.type == "delta" and e.delta.type == "text" and e.delta.text
        )
        return LMResponse(completions=[LMCompletion(parts=[TextPart(text=text)])])

    async def aforward(self, messages, config):
        return self.forward(messages, config)

    async def stream(self, messages, config=None):
        for event in self._events:
            yield event


@pytest.fixture
def simple_signature():
    class Sig(dspy.Signature):
        """Answer."""
        q: str = dspy.InputField()
        a: str = dspy.OutputField()
    return Sig


class TestStreamCall:
    def test_chat_style_streams_field_chunks(self, simple_signature):
        events = [
            LMStreamEvent(type="start", model="x"),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="[[ ## a ## ]]\n")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="Par")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="is")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="\n\n[[ ## completed ## ]]")),
            LMStreamEvent(type="end", finish_reason="stop"),
        ]
        lm = _StreamingLM(events)
        adapter = from_preset("chat")

        collected = []

        async def run():
            async for chunk in adapter.stream_call(
                lm, LMConfig(), simple_signature, [], {"q": "hi"}
            ):
                collected.append(chunk)

        asyncio.run(run())

        # Should have chunks for field "a"
        a_chunks = [c for c in collected if c.field_name == "a"]
        assert len(a_chunks) >= 1
        text = "".join(c.text for c in a_chunks)
        assert "Paris" in text

    def test_json_style_streams(self, simple_signature):
        events = [
            LMStreamEvent(type="start"),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text='{"a": "P')),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text='aris"}')),
            LMStreamEvent(type="end", finish_reason="stop"),
        ]
        lm = _StreamingLM(events)
        adapter = from_preset("json")

        collected = []

        async def run():
            async for chunk in adapter.stream_call(
                lm, LMConfig(), simple_signature, [], {"q": "hi"}
            ):
                collected.append(chunk)

        asyncio.run(run())
        a_chunks = [c for c in collected if c.field_name == "a"]
        full = "".join(c.text for c in a_chunks)
        assert "Paris" in full

    def test_xml_style_streams(self, simple_signature):
        events = [
            LMStreamEvent(type="start"),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="<a>Par")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="is</a>")),
            LMStreamEvent(type="end", finish_reason="stop"),
        ]
        lm = _StreamingLM(events)
        adapter = from_preset("xml")

        collected = []

        async def run():
            async for chunk in adapter.stream_call(
                lm, LMConfig(), simple_signature, [], {"q": "hi"}
            ):
                collected.append(chunk)

        asyncio.run(run())
        a_chunks = [c for c in collected if c.field_name == "a"]
        assert "Paris" in "".join(c.text for c in a_chunks)


class TestThinkingStream:
    def test_thinking_streams_to_reasoning_field(self):
        from dspy.adapters.types import Reasoning

        class Sig(dspy.Signature):
            """Answer."""
            q: str = dspy.InputField()
            reasoning: Reasoning = dspy.OutputField()
            a: str = dspy.OutputField()

        events = [
            LMStreamEvent(type="start"),
            LMStreamEvent(type="delta", delta=PartDelta(type="thinking", text="let me ")),
            LMStreamEvent(type="delta", delta=PartDelta(type="thinking", text="think...")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="[[ ## a ## ]]\n")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="Answer\n\n[[ ## completed ## ]]")),
            LMStreamEvent(type="end", finish_reason="stop"),
        ]
        lm = _StreamingLM(events)
        adapter = from_preset("chat")

        collected = []

        async def run():
            async for chunk in adapter.stream_call(
                lm, LMConfig(reasoning_effort="low"), Sig, [], {"q": "hi"}
            ):
                collected.append(chunk)

        asyncio.run(run())

        # Check reasoning chunks
        reasoning = [c for c in collected if c.field_name == "reasoning"]
        assert len(reasoning) >= 1
        assert "think" in "".join(c.text for c in reasoning)


class TestToolCallsInStream:
    def test_tool_calls_accumulated_and_emitted(self):
        from dspy.adapters.types import Tool, ToolCalls

        class Sig(dspy.Signature):
            query: str = dspy.InputField()
            tools: list[Tool] = dspy.InputField()
            tool_calls: ToolCalls = dspy.OutputField()

        events = [
            LMStreamEvent(type="start"),
            LMStreamEvent(type="delta", delta=PartDelta(
                type="tool_call", part_index=0, id="c1", name="search", input='{"q":',
            )),
            LMStreamEvent(type="delta", delta=PartDelta(
                type="tool_call", part_index=0, input=' "hello"}',
            )),
            LMStreamEvent(type="end", finish_reason="tool_call"),
        ]
        lm = _StreamingLM(events)
        adapter = AdapterV2(style="chat")

        def search(q: str) -> str:
            return q

        collected = []

        async def run():
            async for chunk in adapter.stream_call(
                lm, LMConfig(), Sig, [], {"query": "hi", "tools": [Tool(search)]}
            ):
                collected.append(chunk)

        asyncio.run(run())
        tool_chunks = [c for c in collected if c.field_name == "tool_calls"]
        assert len(tool_chunks) >= 1


class TestStreamError:
    def test_stream_error_raises(self, simple_signature):
        from dspy.lm_types import LMStreamError, RateLimitError

        events = [
            LMStreamEvent(type="start"),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="[[ ## a ## ]]\nPart")),
            LMStreamEvent(
                type="error",
                error=LMStreamError(code="rate_limit", message="slow down"),
            ),
        ]
        lm = _StreamingLM(events)
        adapter = from_preset("chat")

        async def run():
            async for _ in adapter.stream_call(
                lm, LMConfig(), simple_signature, [], {"q": "hi"}
            ):
                pass

        with pytest.raises(RateLimitError):
            asyncio.run(run())


class TestCSVStreaming:
    def test_multi_row_streaming(self):
        class Sig(dspy.Signature):
            """Extract events."""
            text: str = dspy.InputField()
            events: list = dspy.OutputField()

        events_data = [
            LMStreamEvent(type="start"),
            LMStreamEvent(type="delta", delta=PartDelta(
                type="text", text="[[ ## rows ## ]]\nname,year\n",
            )),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="Eclipse,2024\n")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="Olympics,2024\n")),
            LMStreamEvent(type="end", finish_reason="stop"),
        ]
        lm = _StreamingLM(events_data)
        adapter = AdapterV2(style="csv")

        collected = []

        async def run():
            async for chunk in adapter.stream_call(
                lm, LMConfig(), Sig, [], {"text": "x"}
            ):
                collected.append(chunk)

        asyncio.run(run())
        rows = [c for c in collected if c.field_name == "events"]
        assert len(rows) >= 2


class TestPartDeltaAccumulation:
    def test_accumulator_in_stream_call(self, simple_signature):
        """PartAccumulator should track full completion across stream."""
        events = [
            LMStreamEvent(type="start"),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="[[ ## a ## ]]\n")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="Paris")),
            LMStreamEvent(type="delta", delta=PartDelta(type="text", text="\n\n[[ ## completed ## ]]")),
            LMStreamEvent(type="end", finish_reason="stop"),
        ]
        lm = _StreamingLM(events)
        adapter = from_preset("chat")

        collected = []

        async def run():
            async for chunk in adapter.stream_call(
                lm, LMConfig(), simple_signature, [], {"q": "hi"}
            ):
                collected.append(chunk)

        asyncio.run(run())

        # Last chunks should be marked is_last for field "a"
        a_chunks = [c for c in collected if c.field_name == "a"]
        assert any(c.is_last for c in a_chunks)
