from __future__ import annotations

from typing import Any

from dspy.streaming.buffer import StreamBuffer
from dspy.streaming.chunks import StreamChunk
from dspy.streaming.parsers.base import BaseStreamParser


class ChunkInterceptor:
    """Sits between the LM and the streaming infrastructure.

    Implements the same async ``send`` interface as
    ``anyio.streams.memory.MemoryObjectSendStream`` so it can be installed as
    ``settings.send_stream``.  Receives raw LM response chunks, runs them
    through a :class:`BaseStreamParser` to produce tagged
    :class:`StreamChunk` objects, and deposits them in a
    :class:`StreamBuffer`.
    """

    def __init__(self, parser: BaseStreamParser, buffer: StreamBuffer):
        self.parser = parser
        self.buffer = buffer
        self.received_any: bool = False

    async def send(self, item: Any) -> None:
        """Handle one item from the LM streaming pipeline.

        Items can be ``ModelResponseStream`` chunks (from litellm),
        ``StatusMessage`` objects (from callbacks), or other types.
        """
        if self.buffer.is_cancelled:
            return

        # ── Status messages (from StatusStreamingCallback) ──
        from dspy.streaming.messages import StatusMessage

        if isinstance(item, StatusMessage):
            self.buffer.put(StreamChunk(type="status", text=item.message))
            return

        # ── LM response stream chunks ──
        from litellm import ModelResponseStream

        if not isinstance(item, ModelResponseStream):
            # Unknown item type — ignore silently
            return

        self.received_any = True

        # Content tokens → adapter-specific field tagging
        content = _get_content(item)
        if content:
            for chunk in self.parser.feed(content):
                self.buffer.put(chunk)

        # Reasoning / thinking tokens
        reasoning = _get_reasoning(item)
        if reasoning:
            self.buffer.put(StreamChunk(type="reasoning", text=reasoning))

        # Tool call fragments
        tool_data = _get_tool_calls(item)
        if tool_data:
            self.buffer.put(
                StreamChunk(type="tool_call", text="", metadata={"tool_calls": tool_data})
            )

        # Citation data
        citation_data = _get_citations(item)
        if citation_data:
            self.buffer.put(
                StreamChunk(type="citation", text="", metadata={"citation": citation_data})
            )

    def finalize(self) -> None:
        """Flush remaining text in the stream parser."""
        for chunk in self.parser.finalize():
            self.buffer.put(chunk)


# ── Raw-chunk field extractors ──────────────────────────────


def _get_content(chunk: Any) -> str | None:
    try:
        c = chunk.choices[0].delta.content
        return c if c else None
    except (IndexError, AttributeError):
        return None


def _get_reasoning(chunk: Any) -> str | None:
    try:
        return getattr(chunk.choices[0].delta, "reasoning_content", None)
    except (IndexError, AttributeError):
        return None


def _get_tool_calls(chunk: Any) -> list[dict] | None:
    try:
        tcs = chunk.choices[0].delta.tool_calls
        if not tcs:
            return None
        return [
            {
                "index": tc.index,
                "id": getattr(tc, "id", None),
                "function": {
                    "name": getattr(tc.function, "name", None),
                    "arguments": getattr(tc.function, "arguments", ""),
                },
            }
            for tc in tcs
        ]
    except (IndexError, AttributeError):
        return None


def _get_citations(chunk: Any) -> dict | None:
    try:
        delta = chunk.choices[0].delta
        psf = getattr(delta, "provider_specific_fields", None)
        if psf:
            return psf.get("citation")
    except (IndexError, AttributeError):
        pass
    return None
