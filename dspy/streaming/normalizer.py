from __future__ import annotations

from typing import Any

from dspy.streaming.chunks import StreamChunk
from dspy.streaming.parsers.base import BaseStreamParser


class StreamChunkNormalizer:
    """Normalize raw provider stream items into adapter-aware ``StreamChunk`` objects.

    This is the boundary where provider-specific stream objects are translated
    into DSPy-owned streaming primitives. Higher layers should consume
    ``StreamChunk`` rather than provider chunk types directly.
    """

    def __init__(self, parser: BaseStreamParser):
        self.parser = parser
        self.received_any = False

    def feed(self, item: Any) -> list[StreamChunk]:
        from dspy.streaming.messages import StatusMessage

        if isinstance(item, StatusMessage):
            return [StreamChunk(type="status", text=item.message)]

        content = _get_content(item)
        if content is None:
            return []

        self.received_any = True
        return self.parser.feed(content)

    def finalize(self) -> list[StreamChunk]:
        return self.parser.finalize()


def _get_content(chunk: Any) -> str | None:
    try:
        content = chunk.choices[0].delta.content
        return content if content else None
    except (IndexError, AttributeError, TypeError):
        return None
