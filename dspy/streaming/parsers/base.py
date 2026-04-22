from __future__ import annotations

from dspy.streaming.chunks import StreamChunk


class BaseStreamParser:
    """Base class for adapter-specific incremental stream parsing.

    Each adapter format (Chat delimiters, JSON, XML) has its own subclass
    that detects field boundaries in the raw token stream and produces
    tagged :class:`StreamChunk` objects.

    Subclasses must implement :meth:`feed` and :meth:`finalize`.
    """

    def __init__(self, output_fields: list[str]):
        self.output_fields = output_fields
        self._current_field: str | None = None

    def feed(self, text: str) -> list[StreamChunk]:
        """Process incoming text and return zero or more tagged chunks.

        May buffer text internally when a field boundary might be forming.
        """
        raise NotImplementedError

    def finalize(self) -> list[StreamChunk]:
        """Flush any remaining buffered text as final chunks."""
        raise NotImplementedError
