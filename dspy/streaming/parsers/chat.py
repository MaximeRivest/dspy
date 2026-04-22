from __future__ import annotations

import re

from dspy.streaming.chunks import StreamChunk
from dspy.streaming.parsers.base import BaseStreamParser

# Matches  [[ ## field_name ## ]]
_HEADER_RE = re.compile(r"\[\[ ## (\w+) ## \]\]")

# If the buffer ends with any of these strings it *might* be the start of a
# header that hasn't fully arrived yet, so we hold it back.
_PARTIAL_PREFIXES = ["[", "[[", "[[ ", "[[ #", "[[ ##"]


class ChatStreamParser(BaseStreamParser):
    """Stream parser for ChatAdapter format.

    Detects ``[[ ## field_name ## ]]`` headers to determine which output field
    incoming tokens belong to.  Text between headers is tagged with the
    current field name.
    """

    def __init__(self, output_fields: list[str]):
        super().__init__(output_fields)
        self._buffer = ""

    # ── Public API ──────────────────────────────────────────

    def feed(self, text: str) -> list[StreamChunk]:
        self._buffer += text
        return self._drain()

    def finalize(self) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []
        if self._current_field and self._buffer.strip():
            chunks.append(
                StreamChunk(
                    type="output_field",
                    field=self._current_field,
                    text=self._buffer,
                    is_last=True,
                )
            )
        self._buffer = ""
        self._current_field = None
        return chunks

    # ── Internals ───────────────────────────────────────────

    def _drain(self) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []

        while True:
            match = _HEADER_RE.search(self._buffer)

            if match:
                before = self._buffer[: match.start()]
                field_name = match.group(1)
                after = self._buffer[match.end() :]

                # Emit text that preceded the header. If the only thing
                # before the next header is whitespace, still emit a final
                # zero-length marker so consumers know the field ended.
                if self._current_field:
                    text = before.rstrip("\n")
                    if text:
                        chunks.append(
                            StreamChunk(
                                type="output_field",
                                field=self._current_field,
                                text=text,
                                is_last=True,
                            )
                        )
                    else:
                        chunks.append(
                            StreamChunk(
                                type="output_field",
                                field=self._current_field,
                                text="",
                                is_last=True,
                            )
                        )

                # Transition to new field (or end)
                if field_name == "completed":
                    self._current_field = None
                elif field_name in self.output_fields:
                    self._current_field = field_name
                # else: unknown header — ignore

                self._buffer = after.lstrip("\n")
            else:
                # No complete header — emit what's safe
                safe = self._safe_emit_length()
                if self._current_field and safe > 0:
                    text = self._buffer[:safe]
                    if safe < len(self._buffer):
                        # Hold back separator newlines that belong to an
                        # incoming header rather than the field content.
                        text = text.rstrip("\n")
                    if text:
                        chunks.append(
                            StreamChunk(
                                type="output_field",
                                field=self._current_field,
                                text=text,
                            )
                        )
                    self._buffer = self._buffer[safe:]
                break

        return chunks

    def _safe_emit_length(self) -> int:
        """Number of leading characters that can be emitted without risk of
        splitting a field header."""
        buf = self._buffer

        # Could the tail be the start of a header?
        for prefix in _PARTIAL_PREFIXES:
            if buf.endswith(prefix):
                return len(buf) - len(prefix)

        # Could there be a partial header *inside* the buffer?
        # e.g. "hello [[ ## ans"  —  hold from "[[ ##" onward.
        marker = "[[ ##"
        idx = buf.rfind(marker)
        if idx >= 0 and not _HEADER_RE.search(buf[idx:]):
            return idx

        return len(buf)
