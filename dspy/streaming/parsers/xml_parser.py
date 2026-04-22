from __future__ import annotations

import re

from dspy.streaming.chunks import StreamChunk
from dspy.streaming.parsers.base import BaseStreamParser


class XMLStreamParser(BaseStreamParser):
    """Stream parser for XMLAdapter format.

    Detects ``<field_name>`` opening tags and ``</field_name>`` closing tags
    to determine field boundaries.  Text inside a tag pair is tagged with
    the corresponding field name.
    """

    def __init__(self, output_fields: list[str]):
        super().__init__(output_fields)
        self._buffer = ""

        fields_alt = "|".join(re.escape(f) for f in output_fields)
        self._open_re = re.compile(rf"<({fields_alt})>")
        self._close_re = re.compile(rf"</({fields_alt})>")

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
                    text=self._buffer.strip(),
                    is_last=True,
                )
            )
        self._buffer = ""
        self._current_field = None
        return chunks

    def _drain(self) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []

        while True:
            if self._current_field is None:
                # Looking for an opening tag
                match = self._open_re.search(self._buffer)
                if match:
                    self._current_field = match.group(1)
                    self._buffer = self._buffer[match.end() :]
                else:
                    # Hold back a trailing '<' that might be an incomplete tag
                    if self._buffer.rstrip().endswith("<"):
                        break
                    self._buffer = ""
                    break
            else:
                # Inside a field — look for the closing tag
                close = self._close_re.search(self._buffer)
                if close:
                    before = self._buffer[: close.start()]
                    if before.strip():
                        chunks.append(
                            StreamChunk(
                                type="output_field",
                                field=self._current_field,
                                text=before.strip(),
                                is_last=True,
                            )
                        )
                    self._current_field = None
                    self._buffer = self._buffer[close.end() :]
                else:
                    safe = self._safe_emit_length()
                    if safe > 0:
                        chunks.append(
                            StreamChunk(
                                type="output_field",
                                field=self._current_field,
                                text=self._buffer[:safe],
                            )
                        )
                        self._buffer = self._buffer[safe:]
                    break

        return chunks

    def _safe_emit_length(self) -> int:
        """Characters safe to emit without splitting a closing tag."""
        idx = self._buffer.rfind("<")
        if idx >= 0 and not self._close_re.search(self._buffer[idx:]):
            return idx
        return len(self._buffer)
