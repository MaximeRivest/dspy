from __future__ import annotations

import re

from dspy.streaming.chunks import StreamChunk
from dspy.streaming.parsers.base import BaseStreamParser


class JSONStreamParser(BaseStreamParser):
    """Stream parser for JSONAdapter format.

    Tracks top-level output fields in a streamed JSON object and emits raw
    value text for the field currently being generated. The emitted chunks
    preserve the original JSON value tokenization, which matches the current
    ``StreamResponse`` behavior used by ``streamify``.
    """

    def __init__(self, output_fields: list[str]):
        super().__init__(output_fields)
        fields_alt = "|".join(re.escape(f) for f in output_fields)
        self._field_re = re.compile(rf'"({fields_alt})"\s*:')
        self._buffer = ""
        self._emitted = 0

    def feed(self, text: str) -> list[StreamChunk]:
        self._buffer += text
        return self._drain()

    def finalize(self) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []
        if self._current_field is not None:
            final_text = self._buffer[self._emitted :]
            if final_text:
                chunks.append(
                    StreamChunk(
                        type="output_field",
                        field=self._current_field,
                        text=final_text,
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
        self._buffer = ""
        self._current_field = None
        self._emitted = 0
        return chunks

    def _drain(self) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []

        while True:
            if self._current_field is None:
                match = self._field_re.search(self._buffer)
                if not match:
                    self._buffer = self._trim_prefix(self._buffer)
                    break

                self._current_field = match.group(1)
                self._buffer = self._buffer[match.end() :]
                self._buffer = self._buffer.lstrip()
                self._emitted = 0
                continue

            boundary = self._find_value_boundary(self._buffer)
            if boundary is None:
                new_text = self._buffer[self._emitted :]
                if new_text:
                    chunks.append(
                        StreamChunk(
                            type="output_field",
                            field=self._current_field,
                            text=new_text,
                        )
                    )
                    self._emitted = len(self._buffer)
                break

            final_text = self._buffer[:boundary]
            new_text = final_text[self._emitted :]
            chunks.append(
                StreamChunk(
                    type="output_field",
                    field=self._current_field,
                    text=new_text,
                    is_last=True,
                )
            )
            self._current_field = None
            self._buffer = self._buffer[boundary:]
            self._buffer = self._buffer.lstrip(", \n\r\t")
            self._emitted = 0

        return chunks

    def _trim_prefix(self, buf: str) -> str:
        # JSON field identifiers may arrive over many tiny chunks; keep the
        # full buffered prefix rather than trying to trim aggressively.
        return buf

    def _find_value_boundary(self, buf: str) -> int | None:
        in_string = False
        escape = False
        depth = 0

        for i, ch in enumerate(buf):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                if depth == 0:
                    return i
                depth -= 1
            elif ch == "," and depth == 0:
                return i

        return None
