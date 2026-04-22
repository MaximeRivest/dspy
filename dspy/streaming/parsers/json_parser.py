from __future__ import annotations

from dspy.streaming.chunks import StreamChunk
from dspy.streaming.parsers.base import BaseStreamParser


class JSONStreamParser(BaseStreamParser):
    """Stream parser for JSONAdapter format.

    Uses partial JSON parsing (via ``jiter``) to detect field boundaries in
    a JSON object being streamed token-by-token.  As each new key is
    discovered, subsequent tokens are tagged with that field name.
    """

    def __init__(self, output_fields: list[str]):
        super().__init__(output_fields)
        self._accumulated = ""
        self._emitted_per_field: dict[str, int] = {}

    def feed(self, text: str) -> list[StreamChunk]:
        self._accumulated += text
        return self._drain()

    def finalize(self) -> list[StreamChunk]:
        chunks: list[StreamChunk] = []
        if self._current_field:
            # Emit whatever the partial parse last had for the field
            chunks.append(
                StreamChunk(
                    type="output_field",
                    field=self._current_field,
                    text="",
                    is_last=True,
                )
            )
        self._current_field = None
        return chunks

    def _drain(self) -> list[StreamChunk]:
        try:
            import jiter
        except ImportError:
            # Without jiter we can't do incremental JSON parsing —
            # fall back to emitting nothing until finalize.
            return []

        chunks: list[StreamChunk] = []

        try:
            parsed = jiter.from_json(
                self._accumulated.encode("utf-8"),
                partial_mode="trailing-strings",
            )
        except (ValueError, Exception):
            return chunks

        if not isinstance(parsed, dict):
            return chunks

        fields_found = [k for k in parsed if k in self.output_fields]
        if not fields_found:
            return chunks

        # The last key in the parsed dict is the one currently being generated
        current = fields_found[-1]

        if current != self._current_field:
            # Field transition — emit remaining text of the *old* field
            if self._current_field:
                old_value = str(parsed.get(self._current_field, ""))
                already = self._emitted_per_field.get(self._current_field, 0)
                remainder = old_value[already:]
                if remainder:
                    chunks.append(
                        StreamChunk(
                            type="output_field",
                            field=self._current_field,
                            text=remainder,
                            is_last=True,
                        )
                    )
                    self._emitted_per_field[self._current_field] = len(old_value)
                else:
                    chunks.append(
                        StreamChunk(type="output_field", field=self._current_field, text="", is_last=True)
                    )

            self._current_field = current

        # Emit new text for current field
        current_value = str(parsed.get(current, ""))
        already = self._emitted_per_field.get(current, 0)
        new_text = current_value[already:]

        if new_text:
            chunks.append(StreamChunk(type="output_field", field=current, text=new_text))
            self._emitted_per_field[current] = len(current_value)

        return chunks
