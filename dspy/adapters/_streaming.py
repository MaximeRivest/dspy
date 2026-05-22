"""Private adapter-owned streaming field parsers.

These parsers know the textual delimiters emitted by each adapter. The public
streaming facade owns predictor/listener routing; adapters own how their own
rendered output is incrementally sliced into field chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from queue import Queue
from typing import Any

import jiter


@dataclass
class _AdapterStreamChunk:
    chunk: str
    is_last_chunk: bool = False


class _AdapterFieldStreamParser:
    """Incrementally extract one output field from adapter text deltas."""

    start_identifier: str
    end_identifier: Any
    start_indicator: str
    end_pattern_prefixes: tuple[str, ...] = ()
    end_pattern_contains: str | None = None

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.field_start_queue: list[str] = []
        self.field_end_queue: Queue[str] = Queue()
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False

    def reset_for_reuse(self) -> None:
        self.field_start_queue = []
        self.field_end_queue = Queue()
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False

    def receive_text(self, chunk_message: str) -> _AdapterStreamChunk | None:
        if chunk_message and self.start_identifier in chunk_message and not isinstance(self, _JSONAdapterStreamParser):
            message_after_start_identifier = chunk_message[chunk_message.find(self.start_identifier) + len(self.start_identifier) :]
            if re.search(self.end_identifier, message_after_start_identifier):
                self.cache_hit = True
                self.stream_start = True
                self.stream_end = True
                return None

        if len(self.field_start_queue) == 0 and not self.stream_start and self.start_indicator in chunk_message:
            self.field_start_queue.append(chunk_message)
            return None

        if len(self.field_start_queue) > 0 and not self.stream_start:
            self.field_start_queue.append(chunk_message)
            concat_message = "".join(self.field_start_queue)

            if self.start_identifier in concat_message:
                self.stream_start = True
                self.field_start_queue = []
                value_start_index = concat_message.find(self.start_identifier) + len(self.start_identifier)
                chunk_message = concat_message[value_start_index:].lstrip()
                self.on_stream_start()
            elif _buffered_message_end_with_start_identifier(concat_message.strip(), self.start_identifier):
                return None
            else:
                self.field_start_queue = []
                return None

        if self.stream_start and chunk_message:
            self.field_end_queue.put(chunk_message)
            token = None
            concat_message = "".join(self.field_end_queue.queue).strip()

            if not self._could_form_end_identifier(concat_message):
                token = self.flush()
            elif self.field_end_queue.qsize() > 10:
                token = self.field_end_queue.get()

            return self.handle_stream_chunk(token, chunk_message)
        return None

    def on_stream_start(self) -> None:
        return None

    def handle_stream_chunk(self, token: str | None, chunk_message: str) -> _AdapterStreamChunk | None:
        return self._default_handle_stream_chunk(token)

    def _default_handle_stream_chunk(self, token: str | None) -> _AdapterStreamChunk | None:
        concat_message = "".join(self.field_end_queue.queue).strip()

        if re.search(self.end_identifier, concat_message):
            self.stream_end = True
            last_token = self.flush()
            token = token + last_token if token else last_token
            token = token.rstrip()

        if token or self.stream_end:
            return _AdapterStreamChunk(token or "", is_last_chunk=self.stream_end)
        return None

    def flush(self) -> str:
        last_tokens = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()
        return self._trim_at_boundary(last_tokens)

    def finalize(self) -> _AdapterStreamChunk | None:
        if self.stream_end or not self.stream_start:
            return None
        self.stream_end = True
        if self.field_end_queue.qsize() > 0:
            token = self.flush()
            if token:
                return _AdapterStreamChunk(token, is_last_chunk=True)
        return None

    def _trim_at_boundary(self, text: str) -> str:
        return text

    def _could_form_end_identifier(self, concat_message: str) -> bool:
        if any(concat_message.endswith(prefix) for prefix in self.end_pattern_prefixes):
            return True
        if self.end_pattern_contains and self.end_pattern_contains in concat_message:
            return True
        return False


class _ChatAdapterStreamParser(_AdapterFieldStreamParser):
    def __init__(self, field_name: str):
        super().__init__(field_name)
        self.start_identifier = f"[[ ## {field_name} ## ]]"
        self.end_identifier = re.compile(r"\[\[ ## (\w+) ## \]\]")
        self.start_indicator = "["
        self.end_pattern_prefixes = ("[", "[[", "[[ ", "[[ #", "[[ ##")
        self.end_pattern_contains = "[[ ##"

    def _trim_at_boundary(self, text: str) -> str:
        boundary_index = text.find("[[")
        return text[: boundary_index if boundary_index != -1 else len(text)]


class _XMLAdapterStreamParser(_AdapterFieldStreamParser):
    def __init__(self, field_name: str):
        super().__init__(field_name)
        self.start_identifier = f"<{field_name}>"
        self.end_identifier = re.compile(rf"</{field_name}>")
        self.start_indicator = "<"
        self.end_pattern_prefixes = ("<", "</")
        self.end_pattern_contains = "</"

    def _trim_at_boundary(self, text: str) -> str:
        boundary_index = text.find(f"</{self.field_name}>")
        return text[: boundary_index if boundary_index != -1 else len(text)]


class _JSONAdapterStreamParser(_AdapterFieldStreamParser):
    def __init__(self, field_name: str):
        super().__init__(field_name)
        self.start_identifier = f'"{field_name}":'
        self.end_identifier = re.compile(r'\w*"(,|\s*})')
        self.start_indicator = '"'
        self.end_pattern_prefixes = ('"', '",', '" ', '"}')
        self.end_pattern_contains = "}"
        self.field_accumulated_messages = ""

    def reset_for_reuse(self) -> None:
        super().reset_for_reuse()
        self.field_accumulated_messages = ""

    def on_stream_start(self) -> None:
        self.field_accumulated_messages += "{" + self.start_identifier

    def handle_stream_chunk(self, token: str | None, chunk_message: str) -> _AdapterStreamChunk | None:
        self.field_accumulated_messages += chunk_message
        if self.field_accumulated_messages.rstrip().endswith("}"):
            try:
                jiter.from_json(self.field_accumulated_messages.encode("utf-8"))
                self.stream_end = True
                last_token = self.flush()
                right_curly_bracket_index = last_token.rfind("}")
                token = token + last_token[:right_curly_bracket_index] if token else last_token[:right_curly_bracket_index]
                return _AdapterStreamChunk(token or "", is_last_chunk=self.stream_end)
            except ValueError:
                pass

        try:
            parsed = jiter.from_json(
                self.field_accumulated_messages.encode("utf-8"),
                partial_mode="trailing-strings",
            )
            if len(parsed) > 1:
                self.stream_end = True
                last_token = self.flush()
                keys = list(parsed.keys())
                next_field_name = next((key for key in keys if key != self.field_name), None)
                if next_field_name is not None:
                    last_token_index = last_token.find(next_field_name)
                    token = token + last_token[:last_token_index] if token else last_token[:last_token_index]
        except ValueError:
            pass

        if token or self.stream_end:
            return _AdapterStreamChunk(token or "", is_last_chunk=self.stream_end)
        return None

    def _trim_at_boundary(self, text: str) -> str:
        return text


def _buffered_message_end_with_start_identifier(concat_message: str, start_identifier: str) -> bool:
    for i in range(len(concat_message)):
        if start_identifier.startswith(concat_message[len(concat_message) - i - 1 :]):
            return True
    return False
