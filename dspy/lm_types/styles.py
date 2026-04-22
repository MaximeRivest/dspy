"""Style abstraction and built-in styles: chat, json, xml, csv, tsv.

A Style bundles three coupled behaviors:

1. Prompt layout — how fields render in the outgoing prompt
2. Parse — how to extract fields from a complete response
3. Stream parse — how to detect field boundaries during streaming

These three MUST agree because they describe the same textual format.
Bundling them into one ``Style`` class makes the contract explicit.
"""

from __future__ import annotations

import csv
import io
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Union, get_args, get_origin

import pydantic


# ═══════════════════════════════════════════════════════════════
# StreamParser protocol and FieldChunk
# ═══════════════════════════════════════════════════════════════

@dataclass
class FieldChunk:
    """A chunk of streamed content belonging to a specific output field."""
    field_name: str
    text: Any
    is_last: bool = False


class StreamParser(ABC):
    """Feed text deltas, yield field-boundary chunks.

    Stateful (accumulates text across ``feed`` calls).  Has no knowledge
    of litellm, LM providers, or the adapter's prompt format — only
    cares about field boundaries in the text stream.
    """

    @abstractmethod
    def feed(self, text_chunk: str) -> Iterator[FieldChunk]:
        ...

    @abstractmethod
    def finalize(self) -> Iterator[FieldChunk]:
        ...


# ═══════════════════════════════════════════════════════════════
# FieldLayout — how a single field is wrapped in text
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FieldLayout:
    """How to wrap field name+value pairs in text output.

    Templates use ``{name}`` and ``{value}`` as placeholders.
    """
    field_template: str
    separator: str = "\n\n"
    envelope_open: str = ""
    envelope_close: str = ""

    def render(self, items: list[tuple[str, str]]) -> str:
        """Render a list of (name, value) pairs."""
        chunks = [
            self.field_template.format(name=name, value=value)
            for name, value in items
        ]
        return self.envelope_open + self.separator.join(chunks) + self.envelope_close


# ═══════════════════════════════════════════════════════════════
# Style base class
# ═══════════════════════════════════════════════════════════════

class Style(ABC):
    """A complete prompt style: layout + parser + stream parser.

    Implement this to add a new format (e.g., TOML, YAML, custom markup).
    The built-in chat/json/xml/csv styles are just specific implementations.
    """

    name: str

    @abstractmethod
    def field_layout(self) -> FieldLayout:
        """How fields are laid out in the prompt (for value rendering)."""
        ...

    @abstractmethod
    def output_request(self, output_field_names: list[str]) -> str:
        """The 'respond with...' instruction for this style."""
        ...

    @abstractmethod
    def parse(
        self,
        completion: str,
        output_fields: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse a complete LM response back into field values.

        ``output_fields`` maps field name → FieldInfo-like object with
        ``.annotation``.  Unknown fields in the response should be ignored.
        """
        ...

    def stream_parser(
        self,
        output_field_names: list[str],
        output_fields: dict[str, Any] | None = None,
    ) -> StreamParser:
        """Create a streaming parser for this style.

        ``output_fields`` is the full dict (name → FieldInfo) if known;
        styles may use annotations to pick between rendering modes.
        Subclasses should accept both calling conventions for backward
        compatibility.
        """
        raise NotImplementedError

    # Optional hooks for adapter rendering.  Styles that want to
    # customize how output blocks are serialized for demos/history
    # can override ``format_output_block``.
    def format_output_block(
        self,
        outputs: dict[str, Any],
        columns: list[str],
    ) -> str:
        """Render the output side of a demo/history turn as text.

        Default implementation uses ``field_layout().render()`` with
        stringified values.  Override for non-trivial serialization
        (e.g., CSV multi-row, JSON object).
        """
        items = [(c, _stringify(outputs.get(c))) for c in columns]
        return self.field_layout().render(items)


def _stringify(value: Any) -> str:
    """Render a Python value as a text string for prompt layouts."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, pydantic.BaseModel):
        return value.model_dump_json()
    if isinstance(value, (list, tuple)):
        import json as _json
        try:
            return _json.dumps(list(value), ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, dict):
        import json as _json
        try:
            return _json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    return str(value)


# ═══════════════════════════════════════════════════════════════
# ChatStyle
# ═══════════════════════════════════════════════════════════════

class ChatStyle(Style):
    """Chat-style markers: ``[[ ## field ## ]]``.

    Matches the current ``dspy.ChatAdapter`` format.
    """
    name = "chat"

    _FIELD_RE = re.compile(r"\[\[ ## (\w+) ## \]\]")

    def field_layout(self) -> FieldLayout:
        return FieldLayout(
            field_template="[[ ## {name} ## ]]\n{value}",
            separator="\n\n",
            envelope_open="",
            envelope_close="\n\n[[ ## completed ## ]]\n",
        )

    def output_request(self, output_field_names: list[str]) -> str:
        fields = ", then ".join(f"`[[ ## {n} ## ]]`" for n in output_field_names)
        return (
            f"Respond with the corresponding output fields, starting with the field "
            f"{fields}, and then ending with the marker for `[[ ## completed ## ]]`."
        )

    def parse(self, completion: str, output_fields: dict[str, Any]) -> dict[str, Any]:
        sections: list[tuple[str | None, list[str]]] = [(None, [])]
        for line in completion.splitlines():
            m = self._FIELD_RE.match(line.strip())
            if m:
                header = m.group(1)
                remaining = line[m.end():].strip()
                sections.append((header, [remaining] if remaining else []))
            else:
                sections[-1][1].append(line)

        result: dict[str, Any] = {}
        for name, lines in sections:
            if name in output_fields and name not in result:
                result[name] = "\n".join(lines).strip()
        return result

    def stream_parser(
        self,
        output_field_names: list[str],
        output_fields: dict[str, Any] | None = None,
    ) -> StreamParser:
        return _ChatStreamParser(output_field_names)


class _ChatStreamParser(StreamParser):
    """Scans for ``[[ ## field ## ]]`` markers in a stream of text chunks."""

    _FIELD_RE = re.compile(r"\[\[ ## (\w+) ## \]\]")
    _MARKER_PREFIXES = ("[", "[[", "[[ ", "[[ #", "[[ ##", "[[ ## ")

    def __init__(self, output_field_names: list[str]) -> None:
        self.output_fields = set(output_field_names)
        self.current: str | None = None
        self.buffer = ""

    def feed(self, text_chunk: str) -> Iterator[FieldChunk]:
        self.buffer += text_chunk

        while True:
            m = self._FIELD_RE.search(self.buffer)
            if m:
                # Yield any content before the marker for the current field
                if self.current:
                    content = self.buffer[:m.start()].rstrip("\n")
                    # Always signal the end of this field, even when the only
                    # remaining text is whitespace (the previous chunk already
                    # yielded the last non-whitespace piece)
                    yield FieldChunk(self.current, content, is_last=True)
                # Switch to the new field (or None if it's `completed` or unknown)
                name = m.group(1)
                self.buffer = self.buffer[m.end():].lstrip("\n")
                self.current = name if name in self.output_fields else None
            else:
                # No marker in buffer; yield the portion that can't be a marker prefix
                if self.current and self.buffer:
                    keep = self._safe_tail_length(self.buffer)
                    if keep < len(self.buffer):
                        to_yield = self.buffer[:len(self.buffer) - keep]
                        if to_yield:
                            yield FieldChunk(self.current, to_yield)
                        self.buffer = self.buffer[len(self.buffer) - keep:]
                return

    def finalize(self) -> Iterator[FieldChunk]:
        if self.current and self.buffer.strip():
            yield FieldChunk(self.current, self.buffer.strip(), is_last=True)
        self.buffer = ""
        self.current = None

    def _safe_tail_length(self, buf: str) -> int:
        """Length of trailing portion that could be a partial marker."""
        for prefix in self._MARKER_PREFIXES[::-1]:
            if buf.endswith(prefix):
                return len(prefix)
        return 0


# ═══════════════════════════════════════════════════════════════
# JSONStyle
# ═══════════════════════════════════════════════════════════════

class JSONStyle(Style):
    """JSON object output format."""
    name = "json"

    def field_layout(self) -> FieldLayout:
        return FieldLayout(
            field_template='"{name}": {value}',
            separator=",\n",
            envelope_open="{\n",
            envelope_close="\n}",
        )

    def output_request(self, output_field_names: list[str]) -> str:
        fields = ", then ".join(f"`{n}`" for n in output_field_names)
        return f"Respond with a JSON object in the following order of fields: {fields}."

    def parse(self, completion: str, output_fields: dict[str, Any]) -> dict[str, Any]:
        import json_repair

        text = completion.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = _strip_code_fence(text)

        parsed: Any
        try:
            parsed = json_repair.loads(text)
        except Exception:
            parsed = None

        if not isinstance(parsed, dict):
            # Try to find a JSON object in the text
            m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
            if m:
                try:
                    parsed = json_repair.loads(m.group(0))
                except Exception:
                    parsed = None

        if not isinstance(parsed, dict):
            return {}

        return {k: v for k, v in parsed.items() if k in output_fields}

    def stream_parser(
        self,
        output_field_names: list[str],
        output_fields: dict[str, Any] | None = None,
    ) -> StreamParser:
        return _JSONStreamParser(output_field_names)

    def format_output_block(
        self,
        outputs: dict[str, Any],
        columns: list[str],
    ) -> str:
        import json as _json

        def _serialize(v):
            if isinstance(v, pydantic.BaseModel):
                return v.model_dump()
            return v

        d = {c: _serialize(outputs.get(c)) for c in columns}
        try:
            return _json.dumps(d, indent=2, ensure_ascii=False)
        except Exception:
            return _json.dumps({c: str(outputs.get(c, "")) for c in columns}, indent=2)


class _JSONStreamParser(StreamParser):
    """Partial JSON parser using jiter."""

    def __init__(self, output_field_names: list[str]) -> None:
        self.output_fields = list(output_field_names)
        self.buffer = ""
        self.emitted: dict[str, str] = {}

    def feed(self, text_chunk: str) -> Iterator[FieldChunk]:
        self.buffer += text_chunk
        yield from self._try_parse()

    def _try_parse(self) -> Iterator[FieldChunk]:
        text = self.buffer.strip()
        if not text.startswith("{"):
            # Try to find the JSON object start
            idx = text.find("{")
            if idx == -1:
                return
            text = text[idx:]

        try:
            import jiter
            parsed = jiter.from_json(
                text.encode("utf-8"),
                partial_mode="trailing-strings",
            )
        except Exception:
            return

        if not isinstance(parsed, dict):
            return

        for name in self.output_fields:
            if name in parsed:
                current = str(parsed[name])
                previous = self.emitted.get(name, "")
                if len(current) > len(previous):
                    delta = current[len(previous):]
                    self.emitted[name] = current
                    yield FieldChunk(name, delta, is_last=False)

    def finalize(self) -> Iterator[FieldChunk]:
        # Mark each emitted field as complete
        for name in self.emitted:
            yield FieldChunk(name, "", is_last=True)
        self.buffer = ""


# ═══════════════════════════════════════════════════════════════
# XMLStyle
# ═══════════════════════════════════════════════════════════════

class XMLStyle(Style):
    """XML tag output format: ``<field>value</field>``."""
    name = "xml"

    def field_layout(self) -> FieldLayout:
        return FieldLayout(
            field_template="<{name}>\n{value}\n</{name}>",
            separator="\n",
        )

    def output_request(self, output_field_names: list[str]) -> str:
        tags = ", then ".join(f"`<{n}>`" for n in output_field_names)
        return f"Respond with the output fields wrapped in XML tags {tags}."

    def parse(self, completion: str, output_fields: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in output_fields:
            pattern = re.compile(rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>", re.DOTALL)
            m = pattern.search(completion)
            if m:
                result[name] = m.group(1).strip()
        return result

    def stream_parser(
        self,
        output_field_names: list[str],
        output_fields: dict[str, Any] | None = None,
    ) -> StreamParser:
        return _XMLStreamParser(output_field_names)


class _XMLStreamParser(StreamParser):
    def __init__(self, output_field_names: list[str]) -> None:
        self.output_fields = set(output_field_names)
        self.buffer = ""
        self.current: str | None = None
        self.close_tag = ""

    def feed(self, text_chunk: str) -> Iterator[FieldChunk]:
        self.buffer += text_chunk
        while True:
            if self.current is None:
                m = re.search(r"<(\w+)>", self.buffer)
                if not m:
                    return
                name = m.group(1)
                if name not in self.output_fields:
                    # Skip unknown tag; consume up to end of it
                    self.buffer = self.buffer[m.end():]
                    continue
                self.current = name
                self.close_tag = f"</{name}>"
                self.buffer = self.buffer[m.end():]
            # Look for closing tag
            idx = self.buffer.find(self.close_tag)
            if idx == -1:
                # Not complete; yield safe prefix
                keep = len(self.close_tag) + 2  # buffer enough for partial
                if len(self.buffer) > keep:
                    safe = self.buffer[:-keep]
                    if safe:
                        yield FieldChunk(self.current, safe)
                    self.buffer = self.buffer[-keep:]
                return
            # Field complete
            yield FieldChunk(self.current, self.buffer[:idx].strip("\n"), is_last=True)
            self.buffer = self.buffer[idx + len(self.close_tag):]
            self.current = None
            self.close_tag = ""

    def finalize(self) -> Iterator[FieldChunk]:
        if self.current and self.buffer.strip():
            yield FieldChunk(self.current, self.buffer.strip(), is_last=True)
        self.buffer = ""


# ═══════════════════════════════════════════════════════════════
# CSVStyle
# ═══════════════════════════════════════════════════════════════

class CSVStyle(Style):
    """CSV / tabular output format.

    Supports two modes:

    - Single-row: each output field is a column, one data row.
    - Multi-row: single list-valued output field; each row becomes
      one element of the list.

    Mode is detected automatically from the signature.
    """

    name = "csv"

    def __init__(self, delimiter: str = ",", quoting: int = csv.QUOTE_MINIMAL):
        self.delimiter = delimiter
        self.quoting = quoting

    def field_layout(self) -> FieldLayout:
        return FieldLayout(
            field_template="  - {name}: {value}",
            separator="\n",
            envelope_open="Columns:\n",
            envelope_close="",
        )

    def output_request(self, output_field_names: list[str]) -> str:
        cols = self.delimiter.join(output_field_names)
        return (
            f"Respond with a CSV block using delimiter {self.delimiter!r}. "
            f"First output the header row, then one or more data rows. "
            f"Expected columns: {cols}. "
            f"Use double quotes around values containing the delimiter, quotes, or newlines. "
            f"Begin your response with the marker `[[ ## rows ## ]]` followed by the CSV content."
        )

    def format_output_block(
        self,
        outputs: dict[str, Any],
        columns: list[str],
    ) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf, delimiter=self.delimiter, quoting=self.quoting)

        writer.writerow(columns)

        # Multi-row: single list-valued output
        if len(outputs) == 1:
            only_key = next(iter(outputs))
            only_value = outputs[only_key]
            if isinstance(only_value, list) and only_value:
                for item in only_value:
                    writer.writerow(self._row_from_item(item, columns))
                return buf.getvalue().rstrip()

        # Single-row
        writer.writerow([_stringify(outputs.get(c)) for c in columns])
        return buf.getvalue().rstrip()

    @staticmethod
    def _row_from_item(item: Any, columns: list[str]) -> list[str]:
        if isinstance(item, dict):
            return [_stringify(item.get(c)) for c in columns]
        if isinstance(item, pydantic.BaseModel):
            d = item.model_dump()
            return [_stringify(d.get(c)) for c in columns]
        return [_stringify(item)] + [""] * (len(columns) - 1)

    def parse(self, completion: str, output_fields: dict[str, Any]) -> dict[str, Any]:
        text = self._extract_csv_block(completion)

        reader = csv.reader(io.StringIO(text), delimiter=self.delimiter)
        rows = [row for row in reader if row]

        if not rows:
            return {}

        header = [h.strip() for h in rows[0]]
        data_rows = rows[1:]

        if self._is_multi_row_mode(output_fields):
            return self._parse_multi_row(header, data_rows, output_fields)
        return self._parse_single_row(header, data_rows, output_fields)

    def _extract_csv_block(self, completion: str) -> str:
        marker = "[[ ## rows ## ]]"
        if marker in completion:
            completion = completion.split(marker, 1)[1]

        stripped = completion.strip()
        if stripped.startswith("```"):
            stripped = _strip_code_fence(stripped)

        # Stop at next `[[ ## ... ## ]]` marker
        m = re.search(r"\[\[ ## \w+ ## \]\]", stripped)
        if m:
            stripped = stripped[:m.start()]
        return stripped.strip()

    @staticmethod
    def _is_multi_row_mode(output_fields: dict[str, Any]) -> bool:
        if len(output_fields) != 1:
            return False
        only = next(iter(output_fields.values()))
        annotation = getattr(only, "annotation", only)
        return get_origin(annotation) is list

    def _parse_single_row(
        self,
        header: list[str],
        data_rows: list[list[str]],
        output_fields: dict[str, Any],
    ) -> dict[str, Any]:
        if not data_rows:
            return {}
        row = data_rows[0]
        return {
            col: (row[idx] if idx < len(row) else "")
            for idx, col in enumerate(header)
            if col in output_fields
        }

    def _parse_multi_row(
        self,
        header: list[str],
        data_rows: list[list[str]],
        output_fields: dict[str, Any],
    ) -> dict[str, Any]:
        only_key = next(iter(output_fields))
        only_field = output_fields[only_key]
        annotation = getattr(only_field, "annotation", only_field)
        args = get_args(annotation)
        elem_type = args[0] if args else dict

        items: list[Any] = []
        for row in data_rows:
            item = {
                header[i]: (row[i] if i < len(row) else "")
                for i in range(min(len(header), len(row)))
            }
            if isinstance(elem_type, type) and issubclass(elem_type, pydantic.BaseModel):
                items.append(elem_type.model_validate(item))
            else:
                items.append(item)
        return {only_key: items}

    def stream_parser(
        self,
        output_field_names: list[str],
        output_fields: dict[str, Any] | None = None,
    ) -> StreamParser:
        multi_row = False
        if output_fields is not None:
            multi_row = self._is_multi_row_mode(output_fields)
        elif len(output_field_names) == 1:
            # Ambiguous without annotations; default to single-row (cell streaming)
            multi_row = False
        return _CSVStreamParser(
            output_field_names,
            delimiter=self.delimiter,
            multi_row=multi_row,
        )


class _CSVStreamParser(StreamParser):
    """Streams CSV output row-by-row using a quote-state machine.

    When ``multi_row`` is True, emits one ``FieldChunk`` per row under
    the single output field name (with ``text`` being the row dict).
    Otherwise emits one ``FieldChunk`` per cell, keyed by column name.
    """

    _MARKER = "[[ ## rows ## ]]"

    def __init__(
        self,
        output_field_names: list[str],
        delimiter: str = ",",
        multi_row: bool = False,
    ) -> None:
        self.output_fields = list(output_field_names)
        self.delimiter = delimiter
        self.buffer = ""
        self.header: list[str] | None = None
        self.started = False
        self.multi_row = multi_row

    def feed(self, text_chunk: str) -> Iterator[FieldChunk]:
        self.buffer += text_chunk

        if not self.started:
            idx = self.buffer.find(self._MARKER)
            if idx == -1:
                # Keep only enough tail to match a partial marker
                if len(self.buffer) > len(self._MARKER):
                    self.buffer = self.buffer[-len(self._MARKER):]
                return
            self.buffer = self.buffer[idx + len(self._MARKER):].lstrip()
            self.started = True

        while True:
            row_end = self._find_row_end(self.buffer)
            if row_end is None:
                return
            raw_row = self.buffer[:row_end]
            self.buffer = self.buffer[row_end + 1:]
            if not raw_row.strip():
                continue

            parsed = next(csv.reader(io.StringIO(raw_row), delimiter=self.delimiter))

            if self.header is None:
                self.header = [c.strip() for c in parsed]
                # Auto-detect mode if not explicitly set:
                # - If single output field AND header doesn't contain it
                #   → multi-row (header has a different schema)
                # - Otherwise → single-row (header matches field names)
                if not self.multi_row and len(self.output_fields) == 1:
                    only_key = self.output_fields[0]
                    if only_key not in self.header:
                        self.multi_row = True
                continue

            if self.multi_row and len(self.output_fields) == 1:
                only_key = self.output_fields[0]
                row_dict = {
                    self.header[i]: (parsed[i] if i < len(parsed) else "")
                    for i in range(len(self.header))
                }
                yield FieldChunk(only_key, row_dict, is_last=False)
            else:
                for i, col in enumerate(self.header):
                    if col in self.output_fields and i < len(parsed):
                        yield FieldChunk(col, parsed[i], is_last=True)

    def finalize(self) -> Iterator[FieldChunk]:
        if self.buffer.strip() and self.header is not None:
            try:
                parsed = next(csv.reader(io.StringIO(self.buffer), delimiter=self.delimiter))
            except Exception:
                parsed = []
            if parsed:
                if self.multi_row and len(self.output_fields) == 1:
                    only_key = self.output_fields[0]
                    row_dict = {
                        self.header[i]: (parsed[i] if i < len(parsed) else "")
                        for i in range(len(self.header))
                    }
                    yield FieldChunk(only_key, row_dict, is_last=True)
                else:
                    for i, col in enumerate(self.header):
                        if col in self.output_fields and i < len(parsed):
                            yield FieldChunk(col, parsed[i], is_last=True)
        self.buffer = ""

    def _find_row_end(self, text: str) -> int | None:
        in_quote = False
        i = 0
        n = len(text)
        while i < n:
            ch = text[i]
            if ch == '"':
                if in_quote and i + 1 < n and text[i + 1] == '"':
                    i += 2
                    continue
                in_quote = not in_quote
            elif ch == "\n" and not in_quote:
                return i
            i += 1
        return None


class TSVStyle(CSVStyle):
    """Tab-separated values — CSV with ``\\t`` delimiter."""
    name = "tsv"

    def __init__(self):
        super().__init__(delimiter="\t")


# ═══════════════════════════════════════════════════════════════
# Style registry
# ═══════════════════════════════════════════════════════════════

_STYLES: dict[str, Style] = {}

StyleLike = Union[str, Style]


def register_style(style: Style) -> Style:
    """Register a style under its ``name``.  Returns the style."""
    _STYLES[style.name] = style
    return style


def get_style(name_or_instance: StyleLike) -> Style:
    """Look up a registered style by name, or return an instance directly."""
    if isinstance(name_or_instance, Style):
        return name_or_instance
    if name_or_instance not in _STYLES:
        raise KeyError(
            f"Unknown style: {name_or_instance!r}. "
            f"Available: {sorted(_STYLES)}. "
            f"Register custom styles with register_style(my_style)."
        )
    return _STYLES[name_or_instance]


def available_styles() -> list[str]:
    """Return all registered style names."""
    return sorted(_STYLES)


# Register built-ins
register_style(ChatStyle())
register_style(JSONStyle())
register_style(XMLStyle())
register_style(CSVStyle())
register_style(TSVStyle())


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _strip_code_fence(text: str) -> str:
    """Strip triple-backtick code fences around a block."""
    lines = text.splitlines()
    if not lines:
        return text
    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


__all__ = [
    "FieldChunk",
    "FieldLayout",
    "StreamParser",
    "Style",
    "StyleLike",
    "ChatStyle",
    "JSONStyle",
    "XMLStyle",
    "CSVStyle",
    "TSVStyle",
    "register_style",
    "get_style",
    "available_styles",
]
