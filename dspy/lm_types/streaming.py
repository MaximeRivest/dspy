"""Streaming primitives: PartDelta, LMStreamError, LMStreamEvent, PartAccumulator."""

from __future__ import annotations

from typing import Any, Literal

import pydantic

from .completion import LMCompletion, LMUsage
from .errors import RETRYABLE_ERRORS, ErrorCode, LMError, error_class
from .parts import (
    AudioPart,
    CitationPart,
    ImagePart,
    Part,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)


class PartDelta(pydantic.BaseModel):
    """A typed fragment of a Part arriving during streaming.

    ``part_index`` identifies which Part in the completion this delta
    belongs to.  Deltas with the same ``part_index`` accumulate into a
    single Part.

    Fields populated by ``type``:

    - ``text``, ``thinking`` → ``text``
    - ``tool_call`` → ``input`` (JSON fragment); plus ``id``/``name`` on first
    - ``audio``, ``image`` → ``data`` (base64 chunk)
    - ``citation`` → ``cited_text``, ``source_index``, ``url``, ``title``
    """

    type: Literal["text", "thinking", "tool_call", "audio", "image", "document", "citation"]
    part_index: int = 0

    # Content fields
    text: str | None = None
    data: str | None = None
    input: str | None = None

    # Identity (first delta only for tool_call)
    id: str | None = None
    name: str | None = None

    # Citation fields
    cited_text: str | None = None
    source_index: int | None = None
    url: str | None = None
    title: str | None = None

    # Media metadata (first delta only)
    media_type: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class LMStreamError(pydantic.BaseModel):
    """Structured error that arrived mid-stream.

    Providers can surface errors during streaming (not just before the
    stream starts).  The consumer can convert this to an exception with
    ``to_exception()`` or inspect ``retryable`` for retry logic.
    """

    code: ErrorCode
    message: str
    provider_code: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)

    def to_exception(self, *, model: str | None = None) -> LMError:
        """Convert to the corresponding ``LMError`` exception."""
        cls = error_class(self.code)
        if isinstance(cls, type) and issubclass(cls, LMError):
            return cls(self.message, model=model, provider_code=self.provider_code)
        return LMError(self.message, model=model, provider_code=self.provider_code)

    @property
    def retryable(self) -> bool:
        """Whether this error is safe to retry."""
        cls = error_class(self.code)
        return isinstance(cls, type) and issubclass(cls, RETRYABLE_ERRORS)


class LMStreamEvent(pydantic.BaseModel):
    """An event in the LM streaming protocol.

    Normal sequence::

        start → delta* → end

    With mid-stream error::

        start → delta* → error      (stream terminates)
    """

    type: Literal["start", "delta", "error", "end"]

    # start
    id: str | None = None
    model: str | None = None

    # delta
    delta: PartDelta | None = None
    completion_index: int = 0

    # error
    error: LMStreamError | None = None

    # end
    finish_reason: Literal["stop", "length", "tool_call", "error", "content_filter"] | None = None
    usage: LMUsage | None = None
    completion: LMCompletion | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class PartAccumulator:
    """Accumulates ``PartDelta``s into completed ``Part`` objects.

    Groups deltas by ``part_index``, concatenates content fragments,
    and produces a list of finalized Parts via ``build()``.
    """

    def __init__(self) -> None:
        self._buffers: dict[int, dict[str, Any]] = {}

    def feed(self, delta: PartDelta) -> None:
        """Feed a delta into the accumulator."""
        idx = delta.part_index
        if idx not in self._buffers:
            self._buffers[idx] = {
                "type": delta.type,
                "text_chunks": [],
                "data_chunks": [],
                "input_chunks": [],
                "id": None,
                "name": None,
                "media_type": None,
                "cited_text": None,
                "source_index": None,
                "url": None,
                "title": None,
            }

        buf = self._buffers[idx]

        # If this delta has a different type than the buffered one, keep
        # the first type (providers occasionally send stray deltas).
        if delta.text is not None:
            buf["text_chunks"].append(delta.text)
        if delta.data is not None:
            buf["data_chunks"].append(delta.data)
        if delta.input is not None:
            buf["input_chunks"].append(delta.input)

        # First-delta identity fields
        for attr in ("id", "name", "media_type", "cited_text",
                     "source_index", "url", "title"):
            v = getattr(delta, attr)
            if v is not None and buf[attr] is None:
                buf[attr] = v

    def build(self) -> list[Part]:
        """Build completed Parts from accumulated deltas."""
        import json as _json
        import json_repair

        parts: list[Part] = []
        for idx in sorted(self._buffers):
            buf = self._buffers[idx]
            t = buf["type"]

            if t == "text":
                parts.append(TextPart(text="".join(buf["text_chunks"])))
            elif t == "thinking":
                parts.append(ThinkingPart(text="".join(buf["text_chunks"])))
            elif t == "tool_call":
                raw_input = "".join(buf["input_chunks"]) or "{}"
                try:
                    parsed = _json.loads(raw_input)
                    if not isinstance(parsed, dict):
                        parsed = {}
                except _json.JSONDecodeError:
                    try:
                        parsed = json_repair.loads(raw_input) or {}
                        if not isinstance(parsed, dict):
                            parsed = {}
                    except Exception:
                        parsed = {}
                parts.append(ToolCallPart(
                    id=buf["id"] or "",
                    name=buf["name"] or "",
                    input=parsed,
                ))
            elif t == "audio":
                parts.append(AudioPart(
                    media_type=buf["media_type"] or "audio/wav",
                    data="".join(buf["data_chunks"]),
                ))
            elif t == "image":
                parts.append(ImagePart(
                    media_type=buf["media_type"] or "image/png",
                    data="".join(buf["data_chunks"]),
                ))
            elif t == "citation":
                parts.append(CitationPart(
                    cited_text=buf["cited_text"],
                    source_index=buf["source_index"],
                    url=buf["url"],
                    title=buf["title"],
                ))
            # document/text_unknown: fallback to text
            elif t == "document":
                # Not commonly emitted in streams; skip.
                pass

        return parts

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._buffers.clear()


__all__ = ["PartDelta", "LMStreamError", "LMStreamEvent", "PartAccumulator"]
