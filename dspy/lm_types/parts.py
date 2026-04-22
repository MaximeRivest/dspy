"""Typed content parts — the atoms of message content.

Each DSPy custom type (Image, Audio, File, Document) can convert to
one of these parts via ``to_part()``.  Parts flow through LMMessage
and LMCompletion as typed building blocks.
"""

from __future__ import annotations

from typing import Any, Literal, Union

import pydantic


class TextPart(pydantic.BaseModel):
    """A block of text content."""
    type: Literal["text"] = "text"
    text: str

    model_config = pydantic.ConfigDict(frozen=True)


class ImagePart(pydantic.BaseModel):
    """An image, addressed by ``data`` (base64), ``url``, or ``file_id``."""
    type: Literal["image"] = "image"
    media_type: str = "image/png"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class AudioPart(pydantic.BaseModel):
    """Audio content, addressed by ``data``, ``url``, or ``file_id``."""
    type: Literal["audio"] = "audio"
    media_type: str = "audio/wav"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class DocumentPart(pydantic.BaseModel):
    """A document (PDF, citation source, etc.).

    Used with citation-capable providers like Anthropic's citations API.
    """
    type: Literal["document"] = "document"
    media_type: str = "text/plain"
    data: str | None = None
    url: str | None = None
    file_id: str | None = None
    title: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class ToolCallPart(pydantic.BaseModel):
    """The model requests a tool invocation."""
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    input: dict[str, Any]

    model_config = pydantic.ConfigDict(frozen=True)


class ToolResultPart(pydantic.BaseModel):
    """Result of a tool invocation, sent back to the model."""
    type: Literal["tool_result"] = "tool_result"
    id: str
    content: list["Part"]
    is_error: bool = False

    model_config = pydantic.ConfigDict(frozen=True)


class ThinkingPart(pydantic.BaseModel):
    """Model reasoning trace (extended thinking / chain-of-thought)."""
    type: Literal["thinking"] = "thinking"
    text: str

    model_config = pydantic.ConfigDict(frozen=True)


class CitationPart(pydantic.BaseModel):
    """A reference to source material."""
    type: Literal["citation"] = "citation"
    cited_text: str | None = None
    source_index: int | None = None
    url: str | None = None
    title: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)


Part = Union[
    TextPart,
    ImagePart,
    AudioPart,
    DocumentPart,
    ToolCallPart,
    ToolResultPart,
    ThinkingPart,
    CitationPart,
]
"""The discriminated union of all content parts.

The ``type`` field acts as the discriminator.  Use ``isinstance()`` to
narrow and access variant-specific fields.
"""


# Rebuild for forward references
ToolResultPart.model_rebuild()


__all__ = [
    "TextPart",
    "ImagePart",
    "AudioPart",
    "DocumentPart",
    "ToolCallPart",
    "ToolResultPart",
    "ThinkingPart",
    "CitationPart",
    "Part",
]
