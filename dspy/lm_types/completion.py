"""LMCompletion and LMResponse — typed LM outputs."""

from __future__ import annotations

from typing import Any, Literal

import pydantic

from .parts import CitationPart, Part, TextPart, ThinkingPart, ToolCallPart


class LMUsage(pydantic.BaseModel):
    """Token usage for one LM call."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cache_read_tokens: int | None = None
    reasoning_tokens: int | None = None
    input_audio_tokens: int | None = None
    output_audio_tokens: int | None = None

    model_config = pydantic.ConfigDict(frozen=True)


class LMCompletion(pydantic.BaseModel):
    """A single completion from the LM.

    Replaces the current ``str | dict`` with a typed object that always
    has the same shape.  ``parts`` contains all content the model produced
    for this completion.
    """

    parts: list[Part]
    finish_reason: Literal["stop", "length", "tool_call", "error", "content_filter"] | None = None
    logprobs: Any | None = None

    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def text(self) -> str | None:
        """Concatenated text from all ``TextPart``s."""
        texts = [p.text for p in self.parts if isinstance(p, TextPart)]
        return "\n".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """All ``ToolCallPart``s in this completion."""
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    @property
    def thinking(self) -> str | None:
        """Concatenated text from all ``ThinkingPart``s."""
        texts = [p.text for p in self.parts if isinstance(p, ThinkingPart)]
        return "\n".join(texts) if texts else None

    @property
    def citations(self) -> list[CitationPart]:
        """All ``CitationPart``s in this completion."""
        return [p for p in self.parts if isinstance(p, CitationPart)]


class LMResponse(pydantic.BaseModel):
    """Complete response from one LM call.

    Contains one or more completions (one per ``n``) plus call-level metadata.
    """

    completions: list[LMCompletion]
    usage: LMUsage = pydantic.Field(default_factory=LMUsage)
    model: str | None = None
    id: str | None = None

    model_config = pydantic.ConfigDict(frozen=True)


__all__ = ["LMCompletion", "LMResponse", "LMUsage"]
