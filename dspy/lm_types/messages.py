"""Typed messages — the unit of LM conversation.

``LMMessage`` replaces untyped ``dict[str, Any]`` messages with a typed
container whose content is always a list of ``Part`` objects.  A plain
text message is just ``[TextPart(text="...")]``.
"""

from __future__ import annotations

from typing import Literal, Union

import pydantic

from .parts import Part, TextPart, ThinkingPart, ToolCallPart


class LMMessage(pydantic.BaseModel):
    """A single message in the LM conversation.

    Roles:
        - ``system``: high-authority system prompt
        - ``user``: end-user input
        - ``assistant``: model output
        - ``tool``: tool execution result
    """

    role: Literal["system", "user", "assistant", "tool"]
    parts: list[Part]

    model_config = pydantic.ConfigDict(frozen=True)

    @pydantic.model_validator(mode="after")
    def _check_parts_nonempty(self) -> "LMMessage":
        if not self.parts:
            raise ValueError("LMMessage requires at least one part")
        return self

    # ── Convenience constructors ──

    @staticmethod
    def system(text: str) -> "LMMessage":
        return LMMessage(role="system", parts=[TextPart(text=text)])

    @staticmethod
    def user(content: Union[str, list[Part]]) -> "LMMessage":
        if isinstance(content, str):
            return LMMessage(role="user", parts=[TextPart(text=content)])
        return LMMessage(role="user", parts=list(content))

    @staticmethod
    def assistant(content: Union[str, list[Part]]) -> "LMMessage":
        if isinstance(content, str):
            return LMMessage(role="assistant", parts=[TextPart(text=content)])
        return LMMessage(role="assistant", parts=list(content))

    # ── Accessors ──

    @property
    def text(self) -> str | None:
        """Concatenated text from all ``TextPart``s, or ``None``."""
        texts = [p.text for p in self.parts if isinstance(p, TextPart)]
        return "\n".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        """All ``ToolCallPart``s in this message."""
        return [p for p in self.parts if isinstance(p, ToolCallPart)]

    @property
    def thinking(self) -> str | None:
        """Concatenated text from all ``ThinkingPart``s, or ``None``."""
        texts = [p.text for p in self.parts if isinstance(p, ThinkingPart)]
        return "\n".join(texts) if texts else None


__all__ = ["LMMessage"]
