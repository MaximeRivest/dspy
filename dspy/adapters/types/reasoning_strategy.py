"""Rendering strategies for `dspy.Reasoning` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMConfig, LMOutput, LMReasoningConfig, LMRequestPatch


@dataclass(frozen=True)
class NativeReasoning(TypeStrategy[Reasoning]):
    """Use the LM's native reasoning channel for `Reasoning` outputs."""

    marker_type: type[Reasoning] = Reasoning
    reasoning_effort: str | None = "low"
    effort: str | None = None
    max_tokens: int | None = None
    summary: str | None = None

    def render_output(self, *, field_name: str, field: Any, adapter: Any) -> LMRequestPatch:
        effort = self.effort if self.effort is not None else self.reasoning_effort
        return LMRequestPatch(
            delete_output_fields=(field_name,),
            config=LMConfig(
                reasoning=LMReasoningConfig(
                    effort=effort,
                    max_tokens=self.max_tokens,
                    summary=self.summary,
                )
            ),
        )

    def parse_output(
        self,
        *,
        field_name: str,
        output: LMOutput | dict[str, Any] | str,
        field: Any | None = None,
        adapter: Any | None = None,
    ) -> Reasoning | None:
        reasoning = _reasoning_content(output)
        if reasoning is None:
            return None
        return Reasoning(reasoning)


@dataclass(frozen=True)
class TextReasoning(TypeStrategy[Reasoning]):
    """Keep `Reasoning` as an ordinary adapter-rendered text field."""

    marker_type: type[Reasoning] = Reasoning
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class CodeCommentReasoning(TextReasoning):
    """Render reasoning text in a code-comment style when adapters opt in."""

    comment_prefix: str = "#"


@dataclass(frozen=True)
class MidMessageReasoning(TextReasoning):
    """Render reasoning as a mid-message scratchpad when adapters opt in."""

    heading: str = "Reason briefly before answering."
    tag_name: str = "reasoning"


def _reasoning_content(output: LMOutput | dict[str, Any] | str) -> str | None:
    if isinstance(output, LMOutput):
        return output.reasoning_content
    if isinstance(output, dict):
        value = output.get("reasoning_content")
        return None if value is None else str(value)
    return None
