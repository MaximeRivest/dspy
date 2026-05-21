"""Rendering strategies for `dspy.History` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dspy.adapters.types.history import History
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMMessage, LMRequestPatch, LMTextPart


@dataclass(frozen=True)
class NativeHistory(TypeStrategy[History]):
    """Expand `History` into prior user/assistant messages."""

    marker_type: type[History] = History

    def render_input(self, *, field_name: str, field: Any, value: History, adapter: Any) -> LMRequestPatch:
        messages: list[LMMessage] = []

        signature = getattr(adapter, "_current_signature", None)
        if signature is not None:
            signature = signature.delete(field_name)

        for turn in value.messages:
            if adapter is not None and signature is not None:
                if hasattr(adapter, "render_demo_user_message") and hasattr(adapter, "render_demo_assistant_message"):
                    user_text = adapter.render_demo_user_message(signature, turn, True)
                    assistant_text = adapter.render_demo_assistant_message(
                        signature,
                        turn,
                        "Not supplied for this conversation history message. ",
                    )
                elif hasattr(adapter, "format_user_message_content") and hasattr(adapter, "format_assistant_message_content"):
                    user_text = adapter.format_user_message_content(signature, turn)
                    assistant_text = adapter.format_assistant_message_content(signature, turn)
                else:
                    user_text = _turn_text(turn, prefix="User")
                    assistant_text = _turn_text(turn, prefix="Assistant")
            else:
                user_text = _turn_text(turn, prefix="User")
                assistant_text = _turn_text(turn, prefix="Assistant")

            messages.append(LMMessage(role="user", parts=_content_to_parts(user_text)))
            messages.append(LMMessage(role="assistant", parts=_content_to_parts(assistant_text)))

        return LMRequestPatch(delete_input_fields=(field_name,), messages=messages)


@dataclass(frozen=True)
class TextHistory(TypeStrategy[History]):
    """Embed `History` as text in the current user message."""

    marker_type: type[History] = History
    heading: str = "Conversation history:"

    def render_input(self, *, field_name: str, field: Any, value: History, adapter: Any) -> LMRequestPatch:
        lines = [self.heading]
        for index, turn in enumerate(value.messages, start=1):
            lines.append(f"\nTurn {index}:")
            for key, item in turn.items():
                lines.append(f"{key}: {item}")
        return LMRequestPatch(
            delete_input_fields=(field_name,),
            user_parts=[LMTextPart(text="\n".join(lines) + "\n\n")],
        )


def _content_to_parts(value: Any) -> list[Any]:
    return [LMTextPart(text=value)] if isinstance(value, str) else list(value)


def _turn_text(turn: dict[str, Any], *, prefix: str) -> str:
    lines = [f"{prefix} turn:"]
    lines.extend(f"{key}: {value}" for key, value in turn.items())
    return "\n".join(lines)
