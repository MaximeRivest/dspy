"""The typed direct-call vocabulary: thin constructors over lm15 messages.

`dspy.System`, `dspy.User`, `dspy.Assistant`, `dspy.ToolCall`, and
`dspy.ToolResult` spell the typed `lm(...)` call surface. They are
constructors, not new types: `User`/`Assistant`/`ToolResult` build
canonical `lm15.Message` values, `ToolCall` builds an `lm15.ToolCallPart`,
and `System` is a small marker that folds into the request's system
prompt (lm15 carries system as request data, not as a message role).

Because the LM IR (lm15) is already the canonical representation, this
layer is a few lines per name — no parallel type universe, no
experimental flag, no forward-contract negotiation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lm15 import ImagePart, Message, TextPart, ToolCallPart, tool_result

__all__ = ["Assistant", "System", "ToolCall", "ToolResult", "User"]


@dataclass(frozen=True)
class System:
    """A system instruction for a typed `lm(...)` call.

    lm15 carries the system prompt on the request, not as a message
    role, so this is a marker the call builder folds in — multiple
    `System` items concatenate in order.
    """

    text: str


def User(*content: Any) -> Message:
    """Build a user message from strings and content parts.

    Args:
        *content: Strings (become text parts), lm15 parts, or
            `dspy.Image` values (converted to lm15 image parts).

    Examples:
        ```python
        dspy.User("Describe this image.", dspy.Image("https://x.test/dog.png"))
        ```
    """
    return Message(role="user", parts=_parts(content))


def Assistant(*content: Any) -> Message:
    """Build an assistant message — prior model turns, tool calls included.

    Args:
        *content: Strings, lm15 parts (e.g. `dspy.ToolCall(...)` results),
            or `dspy.Image` values.
    """
    return Message(role="assistant", parts=_parts(content))


def ToolCall(*, id: str, name: str, args: dict[str, Any]) -> ToolCallPart:
    """Build a tool-call part for an `Assistant(...)` turn.

    Examples:
        ```python
        dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"}))
        ```
    """
    return ToolCallPart(id=id, name=name, input=dict(args))


def ToolResult(
    output: Any,
    *,
    call_id: str,
    name: str | None = None,
    is_error: bool = False,
) -> Message:
    """Build a tool-result message answering a prior `ToolCall`.

    Args:
        output: The tool's output — a string, or anything
            `lm15.tool_result` accepts (parts, JSON-able values).
        call_id: The id of the tool call this result answers.
        name: Optional tool name, echoed for providers that want it.
        is_error: Mark the result as a tool failure.
    """
    return Message(role="tool", parts=(tool_result(call_id, output, name=name, is_error=is_error),))


def _parts(content: tuple[Any, ...]) -> tuple[Any, ...]:
    """Convert mixed content into lm15 parts; refuse the unknown loudly."""
    if not content:
        raise ValueError("A message needs at least one content item.")
    parts: list[Any] = []
    for item in content:
        if isinstance(item, str):
            parts.append(TextPart(item))
        elif _is_adapter_image(item):
            parts.append(_image_part(item))
        else:
            # lm15 parts pass through; Message.__post_init__ refuses
            # anything that is not a part, with a teaching error.
            parts.append(item)
    return tuple(parts)


def _is_adapter_image(item: Any) -> bool:
    from dspy.adapters.types.image import Image

    return isinstance(item, Image)


def _image_part(image: Any) -> ImagePart:
    """Convert a `dspy.Image` (URL or data URI) to an lm15 image part."""
    url = image.url
    if isinstance(url, str) and url.startswith("data:"):
        head, _, data = url.partition(",")
        media_type = head[len("data:") :].split(";", 1)[0] or "image/png"
        return ImagePart(media_type=media_type, data=data)
    return ImagePart(url=url)
