import dataclasses
from typing import Any


@dataclasses.dataclass
class StreamChunk:
    """A tagged chunk from an LM stream.

    Every piece of incremental output — a token of an output field, a tool call
    fragment, reasoning content, a citation, or a status message — is wrapped
    in a StreamChunk with a descriptive ``type`` and optional ``field`` name.

    Attributes:
        type: The kind of chunk. One of ``"output_field"``, ``"tool_call"``,
            ``"reasoning"``, ``"citation"``, or ``"status"``.
        text: The chunk's text content.
        field: Output field name when ``type="output_field"`` (e.g. ``"answer"``).
        is_last: Whether this is the final chunk for the current field or type.
        metadata: Extra info (tool name, citation data, etc.).
    """

    type: str
    text: str = ""
    field: str | None = None
    is_last: bool = False
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type, "text": self.text}
        if self.field is not None:
            d["field"] = self.field
        if self.is_last:
            d["is_last"] = True
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self) -> str:
        import orjson

        return orjson.dumps(self.to_dict()).decode()
