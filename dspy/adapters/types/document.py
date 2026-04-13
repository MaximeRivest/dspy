"""Document type for DSPy signatures — for citation-enabled LM responses."""

from typing import Any, Literal

import pydantic

from dspy.adapters.types.base_type import Type
from dspy.utils.annotation import experimental


@experimental(version="3.0.4")
class Document(Type):
    """A document that can be cited by language models."""

    data: str
    title: str | None = None
    media_type: Literal["text/plain", "application/pdf"] = "text/plain"
    context: str | None = None

    def format(self) -> list[dict[str, Any]]:
        block = {
            "type": "document",
            "source": {"type": "text", "media_type": self.media_type, "data": self.data},
            "citations": {"enabled": True},
        }
        if self.title:
            block["title"] = self.title
        if self.context:
            block["context"] = self.context
        return [block]

    def to_lm15_part(self):
        from lm15.types import Part
        return Part.document(data=self.data, media_type=self.media_type)

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data
        if isinstance(data, str):
            return {"data": data}
        if isinstance(data, dict):
            return data
        raise ValueError(f"Invalid Document value: {data}")

    def __str__(self):
        title = f"'{self.title}': " if self.title else ""
        return f"Document({title}{len(self.data)} chars)"
