"""Document type for DSPy — delegates to lm15.Part.document."""

from typing import Any, Literal

import pydantic

from dspy.adapters.types.base_type import Type

try:
    from lm15.types import Part as _Part
    _HAS_LM15 = True
except ImportError:
    _HAS_LM15 = False


class Document(Type):
    data: str
    title: str | None = None
    media_type: Literal["text/plain", "application/pdf"] = "text/plain"

    def format(self):
        d = {"type": "document", "source": {"type": "text", "media_type": self.media_type, "data": self.data}, "citations": {"enabled": True}}
        if self.title:
            d["title"] = self.title
        return [d]

    def to_lm15_part(self):
        return _Part.document(data=self.data, media_type=self.media_type) if _HAS_LM15 else None

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls): return data
        if isinstance(data, str): return {"data": data}
        if isinstance(data, dict): return data
        raise ValueError(f"Invalid Document: {data}")

    def __str__(self):
        return f"Document({self.title + ': ' if self.title else ''}{len(self.data)} chars)"
