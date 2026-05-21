"""Rendering strategies for experimental `dspy.Document` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args, get_origin

from dspy.adapters.types.document import Document
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMDocumentPart, LMRequestPatch, LMTextPart


@dataclass(frozen=True)
class NativeDocument(TypeStrategy[Document]):
    """Render document inputs as normalized document parts."""

    marker_type: type[Document] = Document

    def matches(self, annotation: Any) -> bool:
        try:
            if annotation == self.marker_type or (isinstance(annotation, type) and issubclass(annotation, self.marker_type)):
                return True
        except TypeError:
            return False
        origin = get_origin(annotation)
        return origin in (list, tuple) and any(self.matches(arg) for arg in get_args(annotation))

    def render_input(self, *, field_name: str, field: Any, value: Any, adapter: Any) -> LMRequestPatch:
        values = value if isinstance(value, list | tuple) else [value]
        parts = [LMTextPart(text=f"\n\n{field_name}:\n")]
        parts.extend(_document_value_to_lm_part(item) for item in values)
        return LMRequestPatch(delete_input_fields=(field_name,), user_parts=parts)


def _document_value_to_lm_part(document: Document) -> LMDocumentPart:
    return LMDocumentPart(
        source={
            "type": "text" if document.media_type == "text/plain" else "base64",
            "media_type": document.media_type,
            "data": document.data,
        },
        citations={"enabled": True},
        title=document.title,
        context=document.context,
    )
