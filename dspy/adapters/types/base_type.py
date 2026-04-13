"""Base type for DSPy custom types (Image, Audio, etc.).

Custom types implement ``format()`` to produce lm15 Parts or text.
The ``serialize_model()`` hook renders them as text when they appear
inside prompt strings; multimodal content is collected separately by
the adapter via ``to_lm15_part()``.
"""

import json
from typing import TYPE_CHECKING, Any, Optional, get_args, get_origin

import pydantic

from dspy.clients.base_lm import BaseLM

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature


class Type(pydantic.BaseModel):
    """Base class for DSPy custom types used in signature fields.

    Subclasses must implement ``format()`` returning either:
    - A list of OpenAI content-block dicts (legacy)
    - A plain string

    For multimodal types, also implement ``to_lm15_part()`` returning
    an ``lm15.Part`` for direct lm15 integration.
    """

    def format(self) -> list[dict[str, Any]] | str:
        raise NotImplementedError

    def to_lm15_part(self):
        """Return an lm15 Part for this value, or None if text-only."""
        return None

    @classmethod
    def description(cls) -> str:
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation):
        """Extract all custom Type subclasses from a (possibly nested) annotation."""
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass
        origin = get_origin(annotation)
        if origin is None:
            return []
        result = []
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))
        return result

    @pydantic.model_serializer()
    def serialize_model(self):
        """Serialize for embedding in prompt text.

        Multimodal types return a short placeholder; the actual content
        is attached as an lm15 Part by the adapter.
        """
        formatted = self.format()
        if isinstance(formatted, str):
            return formatted
        # For multimodal types, return a placeholder — the adapter
        # collects the actual Part via to_lm15_part().
        try:
            if self.to_lm15_part() is not None:
                return "[media attached]"
        except Exception:
            pass
        # Fallback: inline the JSON (for types that produce content blocks
        # but don't implement to_lm15_part — e.g. Document for citations)
        return json.dumps(formatted, ensure_ascii=False)

    @classmethod
    def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
        return signature

    @classmethod
    def is_streamable(cls) -> bool:
        return False

    @classmethod
    def parse_stream_chunk(cls, chunk) -> Optional["Type"]:
        return None

    @classmethod
    def parse_lm_response(cls, response) -> Optional["Type"]:
        return None
