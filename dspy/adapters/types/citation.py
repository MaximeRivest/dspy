"""Citations type for DSPy signatures — wraps lm15's native citation support."""

from typing import Any, Optional

import pydantic

from dspy.adapters.types.base_type import Type
from dspy.utils.annotation import experimental


@experimental(version="3.0.4")
class Citations(Type):
    """Citations extracted from an LM response with source references."""

    class Citation(Type):
        type: str = "char_location"
        cited_text: str
        document_index: int
        document_title: str | None = None
        start_char_index: int
        end_char_index: int
        supported_text: str | None = None

        def format(self) -> dict[str, Any]:
            d = {"type": self.type, "cited_text": self.cited_text,
                 "document_index": self.document_index,
                 "start_char_index": self.start_char_index,
                 "end_char_index": self.end_char_index}
            if self.document_title: d["document_title"] = self.document_title
            if self.supported_text: d["supported_text"] = self.supported_text
            return d

    citations: list[Citation]

    @classmethod
    def from_dict_list(cls, dicts: list[dict[str, Any]]) -> "Citations":
        return cls(citations=[cls.Citation(**d) for d in dicts])

    def format(self) -> list[dict[str, Any]]:
        return [c.format() for c in self.citations]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data
        if isinstance(data, list) and all(isinstance(d, dict) and "cited_text" in d for d in data):
            return {"citations": [cls.Citation(**d) for d in data]}
        if isinstance(data, dict):
            if "citations" in data:
                return {"citations": [cls.Citation(**d) if isinstance(d, dict) else d for d in data["citations"]]}
            if "cited_text" in data:
                return {"citations": [cls.Citation(**data)]}
        raise ValueError(f"Invalid Citations value: {data}")

    def __iter__(self):
        return iter(self.citations)

    def __len__(self):
        return len(self.citations)

    def __getitem__(self, i):
        return self.citations[i]

    @classmethod
    def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
        if lm.model.startswith("anthropic/"):
            return signature.delete(field_name)
        return signature

    @classmethod
    def is_streamable(cls) -> bool:
        return True

    @classmethod
    def parse_stream_chunk(cls, chunk) -> Optional["Citations"]:
        try:
            delta = chunk.choices[0].delta
            if hasattr(delta, "provider_specific_fields") and delta.provider_specific_fields:
                cd = delta.provider_specific_fields.get("citation")
                if cd:
                    return cls.from_dict_list([cd])
        except Exception:
            pass
        return None

    @classmethod
    def parse_lm_response(cls, response: str | dict[str, Any]) -> Optional["Citations"]:
        if isinstance(response, dict) and "citations" in response:
            cd = response["citations"]
            if isinstance(cd, list):
                return cls.from_dict_list(cd)
        return None
