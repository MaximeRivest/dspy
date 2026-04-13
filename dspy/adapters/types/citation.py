"""Citations type for DSPy — wraps lm15's native citation support."""

from typing import Any, Optional

import pydantic

from dspy.adapters.types.base_type import Type


class Citations(Type):
    """Citations from an LM response. lm15 provides these natively via resp.citations."""

    class Citation(Type):
        type: str = "char_location"
        cited_text: str
        document_index: int
        document_title: str | None = None
        start_char_index: int
        end_char_index: int

        def format(self):
            return {"type": self.type, "cited_text": self.cited_text,
                    "document_index": self.document_index,
                    "start_char_index": self.start_char_index,
                    "end_char_index": self.end_char_index}

    citations: list[Citation]

    @classmethod
    def from_dict_list(cls, dicts):
        return cls(citations=[cls.Citation(**d) for d in dicts])

    def format(self):
        return [c.format() for c in self.citations]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls): return data
        if isinstance(data, list) and all(isinstance(d, dict) and "cited_text" in d for d in data):
            return {"citations": [cls.Citation(**d) for d in data]}
        if isinstance(data, dict):
            if "citations" in data:
                return {"citations": [cls.Citation(**d) if isinstance(d, dict) else d for d in data["citations"]]}
            if "cited_text" in data:
                return {"citations": [cls.Citation(**data)]}
        raise ValueError(f"Invalid Citations: {data}")

    def __iter__(self): return iter(self.citations)
    def __len__(self): return len(self.citations)
    def __getitem__(self, i): return self.citations[i]

    @classmethod
    def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
        return signature.delete(field_name) if lm.model.startswith("anthropic/") else signature

    @classmethod
    def is_streamable(cls): return True

    @classmethod
    def parse_stream_chunk(cls, chunk) -> Optional["Citations"]:
        try:
            cd = chunk.choices[0].delta.provider_specific_fields.get("citation")
            return cls.from_dict_list([cd]) if cd else None
        except Exception:
            return None

    @classmethod
    def parse_lm_response(cls, response) -> Optional["Citations"]:
        if isinstance(response, dict) and "citations" in response:
            return cls.from_dict_list(response["citations"]) if isinstance(response["citations"], list) else None
        return None
