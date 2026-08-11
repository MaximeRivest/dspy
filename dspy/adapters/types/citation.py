"""The `Citations` shape: claim spans paired with source-document indices.

The greenfield citations shape is deliberately neutral: a list of
`(span, doc)` pairs, where `doc` is the 1-based index into the call's
source documents. Both citation conducts materialize it — the native
provider channel through a channel routing, and inline `[n]` markers
through the `citations` parse combinator (adapter-ir-stage example 07).
"""

from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type


class Citations(Type):
    """Citations for an answer: claim spans tied to source documents.

    Attributes:
        citations: The individual citations, in emission order.

    Examples:
        ```python
        citations = dspy.Citations(
            [{"span": "Water boils at 100C.", "doc": 1}]
        )
        assert citations.citations[0].doc == 1
        ```
    """

    class Citation(Type):
        span: str
        doc: int | None = None

        def format(self) -> dict[str, Any]:
            return {"span": self.span, "doc": self.doc}

    citations: list[Citation]

    @classmethod
    def description(cls) -> str:
        return (
            "Citations must be a list of objects with `span` (the cited claim text) "
            "and `doc` (the 1-based source-document index)."
        )

    def format(self) -> list[dict[str, Any]]:
        return [citation.format() for citation in self.citations]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls):
            return data
        if isinstance(data, list):
            return {"citations": data}
        if isinstance(data, dict) and "citations" in data:
            return data
        raise ValueError(f"Received invalid value for `dspy.Citations`: {data!r}")

    def __iter__(self):
        return iter(self.citations)

    def __len__(self):
        return len(self.citations)
