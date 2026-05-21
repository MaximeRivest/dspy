"""Rendering strategies for experimental `dspy.Citations` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dspy.adapters.types.citation import Citations
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMOutput, LMRequestPatch


@dataclass(frozen=True)
class NativeCitations(TypeStrategy[Citations]):
    """Use native provider citation metadata for `Citations` outputs."""

    marker_type: type[Citations] = Citations

    def render_output(self, *, field_name: str, field: Any, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(delete_output_fields=(field_name,))

    def parse_output(
        self,
        *,
        field_name: str,
        output: LMOutput | dict[str, Any] | str,
        field: Any | None = None,
        adapter: Any | None = None,
    ) -> Citations | None:
        if isinstance(output, LMOutput):
            if not output.citations:
                return None
            return Citations.from_dict_list([_citation_part_to_dict(citation) for citation in output.citations])
        if isinstance(output, dict):
            citations = output.get("citations") or output.get(field_name)
            if isinstance(citations, Citations):
                return citations
            if isinstance(citations, list):
                return Citations.from_dict_list([_citation_part_to_dict(citation) for citation in citations])
        return None


def _citation_part_to_dict(citation: Any) -> dict[str, Any]:
    data = citation.model_dump(exclude_none=True) if hasattr(citation, "model_dump") else dict(citation)
    cited_text = data.get("cited_text") or data.get("text") or data.get("supported_text") or ""
    return {
        "cited_text": cited_text,
        "document_index": data.get("document_index", 0),
        "document_title": data.get("document_title") or data.get("title"),
        "start_char_index": data.get("start_char_index", 0),
        "end_char_index": data.get("end_char_index", len(cited_text)),
        "supported_text": data.get("supported_text"),
    }
