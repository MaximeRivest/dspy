"""Rendering strategies for `dspy.Code` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import TypeAdapter

from dspy.adapters.types.code import Code
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMOutput, LMRequestPatch, LMTextPart


@dataclass(frozen=True)
class NativeCode(TypeStrategy[Code]):
    """Read/write code as native normalized response text artifacts.

    DSPy does not yet have a dedicated `LMCodePart`, so this strategy uses
    `LMTextPart(metadata={"dspy_field": field_name})` as the temporary native
    artifact convention.
    """

    marker_type: type[Code] = Code

    def render_output(self, *, field_name: str, field: Any, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(
            delete_output_fields=(field_name,),
            system_parts=[
                LMTextPart(
                    text=(
                        f"When producing `{field_name}`, return only the code for that field "
                        "as a native response artifact when supported."
                    )
                )
            ],
        )

    def parse_output(
        self,
        *,
        field_name: str,
        output: LMOutput | dict[str, Any] | str,
        field: Any | None = None,
        adapter: Any | None = None,
    ) -> Code | None:
        annotation = getattr(field, "annotation", self.marker_type) if field is not None else self.marker_type

        if isinstance(output, LMOutput):
            for part in output.parts:
                if isinstance(part, LMTextPart) and part.metadata.get("dspy_field") == field_name:
                    return _parse_code(annotation, part.text)
            if output.text is not None:
                return _parse_code(annotation, output.text)
        elif isinstance(output, dict) and isinstance(output.get(field_name), str):
            return _parse_code(annotation, output[field_name])
        elif isinstance(output, str):
            return _parse_code(annotation, output)
        return None


@dataclass(frozen=True)
class TextCode(TypeStrategy[Code]):
    """Keep `Code` as an ordinary adapter-rendered text field."""

    marker_type: type[Code] = Code


def _parse_code(annotation: type[Code], value: str) -> Code:
    return TypeAdapter(annotation).validate_python(value)
