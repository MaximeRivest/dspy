"""Rendering strategies for `dspy.File` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dspy.adapters.types.file import File
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMBinaryPart, LMOutput, LMRequestPatch, LMTextPart


@dataclass(frozen=True)
class NativeFile(TypeStrategy[File]):
    """Render file fields as normalized binary parts and parse binary outputs."""

    marker_type: type[File] = File

    def render_input(self, *, field_name: str, field: Any, value: File, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(
            delete_input_fields=(field_name,),
            user_parts=[
                LMTextPart(text=f"\n\n{field_name}:\n"),
                _file_value_to_lm_part(value),
            ],
        )

    def render_output(self, *, field_name: str, field: Any, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(
            delete_output_fields=(field_name,),
            system_parts=[
                LMTextPart(
                    text=(
                        f"When producing `{field_name}`, return it as a native binary/file output part "
                        "if the backend supports file generation."
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
    ) -> File | None:
        if isinstance(output, LMOutput):
            if not output.binaries:
                return None
            return _lm_part_to_file(output.binaries[0])
        if isinstance(output, dict):
            value = output.get(field_name)
            if isinstance(value, File):
                return value
            if isinstance(value, dict) or isinstance(value, str) or isinstance(value, bytes):
                return File(value)
        if isinstance(output, str):
            return File(output.strip())
        return None


def _file_value_to_lm_part(file: File) -> LMBinaryPart:
    if file.file_data is not None:
        media_type, data = _split_data_uri(file.file_data)
        return LMBinaryPart(data=data, media_type=media_type, filename=file.filename)
    if file.file_id is not None:
        return LMBinaryPart(file_id=file.file_id, filename=file.filename)
    raise ValueError("File must have file_data or file_id.")


def _lm_part_to_file(binary: LMBinaryPart) -> File:
    if binary.data is not None:
        return File(file_data=f"data:{binary.media_type};base64,{binary.data}", filename=binary.filename)
    if binary.url is not None:
        return File(file_data=binary.url, filename=binary.filename)
    if binary.file_id is not None:
        return File(file_id=binary.file_id, filename=binary.filename)
    if binary.path is not None:
        return File.from_path(binary.path, filename=binary.filename, mime_type=binary.media_type)
    raise ValueError("LMBinaryPart has no binary source.")


def _split_data_uri(value: str) -> tuple[str, str]:
    if value.startswith("data:") and "," in value:
        header, data = value.split(",", 1)
        return header.removeprefix("data:").split(";", 1)[0], data
    return "application/octet-stream", value
