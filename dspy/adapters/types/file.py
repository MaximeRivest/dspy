"""File type for DSPy signatures — thin wrapper for document/file inputs."""

import base64
import mimetypes
import os
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type


class File(Type):
    """A file input type for DSPy.

    Examples:
        ```python
        import dspy
        class QA(dspy.Signature):
            file: dspy.File = dspy.InputField()
            summary = dspy.OutputField()
        result = dspy.Predict(QA)(file=dspy.File.from_path("./research.pdf"))
        ```
    """

    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None

    model_config = pydantic.ConfigDict(frozen=True, str_strip_whitespace=True, extra="forbid")

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        if isinstance(values, cls):
            return {"file_data": values.file_data, "file_id": values.file_id, "filename": values.filename}
        if isinstance(values, dict):
            if any(k in values for k in ("file_data", "file_id", "filename")):
                return values
            raise ValueError("File must have file_data, file_id, or filename")
        if isinstance(values, str) and os.path.isfile(values):
            return _path_to_dict(values)
        if isinstance(values, bytes):
            return {"file_data": f"data:application/octet-stream;base64,{base64.b64encode(values).decode()}"}
        raise ValueError(f"Unsupported File input: {type(values)}")

    def format(self) -> list[dict[str, Any]]:
        d = {}
        if self.file_data: d["file_data"] = self.file_data
        if self.file_id: d["file_id"] = self.file_id
        if self.filename: d["filename"] = self.filename
        return [{"type": "file", "file": d}]

    def to_lm15_part(self):
        from lm15.types import Part
        if self.file_data:
            return Part.document(data=self.file_data)
        if self.file_id:
            return Part.document(file_id=self.file_id)
        raise ValueError("File has no data or file_id")

    @classmethod
    def from_path(cls, path: str, filename: str | None = None, mime_type: str | None = None) -> "File":
        d = _path_to_dict(path, filename, mime_type)
        return cls(**d)

    @classmethod
    def from_bytes(cls, data: bytes, filename: str | None = None, mime_type: str = "application/octet-stream") -> "File":
        encoded = f"data:{mime_type};base64,{base64.b64encode(data).decode()}"
        return cls(file_data=encoded, filename=filename)

    @classmethod
    def from_file_id(cls, file_id: str, filename: str | None = None) -> "File":
        return cls(file_id=file_id, filename=filename)

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        parts = []
        if self.file_data: parts.append(f"file_data=<{len(self.file_data)} chars>")
        if self.file_id: parts.append(f"file_id='{self.file_id}'")
        if self.filename: parts.append(f"filename='{self.filename}'")
        return f"File({', '.join(parts)})"


def _path_to_dict(path: str, filename: str | None = None, mime_type: str | None = None) -> dict:
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    mime = mime_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
    return {
        "file_data": f"data:{mime};base64,{base64.b64encode(data).decode()}",
        "filename": filename or os.path.basename(path),
    }
