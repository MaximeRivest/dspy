"""File type for DSPy — delegates to lm15.Part.document."""

import base64
import mimetypes
import os
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type

try:
    from lm15.types import Part as _Part
    _HAS_LM15 = True
except ImportError:
    _HAS_LM15 = False


class File(Type):
    file_data: str | None = None
    file_id: str | None = None
    filename: str | None = None

    model_config = pydantic.ConfigDict(frozen=True, str_strip_whitespace=True, extra="forbid")

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, v: Any) -> Any:
        if isinstance(v, cls):
            return {"file_data": v.file_data, "file_id": v.file_id, "filename": v.filename}
        if isinstance(v, dict) and any(k in v for k in ("file_data", "file_id", "filename")):
            return v
        if isinstance(v, str) and os.path.isfile(v):
            return _path_to_dict(v)
        if isinstance(v, bytes):
            return {"file_data": f"data:application/octet-stream;base64,{base64.b64encode(v).decode()}"}
        raise ValueError(f"Unsupported File input: {type(v)}")

    def format(self):
        d = {}
        if self.file_data: d["file_data"] = self.file_data
        if self.file_id: d["file_id"] = self.file_id
        if self.filename: d["filename"] = self.filename
        return [{"type": "file", "file": d}]

    def to_lm15_part(self):
        if not _HAS_LM15: return None
        if self.file_data: return _Part.document(data=self.file_data)
        if self.file_id: return _Part.document(file_id=self.file_id)
        return None

    @classmethod
    def from_path(cls, path, filename=None, mime_type=None):
        return cls(**_path_to_dict(path, filename, mime_type))

    @classmethod
    def from_bytes(cls, data, filename=None, mime_type="application/octet-stream"):
        return cls(file_data=f"data:{mime_type};base64,{base64.b64encode(data).decode()}", filename=filename)

    def __repr__(self):
        parts = []
        if self.file_data: parts.append(f"data=<{len(self.file_data)}ch>")
        if self.file_id: parts.append(f"id='{self.file_id}'")
        if self.filename: parts.append(f"'{self.filename}'")
        return f"File({', '.join(parts)})"


def _path_to_dict(path, filename=None, mime_type=None):
    if not os.path.isfile(path): raise ValueError(f"Not found: {path}")
    with open(path, "rb") as f: data = f.read()
    mime = mime_type or mimetypes.guess_type(path)[0] or "application/octet-stream"
    return {"file_data": f"data:{mime};base64,{base64.b64encode(data).decode()}", "filename": filename or os.path.basename(path)}
