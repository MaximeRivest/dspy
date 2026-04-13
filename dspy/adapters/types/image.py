"""Image type for DSPy signatures — delegates to lm15.Part.image."""

import os
import warnings
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type

try:
    from lm15.types import Part as _Part
    _HAS_LM15 = True
except ImportError:
    _HAS_LM15 = False


class Image(Type):
    url: str

    model_config = pydantic.ConfigDict(frozen=True, str_strip_whitespace=True, extra="forbid")

    def __init__(self, url: Any = None, *, download: bool = False, **data):
        if url is not None and "url" not in data:
            if isinstance(url, dict) and "url" in url:
                data["url"] = url["url"]
            else:
                data["url"] = _to_str(url, download)
        elif "url" in data:
            data["url"] = _to_str(data["url"], download)
        super().__init__(**data)

    def format(self):
        return [{"type": "image_url", "image_url": {"url": self.url}}]

    def to_lm15_part(self):
        if not _HAS_LM15:
            return None
        import re
        m = re.match(r"data:([^;]+);base64,(.+)", self.url, re.DOTALL)
        if m:
            return _Part.image(data=m.group(2), media_type=m.group(1))
        return _Part.image(url=self.url)

    def __repr__(self):
        if "base64" in self.url:
            return f"Image(<base64 {len(self.url)} chars>)"
        return f"Image(url='{self.url}')"


def _to_str(v: Any, download: bool = False) -> str:
    """Normalize any input to a URL or data URI string."""
    if isinstance(v, str):
        if v.startswith("data:") or v.startswith(("http://", "https://", "gs://")):
            return v
        if os.path.isfile(v):
            import base64, mimetypes
            mime = mimetypes.guess_type(v)[0] or "application/octet-stream"
            with open(v, "rb") as f:
                return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
        raise ValueError(f"Not a file or URL: {v}")
    if isinstance(v, bytes):
        import base64
        return f"data:image/png;base64,{base64.b64encode(v).decode()}"
    try:
        from PIL import Image as PILImage
        if isinstance(v, PILImage.Image):
            import base64, io
            buf = io.BytesIO()
            v.save(buf, format=v.format or "PNG")
            return f"data:image/{(v.format or 'png').lower()};base64,{base64.b64encode(buf.getvalue()).decode()}"
    except ImportError:
        pass
    raise ValueError(f"Unsupported image type: {type(v)}")


# Backward compat
def encode_image(image, download_images=False, verify=True):
    return _to_str(image, download=download_images)

def is_image(obj):
    if isinstance(obj, str):
        return obj.startswith("data:") or os.path.isfile(obj) or obj.startswith(("http://", "https://"))
    try:
        from PIL import Image as PILImage
        return isinstance(obj, PILImage.Image)
    except ImportError:
        return False
