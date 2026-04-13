"""Image type for DSPy signatures — thin wrapper over lm15.Part.image."""

import io
import os
import warnings
from typing import Any

import pydantic

from dspy.adapters.types.base_type import Type

try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class Image(Type):
    url: str

    model_config = pydantic.ConfigDict(frozen=True, str_strip_whitespace=True, extra="forbid")

    def __init__(self, url: Any = None, *, download: bool = False, verify: bool = True, **data):
        if url is not None and "url" not in data:
            if isinstance(url, dict) and set(url.keys()) == {"url"}:
                data["url"] = url["url"]
            else:
                data["url"] = url
        if "url" in data:
            data["url"] = _normalize(data["url"], download=download, verify=verify)
        super().__init__(**data)

    def format(self) -> list[dict[str, Any]]:
        return [{"type": "image_url", "image_url": {"url": self.url}}]

    def to_lm15_part(self):
        """Return an lm15 ImagePart for direct use with lm15 Messages."""
        try:
            from lm15.types import Part
        except ImportError:
            return None
        if self.url.startswith("data:"):
            import re
            m = re.match(r"data:([^;]+);base64,(.+)", self.url)
            if m:
                return Part.image(data=m.group(2), media_type=m.group(1))
        return Part.image(url=self.url)

    @classmethod
    def from_url(cls, url, download=False):
        warnings.warn("Image.from_url is deprecated; use Image(url) instead.", DeprecationWarning, stacklevel=2)
        return cls(url, download=download)

    @classmethod
    def from_file(cls, file_path):
        warnings.warn("Image.from_file is deprecated; use Image(file_path) instead.", DeprecationWarning, stacklevel=2)
        return cls(file_path)

    @classmethod
    def from_PIL(cls, pil_image):
        warnings.warn("Image.from_PIL is deprecated; use Image(pil_image) instead.", DeprecationWarning, stacklevel=2)
        return cls(pil_image)

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        if "base64" in self.url:
            n = len(self.url.split("base64,")[1])
            t = self.url.split(";")[0].split("/")[-1]
            return f"Image(url=data:image/{t};base64,<BASE64({n})>)"
        return f"Image(url='{self.url}')"


# ---- Encoding helpers (kept for backward compat, but simpler) ----

def _normalize(image: Any, download: bool = False, verify: bool = True) -> str:
    if isinstance(image, str):
        if image.startswith("data:"):
            return image
        if os.path.isfile(image):
            return _file_to_data_uri(image)
        if image.startswith(("http://", "https://", "gs://")):
            return _url_to_data_uri(image, verify) if download else image
        raise ValueError(f"Unrecognized image string: {image}")
    if isinstance(image, dict) and "url" in image:
        return image["url"]
    if PIL_AVAILABLE and isinstance(image, PILImage.Image):
        return _pil_to_data_uri(image)
    if isinstance(image, bytes):
        if not PIL_AVAILABLE:
            raise ImportError("Pillow is required to process image bytes.")
        return _pil_to_data_uri(PILImage.open(io.BytesIO(image)))
    if isinstance(image, Image):
        return image.url
    raise ValueError(f"Unsupported image type: {type(image)}")


def _file_to_data_uri(path: str) -> str:
    import base64, mimetypes
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        raise ValueError(f"Cannot determine MIME type: {path}")
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def _url_to_data_uri(url: str, verify: bool = True) -> str:
    import base64, requests
    resp = requests.get(url, verify=verify)
    resp.raise_for_status()
    mime = resp.headers.get("Content-Type", "image/png")
    return f"data:{mime};base64,{base64.b64encode(resp.content).decode()}"


def _pil_to_data_uri(img) -> str:
    import base64, mimetypes
    buf = io.BytesIO()
    fmt = img.format or "PNG"
    img.save(buf, format=fmt)
    mime, _ = mimetypes.guess_type(f"x.{fmt.lower()}")
    if not mime:
        mime = f"image/{fmt.lower()}"
    return f"data:{mime};base64,{base64.b64encode(buf.getvalue()).decode()}"


def encode_image(image, download_images=False, verify=True) -> str:
    """Backward-compatible entry point."""
    return _normalize(image, download=download_images, verify=verify)


def is_image(obj) -> bool:
    if PIL_AVAILABLE and isinstance(obj, PILImage.Image):
        return True
    if isinstance(obj, str):
        return obj.startswith("data:") or os.path.isfile(obj) or obj.startswith(("http://", "https://"))
    return False
