"""Audio type for DSPy signatures — delegates to lm15.Part.audio."""

import base64
import io
import mimetypes
import os
from typing import Any, Union

import pydantic

from dspy.adapters.types.base_type import Type

try:
    from lm15.types import Part as _Part
    _HAS_LM15 = True
except ImportError:
    _HAS_LM15 = False


class Audio(Type):
    data: str
    audio_format: str

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def format(self):
        return [{"type": "input_audio", "input_audio": {"data": self.data, "format": self.audio_format}}]

    def to_lm15_part(self):
        return _Part.audio(data=self.data, media_type=f"audio/{self.audio_format}") if _HAS_LM15 else None

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, v: Any) -> Any:
        if isinstance(v, cls):
            return {"data": v.data, "audio_format": v.audio_format}
        return encode_audio(v)

    @classmethod
    def from_file(cls, path: str) -> "Audio":
        mime = mimetypes.guess_type(path)[0] or "audio/wav"
        fmt = mime.split("/")[1].removeprefix("x-")
        with open(path, "rb") as f:
            return cls(data=base64.b64encode(f.read()).decode(), audio_format=fmt)

    def __repr__(self):
        return f"Audio(<{len(self.data)} chars>, fmt='{self.audio_format}')"


def encode_audio(audio: Union[str, bytes, dict, Any], sampling_rate: int = 16000, format: str = "wav") -> dict:
    if isinstance(audio, dict) and "data" in audio and "audio_format" in audio:
        return audio
    if isinstance(audio, str):
        if audio.startswith("data:audio/"):
            header, b64 = audio.split(",", 1)
            fmt = header.split(":")[1].split(";")[0].split("/")[1].removeprefix("x-")
            return {"data": b64, "audio_format": fmt}
        if os.path.isfile(audio):
            a = Audio.from_file(audio)
            return {"data": a.data, "audio_format": a.audio_format}
    if isinstance(audio, bytes):
        return {"data": base64.b64encode(audio).decode(), "audio_format": format}
    try:
        import soundfile as sf
        if hasattr(audio, "shape"):
            buf = io.BytesIO()
            sf.write(buf, audio, sampling_rate, format=format.upper(), subtype="PCM_16")
            return {"data": base64.b64encode(buf.getvalue()).decode(), "audio_format": format}
    except ImportError:
        pass
    raise ValueError(f"Unsupported audio type: {type(audio)}")
