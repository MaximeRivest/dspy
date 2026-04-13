"""Audio type for DSPy signatures — thin wrapper over lm15.Part.audio."""

import base64
import io
import mimetypes
import os
from typing import Any, Union

import pydantic

from dspy.adapters.types.base_type import Type

try:
    import soundfile as sf
    SF_AVAILABLE = True
except ImportError:
    SF_AVAILABLE = False


class Audio(Type):
    data: str
    audio_format: str

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    def format(self) -> list[dict[str, Any]]:
        return [{"type": "input_audio", "input_audio": {"data": self.data, "format": self.audio_format}}]

    def to_lm15_part(self):
        from lm15.types import Part
        return Part.audio(data=self.data, media_type=f"audio/{self.audio_format}")

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, values: Any) -> Any:
        if isinstance(values, cls):
            return {"data": values.data, "audio_format": values.audio_format}
        return encode_audio(values)

    @classmethod
    def from_url(cls, url: str) -> "Audio":
        import requests
        resp = requests.get(url); resp.raise_for_status()
        mime = resp.headers.get("Content-Type", "audio/wav")
        fmt = mime.split("/")[1].removeprefix("x-")
        return cls(data=base64.b64encode(resp.content).decode(), audio_format=fmt)

    @classmethod
    def from_file(cls, path: str) -> "Audio":
        if not os.path.isfile(path):
            raise ValueError(f"File not found: {path}")
        mime, _ = mimetypes.guess_type(path)
        if not mime or not mime.startswith("audio/"):
            raise ValueError(f"Not an audio file: {path}")
        fmt = mime.split("/")[1].removeprefix("x-")
        with open(path, "rb") as f:
            return cls(data=base64.b64encode(f.read()).decode(), audio_format=fmt)

    @classmethod
    def from_array(cls, array: Any, sampling_rate: int, format: str = "wav") -> "Audio":
        if not SF_AVAILABLE:
            raise ImportError("soundfile is required to process audio arrays.")
        buf = io.BytesIO()
        sf.write(buf, array, sampling_rate, format=format.upper(), subtype="PCM_16")
        return cls(data=base64.b64encode(buf.getvalue()).decode(), audio_format=format)

    def __str__(self):
        return self.serialize_model()

    def __repr__(self):
        return f"Audio(data=<BASE64({len(self.data)})>, audio_format='{self.audio_format}')"


def encode_audio(audio: Union[str, bytes, dict, "Audio", Any], sampling_rate: int = 16000, format: str = "wav") -> dict:
    if isinstance(audio, dict) and "data" in audio and "audio_format" in audio:
        return audio
    if isinstance(audio, Audio):
        return {"data": audio.data, "audio_format": audio.audio_format}
    if isinstance(audio, str):
        if audio.startswith("data:audio/"):
            header, b64 = audio.split(",", 1)
            fmt = header.split(":")[1].split(";")[0].split("/")[1].removeprefix("x-")
            return {"data": b64, "audio_format": fmt}
        if os.path.isfile(audio):
            a = Audio.from_file(audio)
            return {"data": a.data, "audio_format": a.audio_format}
        if audio.startswith("http"):
            a = Audio.from_url(audio)
            return {"data": a.data, "audio_format": a.audio_format}
    if SF_AVAILABLE and hasattr(audio, "shape"):
        a = Audio.from_array(audio, sampling_rate=sampling_rate, format=format)
        return {"data": a.data, "audio_format": a.audio_format}
    if isinstance(audio, bytes):
        return {"data": base64.b64encode(audio).decode(), "audio_format": format}
    raise ValueError(f"Unsupported type for encode_audio: {type(audio)}")
