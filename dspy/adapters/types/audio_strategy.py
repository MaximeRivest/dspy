"""Rendering strategies for `dspy.Audio` fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dspy.adapters.types.audio import Audio
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.language_models.types import LMAudioPart, LMOutput, LMRequestPatch, LMTextPart


@dataclass(frozen=True)
class NativeAudio(TypeStrategy[Audio]):
    """Render audio fields as normalized audio parts and parse audio outputs."""

    marker_type: type[Audio] = Audio

    def render_input(self, *, field_name: str, field: Any, value: Audio, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(
            delete_input_fields=(field_name,),
            user_parts=[
                LMTextPart(text=f"\n\n{field_name}:\n"),
                _audio_value_to_lm_part(value),
            ],
        )

    def render_output(self, *, field_name: str, field: Any, adapter: Any) -> LMRequestPatch:
        return LMRequestPatch(
            delete_output_fields=(field_name,),
            system_parts=[
                LMTextPart(
                    text=(
                        f"When producing `{field_name}`, return it as a native audio output part "
                        "if the backend supports audio generation."
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
    ) -> Audio | None:
        if isinstance(output, LMOutput):
            if not output.audio:
                return None
            return _lm_part_to_audio(output.audio[0])
        if isinstance(output, dict):
            value = output.get(field_name)
            if isinstance(value, Audio):
                return value
            if isinstance(value, dict) or isinstance(value, str) or isinstance(value, bytes):
                return Audio(value)
        if isinstance(output, str):
            return Audio(output.strip())
        return None


def _audio_value_to_lm_part(audio: Audio) -> LMAudioPart:
    return LMAudioPart(data=audio.data, media_type=f"audio/{audio.audio_format}")


def _lm_part_to_audio(audio: LMAudioPart) -> Audio:
    if audio.data is not None:
        return Audio(data=audio.data, audio_format=_audio_format(audio.media_type))
    if audio.url is not None:
        return Audio.from_url(audio.url)
    if audio.path is not None:
        return Audio.from_file(audio.path)
    if audio.file_id is not None:
        raise ValueError("Cannot convert an audio file_id to dspy.Audio without fetching the file content.")
    raise ValueError("LMAudioPart has no audio source.")


def _audio_format(media_type: str) -> str:
    return media_type.split("/", 1)[1] if "/" in media_type else media_type
