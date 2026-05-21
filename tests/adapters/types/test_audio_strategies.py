import pytest

import dspy
from dspy.clients.language_models import LMAudioPart, LMOutput


def test_native_audio_renders_input_as_lm_audio_part_and_deletes_input_field():
    strategy = dspy.types.NativeAudio()
    value = dspy.Audio(data="abc123", audio_format="wav")

    patch = strategy.render_input(field_name="clip", field=None, value=value, adapter=None)

    assert patch.delete_input_fields == ("clip",)
    assert patch.user_parts[0].text.strip() == "clip:"
    assert patch.user_parts[1].type == "audio"
    assert patch.user_parts[1].data == "abc123"
    assert patch.user_parts[1].media_type == "audio/wav"


def test_native_audio_deletes_output_field_and_parses_lm_audio_part():
    strategy = dspy.types.NativeAudio()

    patch = strategy.render_output(field_name="spoken_answer", field=None, adapter=None)
    value = strategy.parse_output(
        field_name="spoken_answer",
        output=LMOutput(parts=[LMAudioPart(data="abc123", media_type="audio/mp3")]),
    )

    assert patch.delete_output_fields == ("spoken_answer",)
    assert isinstance(value, dspy.Audio)
    assert value.data == "abc123"
    assert value.audio_format == "mp3"


def test_native_audio_refuses_file_id_without_fetching_content():
    strategy = dspy.types.NativeAudio()

    with pytest.raises(ValueError, match="file_id"):
        strategy.parse_output(
            field_name="spoken_answer",
            output=LMOutput(parts=[LMAudioPart(file_id="file_123", media_type="audio/wav")]),
        )
