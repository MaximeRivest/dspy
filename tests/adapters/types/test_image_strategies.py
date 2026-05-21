import dspy
from dspy.clients.language_models import LMImagePart, LMOutput


def test_native_image_renders_input_as_lm_image_part_and_deletes_input_field():
    strategy = dspy.types.NativeImage(detail="high")
    value = dspy.Image("https://example.com/source.png")

    patch = strategy.render_input(field_name="image", field=None, value=value, adapter=None)

    assert patch.delete_input_fields == ("image",)
    assert patch.user_parts[0].text.strip() == "image:"
    assert patch.user_parts[1].type == "image"
    assert patch.user_parts[1].url == "https://example.com/source.png"
    assert patch.user_parts[1].detail == "high"


def test_native_image_renders_data_uri_as_lm_image_data_part():
    strategy = dspy.types.NativeImage()
    value = dspy.Image("data:image/png;base64,abc123")

    patch = strategy.render_input(field_name="image", field=None, value=value, adapter=None)

    image = patch.user_parts[1]
    assert image.media_type == "image/png"
    assert image.data == "abc123"


def test_native_image_deletes_output_field_and_parses_lm_image_part():
    strategy = dspy.types.NativeImage()

    patch = strategy.render_output(field_name="edited_image", field=None, adapter=None)
    value = strategy.parse_output(
        field_name="edited_image",
        output=LMOutput(parts=[LMImagePart(url="https://example.com/edited.png")]),
    )

    assert patch.delete_output_fields == ("edited_image",)
    assert isinstance(value, dspy.Image)
    assert value.url == "https://example.com/edited.png"
