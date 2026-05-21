import dspy
from dspy.clients.language_models import LMBinaryPart, LMOutput


def test_native_file_renders_input_as_binary_part_and_deletes_input_field():
    strategy = dspy.types.NativeFile()
    value = dspy.File(file_data="data:text/plain;base64,aGVsbG8=", filename="hello.txt")

    patch = strategy.render_input(field_name="attachment", field=None, value=value, adapter=None)

    assert patch.delete_input_fields == ("attachment",)
    assert patch.user_parts[0].text.strip() == "attachment:"
    assert patch.user_parts[1].type == "binary"
    assert patch.user_parts[1].data == "aGVsbG8="
    assert patch.user_parts[1].media_type == "text/plain"
    assert patch.user_parts[1].filename == "hello.txt"


def test_native_file_deletes_output_field_and_parses_binary_part():
    strategy = dspy.types.NativeFile()

    patch = strategy.render_output(field_name="attachment", field=None, adapter=None)
    value = strategy.parse_output(
        field_name="attachment",
        output=LMOutput(parts=[LMBinaryPart(data="aGVsbG8=", media_type="text/plain", filename="hello.txt")]),
    )

    assert patch.delete_output_fields == ("attachment",)
    assert isinstance(value, dspy.File)
    assert value.file_data == "data:text/plain;base64,aGVsbG8="
    assert value.filename == "hello.txt"
