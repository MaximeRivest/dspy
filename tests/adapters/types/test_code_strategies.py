from types import SimpleNamespace

import dspy
from dspy.clients.language_models import LMOutput, LMTextPart


def test_native_code_deletes_output_field_and_adds_instruction_part():
    strategy = dspy.types.NativeCode()

    patch = strategy.render_output(field_name="code", field=None, adapter=None)

    assert patch.delete_output_fields == ("code",)
    assert len(patch.system_parts) == 1
    assert "code" in patch.system_parts[0].text


def test_native_code_parses_field_tagged_text_part_with_code_annotation():
    strategy = dspy.types.NativeCode()
    PythonCode = dspy.Code["python"]
    field = SimpleNamespace(annotation=PythonCode)
    output = LMOutput(parts=[LMTextPart(text="def f():\n    return 1", metadata={"dspy_field": "code"})])

    value = strategy.parse_output(field_name="code", field=field, output=output)

    assert isinstance(value, PythonCode)
    assert value.code == "def f():\n    return 1"


def test_native_code_temporarily_falls_back_to_output_text():
    strategy = dspy.types.NativeCode()
    output = LMOutput(parts=[LMTextPart(text="```python\ndef f():\n    return 1\n```")])

    value = strategy.parse_output(field_name="code", output=output)

    assert isinstance(value, dspy.Code)
    assert value.code == "def f():\n    return 1"


def test_text_code_keeps_field_for_outer_adapter():
    strategy = dspy.types.TextCode()

    patch = strategy.render_output(field_name="code", field=None, adapter=None)

    assert patch.delete_output_fields == ()
    assert patch.system_parts == []
