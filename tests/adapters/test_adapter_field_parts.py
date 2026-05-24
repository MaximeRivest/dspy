import dspy
from dspy.core.types import LMOutput, LMTextPart


def test_base_value_helpers_round_trip_text_field_parts():
    class SimpleSignature(dspy.Signature):
        answer: int = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    field_info = SimpleSignature.output_fields["answer"]

    parts = adapter.value_to_lm_parts(42, field_info)

    assert parts == [LMTextPart(text="42")]
    assert adapter.lm_parts_to_value(parts, field_info) == 42


def test_chat_adapter_wraps_and_slices_field_parts():
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter()

    wrapped = adapter.wrap_input_field_parts("question", [LMTextPart(text="What?")])
    assert wrapped == [LMTextPart(text="[[ ## question ## ]]\n"), LMTextPart(text="What?")]

    output = LMOutput(parts=[LMTextPart(text="[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]")])
    fields = adapter.parse_output_fields_to_parts(output, SimpleSignature)

    assert fields == {"answer": [LMTextPart(text="Paris")]}


def test_json_adapter_slices_output_fields_to_text_parts():
    class SimpleSignature(dspy.Signature):
        answer: str = dspy.OutputField()
        score: float = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    output = LMOutput(parts=[LMTextPart(text='{"answer": "Paris", "score": 0.9}')])

    fields = adapter.parse_output_fields_to_parts(output, SimpleSignature)

    assert fields == {
        "answer": [LMTextPart(text="Paris")],
        "score": [LMTextPart(text="0.9")],
    }


def test_xml_adapter_wraps_and_slices_field_parts():
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.XMLAdapter()

    wrapped = adapter.wrap_output_field_parts("answer", [LMTextPart(text="Paris")])
    assert wrapped == [
        LMTextPart(text="<answer>\n"),
        LMTextPart(text="Paris"),
        LMTextPart(text="\n</answer>"),
    ]

    output = LMOutput(parts=[LMTextPart(text="<answer>Paris</answer>")])
    fields = adapter.parse_output_fields_to_parts(output, SimpleSignature)

    assert fields == {"answer": [LMTextPart(text="Paris")]}
