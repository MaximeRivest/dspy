import dspy
from dspy.clients.language_models import LMCitationPart, LMOutput


def test_native_document_renders_document_as_lm_document_part():
    strategy = dspy.types.NativeDocument()
    document = dspy.experimental.Document(data="Source text", title="Source")

    patch = strategy.render_input(field_name="documents", field=None, value=[document], adapter=None)

    assert patch.delete_input_fields == ("documents",)
    assert patch.user_parts[0].text.strip() == "documents:"
    assert patch.user_parts[1].type == "document"
    assert patch.user_parts[1].source == {"type": "text", "media_type": "text/plain", "data": "Source text"}
    assert patch.user_parts[1].citations == {"enabled": True}
    assert patch.user_parts[1].title == "Source"


def test_native_citations_deletes_output_field_and_parses_citations():
    strategy = dspy.types.NativeCitations()

    patch = strategy.render_output(field_name="citations", field=None, adapter=None)
    value = strategy.parse_output(
        field_name="citations",
        output=LMOutput(parts=[LMCitationPart(text="Source text", title="Source")]),
    )

    assert patch.delete_output_fields == ("citations",)
    assert isinstance(value, dspy.experimental.Citations)
    assert value.citations[0].cited_text == "Source text"
    assert value.citations[0].document_title == "Source"
