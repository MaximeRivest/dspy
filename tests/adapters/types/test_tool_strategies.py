import dspy
from dspy.clients.language_models import LMOutput, LMToolCallPart


def test_native_tool_calls_deletes_tool_fields_and_adds_tool_specs():
    class UseTool(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    def search(query: str) -> str:
        return query

    strategy = dspy.types.NativeToolCalls(tool_choice="required")
    patch = strategy.prepare(
        signature=UseTool,
        lm=None,
        lm_kwargs={},
        inputs={"tools": [dspy.Tool(search)]},
        adapter=None,
    )

    assert patch.delete_input_fields == ("tools",)
    assert patch.delete_output_fields == ("tool_calls",)
    assert patch.tools[0].name == "search"
    assert patch.config.tool_choice.mode == "required"


def test_native_tool_calls_parses_lm_tool_call_parts():
    strategy = dspy.types.NativeToolCalls()

    value = strategy.parse_output(
        field_name="tool_calls",
        output=LMOutput(parts=[LMToolCallPart(name="search", args={"query": "DSPy"})]),
    )

    assert isinstance(value, dspy.ToolCalls)
    assert value.tool_calls[0].name == "search"
    assert value.tool_calls[0].args == {"query": "DSPy"}
