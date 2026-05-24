import dspy
from dspy.utils.exceptions import AdapterParseError


def add(a: int, b: int) -> int:
    return a + b


class NativeToolLM:
    model = "openai/gpt-5-nano"
    supports_function_calling = True
    supports_reasoning = False
    supports_response_schema = False
    supported_params = frozenset()
    kwargs = {}

    def __init__(self, output):
        self.output = output
        self.messages = None
        self.kwargs = None

    def __call__(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return [self.output]


def test_native_tool_response_can_combine_visible_text_and_tool_calls():
    class ToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    lm = NativeToolLM(
        {
            "text": "[[ ## answer ## ]]\nworking\n\n[[ ## completed ## ]]",
            "tool_calls": [
                {
                    "id": "call_add",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                }
            ],
        }
    )

    result = dspy.ChatAdapter(use_native_function_calling=True)(
        lm,
        {},
        ToolSignature,
        [],
        {"question": "What is 1+2?", "tools": [dspy.Tool(add)]},
    )[0]

    assert result["answer"] == "working"
    assert result["tool_calls"].tool_calls[0].id == "call_add"
    assert result["tool_calls"].tool_calls[0].args == {"a": 1, "b": 2}


def test_tool_type_handler_does_not_override_explicit_tool_choice_mode():
    class ToolSignature(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    lm = NativeToolLM({"text": None})
    adapter = dspy.ChatAdapter(use_native_function_calling=True, allow_parallel_tool_calls=False)

    try:
        adapter(
            lm,
            {"tool_choice": "none"},
            ToolSignature,
            [],
            {"question": "What is 1+2?", "tools": [dspy.Tool(add)]},
        )
    except AdapterParseError:
        pass

    assert lm.kwargs["tool_choice"] == "none"
    assert lm.kwargs["parallel_tool_calls"] is False
    assert lm.kwargs["tools"][0]["function"]["parameters"]["properties"]["a"]["type"] == "integer"
