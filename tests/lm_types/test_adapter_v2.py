"""Tests for the new typed Adapter with template-as-presets."""


import asyncio

import pytest

import dspy
from dspy.lm_types import (
    AdapterV2,
    BaseLMv2,
    ChatStyle,
    CSVStyle,
    Demos,
    History as HistoryDirective,
    ImagePart,
    Inputs,
    Instruction,
    InputFields,
    JSONStyle,
    LMCompletion,
    LMConfig,
    LMMessage,
    LMResponse,
    LMStreamEvent,
    LMUsage,
    Message,
    OutputFields,
    OutputRequest,
    Outputs,
    PartDelta,
    Preset,
    Structure,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    XMLStyle,
    from_preset,
)


class _EchoLM(BaseLMv2):
    """LM that returns a fixed response, capturing what was sent."""

    def __init__(self, response_text="ok", model="echo", **kw):
        super().__init__(model=model, **kw)
        self.response_text = response_text
        self.last_messages = None
        self.last_config = None

    def forward(self, messages, config):
        self.last_messages = messages
        self.last_config = config
        return LMResponse(
            completions=[LMCompletion(
                parts=[TextPart(text=self.response_text)],
                finish_reason="stop",
            )],
            model=self.model,
        )

    async def aforward(self, messages, config):
        return self.forward(messages, config)


@pytest.fixture
def simple_signature():
    class Sig(dspy.Signature):
        """Answer the question."""
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()
    return Sig


@pytest.fixture
def multi_field_signature():
    class Sig(dspy.Signature):
        """Extract entities."""
        text: str = dspy.InputField()
        entities: str = dspy.OutputField()
        summary: str = dspy.OutputField()
    return Sig


class TestAdapterConstruction:
    def test_default_is_chat_preset(self):
        adapter = AdapterV2()
        assert adapter.style.name == "chat"

    def test_from_preset_chat(self):
        adapter = from_preset("chat")
        assert adapter.style.name == "chat"

    def test_from_preset_json(self):
        adapter = from_preset("json")
        assert adapter.style.name == "json"

    def test_from_preset_xml(self):
        adapter = from_preset("xml")
        assert adapter.style.name == "xml"

    def test_custom_messages(self):
        custom = [
            Message(role="system", content="you are helpful"),
            Message(role="user", content=[Inputs(), Instruction()]),
        ]
        adapter = AdapterV2(messages=custom, style="chat")
        assert len(adapter.messages) == 2

    def test_style_by_instance(self):
        adapter = AdapterV2(style=CSVStyle())
        assert adapter.style.name == "csv"

    def test_style_by_name(self):
        adapter = AdapterV2(style="json")
        assert adapter.style.name == "json"


class TestFormat:
    def test_chat_preset_produces_messages(self, simple_signature):
        adapter = from_preset("chat")
        messages = adapter.format(simple_signature, demos=[], inputs={"question": "What?"})
        # Should have system + user at minimum
        roles = [m.role for m in messages]
        assert "system" in roles
        assert "user" in roles

    def test_format_returns_lm_messages(self, simple_signature):
        adapter = from_preset("chat")
        messages = adapter.format(simple_signature, demos=[], inputs={"question": "What?"})
        assert all(isinstance(m, LMMessage) for m in messages)

    def test_inputs_interpolated(self, simple_signature):
        adapter = from_preset("chat")
        messages = adapter.format(simple_signature, demos=[], inputs={"question": "MYQUESTION"})
        user_msgs = [m for m in messages if m.role == "user"]
        assert any("MYQUESTION" in (m.text or "") for m in user_msgs)

    def test_instruction_interpolated(self, simple_signature):
        adapter = from_preset("chat")
        messages = adapter.format(simple_signature, demos=[], inputs={"question": "x"})
        system = [m for m in messages if m.role == "system"][0]
        assert "Answer the question" in (system.text or "")

    def test_demos_injected(self, simple_signature):
        adapter = from_preset("chat")
        demos = [{"question": "Q1", "answer": "A1"}]
        messages = adapter.format(simple_signature, demos=demos, inputs={"question": "Q2"})
        roles = [m.role for m in messages]
        assert roles.count("user") >= 2  # demo user + current user
        assert "assistant" in roles

    def test_multimodal_image_part(self):
        from dspy.adapters.types import Image
        class SigWithImage(dspy.Signature):
            image: Image = dspy.InputField()
            answer: str = dspy.OutputField()

        adapter = from_preset("chat")
        img = Image(url="https://example.com/a.png")
        messages = adapter.format(SigWithImage, demos=[], inputs={"image": img})
        user = [m for m in messages if m.role == "user"][-1]
        part_types = [p.type for p in user.parts]
        assert "image" in part_types


class TestParse:
    def test_chat_parse(self, simple_signature):
        adapter = from_preset("chat")
        completion = "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
        result = adapter.parse(simple_signature, completion)
        assert result["answer"] == "Paris"

    def test_json_parse(self, simple_signature):
        adapter = from_preset("json")
        completion = '{"answer": "Paris"}'
        result = adapter.parse(simple_signature, completion)
        assert result["answer"] == "Paris"

    def test_xml_parse(self, simple_signature):
        adapter = from_preset("xml")
        completion = "<answer>Paris</answer>"
        result = adapter.parse(simple_signature, completion)
        assert result["answer"] == "Paris"

    def test_missing_field_raises(self, multi_field_signature):
        from dspy.lm_types import AdapterParseError
        adapter = from_preset("chat")
        completion = "[[ ## entities ## ]]\nx\n\n[[ ## completed ## ]]"
        with pytest.raises(AdapterParseError):
            adapter.parse(multi_field_signature, completion)


class TestCall:
    def test_end_to_end_simple(self, simple_signature):
        lm = _EchoLM(response_text="[[ ## answer ## ]]\nHello!\n\n[[ ## completed ## ]]")
        adapter = from_preset("chat")
        results = adapter(lm, LMConfig(), simple_signature, [], {"question": "Hi"})
        assert len(results) == 1
        assert results[0]["answer"] == "Hello!"

    def test_receives_typed_messages(self, simple_signature):
        lm = _EchoLM(response_text="[[ ## answer ## ]]\nx\n\n[[ ## completed ## ]]")
        adapter = from_preset("chat")
        adapter(lm, LMConfig(), simple_signature, [], {"question": "Hi"})
        assert lm.last_messages is not None
        assert all(isinstance(m, LMMessage) for m in lm.last_messages)

    def test_config_passed_through(self, simple_signature):
        lm = _EchoLM(response_text="[[ ## answer ## ]]\nx\n\n[[ ## completed ## ]]")
        adapter = from_preset("chat")
        adapter(lm, LMConfig(temperature=0.7), simple_signature, [], {"question": "Hi"})
        assert lm.last_config.temperature == 0.7

    def test_async_call(self, simple_signature):
        lm = _EchoLM(response_text="[[ ## answer ## ]]\nA\n\n[[ ## completed ## ]]")
        adapter = from_preset("chat")
        results = asyncio.run(
            adapter.acall(lm, LMConfig(), simple_signature, [], {"question": "Q"})
        )
        assert results[0]["answer"] == "A"


class TestPostprocessNativeFeatures:
    def test_tool_calls_surfaced(self, simple_signature):
        class _ToolLM(BaseLMv2):
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[
                    ToolCallPart(id="c1", name="search", input={"q": "x"}),
                ])])
            async def aforward(self, messages, config):
                return self.forward(messages, config)

        from dspy.adapters.types import Tool, ToolCalls
        class SigWithTool(dspy.Signature):
            query: str = dspy.InputField()
            tools: list[Tool] = dspy.InputField()
            tool_calls: ToolCalls = dspy.OutputField()

        def search(q: str) -> str:
            return q
        adapter = AdapterV2(style="chat")
        lm = _ToolLM(model="tool")
        results = adapter(
            lm, LMConfig(), SigWithTool, [],
            {"query": "hi", "tools": [Tool(search)]},
        )
        assert len(results) == 1
        # tool_calls should be populated from completion.tool_calls
        assert results[0].get("tool_calls") is not None

    def test_thinking_surfaced(self):
        from dspy.adapters.types import Reasoning
        class Sig(dspy.Signature):
            """Answer with reasoning."""
            q: str = dspy.InputField()
            reasoning: Reasoning = dspy.OutputField()
            a: str = dspy.OutputField()

        class _ReasoningLM(BaseLMv2):
            @property
            def supports_reasoning(self): return True
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[
                    ThinkingPart(text="pondering..."),
                    TextPart(text="[[ ## a ## ]]\nAnswer\n\n[[ ## completed ## ]]"),
                ])])
            async def aforward(self, messages, config):
                return self.forward(messages, config)

        adapter = from_preset("chat")
        lm = _ReasoningLM(model="reason")
        results = adapter(lm, LMConfig(reasoning_effort="low"), Sig, [], {"q": "x"})
        # reasoning should be populated from ThinkingPart
        assert results[0].get("reasoning") is not None


class TestCustomTemplate:
    def test_custom_template_format(self, simple_signature):
        custom = [
            Message(role="system", content=[Text_literal("you extract entities. "), Instruction()]),
            Message(role="user", content=[Text_literal("<q>"), Inputs(), Text_literal("</q> "), OutputRequest(style="xml")]),
        ]
        adapter = AdapterV2(messages=custom, style="xml")
        messages = adapter.format(simple_signature, [], {"question": "X"})
        assert any("extract entities" in (m.text or "") for m in messages)

    def test_for_each_loop(self, multi_field_signature):
        from dspy.lm_types import ForEach
        custom = [
            Message(role="system", content=[
                ForEach(fields="output", template="Field: {name}", separator="; "),
            ]),
            Message(role="user", content=Inputs()),
        ]
        adapter = AdapterV2(messages=custom, style="chat")
        messages = adapter.format(multi_field_signature, [], {"text": "x"})
        system = [m for m in messages if m.role == "system"][0]
        assert "Field: entities" in (system.text or "")
        assert "Field: summary" in (system.text or "")


# Helper for tests
def Text_literal(s):
    from dspy.lm_types import Text
    return Text(s)
