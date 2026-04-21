"""Tests for LMMessage — typed messages with parts."""


import pytest

from dspy.lm_types import (
    ImagePart,
    LMMessage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)


class TestLMMessage:
    def test_system_constructor(self):
        m = LMMessage.system("You are helpful")
        assert m.role == "system"
        assert len(m.parts) == 1
        assert isinstance(m.parts[0], TextPart)
        assert m.parts[0].text == "You are helpful"

    def test_user_string(self):
        m = LMMessage.user("hello")
        assert m.role == "user"
        assert m.text == "hello"

    def test_user_parts(self):
        m = LMMessage.user([TextPart(text="what is this? "), ImagePart(url="https://x.com/a.png")])
        assert m.role == "user"
        assert len(m.parts) == 2
        assert isinstance(m.parts[1], ImagePart)

    def test_assistant(self):
        m = LMMessage.assistant("ok")
        assert m.role == "assistant"
        assert m.text == "ok"

    def test_text_property_concatenates_text_parts(self):
        m = LMMessage(
            role="assistant",
            parts=[TextPart(text="one"), ThinkingPart(text="skip"), TextPart(text="two")],
        )
        assert m.text == "one\ntwo"

    def test_text_property_returns_none_when_no_text(self):
        m = LMMessage(role="assistant", parts=[ThinkingPart(text="only thinking")])
        assert m.text is None

    def test_tool_calls_property(self):
        m = LMMessage(
            role="assistant",
            parts=[
                TextPart(text="calling"),
                ToolCallPart(id="1", name="fn", input={"x": 1}),
                ToolCallPart(id="2", name="fn", input={"x": 2}),
            ],
        )
        assert len(m.tool_calls) == 2
        assert m.tool_calls[0].name == "fn"

    def test_thinking_property(self):
        m = LMMessage(
            role="assistant",
            parts=[TextPart(text="answer"), ThinkingPart(text="reasoning here")],
        )
        assert m.thinking == "reasoning here"

    def test_requires_nonempty_parts(self):
        with pytest.raises(ValueError):
            LMMessage(role="user", parts=[])

    def test_invalid_role(self):
        with pytest.raises(Exception):
            LMMessage(role="bogus", parts=[TextPart(text="x")])

    def test_is_frozen(self):
        m = LMMessage.user("hi")
        with pytest.raises(Exception):
            m.role = "assistant"
