"""Tests for LMCompletion and LMResponse."""


import pytest

from dspy.lm_types import (
    CitationPart,
    LMCompletion,
    LMResponse,
    LMUsage,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)


class TestLMCompletion:
    def test_text_only(self):
        c = LMCompletion(parts=[TextPart(text="hello")])
        assert c.text == "hello"
        assert c.thinking is None
        assert c.tool_calls == []
        assert c.citations == []

    def test_text_and_thinking(self):
        c = LMCompletion(parts=[
            ThinkingPart(text="reasoning"),
            TextPart(text="answer"),
        ])
        assert c.text == "answer"
        assert c.thinking == "reasoning"

    def test_tool_calls(self):
        c = LMCompletion(parts=[
            ToolCallPart(id="1", name="search", input={"q": "x"}),
            ToolCallPart(id="2", name="get", input={}),
        ])
        assert len(c.tool_calls) == 2
        assert c.tool_calls[0].name == "search"
        assert c.text is None  # no text parts

    def test_citations(self):
        c = LMCompletion(parts=[
            TextPart(text="answer"),
            CitationPart(cited_text="source1"),
            CitationPart(cited_text="source2"),
        ])
        assert len(c.citations) == 2
        assert c.text == "answer"

    def test_finish_reason(self):
        c = LMCompletion(parts=[TextPart(text="x")], finish_reason="stop")
        assert c.finish_reason == "stop"

    def test_logprobs(self):
        c = LMCompletion(parts=[TextPart(text="x")], logprobs={"tokens": []})
        assert c.logprobs == {"tokens": []}

    def test_multiple_text_parts_concatenate(self):
        c = LMCompletion(parts=[
            TextPart(text="part1"),
            TextPart(text="part2"),
        ])
        assert c.text == "part1\npart2"

    def test_frozen(self):
        c = LMCompletion(parts=[TextPart(text="x")])
        with pytest.raises(Exception):
            c.finish_reason = "length"


class TestLMUsage:
    def test_defaults(self):
        u = LMUsage()
        assert u.input_tokens == 0
        assert u.output_tokens == 0
        assert u.total_tokens == 0
        assert u.cache_read_tokens is None
        assert u.reasoning_tokens is None

    def test_with_values(self):
        u = LMUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert u.input_tokens == 100


class TestLMResponse:
    def test_basic(self):
        r = LMResponse(
            completions=[LMCompletion(parts=[TextPart(text="hi")])],
            usage=LMUsage(total_tokens=10),
        )
        assert len(r.completions) == 1
        assert r.usage.total_tokens == 10

    def test_multiple_completions(self):
        r = LMResponse(completions=[
            LMCompletion(parts=[TextPart(text="a")]),
            LMCompletion(parts=[TextPart(text="b")]),
            LMCompletion(parts=[TextPart(text="c")]),
        ])
        assert len(r.completions) == 3
        assert r.completions[1].text == "b"

    def test_default_usage(self):
        r = LMResponse(completions=[LMCompletion(parts=[TextPart(text="x")])])
        assert r.usage.total_tokens == 0

    def test_model_and_id(self):
        r = LMResponse(
            completions=[LMCompletion(parts=[TextPart(text="x")])],
            model="gpt-4",
            id="resp-123",
        )
        assert r.model == "gpt-4"
        assert r.id == "resp-123"

    def test_frozen(self):
        r = LMResponse(completions=[LMCompletion(parts=[TextPart(text="x")])])
        with pytest.raises(Exception):
            r.model = "other"
