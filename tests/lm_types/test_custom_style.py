"""Test that a user-defined Style works end-to-end."""

import asyncio
import re

import pytest

import dspy
from dspy.lm_types import (
    AdapterV2,
    BaseLMv2,
    FieldChunk,
    FieldLayout,
    LMCompletion,
    LMConfig,
    LMResponse,
    Style,
    TextPart,
    register_style,
)
from dspy.lm_types.styles import StreamParser


class TOMLStyle(Style):
    """Custom TOML-like style: ``key = "value"``."""
    name = "_test_toml"

    def field_layout(self) -> FieldLayout:
        return FieldLayout(field_template='{name} = "{value}"', separator="\n")

    def output_request(self, output_field_names):
        return f"Respond in TOML with keys: {', '.join(output_field_names)}."

    def parse(self, completion, output_fields):
        result = {}
        for line in completion.splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key in output_fields:
                    result[key] = value
        return result

    def stream_parser(self, output_field_names, output_fields=None):
        return _TOMLStreamParser(output_field_names)


class _TOMLStreamParser(StreamParser):
    def __init__(self, names):
        self.names = names
        self.buffer = ""

    def feed(self, text_chunk):
        self.buffer += text_chunk
        while "\n" in self.buffer:
            line, _, self.buffer = self.buffer.partition("\n")
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"')
                if key in self.names:
                    yield FieldChunk(key, value, is_last=True)

    def finalize(self):
        if "=" in self.buffer:
            key, _, value = self.buffer.partition("=")
            key = key.strip()
            value = value.strip().strip('"')
            if key in self.names:
                yield FieldChunk(key, value, is_last=True)
        self.buffer = ""


# Register at import time
register_style(TOMLStyle())


class TestCustomTOMLStyle:
    def test_registered(self):
        from dspy.lm_types import get_style
        assert get_style("_test_toml").name == "_test_toml"

    def test_render_output_layout(self):
        style = TOMLStyle()
        layout = style.field_layout()
        rendered = layout.render([("answer", "Paris"), ("score", "0.9")])
        assert rendered == 'answer = "Paris"\nscore = "0.9"'

    def test_parse_single_line(self):
        style = TOMLStyle()

        class _F:
            annotation = str

        fields = {"answer": _F(), "score": _F()}
        completion = 'answer = "Paris"\nscore = "0.9"'
        result = style.parse(completion, fields)
        assert result == {"answer": "Paris", "score": "0.9"}

    def test_end_to_end_adapter(self):
        class Sig(dspy.Signature):
            """Answer the question."""
            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        class _LM(BaseLMv2):
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[TextPart(
                    text='a = "Paris"'
                )])])

        adapter = AdapterV2(style="_test_toml")
        lm = _LM(model="test")
        results = adapter(lm, LMConfig(), Sig, [], {"q": "x"})
        assert results[0]["a"] == "Paris"

    def test_streaming(self):
        class Sig(dspy.Signature):
            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        from dspy.lm_types import LMStreamEvent, PartDelta

        class _LM(BaseLMv2):
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[TextPart(text='a = "Paris"')])])
            async def stream(self, messages, config=None):
                yield LMStreamEvent(type="start")
                yield LMStreamEvent(type="delta", delta=PartDelta(type="text", text='a = "Par'))
                yield LMStreamEvent(type="delta", delta=PartDelta(type="text", text='is"\n'))
                yield LMStreamEvent(type="end", finish_reason="stop")

        adapter = AdapterV2(style="_test_toml")
        lm = _LM(model="test")
        collected = []

        async def run():
            async for c in adapter.stream_call(lm, LMConfig(), Sig, [], {"q": "x"}):
                collected.append(c)

        asyncio.run(run())
        answers = [c.text for c in collected if c.field_name == "a"]
        assert "Paris" in "".join(answers)
