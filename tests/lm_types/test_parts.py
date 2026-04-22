"""Tests for typed content Parts."""


import pytest

from dspy.lm_types import (
    AudioPart,
    CitationPart,
    DocumentPart,
    ImagePart,
    Part,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)


class TestTextPart:
    def test_basic_construction(self):
        p = TextPart(text="hello")
        assert p.type == "text"
        assert p.text == "hello"

    def test_is_frozen(self):
        p = TextPart(text="hello")
        with pytest.raises(Exception):
            p.text = "world"

    def test_empty_text_allowed(self):
        p = TextPart(text="")
        assert p.text == ""

    def test_multiline(self):
        p = TextPart(text="line1\nline2")
        assert "\n" in p.text


class TestImagePart:
    def test_from_url(self):
        p = ImagePart(url="https://example.com/img.png")
        assert p.type == "image"
        assert p.url == "https://example.com/img.png"
        assert p.data is None
        assert p.file_id is None

    def test_from_data(self):
        p = ImagePart(data="base64data", media_type="image/jpeg")
        assert p.type == "image"
        assert p.data == "base64data"
        assert p.media_type == "image/jpeg"
        assert p.url is None

    def test_from_file_id(self):
        p = ImagePart(file_id="file-abc123")
        assert p.type == "image"
        assert p.file_id == "file-abc123"

    def test_default_media_type(self):
        p = ImagePart(url="https://example.com/a.png")
        assert p.media_type == "image/png"


class TestAudioPart:
    def test_from_data(self):
        p = AudioPart(data="base64audio", media_type="audio/wav")
        assert p.type == "audio"
        assert p.data == "base64audio"
        assert p.media_type == "audio/wav"

    def test_default_media_type(self):
        p = AudioPart(data="abc")
        assert p.media_type == "audio/wav"


class TestDocumentPart:
    def test_basic(self):
        p = DocumentPart(
            data="some text content",
            title="My Doc",
            media_type="text/plain",
        )
        assert p.type == "document"
        assert p.title == "My Doc"

    def test_optional_title(self):
        p = DocumentPart(data="content")
        assert p.title is None


class TestToolCallPart:
    def test_basic(self):
        p = ToolCallPart(id="call_1", name="get_weather", input={"city": "Paris"})
        assert p.type == "tool_call"
        assert p.id == "call_1"
        assert p.name == "get_weather"
        assert p.input == {"city": "Paris"}

    def test_empty_input(self):
        p = ToolCallPart(id="call_1", name="noop", input={})
        assert p.input == {}


class TestToolResultPart:
    def test_basic(self):
        p = ToolResultPart(id="call_1", content=[TextPart(text="72F")])
        assert p.type == "tool_result"
        assert p.id == "call_1"
        assert len(p.content) == 1
        assert p.content[0].text == "72F"

    def test_error_flag(self):
        p = ToolResultPart(id="call_1", content=[TextPart(text="err")], is_error=True)
        assert p.is_error is True


class TestThinkingPart:
    def test_basic(self):
        p = ThinkingPart(text="let me think...")
        assert p.type == "thinking"
        assert p.text == "let me think..."


class TestCitationPart:
    def test_basic(self):
        p = CitationPart(
            cited_text="the sky is blue",
            url="https://example.com/fact",
            title="Sky Facts",
        )
        assert p.type == "citation"
        assert p.cited_text == "the sky is blue"

    def test_minimal(self):
        p = CitationPart(cited_text="fact")
        assert p.cited_text == "fact"
        assert p.url is None


class TestPartUnion:
    def test_all_parts_have_type_discriminator(self):
        parts: list[Part] = [
            TextPart(text="a"),
            ImagePart(url="b"),
            AudioPart(data="c"),
            DocumentPart(data="d"),
            ToolCallPart(id="e", name="f", input={}),
            ToolResultPart(id="g", content=[TextPart(text="h")]),
            ThinkingPart(text="i"),
            CitationPart(cited_text="j"),
        ]
        types = [p.type for p in parts]
        assert types == [
            "text", "image", "audio", "document",
            "tool_call", "tool_result", "thinking", "citation",
        ]

    def test_isinstance_discrimination(self):
        parts: list[Part] = [
            TextPart(text="a"),
            ImagePart(url="b"),
        ]
        texts = [p for p in parts if isinstance(p, TextPart)]
        images = [p for p in parts if isinstance(p, ImagePart)]
        assert len(texts) == 1
        assert len(images) == 1
