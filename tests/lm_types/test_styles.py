"""Tests for Style abstraction: ChatStyle, JSONStyle, XMLStyle, CSVStyle, custom."""


import pytest
from pydantic.fields import FieldInfo

from dspy.lm_types import (
    ChatStyle,
    CSVStyle,
    FieldChunk,
    FieldLayout,
    JSONStyle,
    Style,
    XMLStyle,
    get_style,
    register_style,
)


def _field(ann=str):
    """Fake pydantic FieldInfo-like object for testing."""
    class _F:
        annotation = ann
    return _F()


class TestChatStyle:
    def test_name(self):
        assert ChatStyle().name == "chat"

    def test_field_layout(self):
        layout = ChatStyle().field_layout()
        assert isinstance(layout, FieldLayout)
        assert "[[ ##" in layout.field_template
        assert layout.envelope_close.strip().endswith("[[ ## completed ## ]]")

    def test_layout_render(self):
        layout = ChatStyle().field_layout()
        rendered = layout.render([("answer", "Paris"), ("score", "0.9")])
        assert "[[ ## answer ## ]]" in rendered
        assert "Paris" in rendered
        assert "[[ ## score ## ]]" in rendered
        assert rendered.rstrip().endswith("[[ ## completed ## ]]")

    def test_output_request(self):
        req = ChatStyle().output_request(["answer", "score"])
        assert "[[ ## answer ## ]]" in req
        assert "[[ ## score ## ]]" in req
        assert "completed" in req

    def test_parse_basic(self):
        completion = (
            "[[ ## answer ## ]]\n"
            "Paris\n\n"
            "[[ ## score ## ]]\n"
            "0.9\n\n"
            "[[ ## completed ## ]]\n"
        )
        result = ChatStyle().parse(completion, {"answer": _field(), "score": _field()})
        assert result["answer"] == "Paris"
        assert result["score"] == "0.9"

    def test_parse_ignores_unknown_fields(self):
        completion = "[[ ## answer ## ]]\nParis\n\n[[ ## extra ## ]]\nignored"
        result = ChatStyle().parse(completion, {"answer": _field()})
        assert result == {"answer": "Paris"}


class TestJSONStyle:
    def test_name(self):
        assert JSONStyle().name == "json"

    def test_field_layout(self):
        layout = JSONStyle().field_layout()
        assert "{name}" in layout.field_template
        assert layout.envelope_open.strip().startswith("{")
        assert layout.envelope_close.strip().endswith("}")

    def test_output_request(self):
        req = JSONStyle().output_request(["answer"])
        assert "JSON" in req
        assert "answer" in req

    def test_parse_basic(self):
        completion = '{"answer": "Paris", "score": 0.9}'
        result = JSONStyle().parse(completion, {"answer": _field(), "score": _field()})
        assert result["answer"] == "Paris"
        assert result["score"] == 0.9

    def test_parse_repairs_malformed(self):
        completion = '{"answer": "Paris",}'
        result = JSONStyle().parse(completion, {"answer": _field()})
        assert result["answer"] == "Paris"

    def test_parse_extracts_from_prose(self):
        completion = 'Here is my answer: {"answer": "Paris"} hope that helps'
        result = JSONStyle().parse(completion, {"answer": _field()})
        assert result["answer"] == "Paris"


class TestXMLStyle:
    def test_name(self):
        assert XMLStyle().name == "xml"

    def test_output_request(self):
        req = XMLStyle().output_request(["answer"])
        assert "<answer>" in req

    def test_parse_basic(self):
        completion = "<answer>Paris</answer>\n<score>0.9</score>"
        result = XMLStyle().parse(completion, {"answer": _field(), "score": _field()})
        assert result["answer"] == "Paris"
        assert result["score"] == "0.9"

    def test_parse_multiline_content(self):
        completion = "<answer>\nline 1\nline 2\n</answer>"
        result = XMLStyle().parse(completion, {"answer": _field()})
        assert "line 1" in result["answer"]
        assert "line 2" in result["answer"]


class TestStyleRegistry:
    def test_builtins_registered(self):
        assert get_style("chat").name == "chat"
        assert get_style("json").name == "json"
        assert get_style("xml").name == "xml"
        assert get_style("csv").name == "csv"

    def test_unknown_raises(self):
        with pytest.raises(KeyError):
            get_style("nonexistent")

    def test_instance_passthrough(self):
        s = ChatStyle()
        assert get_style(s) is s

    def test_custom_style_registration(self):
        class MyStyle(Style):
            name = "_test_mystyle"

            def field_layout(self):
                return FieldLayout(field_template="{name}: {value}")

            def output_request(self, output_field_names):
                return f"Fields: {', '.join(output_field_names)}"

            def parse(self, completion, output_fields):
                return {}

            def stream_parser(self, output_field_names):
                from dspy.lm_types import StreamParser

                class _P:
                    def __init__(self, names): self.names = names
                    def feed(self, t): return iter([])
                    def finalize(self): return iter([])
                return _P(output_field_names)

        register_style(MyStyle())
        assert get_style("_test_mystyle").name == "_test_mystyle"


class TestChatStreamParser:
    def test_single_chunk(self):
        parser = ChatStyle().stream_parser(["answer"])
        chunks = list(parser.feed("[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"))
        # finalize produces any final chunks
        chunks += list(parser.finalize())
        assert any(c.field_name == "answer" and "Paris" in c.text for c in chunks)

    def test_split_across_chunks(self):
        parser = ChatStyle().stream_parser(["answer"])
        chunks = []
        for text in ["[[ ## ans", "wer ## ]]\nP", "aris\n\n[[ ## com", "pleted ## ]]"]:
            chunks.extend(parser.feed(text))
        chunks.extend(parser.finalize())
        full = "".join(c.text for c in chunks if c.field_name == "answer")
        assert "Paris" in full

    def test_multiple_fields(self):
        parser = ChatStyle().stream_parser(["answer", "score"])
        text = (
            "[[ ## answer ## ]]\nParis\n\n"
            "[[ ## score ## ]]\n0.9\n\n"
            "[[ ## completed ## ]]"
        )
        chunks = list(parser.feed(text)) + list(parser.finalize())
        answers = [c.text for c in chunks if c.field_name == "answer"]
        scores = [c.text for c in chunks if c.field_name == "score"]
        assert "".join(answers).strip() == "Paris"
        assert "".join(scores).strip() == "0.9"


class TestJSONStreamParser:
    def test_single_chunk(self):
        parser = JSONStyle().stream_parser(["answer"])
        chunks = list(parser.feed('{"answer": "Paris"}'))
        # Validate at least one chunk emitted
        assert any(c.field_name == "answer" for c in chunks)

    def test_split_across_chunks(self):
        parser = JSONStyle().stream_parser(["answer"])
        chunks = []
        for text in ['{"ans', 'wer": "Par', 'is"}']:
            chunks.extend(parser.feed(text))
        chunks.extend(parser.finalize())
        total = "".join(c.text for c in chunks if c.field_name == "answer")
        assert "Paris" in total


class TestXMLStreamParser:
    def test_single_field(self):
        parser = XMLStyle().stream_parser(["answer"])
        chunks = list(parser.feed("<answer>Paris</answer>"))
        chunks.extend(parser.finalize())
        answers = [c.text for c in chunks if c.field_name == "answer"]
        assert "".join(answers).strip() == "Paris"

    def test_split_across_chunks(self):
        parser = XMLStyle().stream_parser(["answer"])
        chunks = []
        for text in ["<answ", "er>Par", "is</answer>"]:
            chunks.extend(parser.feed(text))
        chunks.extend(parser.finalize())
        total = "".join(c.text for c in chunks if c.field_name == "answer")
        assert "Paris" in total


class TestCSVStyle:
    def test_name(self):
        assert CSVStyle().name == "csv"

    def test_format_output_single_row(self):
        block = CSVStyle().format_output_block(
            outputs={"answer": "Paris", "score": 0.9},
            columns=["answer", "score"],
        )
        assert "answer,score" in block
        assert "Paris" in block

    def test_format_output_multi_row(self):
        block = CSVStyle().format_output_block(
            outputs={"events": [
                {"name": "A", "year": 2024},
                {"name": "B", "year": 2025},
            ]},
            columns=["name", "year"],
        )
        lines = block.strip().splitlines()
        assert lines[0] == "name,year"
        assert "A,2024" in lines
        assert "B,2025" in lines

    def test_parse_single_row(self):
        completion = "[[ ## rows ## ]]\nanswer,score\nParis,0.9\n"
        result = CSVStyle().parse(completion, {"answer": _field(), "score": _field()})
        assert result["answer"] == "Paris"
        assert result["score"] == "0.9"

    def test_parse_multi_row(self):
        completion = "[[ ## rows ## ]]\nname,year\nA,2024\nB,2025\n"
        result = CSVStyle().parse(
            completion,
            {"events": _field(list[dict])},
        )
        assert len(result["events"]) == 2
        assert result["events"][0] == {"name": "A", "year": "2024"}

    def test_parse_tolerates_quoted_commas(self):
        completion = '[[ ## rows ## ]]\nanswer,tags\nParis,"a,b"\n'
        result = CSVStyle().parse(completion, {"answer": _field(), "tags": _field()})
        assert result["answer"] == "Paris"
        assert result["tags"] == "a,b"

    def test_tsv_variant(self):
        from dspy.lm_types import TSVStyle
        block = TSVStyle().format_output_block(
            outputs={"a": "x", "b": "y"},
            columns=["a", "b"],
        )
        assert "\t" in block

    def test_stream_single_row(self):
        parser = CSVStyle().stream_parser(["answer", "score"])
        chunks = list(parser.feed("[[ ## rows ## ]]\nanswer,score\nParis,0.9\n"))
        chunks.extend(parser.finalize())
        fields = {c.field_name: c.text for c in chunks}
        assert fields.get("answer") == "Paris"
        assert fields.get("score") == "0.9"

    def test_stream_multi_row(self):
        parser = CSVStyle().stream_parser(["events"])
        text = "[[ ## rows ## ]]\nname,year\nA,2024\nB,2025\n"
        chunks = list(parser.feed(text))
        chunks.extend(parser.finalize())
        events = [c for c in chunks if c.field_name == "events"]
        assert len(events) >= 2

    def test_stream_row_split_across_chunks(self):
        parser = CSVStyle().stream_parser(["answer"])
        chunks = []
        for text in ["[[ ## rows", " ## ]]\nan", "swer\nPar", "is\n"]:
            chunks.extend(parser.feed(text))
        chunks.extend(parser.finalize())
        answers = [c.text for c in chunks if c.field_name == "answer"]
        assert "Paris" in "".join(answers)
