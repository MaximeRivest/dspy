import dspy

from dspy.streaming.chunks import StreamChunk
from dspy.streaming.parser_factory import create_stream_parser
from dspy.streaming.parsers.chat import ChatStreamParser
from dspy.streaming.parsers.json_parser import JSONStreamParser
from dspy.streaming.parsers.xml_parser import XMLStreamParser


class Sig(dspy.Signature):
    q: str = dspy.InputField()
    a: str = dspy.OutputField()


class TestChatStreamParser:
    def test_holds_partial_headers(self):
        parser = ChatStreamParser(["a"])

        chunks = []
        chunks.extend(parser.feed("[[ ## a ## ]]\nPar"))
        chunks.extend(parser.feed("is\n\n[[ ## comp"))
        chunks.extend(parser.feed("leted ## ]]"))
        chunks.extend(parser.finalize())

        assert all(c.field == "a" for c in chunks)
        assert "".join(c.text for c in chunks) == "Paris"
        assert any(c.is_last for c in chunks)


class TestJSONStreamParser:
    def test_streams_incremental_field_values(self):
        parser = JSONStreamParser(["a"])

        chunks = []
        chunks.extend(parser.feed('{"a": "P'))
        chunks.extend(parser.feed('aris"}'))
        chunks.extend(parser.finalize())

        assert ''.join(c.text for c in chunks if c.field == 'a') == '"Paris"'
        assert chunks[-1].is_last is True


class TestXMLStreamParser:
    def test_streams_incremental_field_values(self):
        parser = XMLStreamParser(["a"])

        chunks = []
        chunks.extend(parser.feed("<a>Par"))
        chunks.extend(parser.feed("is</a>"))
        chunks.extend(parser.finalize())

        assert ''.join(c.text for c in chunks if c.field == 'a') == 'Paris'
        assert chunks[-1].is_last is True


class TestParserFactory:
    def test_uses_chat_parser_by_default(self):
        parser = create_stream_parser(dspy.ChatAdapter(), Sig)
        assert isinstance(parser, ChatStreamParser)

    def test_uses_json_parser(self):
        parser = create_stream_parser(dspy.JSONAdapter(), Sig)
        assert isinstance(parser, JSONStreamParser)

    def test_uses_xml_parser(self):
        parser = create_stream_parser(dspy.XMLAdapter(), Sig)
        assert isinstance(parser, XMLStreamParser)


class TestStreamChunk:
    def test_serialization_helpers(self):
        chunk = StreamChunk(type="output_field", field="a", text="Paris", is_last=True, metadata={"x": 1})
        assert chunk.to_dict() == {
            "type": "output_field",
            "text": "Paris",
            "field": "a",
            "is_last": True,
            "metadata": {"x": 1},
        }
        assert '"field":"a"' in chunk.to_json()
