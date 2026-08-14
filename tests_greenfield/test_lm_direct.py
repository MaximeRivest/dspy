"""The typed direct-call surface: `lm(...)` over lm15 messages.

`dspy.System`/`User`/`Assistant`/`ToolCall`/`ToolResult` are thin
constructors over canonical lm15 values; the typed positional face of
`LM.__call__` returns the lm15 `Response`. The legacy keyword face
(`messages=`/`prompt=`) keeps returning `list[str]` for adapters.
"""

import pytest
from lm15 import ImagePart, Message, Response, TextPart, ToolCallPart, ToolResultPart

import dspy


# ---------------------------------------------------------------------------
# The vocabulary: constructors over lm15, not new types
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_user_builds_lm15_message(self):
        m = dspy.User("hello")
        assert isinstance(m, Message)
        assert m.role == "user"
        assert m.parts == (TextPart("hello"),)

    def test_assistant_builds_lm15_message(self):
        m = dspy.Assistant("prior answer")
        assert m.role == "assistant"
        assert m.text == "prior answer"

    def test_system_is_a_marker_not_a_message(self):
        s = dspy.System("Be concise.")
        assert not isinstance(s, Message)  # lm15 carries system on the request
        assert s.text == "Be concise."

    def test_tool_call_is_an_lm15_part(self):
        part = dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})
        assert part == ToolCallPart(id="call_1", name="get_weather", input={"city": "Paris"})

    def test_assistant_carries_tool_calls(self):
        m = dspy.Assistant(dspy.ToolCall(id="c1", name="f", args={}))
        assert m.role == "assistant"
        assert isinstance(m.parts[0], ToolCallPart)

    def test_tool_result_builds_tool_message(self):
        m = dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather")
        assert m.role == "tool"
        part = m.parts[0]
        assert isinstance(part, ToolResultPart)
        assert part.id == "call_1"
        assert part.name == "get_weather"
        assert not part.is_error

    def test_tool_result_error_flag(self):
        m = dspy.ToolResult("boom", call_id="c1", is_error=True)
        assert m.parts[0].is_error

    def test_user_mixed_content_with_dspy_image(self):
        image = dspy.Image("https://example.com/dog.png")
        m = dspy.User("Describe this image.", image)
        assert m.parts[0] == TextPart("Describe this image.")
        assert m.parts[1] == ImagePart(url="https://example.com/dog.png")

    def test_data_uri_image_becomes_data_part(self):
        image = dspy.Image("data:image/jpeg;base64,QUJD")
        part = dspy.User("x", image).parts[1]
        assert part == ImagePart(media_type="image/jpeg", data="QUJD")

    def test_lm15_parts_pass_through(self):
        part = ImagePart(url="https://example.com/cat.png")
        assert dspy.User("look", part).parts[1] is part

    def test_empty_message_refuses(self):
        with pytest.raises(ValueError, match="at least one content item"):
            dspy.User()

    def test_non_part_content_refuses_loudly(self):
        with pytest.raises(TypeError):
            dspy.User(42)


# ---------------------------------------------------------------------------
# The typed call face: positional in, lm15 Response out
# ---------------------------------------------------------------------------


class TestTypedCall:
    def test_bare_string_returns_response(self):
        lm = dspy.DummyLM(["Hello!"])
        response = lm("hi")
        assert isinstance(response, Response)
        assert response.text == "Hello!"
        assert response.usage is not None

    def test_legacy_faces_still_return_strings(self):
        lm = dspy.DummyLM(["a", "b", "c"])
        assert lm(prompt="hi") == ["a"]
        assert lm(messages=[{"role": "user", "content": "hi"}]) == ["b"]
        assert lm([{"role": "user", "content": "hi"}]) == ["c"]  # positional list

    def test_system_folds_into_request_system(self):
        lm = dspy.DummyLM(["ok"])
        lm(dspy.System("Be concise."), dspy.User("hi"))
        request = lm.calls[0]["request"]
        assert request.system == "Be concise."
        assert [m.role for m in request.messages] == ["user"]

    def test_multi_turn_conversation(self):
        lm = dspy.DummyLM(["Programming, not prompting."])
        response = lm(
            dspy.User("What is DSPy?"),
            dspy.Assistant("DSPy is a framework for programming LM pipelines."),
            dspy.User("Say that in five words."),
        )
        assert response.text == "Programming, not prompting."
        request = lm.calls[0]["request"]
        assert [m.role for m in request.messages] == ["user", "assistant", "user"]

    def test_previous_response_folds_in_as_assistant_turn(self):
        lm = dspy.DummyLM(["First answer.", "Shorter."])
        first = lm("Explain DSPy in one sentence.")
        follow = lm(
            dspy.User("Explain DSPy in one sentence."),
            first,
            dspy.User("Now make it even shorter."),
        )
        assert follow.text == "Shorter."
        request = lm.calls[1]["request"]
        assert request.messages[1] is first.message
        assert request.messages[1].role == "assistant"

    def test_tool_transcript_flows_through(self):
        lm = dspy.DummyLM(["It is 22 C in Paris."])
        response = lm(
            dspy.User("What is the weather in Paris?"),
            dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
            dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
            dspy.User("Summarize the result."),
        )
        assert response.text == "It is 22 C in Paris."
        request = lm.calls[0]["request"]
        assert [m.role for m in request.messages] == ["user", "assistant", "tool", "user"]

    def test_typed_call_records_history(self):
        lm = dspy.DummyLM(["ok"])
        response = lm("hi")
        record = lm.history[0]
        assert record["outputs"] == ["ok"]
        assert record["response"] is response

    def test_mixing_typed_and_keyword_refuses(self):
        lm = dspy.DummyLM(["ok"])
        with pytest.raises(ValueError, match="not both"):
            lm(dspy.User("hi"), prompt="hi")
        with pytest.raises(ValueError, match="not both"):
            lm(dspy.User("hi"), messages=[{"role": "user", "content": "hi"}])

    def test_unknown_item_refuses_with_teaching_error(self):
        lm = dspy.DummyLM(["ok"])
        with pytest.raises(TypeError, match="dspy.ToolCall goes inside dspy.Assistant"):
            lm(dspy.ToolCall(id="c1", name="f", args={}))

    def test_script_exhaustion_refuses_on_typed_face(self):
        lm = dspy.DummyLM(["only one"])
        lm("hi")
        with pytest.raises(dspy.LMError, match="script exhausted"):
            lm("again")


# ---------------------------------------------------------------------------
# The typed face over the real LM class (FakeLM transport)
# ---------------------------------------------------------------------------


class TestTypedCallTransport:
    def test_typed_call_builds_canonical_request(self):
        from lm15.testing import FakeLM

        fake = FakeLM(["hello"])
        lm = dspy.LM("openai/gpt-4o-mini", temperature=0.2, router=fake)
        response = lm(dspy.System("Be terse."), dspy.User("hi"), max_tokens=5)
        assert isinstance(response, Response)
        assert response.text == "hello"
        request = fake.requests[0]
        assert request.model == "openai/gpt-4o-mini"
        assert request.system == "Be terse."
        assert request.config.temperature == 0.2
        assert request.config.max_tokens == 5

    def test_provider_failure_maps_to_typed_lm_error(self):
        from lm15.errors import RateLimitError
        from lm15.testing import FakeLM

        lm = dspy.LM("m", router=FakeLM([RateLimitError("boom")]))
        with pytest.raises(dspy.LMError, match="rate_limit"):
            lm(dspy.User("hi"))

    def test_contentless_typed_response_is_returned_not_refused(self):
        # The typed face hands back the canonical Response as-is:
        # tool-call-only responses are data, not an error.
        from lm15 import Usage
        from lm15.testing import FakeLM

        tool_response = Response(
            id=None,
            model="m",
            message=Message(role="assistant", parts=(ToolCallPart(id="c1", name="f", input={}),)),
            finish_reason="tool_call",
            usage=Usage(),
        )
        lm = dspy.LM("m", router=FakeLM([tool_response]))
        response = lm(dspy.User("hi"))
        assert response.text is None
        assert response.tool_calls


# ---------------------------------------------------------------------------
# Streaming: a separate method, one event vocabulary, history on completion
# ---------------------------------------------------------------------------


class TestStream:
    def test_iterating_yields_text_then_response(self):
        lm = dspy.DummyLM(["Rivers run to the sea."])
        stream = lm.stream("Write a haiku about rivers.")
        chunks = list(stream)
        assert "".join(chunks) == "Rivers run to the sea."
        assert stream.response.text == "Rivers run to the sea."
        assert stream.finish_reason == "stop"

    def test_events_speak_the_lm15_vocabulary(self):
        lm = dspy.DummyLM(["hello"])
        events = list(lm.stream("hi").events())
        assert [e.type for e in events] == ["start", "delta", "end"]
        assert events[1].delta.type == "text"
        assert events[1].delta.text == "hello"

    def test_accepts_every_input_face(self):
        lm = dspy.DummyLM(["a", "b", "c"])
        assert lm.stream(dspy.System("terse"), dspy.User("hi")).text == "a"
        assert lm.stream(prompt="hi").text == "b"
        assert lm.stream(messages=[{"role": "user", "content": "hi"}]).text == "c"

    def test_legacy_system_dict_folds_into_request_system(self):
        lm = dspy.DummyLM(["ok"])
        stream = lm.stream(messages=[{"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"}])
        assert stream.response.text == "ok"
        assert lm.calls[0]["request"].system == "Be terse."

    def test_history_records_once_on_completion(self):
        lm = dspy.DummyLM(["hello"])
        stream = lm.stream("hi")
        assert lm.history == []  # nothing recorded before the stream ends
        list(stream)
        assert stream.response is stream.response
        assert len(lm.history) == 1
        assert lm.history[0]["outputs"] == ["hello"]

    def test_streamed_and_buffered_calls_record_the_same_shape(self):
        lm = dspy.DummyLM(["same", "same"])
        lm("hi")
        lm.stream("hi").response
        buffered, streamed = lm.history
        assert buffered.keys() == streamed.keys()
        assert buffered["outputs"] == streamed["outputs"]

    def test_native_stream_via_fake_transport(self):
        from lm15.testing import FakeLM

        fake = FakeLM(["hello"])
        lm = dspy.LM("m", router=fake)
        stream = lm.stream(dspy.User("hi"))
        assert "".join(stream) == "hello"
        assert stream.response.text == "hello"

    def test_provider_failure_maps_to_typed_lm_error(self):
        from lm15.errors import RateLimitError
        from lm15.testing import FakeLM

        lm = dspy.LM("m", router=FakeLM([RateLimitError("boom")]))
        with pytest.raises(dspy.LMError, match="rate_limit"):
            list(lm.stream("hi"))
        assert lm.history == []  # a failed stream records nothing

    def test_mixing_typed_and_keyword_refuses(self):
        lm = dspy.DummyLM(["ok"])
        with pytest.raises(ValueError, match="not both"):
            lm.stream(dspy.User("hi"), prompt="hi")

    def test_script_exhaustion_refuses(self):
        lm = dspy.DummyLM([])
        with pytest.raises(dspy.LMError, match="script exhausted"):
            list(lm.stream("hi"))
