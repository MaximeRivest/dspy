"""The LM API promise suite: one test per promise in the contract doc.

Mirrors `docs/docs/community/normalized-lm-api-migration.md` section by
section. Every behavior that page promises users is asserted here,
offline, through the production seams (`DummyLM`, `FakeLM`, scripted
engines). The live provider matrix (`tests/live/`) proves the same
promises against real wires.

If a test here must change, the doc page must change in the same
commit — this file is the doc's enforcement arm.
"""

from __future__ import annotations

import pytest
from lm15 import (
    ImagePart,
    Message,
    Request,
    Response,
    StreamEndEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
    Usage,
)
from lm15.testing import FakeLM

import dspy
from dspy.core.errors import LMError


def _response(text: str, model: str = "fake") -> Response:
    return Response(
        id=None,
        model=model,
        message=Message.assistant(text),
        finish_reason="stop",
        usage=Usage(input_tokens=3, output_tokens=2, total_tokens=5),
    )


# ── The call surface ─────────────────────────────────────────────────


def test_public_symbols_exist():
    for name in ("LM", "DummyLM", "System", "User", "Assistant", "ToolCall", "ToolResult", "Image"):
        assert hasattr(dspy, name), f"dspy.{name} is promised by the contract doc"


def test_typed_face_returns_canonical_response_by_default():
    lm = dspy.DummyLM(["Hello!"])
    response = lm("hello")
    assert isinstance(response, Response)
    assert response.text == "Hello!"
    assert response.finish_reason == "stop"
    assert response.usage is not None


def test_no_experimental_gate_exists():
    # The old plan gated typed returns behind dspy.context(experimental=True).
    # The landed contract has no such flag; typed is the default.
    assert not hasattr(dspy, "context")


def test_legacy_keyword_face_returns_strings_with_response_riding():
    lm = dspy.DummyLM(["Hello!"])
    outputs = lm(prompt="hello")
    assert outputs == ["Hello!"]  # equal to a plain list
    assert isinstance(outputs.response, Response)
    assert outputs.response.text == "Hello!"


def test_mixing_the_two_faces_refuses():
    lm = dspy.DummyLM(["x"])
    with pytest.raises(ValueError, match="not both"):
        lm(dspy.User("hi"), prompt="hi")


# ── Conversations are typed turns ────────────────────────────────────


def test_turns_and_previous_response_fold_into_one_conversation():
    fake = FakeLM([_response("alpha"), _response("beta")])
    lm = dspy.LM("fake", router=fake)
    first = lm(dspy.User("Say alpha."))
    follow = lm(
        dspy.System("Be terse."),
        dspy.User("Say alpha."),
        first,
        dspy.User("Now say beta."),
    )
    assert follow.text == "beta"

    request = fake.requests[-1]
    assert request.system == "Be terse."
    assert [m.role for m in request.messages] == ["user", "assistant", "user"]
    assert request.messages[1].text == "alpha"  # the Response became its turn


def test_tool_transcript_turns_build_canonical_parts():
    fake = FakeLM([_response("22 C and sunny")])
    lm = dspy.LM("fake", router=fake)
    lm(
        dspy.User("Weather in Paris?"),
        dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
        dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
        dspy.User("Summarize."),
    )
    request = fake.requests[-1]
    assistant = request.messages[1]
    assert assistant.parts[0].type == "tool_call"
    assert assistant.parts[0].name == "get_weather"
    tool_result = request.messages[2]
    assert tool_result.parts[0].type == "tool_result"
    assert tool_result.parts[0].id == "call_1"


def test_image_values_become_image_parts():
    fake = FakeLM([_response("a dog")])
    lm = dspy.LM("fake", router=fake)
    lm(dspy.User("Describe this image.", dspy.Image("https://example.com/dog.png")))
    parts = fake.requests[-1].messages[0].parts
    assert isinstance(parts[0], TextPart)
    assert isinstance(parts[1], ImagePart)
    assert parts[1].url == "https://example.com/dog.png"


# ── The full-surface escape hatch ────────────────────────────────────


def test_complete_speaks_full_canonical_lm15():
    fake = FakeLM([_response("ok")])
    lm = dspy.LM("fake", router=fake)
    response = lm.complete(
        Request(model="fake", messages=(Message(role="user", parts=(TextPart(text="hi"),)),))
    )
    assert isinstance(response, Response)
    assert fake.requests[-1].messages[0].text == "hi"


# ── Streaming ────────────────────────────────────────────────────────


def test_stream_is_a_method_never_a_flag():
    assert hasattr(dspy.LM, "stream")
    lm = dspy.DummyLM(["Hi there"])
    outputs = lm(prompt="hello", stream=True)  # a kwarg named stream is just a kwarg
    assert outputs == ["Hi there"]  # __call__ never forks its return type


def test_stream_yields_text_then_the_finished_response():
    lm = dspy.DummyLM(["river stones whisper"])
    stream = lm.stream("haiku please")
    chunks = list(stream)
    assert "".join(chunks) == "river stones whisper"
    assert isinstance(stream.response, Response)
    assert stream.response.text == "river stones whisper"


def test_stream_events_read_start_deltas_end():
    lm = dspy.DummyLM(["one two"])
    events = list(lm.stream("count").events())
    assert isinstance(events[0], StreamStartEvent)
    assert isinstance(events[-1], StreamEndEvent)
    assert any(isinstance(getattr(e, "delta", None), TextDelta) for e in events)


def test_streamed_and_buffered_calls_record_identical_history_shape():
    lm = dspy.DummyLM(["a", "b"])
    lm("hello")
    stream = lm.stream("hello again")
    list(stream)
    assert len(lm.history) == 2
    assert set(lm.history[0]) == set(lm.history[1])
    assert lm.history[1]["outputs"] == ["b"]


def test_failed_stream_records_nothing():
    lm = dspy.DummyLM(["only one"])
    list(lm.stream("first"))
    with pytest.raises(LMError, match="exhausted"):
        list(lm.stream("second"))  # script is empty now
    assert len(lm.history) == 1


def test_astream_is_not_promised_yet():
    # The contract doc lists astream as an open item, not a surface.
    assert not hasattr(dspy.LM, "astream")


# ── Model strings, routing, the engine seam ──────────────────────────


def test_any_object_with_complete_is_an_engine():
    class Minimal:
        def complete(self, request: Request) -> Response:
            return _response("from the seam", model=request.model)

    lm = dspy.LM("anything:at-all", router=Minimal())
    assert lm("hi").text == "from the seam"


def test_explicit_router_always_wins_over_endpoint_kwargs():
    class Minimal:
        def complete(self, request: Request) -> Response:
            return _response("mine")

    router = Minimal()
    lm = dspy.LM("openai-chat:stub-model", router=router, api_base="http://ignored.test/v1")
    assert lm.router is router
    assert lm("hi").text == "mine"


def test_endpoint_pin_kwargs_are_held_for_the_call_path():
    # The pin itself is exercised in test_complete_endpoint.py and live
    # in tests/live/test_vllm.py; here we pin the public shape.
    lm = dspy.LM("openai-chat:m", api_base="http://lab.test/v1", api_key="k")
    assert lm.kwargs["api_base"] == "http://lab.test/v1"
    assert lm.kwargs["api_key"] == "k"


# ── Usage, cost, history ─────────────────────────────────────────────

_HISTORY_KEYS = {"model", "messages", "kwargs", "outputs", "usage", "cost", "response", "timestamp"}


def test_every_call_appends_one_history_entry_with_the_promised_keys():
    lm = dspy.DummyLM(["a", "b"])
    lm("typed call")
    lm(prompt="legacy call")
    assert len(lm.history) == 2
    for entry in lm.history:
        assert _HISTORY_KEYS <= set(entry)
        assert isinstance(entry["usage"], Usage)


def test_cost_none_is_the_honest_answer_for_unpriced_models():
    lm = dspy.DummyLM(["a"])
    lm("hello")
    assert lm.history[-1]["cost"] is None  # no catalog price for 'dummy'


# ── Errors ───────────────────────────────────────────────────────────


def test_provider_failures_surface_as_typed_lmerror_with_the_canonical_code():
    from lm15.errors import BillingError

    class Broke:
        def complete(self, request: Request) -> Response:
            raise BillingError("no credits")

    lm = dspy.LM("fake", router=Broke())
    with pytest.raises(LMError, match=r"failed \(billing\)") as info:
        lm("hi")
    assert isinstance(info.value.__cause__, BillingError)  # structured fields ride the cause


# ── Testing seam ─────────────────────────────────────────────────────


def test_dummylm_is_an_ordinary_lm_on_the_engine_seam():
    lm = dspy.DummyLM(["Paris"])
    assert isinstance(lm, dspy.LM)
    assert lm(prompt="Capital of France?") == ["Paris"]
    assert lm.calls[0]["messages"][0]["content"] == "Capital of France?"


def test_exhausted_dummylm_refuses_loudly():
    lm = dspy.DummyLM(["only"])
    lm("first")
    with pytest.raises(LMError, match="exhausted"):
        lm("second")


# ── Credential hygiene at export ─────────────────────────────────────


def test_export_scanner_collects_per_lm_api_keys_for_leak_refusal():
    from dspy.programir.export import _credential_values

    lm = dspy.DummyLM(["a"], api_key="sk-secret-value")
    program = dspy.Predict("question -> answer", lm=lm)
    values = _credential_values(program)
    assert "sk-secret-value" in values.values()
