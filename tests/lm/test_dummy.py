"""DummyLM rides the engine seam: production faces, scripted transport."""

import pytest
from lm15 import Response

import dspy
from dspy.core.errors import LMError


def test_script_plays_in_order_and_records_calls():
    lm = dspy.DummyLM(["Paris", "Berlin"])
    assert lm(prompt="Capital of France?") == ["Paris"]
    assert lm(prompt="Capital of Germany?") == ["Berlin"]
    assert lm.calls[0]["messages"][0]["content"] == "Capital of France?"
    assert len(lm.calls) == 2


def test_exhausted_script_refuses_loudly():
    lm = dspy.DummyLM(["only one"])
    lm(prompt="first")
    with pytest.raises(LMError, match="script exhausted"):
        lm(prompt="second")


def test_callable_outputs_compute_per_call():
    lm = dspy.DummyLM(lambda messages: messages[-1]["content"].upper())
    assert lm(prompt="echo me") == ["ECHO ME"]


def test_kwargs_are_recorded_for_assertions():
    lm = dspy.DummyLM(["ok"], temperature=0.7)
    lm(prompt="hi", max_tokens=32)
    assert lm.calls[0]["kwargs"] == {"temperature": 0.7, "max_tokens": 32}


def test_typed_face_returns_the_canonical_response():
    lm = dspy.DummyLM(["typed answer"])
    response = lm(dspy.System("Be terse."), dspy.User("Question?"))
    assert isinstance(response, Response)
    assert response.text == "typed answer"
    assert lm.calls[0]["messages"][0] == {"role": "system", "content": "Be terse."}


def test_stream_face_replays_the_script_and_records_history():
    lm = dspy.DummyLM(["streamed"])
    stream = lm.stream("Say it.")
    assert "".join(stream) == "streamed"
    assert stream.response.text == "streamed"
    assert len(lm.history) == 1


def test_lm15_fakelm_plugs_into_the_same_seam():
    from lm15.testing import FakeLM

    lm = dspy.LM("fake", router=FakeLM(["canonical answer"]))
    assert lm(prompt="anything") == ["canonical answer"]
