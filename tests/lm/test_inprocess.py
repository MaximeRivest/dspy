"""The one-line in-process UX: model string in, engine picked, nothing loaded."""

import pytest

import dspy
from dspy.core.errors import LMError
from dspy.lm.inprocess import InProcessEngine, _truncate_at_stop
from dspy.programir.weights import has_weight_spec


def test_hf_prefix_builds_lazy_inprocess_engine():
    lm = dspy.LM("hf:PleIAs/Baguettotron")
    assert isinstance(lm.router, InProcessEngine)
    assert lm.router.model_id == "PleIAs/Baguettotron"
    assert lm.router.transformer is None  # lazy: constructing loads nothing


def test_local_model_directory_is_detected(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    lm = dspy.LM(str(tmp_path))
    assert isinstance(lm.router, InProcessEngine)
    assert lm.router.model_id == str(tmp_path)


def test_plain_directory_without_config_is_not_inprocess(tmp_path):
    sentinel = object()
    lm = dspy.LM(str(tmp_path), router=sentinel)
    assert lm.router is sentinel


def test_inprocess_lm_forwards_the_weight_baking_hook():
    lm = dspy.LM("hf:PleIAs/Baguettotron")
    assert has_weight_spec(lm)
    assert lm.programir_weight_spec.__self__ is lm.router


def test_device_pin_travels_to_the_engine():
    lm = dspy.LM("hf:PleIAs/Baguettotron", device="cpu")
    assert lm.router.requested_device == "cpu"


def test_device_on_a_served_model_refuses():
    with pytest.raises(ValueError, match="in-process models only"):
        dspy.LM("gpt-4o-mini", device="cuda")


def test_inprocess_string_plus_router_refuses():
    with pytest.raises(ValueError, match="not both"):
        dspy.LM("hf:PleIAs/Baguettotron", router=object())


def test_engine_refuses_non_text_parts_loudly():
    from lm15 import ImagePart, Message, Request

    engine = InProcessEngine("test/tiny")
    request = Request(
        model="test/tiny",
        messages=(Message(role="user", parts=(ImagePart(url="https://x.test/dog.png"),)),),
    )
    with pytest.raises(LMError, match="text-only"):
        engine._chat_messages(request)


def test_stop_truncation_cuts_at_the_earliest_stop():
    assert _truncate_at_stop("hello STOP world", ["STOP"]) == ("hello ", True)
    assert _truncate_at_stop("a B c A", ["A", "B"]) == ("a ", True)
    assert _truncate_at_stop("no stops here", ["ZZZ"]) == ("no stops here", False)
    assert _truncate_at_stop("plain", None) == ("plain", False)
