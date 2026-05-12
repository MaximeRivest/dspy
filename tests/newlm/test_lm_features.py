import json
from typing import Iterator

import dspy


class EchoLM(dspy.LanguageModel):
    def __init__(self, response: dspy.LMResponse | None = None):
        super().__init__(model="test/echo", cache=False)
        self.response = response or dspy.LMResponse.from_text("hello", model="test/echo")

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return self.response


class StreamingLM(EchoLM):
    def forward_stream(self, request: dspy.LMRequest) -> Iterator[dspy.LMStreamEvent]:
        yield dspy.LMStreamStartEvent(model=request.model)
        yield dspy.LMStreamDeltaEvent(output_index=0, part_index=0, delta=dspy.LMTextDelta(text="hello"))
        yield dspy.LMStreamEndEvent()


class AsyncLM(EchoLM):
    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return self.response


class AsyncStreamingLM(EchoLM):
    async def aforward_stream(self, request: dspy.LMRequest):
        yield dspy.LMStreamStartEvent(model=request.model)
        yield dspy.LMStreamEndEvent(response=self.response)


def test_feature_status_is_truthy_only_when_supported():
    assert bool(dspy.FeatureStatus("usage", "observed")) is True
    assert bool(dspy.FeatureStatus("usage", "inferred")) is True
    assert bool(dspy.FeatureStatus("usage", "unknown")) is False
    assert bool(dspy.FeatureStatus("usage", "unsupported")) is False


def test_usage_and_cost_are_unknown_before_observation():
    lm = EchoLM()

    assert lm.features.usage.status == "unknown"
    assert lm.features.cost.status == "unknown"
    assert not lm.features.usage
    assert not lm.features.cost


def test_usage_and_cost_are_observed_from_lm_response():
    lm = EchoLM(
        dspy.LMResponse.from_text(
            "hello",
            model="test/echo",
            usage=dspy.LMUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            cost=0.0001,
        )
    )

    lm("hello")

    assert lm.features.usage.status == "observed"
    assert lm.features.cost.status == "observed"
    assert lm.features.supports("usage") is True
    assert lm.features.supports("cost") is True
    assert "Latest LMResponse" in lm.features.explain("usage")


def test_streaming_support_is_inferred_from_forward_stream_override():
    assert StreamingLM().features.streaming.status == "inferred"
    assert StreamingLM().features.streaming
    assert EchoLM().features.streaming.status == "unsupported"
    assert not EchoLM().features.streaming


def test_native_async_support_is_inferred_from_aforward_override():
    assert AsyncLM().features.native_async.status == "inferred"
    assert AsyncLM().features.native_async
    assert EchoLM().features.native_async.status == "unsupported"
    assert not EchoLM().features.native_async


def test_async_streaming_support_is_inferred_from_aforward_stream_override():
    assert AsyncStreamingLM().features.async_streaming.status == "inferred"
    assert AsyncStreamingLM().features.async_streaming
    assert EchoLM().features.async_streaming.status == "unsupported"
    assert not EchoLM().features.async_streaming


def test_report_returns_text_or_json_ready_data():
    lm = EchoLM()

    text = lm.features.report()
    data = lm.features.report(format="json")

    assert "DSPy LM feature report for test/echo" in text
    assert data["model"] == "test/echo"
    assert data["features"]["text_generation"]["status"] == "inferred"
    assert data["features"]["usage"]["status"] == "unknown"
    assert json.loads(lm.features.to_json())["model"] == "test/echo"


def test_report_rejects_unknown_format():
    lm = EchoLM()

    try:
        lm.features.report(format="yaml")
    except ValueError as exc:
        assert "format" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
