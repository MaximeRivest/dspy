"""Tests demonstrating that a custom (non-litellm) BaseLMv2 works fully in DSPy."""


import asyncio

import pytest

import dspy
from dspy.lm_types import (
    AdapterV2,
    AuthError,
    BaseLMv2,
    LMCompletion,
    LMConfig,
    LMMessage,
    LMResponse,
    LMStreamEvent,
    LMUsage,
    PartDelta,
    RateLimitError,
    TextPart,
    ThinkingPart,
    from_preset,
    map_http_status,
)


class FakeProviderLM(BaseLMv2):
    """A BaseLMv2 subclass that wraps a fake HTTP-like provider.

    This is what a user writing a non-litellm LM would do.
    """

    def __init__(self, model="fake/test", responses=None, error_on_call=None):
        super().__init__(model=model, num_retries=2)
        self.responses = responses or ["hello"]
        self.error_on_call = error_on_call or []  # list of (call_index, status_code, message)
        self.call_count = 0

    def forward(self, messages, config):
        idx = self.call_count
        self.call_count += 1

        for err_idx, status, msg in self.error_on_call:
            if err_idx == idx:
                raise map_http_status(status, msg, model=self.model)

        # Fake "call the provider"
        text = self.responses[idx % len(self.responses)]
        return LMResponse(
            completions=[LMCompletion(
                parts=[TextPart(text=text)],
                finish_reason="stop",
            )],
            usage=LMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
            model=self.model,
        )

    async def aforward(self, messages, config):
        return self.forward(messages, config)


class TestCustomLMBasic:
    def test_simple_call(self):
        lm = FakeProviderLM(responses=["[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"])
        response = lm([LMMessage.user("What?")])
        assert "Paris" in response.completions[0].text

    def test_config_handled(self):
        lm = FakeProviderLM()
        lm([LMMessage.user("hi")], LMConfig(temperature=0.8, max_tokens=500))
        # Just ensure it doesn't crash

    def test_multiple_messages(self):
        lm = FakeProviderLM(responses=["ok"])
        messages = [
            LMMessage.system("system"),
            LMMessage.user("user"),
            LMMessage.assistant("assistant"),
            LMMessage.user("user2"),
        ]
        response = lm(messages)
        assert response.completions[0].text == "ok"


class TestCustomLMWithAdapter:
    def test_full_pipeline(self):
        class Sig(dspy.Signature):
            """Answer."""
            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        lm = FakeProviderLM(responses=[
            "[[ ## a ## ]]\nParis\n\n[[ ## completed ## ]]",
        ])
        adapter = from_preset("chat")
        results = adapter(lm, LMConfig(), Sig, [], {"q": "What?"})
        assert results[0]["a"] == "Paris"

    def test_json_adapter_with_custom_lm(self):
        class Sig(dspy.Signature):
            """Answer."""
            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        lm = FakeProviderLM(responses=['{"a": "Paris"}'])
        adapter = from_preset("json")
        results = adapter(lm, LMConfig(), Sig, [], {"q": "What?"})
        assert results[0]["a"] == "Paris"


class TestCustomLMRetries:
    def test_retries_then_succeeds(self):
        lm = FakeProviderLM(
            responses=["ok"],
            error_on_call=[(0, 429, "rate limit")],
        )
        import dspy.lm_types.base_lm_v2 as _bl
        orig = _bl.time.sleep
        _bl.time.sleep = lambda _: None
        try:
            response = lm([LMMessage.user("hi")])
        finally:
            _bl.time.sleep = orig
        assert response.completions[0].text == "ok"
        assert lm.call_count == 2

    def test_no_retry_on_auth(self):
        lm = FakeProviderLM(error_on_call=[(0, 401, "unauthorized")])
        with pytest.raises(AuthError):
            lm([LMMessage.user("hi")])


class TestCustomLMStreaming:
    """A custom BaseLMv2 with full streaming support."""

    def test_custom_streaming(self):
        class StreamingFakeLM(BaseLMv2):
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[TextPart(text="a b c")])])

            async def aforward(self, messages, config):
                return self.forward(messages, config)

            async def stream(self, messages, config=None):
                yield LMStreamEvent(type="start", model=self.model)
                for word in ["a", " b", " c"]:
                    yield LMStreamEvent(
                        type="delta",
                        delta=PartDelta(type="text", text=word),
                    )
                yield LMStreamEvent(
                    type="end",
                    finish_reason="stop",
                    usage=LMUsage(input_tokens=5, output_tokens=3, total_tokens=8),
                )

        lm = StreamingFakeLM(model="custom")
        events = []

        async def collect():
            async for e in lm.stream([LMMessage.user("hi")]):
                events.append(e)

        asyncio.run(collect())

        deltas = [e.delta.text for e in events if e.type == "delta"]
        assert "".join(deltas) == "a b c"

    def test_custom_streaming_through_adapter(self):
        class Sig(dspy.Signature):
            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        class StreamingFakeLM(BaseLMv2):
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[TextPart(
                    text="[[ ## a ## ]]\nResult\n\n[[ ## completed ## ]]"
                )])])
            async def aforward(self, messages, config):
                return self.forward(messages, config)
            async def stream(self, messages, config=None):
                yield LMStreamEvent(type="start")
                yield LMStreamEvent(type="delta", delta=PartDelta(type="text", text="[[ ## a ## ]]\n"))
                yield LMStreamEvent(type="delta", delta=PartDelta(type="text", text="Res"))
                yield LMStreamEvent(type="delta", delta=PartDelta(type="text", text="ult"))
                yield LMStreamEvent(type="delta", delta=PartDelta(type="text", text="\n\n[[ ## completed ## ]]"))
                yield LMStreamEvent(type="end", finish_reason="stop")

        lm = StreamingFakeLM(model="x")
        adapter = from_preset("chat")
        collected = []

        async def run():
            async for c in adapter.stream_call(lm, LMConfig(), Sig, [], {"q": "hi"}):
                collected.append(c)

        asyncio.run(run())
        a = "".join(c.text for c in collected if c.field_name == "a")
        assert "Result" in a
