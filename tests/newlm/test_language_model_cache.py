import uuid

import pytest

import dspy


class CountingLM(dspy.LanguageModel):
    def __init__(self, *, cache: bool = True):
        super().__init__(model=f"test/counting-{uuid.uuid4()}", cache=cache)
        self.forward_calls = 0

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_calls += 1
        return dspy.LMResponse.from_text(
            f"call {self.forward_calls}",
            model=request.model,
            usage=dspy.LMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=0.001,
        )


class AsyncCountingLM(CountingLM):
    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_calls += 1
        return dspy.LMResponse.from_text(
            f"async call {self.forward_calls}",
            model=request.model,
            usage=dspy.LMUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            cost=0.001,
        )


def test_language_model_uses_dspy_request_cache_before_forward():
    lm = CountingLM(cache=True)

    first = lm("hello")
    second = lm("hello")

    assert lm.forward_calls == 1
    assert first.text == "call 1"
    assert first.cache_hit is False
    assert second.text == "call 1"
    assert second.cache_hit is True
    assert second.usage == {}
    assert second.cost is None


def test_language_model_cache_can_be_disabled_on_instance():
    lm = CountingLM(cache=False)

    first = lm("hello")
    second = lm("hello")

    assert lm.forward_calls == 2
    assert first.text == "call 1"
    assert second.text == "call 2"
    assert second.cache_hit is False


def test_language_model_cache_can_be_overridden_per_call():
    lm = CountingLM(cache=True)

    first = lm("hello", cache=False)
    second = lm("hello", cache=False)

    assert lm.forward_calls == 2
    assert first.text == "call 1"
    assert second.text == "call 2"
    assert second.cache_hit is False


def test_language_model_cache_key_includes_lm_state():
    first_lm = CountingLM(cache=True)
    second_lm = CountingLM(cache=True)

    first = first_lm("hello")
    second = second_lm("hello")

    assert first.text == "call 1"
    assert second.text == "call 1"
    assert first_lm.forward_calls == 1
    assert second_lm.forward_calls == 1


@pytest.mark.asyncio
async def test_async_language_model_uses_dspy_request_cache_before_aforward():
    lm = AsyncCountingLM(cache=True)

    first = await lm.acall("hello")
    second = await lm.acall("hello")

    assert lm.forward_calls == 1
    assert first.text == "async call 1"
    assert second.text == "async call 1"
    assert second.cache_hit is True
