import pytest

import dspy


class EchoNormalizedLM(dspy.BaseLM):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        text = request.messages[-1].text or ""
        return dspy.LMResponse.from_text(f"echo: {text}", model=request.model)

    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return self.forward(request)


def test_baselm_v2_call_returns_lm_response():
    lm = EchoNormalizedLM("test/echo")

    response = lm("hello", temperature=0.2)

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "echo: hello"
    assert response.model == "test/echo"
    assert lm.history[-1].request.config.temperature == 0.2


@pytest.mark.asyncio
async def test_baselm_v2_acall_returns_lm_response():
    lm = EchoNormalizedLM("test/echo")

    response = await lm.acall(dspy.User("hello async"))

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "echo: hello async"


def test_baselm_v2_constructor_omits_generation_defaults():
    lm = EchoNormalizedLM("test/echo")

    assert "temperature" not in lm.kwargs
    assert "max_tokens" not in lm.kwargs


def test_predict_accepts_baselm_v2_lm_response():
    class PredictLM(dspy.BaseLM):
        def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
            return dspy.LMResponse.from_text("[[ ## answer ## ]]\nblue\n\n[[ ## completed ## ]]", model=request.model)

    predict = dspy.Predict("question -> answer")
    result = predict(question="What color?", lm=PredictLM("test/predict"))

    assert result.answer == "blue"


def test_baselm_v1_constructor_keeps_legacy_generation_defaults():
    with pytest.warns(DeprecationWarning):
        class LegacyLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                raise NotImplementedError

    lm = LegacyLM("test/legacy")

    assert lm.kwargs["temperature"] == 0.0
    assert lm.kwargs["max_tokens"] == 1000
