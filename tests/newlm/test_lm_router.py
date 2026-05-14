import pytest

import dspy
from dspy.clients import lm as legacy_lm_module


@pytest.fixture(autouse=True)
def experimental_lm_context():
    with dspy.context(experimental_lm=True):
        yield


class AcmeLM(dspy.LanguageModel):
    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("acme", model=request.model)


def test_lm_uses_legacy_lm_by_default():
    with dspy.context(experimental_lm=False):
        lm = dspy.LM("openai/gpt-4o-mini", cache=False)

    assert isinstance(lm, legacy_lm_module.LM)


def test_lm_constructor_returns_openai_chat_backend():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)

    assert isinstance(lm, dspy.OpenAICompletionsLM)
    assert isinstance(lm, dspy.LanguageModel)
    assert lm.protocol == "chat"
    assert lm.model == "openai/gpt-4o-mini"


def test_lm_model_type_is_deprecated_but_still_routes_temporarily():
    with pytest.warns(DeprecationWarning, match="model_type"):
        lm = dspy.LM("openai/gpt-4o-mini", model_type="responses", cache=False)

    assert isinstance(lm, dspy.OpenAIResponsesLM)


def test_openai_completion_lms_can_be_constructed_directly():
    lm = dspy.OpenAICompletionsLM("openai/gpt-4o-mini", cache=False)

    assert isinstance(lm, dspy.OpenAICompletionsLM)
    assert isinstance(lm, dspy.LanguageModel)
    assert lm.protocol == "chat"
    assert lm.model == "openai/gpt-4o-mini"


def test_lm_returns_backend_state_directly():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=0.7, max_tokens=123)

    assert lm.cache is False
    assert lm.kwargs["temperature"] == 0.7
    assert lm.kwargs["max_tokens"] == 123

    lm.cache = True
    lm.kwargs["temperature"] = 0.2

    assert lm.cache is True
    assert lm.kwargs["temperature"] == 0.2


def test_lm_backend_registry_can_route_custom_models():
    @dspy.register_lm_backend
    def route_acme(model: str, *args, **kwargs):
        if model.startswith("acme/"):
            return AcmeLM(model, *args, **kwargs)
        return None

    lm = dspy.LM("acme/small", cache=False)

    assert isinstance(lm, AcmeLM)
    assert isinstance(lm, dspy.LanguageModel)
    assert lm.model == "acme/small"
    assert lm("hello").text == "acme"


def test_lm_copy_returns_copied_backend():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=0.1)

    copied = lm.copy(temperature=0.9, rollout_id=7)

    assert isinstance(copied, dspy.OpenAICompletionsLM)
    assert copied is not lm
    assert copied.kwargs["temperature"] == 0.9
    assert copied.kwargs["rollout_id"] == 7
    assert lm.kwargs["temperature"] == 0.1
