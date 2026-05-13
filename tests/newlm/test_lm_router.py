import pytest

import dspy
from dspy.clients.lm import LM as LegacyLM


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

    assert isinstance(lm, LegacyLM)
    assert not isinstance(lm, dspy.LMRouter)


def test_lm_constructor_returns_router_with_litellm_chat_backend():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False)

    assert isinstance(lm, dspy.LMRouter)
    assert isinstance(lm, dspy.LanguageModel)
    assert isinstance(lm.backend, dspy.SDKLanguageModel)
    assert lm.backend.metadata["provider"] == "litellm"
    assert lm.backend.metadata["protocol"] == "openai_chat"
    assert isinstance(lm.backend, dspy.LanguageModel)
    assert lm.model == "openai/gpt-4o-mini"
    assert lm.backend.model == "openai/gpt-4o-mini"


def test_litellm_factory_lms_can_be_constructed_directly():
    lm = dspy.litellm_chat_lm("openai/gpt-4o-mini", cache=False)

    assert isinstance(lm, dspy.SDKLanguageModel)
    assert isinstance(lm, dspy.LanguageModel)
    assert not isinstance(lm, dspy.LMRouter)
    assert lm.metadata["protocol"] == "openai_chat"
    assert lm.model == "openai/gpt-4o-mini"


def test_lm_router_delegates_state_to_backend():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=0.7, max_tokens=123)

    assert lm.cache is False
    assert lm.kwargs["temperature"] == 0.7
    assert lm.kwargs["max_tokens"] == 123

    lm.cache = True
    lm.kwargs["temperature"] = 0.2

    assert lm.backend.cache is True
    assert lm.backend.kwargs["temperature"] == 0.2


def test_lm_backend_registry_can_route_custom_models():
    @dspy.register_lm_backend
    def route_acme(model: str, *args, **kwargs):
        if model.startswith("acme/"):
            return AcmeLM(model, *args, **kwargs)
        return None

    lm = dspy.LM("acme/small", cache=False)

    assert isinstance(lm, dspy.LMRouter)
    assert isinstance(lm.backend, AcmeLM)
    assert isinstance(lm.backend, dspy.LanguageModel)
    assert lm.model == "acme/small"
    assert lm("hello").text == "acme"


def test_lm_router_copy_wraps_copied_backend():
    lm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=0.1)

    copied = lm.copy(temperature=0.9, rollout_id=7)

    assert isinstance(copied, dspy.LMRouter)
    assert isinstance(copied.backend, dspy.SDKLanguageModel)
    assert copied.backend.metadata["provider"] == "litellm"
    assert copied is not lm
    assert copied.backend is not lm.backend
    assert copied.kwargs["temperature"] == 0.9
    assert copied.kwargs["rollout_id"] == 7
    assert lm.kwargs["temperature"] == 0.1
