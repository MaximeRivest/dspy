import types

import pytest

import dspy
from dspy.core.types import LMResponse, LMUsage
from dspy.utils.callback import BaseCallback


class V2EchoLM(dspy.BaseLM):
    def __init__(self, **kwargs):
        super().__init__(model="test/echo", cache=False, **kwargs)
        self.requests = []

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.requests.append(request)
        return dspy.LMResponse.from_text("ok", model=request.model, usage=LMUsage(input_tokens=1))


def _completion_response(text="ok"):
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=text, tool_calls=None),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        usage={},
        model="test/legacy",
        _hidden_params={},
    )


def test_legacy_subclass_direct_and_request_calls_keep_contracts():
    class LegacyLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="test/legacy", cache=False)
            self.calls = []

        def forward(self, prompt=None, messages=None, **kwargs):
            self.calls.append({"prompt": prompt, "messages": messages, "kwargs": kwargs})
            return _completion_response("legacy ok")

    lm = LegacyLM()

    assert lm("hello") == ["legacy ok"]

    request = dspy.LMRequest.from_call(model="test/legacy", prompt="hello", temperature=0.2)
    with dspy.context(experimental=True):
        response = lm(request=request)

    assert isinstance(response, LMResponse)
    assert response.text == "legacy ok"
    assert lm.calls[-1]["messages"] == [{"role": "user", "content": "hello"}]
    assert lm.calls[-1]["kwargs"]["temperature"] == 0.2


def test_lm_request_requires_experimental_mode():
    lm = V2EchoLM()
    request = dspy.LMRequest.from_call(model="test/echo", prompt="hello")

    with pytest.warns(UserWarning, match="experimental=True"):
        with pytest.raises(ValueError, match="experimental=True"):
            lm(request=request)


def test_v2_subclass_must_return_lm_response():
    class BadLM(dspy.BaseLM):
        def forward(self, request):
            return ["legacy-shaped output"]

    with dspy.context(experimental=True):
        with pytest.raises(TypeError, match="must return an LMResponse"):
            BadLM("test/bad", cache=False)("hello")


def test_normalized_history_redacts_secrets():
    lm = V2EchoLM()

    with dspy.context(experimental=True):
        lm("hello", api_key="secret", anthropic_api_key="secret2", headers={"Authorization": "Bearer secret"})

    entry = lm.history[-1]
    kwargs = entry["kwargs"]
    assert kwargs["api_key"] == "<redacted>"
    assert kwargs["anthropic_api_key"] == "<redacted>"
    assert kwargs["headers"]["Authorization"] == "<redacted>"


def test_lm_callbacks_receive_redacted_request_and_raw_kwargs():
    class RecordingCallback(BaseCallback):
        def __init__(self):
            self.inputs = None

        def on_lm_start(self, call_id, instance, inputs):
            self.inputs = inputs

    callback = RecordingCallback()
    lm = V2EchoLM(callbacks=[callback])

    with dspy.context(experimental=True):
        lm("hello", api_key="secret")

    assert callback.inputs["request"].config.extensions["api_key"] == "<redacted>"
    assert callback.inputs["raw"]["kwargs"]["api_key"] == "<redacted>"


def test_retry_uses_structured_lm_errors(monkeypatch):
    class FlakyLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="test/flaky", cache=False, num_retries=1)
            self.calls = 0

        def forward(self, request):
            self.calls += 1
            if self.calls == 1:
                raise dspy.LMRateLimitError("slow down", model=request.model, status=429)
            return dspy.LMResponse.from_text("ok", model=request.model)

    monkeypatch.setattr("dspy.clients.base_lm._sleep_before_retry", lambda attempt: None)
    lm = FlakyLM()

    with dspy.context(experimental=True):
        assert lm("hello").text == "ok"
    assert lm.calls == 2


def test_normalized_errors_are_not_normalized_twice():
    class WrappingLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model="test/wrap", cache=False, num_retries=0)
            self.normalize_calls = 0

        def forward(self, request):
            raise RuntimeError("native provider error")

        def normalize_error(self, error, request):
            self.normalize_calls += 1
            return dspy.LMProviderError("wrapped", model=request.model, status=500)

    lm = WrappingLM()

    with dspy.context(experimental=True):
        with pytest.raises(dspy.LMProviderError, match="wrapped"):
            lm("hello")
    assert lm.normalize_calls == 1


def test_streaming_support_requires_stream_method_override():
    class NonStreamingLM(dspy.BaseLM):
        def forward(self, request):
            return dspy.LMResponse.from_text("ok", model=request.model)

    lm = NonStreamingLM("test/non-streaming", cache=False)

    assert lm.supports_streaming is False
    with dspy.context(experimental=True):
        with pytest.raises(NotImplementedError, match="does not support streaming"):
            list(lm.stream("hello"))


def test_lm_capabilities_and_legacy_boolean_properties_agree(monkeypatch):
    class FakeLiteLLM:
        def supports_function_calling(self, model):
            return True

        def supports_reasoning(self, model):
            return True

        def supports_response_schema(self, model, custom_llm_provider):
            return True

        def get_supported_openai_params(self, model, custom_llm_provider):
            return ["response_format"]

    monkeypatch.setattr("dspy.clients.lm._get_litellm", lambda: FakeLiteLLM())

    lm = dspy.LM("openai/gpt-4o-mini", cache=False)

    assert lm.capabilities.function_calling is True
    assert lm.supports_function_calling is True
    assert lm.capabilities.reasoning is True
    assert lm.supports_reasoning is True
    assert lm.capabilities.response_schema is True
    assert lm.supports_response_schema is True
