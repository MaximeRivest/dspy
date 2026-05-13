import pytest

import dspy
from dspy.clients.language_models.openai_format import provider_tool_call_to_part


class TextOnlyLM(dspy.LanguageModel):
    def __init__(self):
        super().__init__(model="test/text-only", cache=False)
        self.forward_called = False

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        self.forward_called = True
        return dspy.LMResponse.from_text("ok", model=request.model)


class StreamingLM(TextOnlyLM):
    def forward_stream(self, request):
        yield dspy.LMStreamEndEvent(response=dspy.LMResponse.from_text("ok", model=request.model))


class NonDeepcopyableClient:
    def __deepcopy__(self, memo):
        raise RuntimeError("SDK clients should not be deep-copied")


class ClientBackedLM(dspy.LanguageModel):
    def __init__(self, client):
        super().__init__(model="test/client-backed", cache=False, temperature=0.1)
        self.client = client

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("ok", model=request.model)


def test_language_model_copy_is_shallow_for_provider_resources_and_isolates_dspy_state():
    callback = object()
    client = NonDeepcopyableClient()
    lm = ClientBackedLM(client)
    lm.callbacks.append(callback)
    lm.history.append({"old": "entry"})

    copied = lm.copy(temperature=0.9, rollout_id=7)

    assert copied is not lm
    assert copied.client is client
    assert copied.history == []
    assert copied.callbacks == [callback]
    assert copied.callbacks is not lm.callbacks
    assert copied.kwargs == {"temperature": 0.9, "rollout_id": 7}
    assert lm.kwargs == {"temperature": 0.1}
    assert copied.features is not lm.features
    assert copied.features._lm is copied


def test_call_enforces_implementation_request_support_before_forward():
    lm = TextOnlyLM()

    with pytest.raises(dspy.LMUnsupportedFeatureError) as exc_info:
        lm("describe", dspy.Image("data:image/png;base64,abc"))

    assert not lm.forward_called
    assert exc_info.value.features == ["request.input_image"]


def test_stream_requires_streaming_implementation_support():
    lm = TextOnlyLM()

    with pytest.raises(dspy.LMUnsupportedFeatureError) as exc_info:
        lm.stream("hello")

    assert exc_info.value.features == ["streaming"]
    assert "forward_stream" in exc_info.value.issues[0]


def test_stream_allowed_when_forward_stream_is_overridden():
    stream = StreamingLM().stream("hello")

    assert list(stream)[-1].type == "end"
    assert stream.result().text == "ok"


def test_litellm_chat_lm_advertises_normalized_streaming():
    pytest.importorskip("litellm")
    lm = dspy.litellm_chat_lm("openai/gpt-4o-mini", cache=False)

    assert lm.support.streaming is True
    assert lm.features.streaming.status == "inferred"


def test_responses_request_preserves_tool_continuation_items():
    lm = dspy.openai_responses_lm("openai/gpt-4o-mini", responses=lambda **kwargs: kwargs, cache=False)
    request = dspy.LMRequest.from_call(
        model="openai/gpt-4o-mini",
        items=(
            dspy.User("What is the weather?"),
            dspy.Assistant(dspy.ToolCall(id="call_1", name="weather", args={"city": "Paris"})),
            dspy.ToolResult('{"temp": 22}', call_id="call_1", name="weather"),
            dspy.User("Summarize it."),
        ),
    )

    provider_request = lm.explain_provider_request(request=request).kwargs

    assert provider_request["input"][0] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "What is the weather?"}],
    }
    assert provider_request["input"][1] == {
        "type": "function_call",
        "name": "weather",
        "arguments": '{"city": "Paris"}',
        "call_id": "call_1",
    }
    assert provider_request["input"][2] == {
        "type": "function_call_output",
        "output": '{"temp": 22}',
        "call_id": "call_1",
    }
    assert provider_request["input"][3] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Summarize it."}],
    }


def test_prompt_cache_is_separate_from_dspy_cache_config():
    lm = TextOnlyLM()
    request = lm.normalize_request("hello", cache=False, prompt_cache=True, prompt_cache_key="prefix-1")

    assert request.config.cache.enabled is False
    assert request.config.prompt_cache.enabled is True
    assert request.config.prompt_cache.key == "prefix-1"


def test_prompt_cache_requires_explicit_implementation_support():
    lm = TextOnlyLM()

    with pytest.raises(dspy.LMUnsupportedFeatureError) as exc_info:
        lm("hello", prompt_cache=True)

    assert exc_info.value.features == ["request.prompt_cache"]


def test_lm15_rejects_logprobs_and_multiple_outputs_before_forward():
    pytest.importorskip("lm15")
    lm = dspy.lm15_lm("openai/gpt-4o-mini", cache=False)

    with pytest.raises(dspy.LMUnsupportedFeatureError) as logprobs_error:
        lm("hello", logprobs=True)
    assert logprobs_error.value.features == ["request.logprobs"]

    with pytest.raises(dspy.LMUnsupportedFeatureError) as multiple_outputs_error:
        lm("hello", n=2)
    assert multiple_outputs_error.value.features == ["request.multiple_outputs"]


def test_lm15_maps_prompt_cache_to_provider_cache_not_dspy_cache():
    pytest.importorskip("lm15")
    from dspy.clients.language_models.lm15 import to_lm15_config

    lm = dspy.lm15_lm("openai/gpt-4o-mini", cache=False)
    request = lm.normalize_request("hello", cache=False, prompt_cache=True, prompt_cache_key="prefix-1")

    config = to_lm15_config(request.config)

    assert config.cache is not None
    assert config.cache.mode == "auto"
    assert config.cache.key == "prefix-1"


def test_malformed_tool_call_arguments_preserve_raw_provider_data():
    part = provider_tool_call_to_part(
        {
            "id": "call_1",
            "function": {
                "name": "search",
                "arguments": "{not json",
            },
        }
    )

    assert part.args == {}
    assert part.provider_data["raw_arguments"] == "{not json"
    assert "arguments_parse_error" in part.provider_data
