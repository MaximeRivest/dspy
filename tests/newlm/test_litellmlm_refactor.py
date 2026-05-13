import pytest
from litellm.utils import Choices, Message, ModelResponse

import dspy
from dspy.clients.language_models.litellm import (
    _CompletionStreamState,
    _prepare_litellm_text_request,
    _ResponsesStreamState,
)


def test_lm_routes_to_protocol_specific_litellm_factories():
    with dspy.context(experimental_lm=True):
        assert dspy.LM("test/model", cache=False).backend.metadata["protocol"] == "openai_chat"
        assert dspy.LM("test/model", model_type="text", cache=False).backend.metadata["protocol"] == "openai_text"
        assert dspy.LM("test/model", model_type="responses", cache=False).backend.metadata["protocol"] == "openai_responses"


def test_litellm_factories_expose_provider_request_mapping():
    request = dspy.litellm_chat_lm("test/model", cache=False).explain_provider_request("hello")

    assert isinstance(request, dspy.ProviderRequest)
    assert request.kwargs["messages"] == [{"role": "user", "content": "hello"}]


def test_litellm_support_is_explicit_protocol_metadata():
    lm = dspy.litellm_chat_lm("test/model", cache=False)

    assert lm.support.tools.schemas is True
    assert lm.support.response_schema is True
    assert lm.support.images is not None
    assert lm.metadata == {"provider": "litellm", "protocol": "openai_chat", "num_retries": 3}


def test_litellm_text_lm_reports_text_only_request_support():
    lm = dspy.litellm_text_lm("test/model", cache=False)

    assert lm.features.request.text
    assert lm.features.request.input_image.status == "unsupported"
    assert lm.features.request.tools.status == "unsupported"
    assert lm.features.request.response_schema.status == "unsupported"
    assert lm.features.response.text
    assert lm.features.response.tool_calls.status == "unsupported"

    with pytest.raises(dspy.LMUnsupportedFeatureError) as exc_info:
        lm("describe", dspy.Image("data:image/png;base64,abc"))
    assert exc_info.value.features == ["request.input_image"]


def test_litellm_chat_maps_file_url_without_empty_file_block():
    lm = dspy.litellm_chat_lm("test/model", cache=False)
    request = lm.normalize_request(dspy.User("read", dspy.LMFilePart(url="https://example.com/a.pdf", filename="a.pdf")))

    provider_request = lm.explain_provider_request(request=request).kwargs

    file_block = provider_request["messages"][0]["content"][1]
    assert file_block == {
        "type": "file",
        "file": {"file_data": "https://example.com/a.pdf", "filename": "a.pdf"},
    }


def test_litellm_chat_maps_single_allowed_tool_choice():
    lm = dspy.litellm_chat_lm("test/model", cache=False)
    request = lm.normalize_request("use the tool", tool_choice={"mode": "required", "allowed": ["search"]})

    provider_request = lm.explain_provider_request(request=request).kwargs

    assert provider_request["tool_choice"] == {"type": "function", "function": {"name": "search"}}


def test_litellm_chat_rejects_ambiguous_allowed_tool_choice():
    lm = dspy.litellm_chat_lm("test/model", cache=False)
    request = lm.normalize_request("use tools", tool_choice={"mode": "required", "allowed": ["a", "b"]})

    with pytest.raises(ValueError, match="single allowed tool"):
        lm.explain_provider_request(request=request)


def test_litellm_chat_reports_streaming_and_output_artifact_hooks():
    lm = dspy.litellm_chat_lm("test/model", cache=False)

    assert lm.features.streaming.status == "inferred"
    assert lm.features.async_streaming.status == "inferred"
    assert lm.features.request.prompt_cache.status == "inferred"
    assert lm.features.response.output_image.status == "inferred"
    assert lm.features.response.output_audio.status == "inferred"
    assert lm.features.response.output_file.status == "inferred"
    assert lm.features.response.refusal.status == "inferred"


def test_litellm_maps_prompt_cache_to_provider_kwargs():
    lm = dspy.litellm_chat_lm("test/model", cache=False)
    request = lm.normalize_request("hello", prompt_cache=True, prompt_cache_key="prefix-1")

    provider_request = lm.explain_provider_request(request=request).kwargs

    assert provider_request["prompt_cache_key"] == "prefix-1"


def test_litellm_responses_preserves_provider_tool_call_metadata():
    from dspy.clients.language_models.openai_format import _responses_function_call_to_part

    part = _responses_function_call_to_part(
        {
            "type": "function_call",
            "id": "item_1",
            "call_id": "call_1",
            "status": "completed",
            "name": "search",
            "arguments": '{"query": "DSPy"}',
        }
    )

    assert part.id == "call_1"
    assert part.name == "search"
    assert part.args == {"query": "DSPy"}
    assert part.provider_data["status"] == "completed"
    assert part.provider_data["call_id"] == "call_1"


def test_completion_stream_uses_stable_part_indices_for_reasoning_then_text():
    state = _CompletionStreamState()
    builder = dspy.LMOutputBuilder()
    builder.apply(dspy.LMStreamStartEvent(model="test/model"))

    for event in state.chunk_to_events({"choices": [{"index": 0, "delta": {"reasoning_content": "think "}}]}):
        builder.apply(event)
    for event in state.chunk_to_events({"choices": [{"index": 0, "delta": {"content": "answer"}}]}):
        builder.apply(event)
    for event in state.missing_output_end_events():
        builder.apply(event)
    response = builder.apply(dspy.LMStreamEndEvent())

    assert response.reasoning_content == "think "
    assert response.text == "answer"


def test_completion_stream_uses_provider_tool_indices_for_interleaved_tool_calls():
    state = _CompletionStreamState()
    builder = dspy.LMOutputBuilder()
    builder.apply(dspy.LMStreamStartEvent(model="test/model"))

    chunks = [
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "id": "call_0", "function": {"name": "first", "arguments": '{"a"'}},
                            {"index": 1, "id": "call_1", "function": {"name": "second", "arguments": '{"b"'}},
                        ]
                    },
                }
            ]
        },
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": ": 2}"}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}}]},
    ]
    for chunk in chunks:
        for event in state.chunk_to_events(chunk):
            builder.apply(event)
    for event in state.missing_output_end_events():
        builder.apply(event)
    response = builder.apply(dspy.LMStreamEndEvent())

    assert [(call.id, call.name, call.args) for call in response.tool_calls] == [
        ("call_0", "first", {"a": 1}),
        ("call_1", "second", {"b": 2}),
    ]


def test_responses_stream_keeps_reasoning_text_and_tool_call_parts_distinct():
    state = _ResponsesStreamState()
    builder = dspy.LMOutputBuilder()
    builder.apply(dspy.LMStreamStartEvent(model="test/model"))

    events = [
        {"type": "response.reasoning_text.delta", "delta": "think "},
        {"type": "response.output_text.delta", "delta": "answer"},
        {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "item_1", "call_id": "call_1", "name": "search"},
        },
        {"type": "response.function_call_arguments.delta", "item_id": "item_1", "delta": '{"query": "DSPy"}'},
    ]
    for raw_event in events:
        for event in state.event_to_events(raw_event):
            builder.apply(event)
    for event in state.finish_events():
        response = builder.apply(event) or locals().get("response")

    assert response.reasoning_content == "think "
    assert response.text == "answer"
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "search"
    assert response.tool_calls[0].args == {"query": "DSPy"}


def test_responses_stream_preserves_completed_response_usage_and_cost():
    state = _ResponsesStreamState()
    events = state.event_to_events(
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "model": "test/model",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}],
                "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
            },
        }
    )
    events.extend(state.finish_events())

    end = events[-1]
    assert isinstance(end, dspy.LMStreamEndEvent)
    assert end.response.text == "done"
    assert end.response.usage.total_tokens == 3


def test_litellm_text_request_uses_legacy_text_completion_model_prefix(monkeypatch):
    monkeypatch.setenv("openai_API_KEY", "env-key")
    monkeypatch.setenv("openai_API_BASE", "https://example.test")

    request, headers = _prepare_litellm_text_request(
        {"model": "openai/davinci", "prompt": "hello", "headers": {"X-Test": "1"}, "rollout_id": 7}
    )

    assert request["model"] == "text-completion-openai/davinci"
    assert request["api_key"] == "env-key"
    assert request["api_base"] == "https://example.test"
    assert "rollout_id" not in request
    assert headers["User-Agent"].startswith("DSPy/")
    assert headers["X-Test"] == "1"


def test_normalized_litellm_cache_ignores_credentials_in_request_cache_key(tmp_path, monkeypatch):
    original_cache = dspy.cache
    dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True, disk_cache_dir=tmp_path / ".dspy_cache")

    try:
        calls = 0

        def fake_completion(**kwargs):
            nonlocal calls
            calls += 1
            return ModelResponse(
                choices=[Choices(message=Message(role="assistant", content="cached"))],
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                model="test/model",
            )

        monkeypatch.setattr("litellm.completion", fake_completion)

        dspy.litellm_chat_lm("test/model", api_key="first")("hello")
        second = dspy.litellm_chat_lm("test/model", api_key="second")("hello")

        assert calls == 1
        assert second.cache_hit is True
    finally:
        dspy.cache = original_cache
