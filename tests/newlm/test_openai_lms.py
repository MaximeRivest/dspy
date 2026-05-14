import json
from types import SimpleNamespace

import pytest

import dspy
from dspy.clients.language_models.openai import completion_stream_to_events, responses_stream_to_events


class FakeHTTPResponse:
    def __init__(self, payload):
        self.payload = payload

    def read(self):
        return json.dumps(self.payload).encode("utf-8")

    def __iter__(self):
        for item in self.payload:
            yield f"data: {json.dumps(item)}\n".encode()
            yield b"\n"


def test_openai_completions_class_calls_chat_completions_and_maps_response():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {
            "id": "cmpl_1",
            "model": "gpt-4o-mini",
            "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    lm = dspy.OpenAICompletionsLM("openai/gpt-4o-mini", completions=completions, cache=False)

    response = lm("say hello", temperature=0.1)

    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["messages"] == [{"role": "user", "content": "say hello"}]
    assert calls[0]["temperature"] == 0.1
    assert response.text == "hello"
    assert response.usage.total_tokens == 2


def test_openai_completions_class_can_use_text_protocol():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {
            "model": "gpt-3.5-turbo-instruct",
            "choices": [{"text": "hello", "finish_reason": "stop"}],
        }

    lm = dspy.OpenAICompletionsLM(
        "openai/gpt-3.5-turbo-instruct",
        completions=completions,
        protocol="text",
        cache=False,
    )

    response = lm("say hello")

    assert calls[0]["model"] == "gpt-3.5-turbo-instruct"
    assert calls[0]["prompt"] == "say hello\n\nBEGIN RESPONSE:"
    assert response.text == "hello"


def test_openai_responses_class_calls_responses_and_maps_response():
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return {
            "id": "resp_1",
            "model": "gpt-4o-mini",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hello"}],
                }
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        }

    lm = dspy.OpenAIResponsesLM("openai/gpt-4o-mini", responses=responses, cache=False)

    response = lm("say hello")

    assert calls[0]["model"] == "gpt-4o-mini"
    assert calls[0]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "say hello"}]}]
    assert response.text == "hello"
    assert response.usage.total_tokens == 2


def test_openai_completions_class_calls_openai_compatible_endpoint_directly(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeHTTPResponse(
            {
                "id": "cmpl_1",
                "model": "local-model",
                "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lm = dspy.OpenAICompletionsLM("local-model", api_key="local", base_url="http://localhost:8000/v1", cache=False)

    response = lm("say hello")

    request, timeout = calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:8000/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer local"
    assert timeout == 60
    assert payload["model"] == "local-model"
    assert response.text == "hello"


def test_openai_responses_class_calls_openai_compatible_endpoint_directly(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return FakeHTTPResponse(
            {
                "id": "resp_1",
                "model": "local-model",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "hello"}]}],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    lm = dspy.OpenAIResponsesLM("local-model", api_key="local", base_url="http://localhost:8000/v1", cache=False)

    response = lm("say hello")

    request, timeout = calls[0]
    payload = json.loads(request.data.decode("utf-8"))
    assert request.full_url == "http://localhost:8000/v1/responses"
    assert timeout == 60
    assert payload["model"] == "local-model"
    assert response.text == "hello"


@pytest.mark.asyncio
async def test_openai_async_call_and_stream_use_anyio_thread_bridge():
    def completions(**kwargs):
        if kwargs.get("stream"):
            return [
                {"choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            ]
        return {"model": "gpt-4o-mini", "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}

    lm = dspy.OpenAICompletionsLM("openai/gpt-4o-mini", completions=completions, cache=False)

    response = await lm.acall("say hello")
    stream = lm.astream("say hello")
    events = [event async for event in stream]

    assert response.text == "hello"
    assert events[-1].type == "end"
    assert stream.result().text == "hello"


def test_completion_stream_to_events_builds_response():
    stream = iter(
        [
            SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="hel"), finish_reason=None)]
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(index=0, delta=SimpleNamespace(content="lo"), finish_reason="stop")]
            ),
        ]
    )

    builder = dspy.LMOutputBuilder()
    for event in completion_stream_to_events(stream, model="gpt-4o-mini"):
        response = builder.apply(event)

    assert response.text == "hello"
    assert response.output.finish_reason == "stop"


def test_responses_stream_to_events_builds_response_from_completed_event():
    completed_response = {
        "model": "gpt-4o-mini",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "hello"}],
            }
        ],
    }
    stream = iter([{"type": "response.completed", "response": completed_response}])

    builder = dspy.LMOutputBuilder()
    for event in responses_stream_to_events(stream, model="gpt-4o-mini"):
        response = builder.apply(event)

    assert response.text == "hello"
