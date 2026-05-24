import dspy
from dspy.clients.openai_compatible import OpenAIChatLM, OpenAIResponsesLM, OpenAITextLM


def test_openai_compatible_chat_lm_uses_normalized_baselm_contract():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {
            "id": "chatcmpl_1",
            "model": kwargs["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hi!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        }

    lm = OpenAIChatLM("groq/llama-3.1-8b-instant", completions=completions, temperature=0.2)
    response = lm("hello")

    assert isinstance(response, dspy.LMResponse)
    assert response.text == "Hi!"
    assert calls[0]["model"] == "llama-3.1-8b-instant"
    assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert calls[0]["temperature"] == 0.2
    assert lm.api_base == "https://api.groq.com/openai/v1"


def test_openai_compatible_text_lm_uses_text_wire_format():
    calls = []

    def completions(**kwargs):
        calls.append(kwargs)
        return {
            "id": "cmpl_1",
            "model": kwargs["model"],
            "choices": [{"index": 0, "text": "Text answer", "finish_reason": "stop"}],
        }

    lm = OpenAITextLM("openai/gpt-3.5-turbo-instruct", completions=completions)
    response = lm("hello")

    assert response.text == "Text answer"
    assert calls[0]["model"] == "gpt-3.5-turbo-instruct"
    assert calls[0]["prompt"] == "hello\n\nBEGIN RESPONSE:"


def test_openai_compatible_responses_lm_uses_responses_wire_format():
    calls = []

    def responses(**kwargs):
        calls.append(kwargs)
        return {
            "id": "resp_1",
            "model": kwargs["model"],
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Response answer"}],
                }
            ],
        }

    lm = OpenAIResponsesLM("openai/gpt-4.1-mini", responses=responses)
    response = lm(dspy.User("hello"))

    assert response.text == "Response answer"
    assert calls[0]["model"] == "gpt-4.1-mini"
    assert calls[0]["input"] == [{"role": "user", "content": [{"type": "input_text", "text": "hello"}]}]


def test_openai_compatible_chat_stream_events():
    def completions(**kwargs):
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}
        return iter(
            [
                {"choices": [{"index": 0, "delta": {"content": "Hel"}, "finish_reason": None}]},
                {"choices": [{"index": 0, "delta": {"content": "lo"}, "finish_reason": "stop"}]},
            ]
        )

    lm = OpenAIChatLM("openai/gpt-4o-mini", completions=completions)
    stream = lm.stream("hello")
    events = list(stream)

    assert [event.type for event in events] == ["start", "delta", "delta", "output_end", "end"]
    assert stream.result().text == "Hello"


def test_ollama_defaults_to_local_openai_compatible_base_and_dummy_key():
    lm = OpenAIChatLM("ollama/llama3.2")

    assert lm.api_base == "http://localhost:11434/v1"
    assert lm.provider == "ollama"
