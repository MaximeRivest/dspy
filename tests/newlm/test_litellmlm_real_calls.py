import os
from pathlib import Path

import pytest

import dspy

pytestmark = pytest.mark.llm_call


def _load_dotenv() -> None:
    """Load simple KEY=VALUE pairs from the repository `.env` file."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _real_model() -> str:
    _load_dotenv()
    model = os.environ.get("LITELLM_REAL_MODEL") or os.environ.get("LM_FOR_TEST") or "openai/gpt-4o-mini"
    _skip_if_missing_common_provider_key(model)
    return model


def _skip_if_missing_common_provider_key(model: str) -> None:
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    required_env_by_provider = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
    }
    required_env = required_env_by_provider.get(provider)
    if required_env and not os.environ.get(required_env):
        pytest.skip(f"{required_env} is not set; add it to .env or the environment")


def test_litellmlm_real_text_call_returns_normalized_response():
    lm = dspy.LiteLLMChatLM(_real_model(), cache=False, temperature=0.0, max_tokens=20)

    response = lm("Output the exact lowercase string pong and nothing else.")

    assert isinstance(response, dspy.LMResponse)
    assert response.outputs
    assert response.text is not None
    assert "pong" in response.text.lower()
    assert response.cache_hit is False


def test_litellmlm_real_multiturn_call_accepts_message_constructors():
    lm = dspy.LiteLLMChatLM(_real_model(), cache=False, temperature=0.0, max_tokens=30)

    response = lm(
        dspy.System("Answer with exactly one word."),
        dspy.User("What word comes after ping in the phrase ping pong?"),
    )

    assert isinstance(response, dspy.LMResponse)
    assert response.text is not None
    assert "pong" in response.text.lower()


def test_litellmlm_real_chat_stream_returns_normalized_response():
    lm = dspy.LiteLLMChatLM(_real_model(), cache=False, temperature=0.0, max_tokens=10)

    stream = lm.stream("Output the exact lowercase string pong and nothing else.")
    events = list(stream)
    response = stream.result()

    assert events[0].type == "start"
    assert events[-1].type == "end"
    assert response.text is not None
    assert "pong" in response.text.lower()
    assert response.usage is not None


def test_litellmlm_real_responses_stream_preserves_usage():
    _load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set; add it to .env or the environment")
    model = os.environ.get("LITELLM_REAL_RESPONSES_MODEL") or "openai/gpt-4o-mini"
    lm = dspy.LiteLLMResponsesLM(model, cache=False, temperature=0.0, max_tokens=10)

    stream = lm.stream("Output the exact lowercase string pong and nothing else.")
    events = list(stream)
    response = stream.result()

    assert events[0].type == "start"
    assert events[-1].type == "end"
    assert response.text is not None
    assert "pong" in response.text.lower()
    assert response.usage is not None
