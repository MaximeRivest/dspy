"""Live matrix: vLLM on the lab server — an OpenAI-compat server with an api_base.

This is the self-hosted credential style: `api_base` pins the chat-compat
dialect at the server's URL, and the server's own `api_key` rides along.
The lab server (192.168.2.24) serves gemma-4-26b-a4b with tool calling
and one-image vision; `max-model-len` is 4096, so keep prompts short.
Local serving has no per-token price, so `cost is None` is correct.
"""

import os
import urllib.request

import pytest

from tests.live import battery
from tests.live.battery import ProviderSpec

pytestmark = pytest.mark.llm_call

BASE_URL = os.getenv("DSPY_LIVE_VLLM_BASE_URL", "http://192.168.2.24:8000/v1")
API_KEY = os.getenv("DSPY_LIVE_VLLM_API_KEY", "inktype-local")


def vllm_gate() -> str | None:
    """Skip unless the vLLM server answers /models with our key."""
    try:
        request = urllib.request.Request(
            f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"}
        )
        urllib.request.urlopen(request, timeout=3)
        return None
    except Exception as e:
        return f"no vLLM server at {BASE_URL}: {e}"


SPEC = ProviderSpec(
    name="vllm",
    model="openai-chat:gemma/gemma-4-26b-a4b-it",
    gate=vllm_gate,
    priced=False,
    tools=True,
    reasoning=False,
    images=True,
    lm_kwargs={"api_base": BASE_URL, "api_key": API_KEY},
)


def test_hello_roundtrip():
    battery.hello(SPEC)


def test_multiturn_typed_conversation():
    battery.multiturn(SPEC)


def test_tool_transcript_replay():
    battery.tool_transcript(SPEC)


def test_model_emits_tool_call():
    battery.tool_emission(SPEC)


def test_streaming_text_events_and_history():
    battery.streaming(SPEC)


def test_reasoning_levels():
    battery.reasoning_levels(SPEC)


def test_image_understanding():
    battery.image_understanding(SPEC)


def test_usage_and_cost():
    battery.usage_and_cost(SPEC)
