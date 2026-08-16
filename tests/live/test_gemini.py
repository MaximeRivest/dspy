"""Live matrix: Google Gemini, API key from `GEMINI_API_KEY` or `GOOGLE_API_KEY`."""

import pytest

from tests.live import battery
from tests.live.battery import ProviderSpec, env_gate

pytestmark = pytest.mark.llm_call

SPEC = ProviderSpec(
    name="gemini",
    model="gemini:gemini-flash-lite-latest",
    gate=env_gate("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    priced=True,
    tools=True,
    reasoning=True,
    images=True,
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
