"""Live matrix: OpenAI Codex — the tutorial's provider.

Credential: the Codex CLI OAuth file (`~/.codex/auth.json`), not an API
key. Billing rides the ChatGPT subscription, so `cost is None` is the
honest answer and the battery asserts exactly that.
"""

import pytest

from tests.live import battery
from tests.live.battery import ProviderSpec, codex_gate

pytestmark = pytest.mark.llm_call

SPEC = ProviderSpec(
    name="openai-codex",
    model="openai-codex:gpt-5.6-luna",
    gate=codex_gate,
    priced=False,
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
