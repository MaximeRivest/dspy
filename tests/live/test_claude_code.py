"""Live matrix: Claude Code OAuth — subscription billing, no API key.

Credential: the local Claude Code OAuth credential. Like the codex
provider, billing rides the subscription, so `cost is None` is correct.
"""

import pytest

from tests.live import battery
from tests.live.battery import ProviderSpec, claude_code_gate

pytestmark = pytest.mark.llm_call

SPEC = ProviderSpec(
    name="claude-code",
    model="claude-code:claude-haiku-4-5",
    gate=claude_code_gate,
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
