"""Live matrix: Ollama — a local server, no credential at all.

Skips when no server answers at `OLLAMA_HOST` (default
`http://localhost:11434`). Local inference has no per-token price, so
`cost is None` is correct.
"""

import pytest

from tests.live import battery
from tests.live.battery import ProviderSpec, ollama_gate

pytestmark = pytest.mark.llm_call

SPEC = ProviderSpec(
    name="ollama",
    model="ollama:llama3.2",
    gate=ollama_gate,
    priced=False,
    tools=True,
    reasoning=False,
    images=False,
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
