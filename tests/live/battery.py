"""The shared live-provider battery: one contract, many providers.

Each provider file in this folder declares a `ProviderSpec` — which model
to run, how its credential arrives, and which capabilities it has — then
exposes one flat test per battery check. The checks here are the tutorial
surface (`explore_lm.md`) plus the transport corners a tutorial never
shows: real tool-call emission, reasoning levels, image input, and the
usage/cost ledger.

Every check runs against the real provider. Nothing here is mocked.
A missing credential skips; an unsupported capability skips with the
reason in the spec.
"""

from __future__ import annotations

import json
import os
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest
from lm15 import (
    Config,
    FunctionTool,
    Message,
    Reasoning,
    Request,
    Response,
    StreamEndEvent,
    StreamStartEvent,
    TextDelta,
    TextPart,
)

import dspy

# A 64x64 solid red PNG, generated offline. Self-contained: the image
# check never depends on an external URL staying up.
RED_SQUARE_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9f"
    "eoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjx"
    "BQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElF"
    "TkSuQmCC"
)


@dataclass
class ProviderSpec:
    """One provider's entry in the live matrix.

    Attributes:
        name: Short provider label, used in skip messages.
        model: Full model string with provider prefix. Overridable via
            the `DSPY_LIVE_<NAME>_MODEL` environment variable.
        gate: Returns a skip reason when the credential (or local
            server) is absent, else None.
        priced: The catalog carries per-token prices, so `cost` must be
            a positive float. False means `cost is None` is the honest
            answer (subscription or local billing).
        tools: The provider emits native tool calls.
        reasoning: The provider accepts reasoning-effort levels.
        reasoning_tokens_visible: Usage reports `reasoning_tokens > 0`
            when effort is high.
        images: The provider accepts image input.
        reasoning_model: Model for the reasoning check; defaults to `model`.
        image_model: Model for the image check; defaults to `model`.
        lm_kwargs: Extra constructor kwargs for every LM in this spec.
    """

    name: str
    model: str
    gate: Callable[[], str | None]
    priced: bool
    tools: bool = True
    reasoning: bool = False
    reasoning_tokens_visible: bool = True
    images: bool = False
    reasoning_model: str | None = None
    image_model: str | None = None
    lm_kwargs: dict = field(default_factory=dict)

    def resolved_model(self) -> str:
        env = f"DSPY_LIVE_{self.name.upper().replace('-', '_')}_MODEL"
        return os.getenv(env, self.model)


# --- credential gates -------------------------------------------------


def env_gate(*keys: str) -> Callable[[], str | None]:
    """Skip unless at least one of `keys` is set in the environment."""

    def check() -> str | None:
        if any(os.getenv(key) for key in keys):
            return None
        return f"missing credential: set one of {', '.join(keys)}"

    return check


def codex_gate() -> str | None:
    """Skip unless the Codex CLI OAuth credential loads."""
    try:
        from lm15.auth import load_codex_cli_credential

        load_codex_cli_credential()
        return None
    except Exception as e:
        return f"no Codex CLI credential: {e}"


def claude_code_gate() -> str | None:
    """Skip unless the Claude Code OAuth credential loads."""
    try:
        from lm15.auth import load_claude_code_credential

        load_claude_code_credential()
        return None
    except Exception as e:
        return f"no Claude Code credential: {e}"


def ollama_gate() -> str | None:
    """Skip unless a local Ollama server answers."""
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=2)
        return None
    except Exception as e:
        return f"no Ollama server at {host}: {e}"


# --- helpers ----------------------------------------------------------


def make_lm(spec: ProviderSpec, model: str | None = None, **overrides) -> dspy.LM:
    reason = spec.gate()
    if reason:
        pytest.skip(f"[{spec.name}] {reason}")
    kwargs = {"max_tokens": 256, **spec.lm_kwargs, **overrides}
    return dspy.LM(model or spec.resolved_model(), **kwargs)


def _text(response: Response) -> str:
    assert isinstance(response, Response)
    assert response.text is not None, f"no text (finish_reason={response.finish_reason!r})"
    return response.text.strip()


def _skip_unless(flag: bool, spec: ProviderSpec, capability: str) -> None:
    if not flag:
        pytest.skip(f"[{spec.name}] no {capability} capability declared")


# --- the battery ------------------------------------------------------


def hello(spec: ProviderSpec) -> None:
    """Tutorial section 1: string in, rich Response out, usage on."""
    lm = make_lm(spec)
    response = lm("Reply with exactly one word: hello")
    assert "hello" in _text(response).lower()
    assert response.finish_reason is not None
    assert (response.usage.total_tokens or 0) > 0


def multiturn(spec: ProviderSpec) -> None:
    """Tutorial section 2: typed turns, and a Response reused as a turn."""
    lm = make_lm(spec)
    first = lm(dspy.User("Reply with exactly: alpha"))
    follow = lm(
        dspy.System("Follow the user's requested exact final token. No punctuation."),
        dspy.User("Reply with exactly: alpha"),
        first,
        dspy.User("Now reply with exactly: beta"),
    )
    assert "beta" in _text(follow).lower()


def tool_transcript(spec: ProviderSpec) -> None:
    """Tutorial section 3: a replayed tool exchange reads back correctly."""
    _skip_unless(spec.tools, spec, "tool")
    lm = make_lm(spec)
    answer = lm(
        dspy.User("What is the weather in Paris?"),
        dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
        dspy.ToolResult('{"temperature": "22 C", "sky": "sunny"}', call_id="call_1", name="get_weather"),
        dspy.User("Answer with the temperature string from the tool result."),
    )
    assert "22" in _text(answer)


def tool_emission(spec: ProviderSpec) -> None:
    """Past the tutorial: the model itself emits a native tool call."""
    _skip_unless(spec.tools, spec, "tool")
    lm = make_lm(spec)
    tool = FunctionTool(
        name="get_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    response = lm.complete(
        Request(
            model=spec.resolved_model(),
            messages=(
                Message(
                    role="user",
                    parts=(TextPart(text="What is the weather in Paris? You must call the get_weather tool."),),
                ),
            ),
            tools=(tool,),
            config=Config(max_tokens=256),
        )
    )
    calls = response.tool_calls
    assert calls, f"expected a tool call, got text: {response.text!r}"
    assert calls[0].name == "get_weather"
    args = calls[0].input
    if isinstance(args, str):
        args = json.loads(args)
    assert "paris" in str(args.get("city", "")).lower()


def streaming(spec: ProviderSpec) -> None:
    """Tutorial section 4: text chunks, then the finished Response and events."""
    lm = make_lm(spec)
    stream = lm.stream("Count from 1 to 3, digits only, one number per line.")
    chunks = list(stream)
    joined = "".join(chunks)
    assert "3" in joined
    assert stream.response.text is not None
    assert (stream.response.usage.total_tokens or 0) > 0
    assert len(lm.history) == 1

    events = list(lm.stream("Reply with exactly: ok").events())
    assert isinstance(events[0], StreamStartEvent)
    assert isinstance(events[-1], StreamEndEvent)
    assert any(isinstance(getattr(e, "delta", None), TextDelta) for e in events)
    assert len(lm.history) == 2


def reasoning_levels(spec: ProviderSpec) -> None:
    """Past the tutorial: effort levels flow through, and the meter shows work."""
    _skip_unless(spec.reasoning, spec, "reasoning")
    lm = make_lm(spec, model=spec.reasoning_model or spec.resolved_model(), max_tokens=4000)
    # Hard enough that high effort reliably spends thinking tokens;
    # trivial arithmetic sometimes gets zero even at high effort.
    question = "What is the sum of the first 40 odd numbers? Answer with the number only."
    low = lm(question, reasoning=Reasoning(effort="low"))
    high = lm(question, reasoning=Reasoning(effort="high"))
    assert "1600" in _text(low)
    assert "1600" in _text(high)
    if spec.reasoning_tokens_visible:
        assert (high.usage.reasoning_tokens or 0) > 0


def image_understanding(spec: ProviderSpec) -> None:
    """Past the tutorial: an image part rides the same typed face."""
    _skip_unless(spec.images, spec, "image")
    lm = make_lm(spec, model=spec.image_model or spec.resolved_model())
    response = lm(
        dspy.User(
            "What color is this square? Answer with one lowercase word.",
            dspy.Image(RED_SQUARE_DATA_URI),
        )
    )
    assert "red" in _text(response).lower()


def usage_and_cost(spec: ProviderSpec) -> None:
    """Tutorial section 8: the ledger is always on, and cost is honest."""
    lm = make_lm(spec)
    lm("Reply with exactly: ok")
    entry = lm.history[-1]
    assert (entry["usage"].total_tokens or 0) > 0
    if spec.priced:
        assert isinstance(entry["cost"], float) and entry["cost"] > 0
    else:
        assert entry["cost"] is None
