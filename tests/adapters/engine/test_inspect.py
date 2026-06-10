"""The internal plan pretty-printer (developer tooling; future explain() seed)."""

from golden.harness import Recorder, StubLM

import dspy
from dspy.adapters._engine.builder import build_plan
from dspy.adapters._engine.inspect import describe_plan
from dspy.adapters._engine.ir import AdapterPlan
from dspy.adapters.chat_adapter import ChatAdapter


def test_empty_plan():
    assert describe_plan(AdapterPlan()) == "AdapterPlan\n  (empty)"


def test_describes_fields_transforms_and_tools():
    class AgentStep(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        next_thought: str = dspy.OutputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    def search(query: str) -> str:
        """Search."""
        return query

    lm = StubLM(Recorder(), supports_function_calling=True)
    built = build_plan(
        ChatAdapter(use_native_function_calling=True),
        lm,
        {},
        AgentStep,
        {"question": "Q?", "tools": [dspy.Tool(search)]},
    )
    text = describe_plan(built.plan)
    assert "tools [hidden]" in text
    assert "tool_calls [hidden]" in text
    assert "HideOutputField" in text
    assert "search" in text
