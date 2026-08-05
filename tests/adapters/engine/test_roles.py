"""Semantic-role derivation and recording (derive-and-record only).

Roles are recorded onto ``RenderField.metadata`` and consulted by nothing:
these tests pin the derivation table, the explicit ``role=`` declaration,
and the no-behavior-change guarantee (the golden corpus is the byte-level
backstop for the latter).
"""

from typing import Optional, Union

import pytest

import dspy
from dspy.adapters._engine.ir import AdapterPlan
from dspy.adapters._engine.roles import resolve_semantic_role, semantic_role_for
from dspy.adapters.types.audio import Audio
from dspy.adapters.types.code import Code
from dspy.adapters.types.document import Document
from dspy.adapters.types.file import File
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.experimental import Citations
from dspy.signatures.field import SEMANTIC_ROLES


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Reasoning, "reasoning"),
        (Tool, "tools"),
        (list[Tool], "tools"),
        (ToolCalls, "tool_calls"),
        (Citations, "citations"),
        (History, "history"),
        (Image, "media"),
        (Audio, "media"),
        (File, "media"),
        (Document, "media"),
        (list[Image], "media"),
        (Code, "code"),
        (Code["python"], "code"),
        (str, "plain"),
        (int, "plain"),
        (list[str], "plain"),
        (dict[str, int], "plain"),
        (Optional[Image], "media"),
        (Image | None, "media"),
        (Union[Image, Audio], "media"),
        (Union[Image, str], "plain"),
    ],
)
def test_derivation_table(annotation, expected):
    assert semantic_role_for(annotation) == expected


def test_every_derived_role_is_in_the_vocabulary():
    for annotation in (Reasoning, Tool, ToolCalls, Citations, History, Image, Code, str):
        assert semantic_role_for(annotation) in SEMANTIC_ROLES


def test_explicit_role_recorded_and_validated():
    field = dspy.OutputField(role="citations")
    assert field.json_schema_extra["semantic_role"] == "citations"

    with pytest.raises(ValueError, match="Unknown semantic role"):
        dspy.OutputField(role="chain_of_thought")


def test_explicit_role_wins_over_plain_derivation():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(role="citations")

    assert resolve_semantic_role(Sig.output_fields["answer"]) == "citations"


def test_explicit_role_conflicting_with_derived_role_raises():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        thinking: Reasoning = dspy.OutputField(role="citations")

    with pytest.raises(ValueError, match="annotation implies"):
        AdapterPlan.from_signature(Sig, {"question": "q"})


def test_explicit_role_agreeing_with_derived_role_is_fine():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        thinking: Reasoning = dspy.OutputField(role="reasoning")

    plan = AdapterPlan.from_signature(Sig, {"question": "q"})
    assert plan.output_fields[0].metadata["semantic_role"] == "reasoning"


def test_plan_records_roles_for_every_field():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[Tool] = dspy.InputField()
        chat: History = dspy.InputField()
        reasoning: Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    plan = AdapterPlan.from_signature(Sig, {"question": "q", "tools": [], "chat": History(messages=[])})
    roles = {f.name: f.metadata["semantic_role"] for f in plan.input_fields + plan.output_fields}
    assert roles == {
        "question": "plain",
        "tools": "tools",
        "chat": "history",
        "reasoning": "reasoning",
        "answer": "plain",
    }


def test_role_recording_changes_no_rendered_bytes():
    """Recording is inert: rendering a role-annotated signature is identical
    to rendering the same signature without the explicit role."""

    class Bare(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class Annotated(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(role="citations")

    adapter = dspy.ChatAdapter()
    bare = adapter.format(Bare, [], {"question": "q"})
    annotated = adapter.format(Annotated, [], {"question": "q"})
    assert bare == annotated
