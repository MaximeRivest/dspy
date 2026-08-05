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


# ---- Annotated role markers (the cross-surface primitive) ----

from typing import Annotated  # noqa: E402

from dspy.adapters._engine.roles import _marker_roles  # noqa: E402
from dspy.signatures import roles as R  # noqa: E402


def test_marker_subscript_sugar_is_annotated():
    assert R.citations[str] == Annotated[str, R.citations]
    assert R.media[list[dspy.Image]] == Annotated[list[dspy.Image], R.media]


def test_marker_repr_reads_well():
    assert repr(R.citations) == "dspy.roles.citations"


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        (Annotated[str, R.citations], {"citations"}),
        (R.citations[str], {"citations"}),
        (list[R.citations[str]], {"citations"}),
        (R.citations[list[str]], {"citations"}),
        (Optional[R.citations[str]], {"citations"}),
        (R.citations[str] | None, {"citations"}),
        (list[str], set()),
        (Annotated[str, "unrelated metadata"], set()),
    ],
)
def test_marker_roles_found_through_nesting(annotation, expected):
    assert _marker_roles(annotation) == expected


def test_top_level_marker_resolves_via_field_metadata():
    """Pydantic hoists a top-level Annotated's metadata onto FieldInfo.metadata."""

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: Annotated[str, R.citations] = dspy.OutputField()

    assert Sig.output_fields["answer"].annotation is str
    assert resolve_semantic_role(Sig.output_fields["answer"]) == "citations"


def test_nested_marker_resolves_via_annotation_tree():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        quotes: list[R.citations[str]] = dspy.OutputField()

    assert resolve_semantic_role(Sig.output_fields["quotes"]) == "citations"


def test_marker_and_kwarg_agreeing_is_fine():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: R.citations[str] = dspy.OutputField(role="citations")

    plan = AdapterPlan.from_signature(Sig, {"question": "q"})
    assert plan.output_fields[0].metadata["semantic_role"] == "citations"


def test_marker_and_kwarg_conflicting_raises():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        answer: R.citations[str] = dspy.OutputField(role="reasoning")

    with pytest.raises(ValueError, match="conflicting semantic roles"):
        AdapterPlan.from_signature(Sig, {"question": "q"})


def test_marker_conflicting_with_legacy_derivation_raises():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        thinking: Annotated[Reasoning, R.citations] = dspy.OutputField()

    with pytest.raises(ValueError, match="annotation implies"):
        AdapterPlan.from_signature(Sig, {"question": "q"})


def test_marker_agreeing_with_legacy_derivation_is_fine():
    assert semantic_role_for(Annotated[Reasoning, R.reasoning]) == "reasoning"


def test_marker_in_string_signature_via_namespace():
    sig = dspy.Signature(
        "question: str -> answer: Annotated[str, citations]",
        custom_types={"Annotated": Annotated, "citations": R.citations},
    )
    assert resolve_semantic_role(sig.output_fields["answer"]) == "citations"


def test_unknown_marker_name_rejected_at_construction():
    with pytest.raises(ValueError, match="Unknown semantic role"):
        R.SemanticRole("chain_of_thought")


def test_marker_annotation_changes_no_rendered_bytes():
    class Bare(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    class Marked(dspy.Signature):
        question: str = dspy.InputField()
        answer: Annotated[str, R.citations] = dspy.OutputField()

    adapter = dspy.ChatAdapter()
    assert adapter.format(Bare, [], {"question": "q"}) == adapter.format(Marked, [], {"question": "q"})
