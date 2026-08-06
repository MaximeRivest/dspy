"""The `@role` string-signature shorthand and the public `dspy.roles` door.

Epic D (D-6): `"answer: str @citations"` is the fourth role spelling —
pre-tokenized around `ast.parse` (which cannot see `@word` in an argument
list) into the canonical `Annotated[shape, marker]` form. The hazards
documented in roadmap/epic-C-semantic-roles.md section 2a are each pinned
here: depth-aware comma splitting, the untyped default-to-str form, and
eager vocabulary validation. Signatures not using `@` take the exact
pre-existing path — the rest of the signature suite is that gate.
"""

from typing import Annotated, Literal

import pydantic
import pytest

import dspy
from dspy.adapters._engine.roles import resolve_semantic_role
from dspy.signatures import roles as R


def _role_of(signature, field_name):
    fields = {**signature.input_fields, **signature.output_fields}
    return resolve_semantic_role(fields[field_name])


# ---------------------------------------------------------------------------
# The public dspy.roles door
# ---------------------------------------------------------------------------


def test_dspy_roles_is_importable_as_a_module():
    import dspy.roles

    assert dspy.roles.citations is R.citations


def test_dspy_roles_attribute_carries_every_marker():
    for name, marker in R.ALL_ROLE_MARKERS.items():
        assert getattr(dspy.roles, name) is marker


def test_dspy_roles_subscript_sugar_from_the_public_path():
    assert dspy.roles.citations[str] == Annotated[str, R.citations]


# ---------------------------------------------------------------------------
# The @role shorthand: happy paths
# ---------------------------------------------------------------------------


def test_typed_form_resolves_role_and_keeps_shape():
    signature = dspy.Signature("question -> answer: str @citations")
    assert _role_of(signature, "answer") == "citations"
    annotation = signature.output_fields["answer"].annotation
    assert annotation is str


def test_untyped_form_defaults_shape_to_str():
    signature = dspy.Signature("question -> answer @reasoning")
    assert _role_of(signature, "answer") == "reasoning"
    assert signature.output_fields["answer"].annotation is str


def test_role_on_input_field():
    signature = dspy.Signature("question, docs: list[str] @media -> answer")
    assert _role_of(signature, "docs") == "media"
    assert signature.input_fields["docs"].annotation == list[str]
    assert _role_of(signature, "question") == "plain"


def test_multiple_roles_on_distinct_fields():
    signature = dspy.Signature("question -> thinking @reasoning, answer: str @citations")
    assert _role_of(signature, "thinking") == "reasoning"
    assert _role_of(signature, "answer") == "citations"
    assert list(signature.output_fields) == ["thinking", "answer"]


def test_subscript_comma_does_not_split_the_field():
    signature = dspy.Signature("counts: dict[str, int] @plain, question -> answer")
    assert signature.input_fields["counts"].annotation == dict[str, int]
    assert list(signature.input_fields) == ["counts", "question"]


def test_shorthand_composes_with_custom_types():
    class Payload(pydantic.BaseModel):
        text: str

    signature = dspy.Signature(
        "payload: Payload @plain -> answer: str @citations",
        custom_types={"Payload": Payload},
    )
    assert signature.input_fields["payload"].annotation is Payload
    assert _role_of(signature, "answer") == "citations"


def test_shorthand_field_mixes_with_plain_fields():
    signature = dspy.Signature("a, b: int -> c @code, d")
    assert signature.input_fields["b"].annotation is int
    assert _role_of(signature, "c") == "code"
    assert signature.output_fields["d"].annotation is str


def test_marker_lands_in_field_metadata():
    signature = dspy.Signature("question -> answer: str @citations")
    metadata = signature.output_fields["answer"].metadata
    assert R.citations in list(metadata or [])


def test_shorthand_predicts_normally():
    from dspy.utils.dummies import DummyLM

    adapter = dspy.ChatAdapter()
    lm = DummyLM([{"answer": "cited"}], adapter=adapter)
    with dspy.context(lm=lm, adapter=adapter):
        result = dspy.Predict("question -> answer: str @citations")(question="q")
    assert result.answer == "cited"


# ---------------------------------------------------------------------------
# Refusals: eager, teaching
# ---------------------------------------------------------------------------


def test_unknown_role_refuses_eagerly_listing_the_vocabulary():
    with pytest.raises(ValueError, match=r"Unknown semantic role 'vibes'.*citations"):
        dspy.Signature("question -> answer: str @vibes")


def test_two_roles_on_one_field_refuse():
    with pytest.raises(ValueError, match="at most one @role"):
        dspy.Signature("question -> answer @reasoning @citations")


def test_role_not_at_field_end_refuses():
    with pytest.raises(ValueError, match="end of a field"):
        dspy.Signature("question -> answer @citations: str")


def test_bare_at_with_no_name_refuses():
    with pytest.raises(ValueError, match="end of a field"):
        dspy.Signature("question -> @citations")


# ---------------------------------------------------------------------------
# Containment: signatures not using the shorthand are untouched
# ---------------------------------------------------------------------------


def test_at_inside_quoted_literal_is_not_a_role():
    signature = dspy.Signature('question -> answer: Literal["x@y", "z"]')
    assert signature.output_fields["answer"].annotation == Literal["x@y", "z"]


def test_plain_signature_unchanged():
    signature = dspy.Signature("question, hint: int -> answer: list[str]")
    assert signature.input_fields["question"].annotation is str
    assert signature.input_fields["hint"].annotation is int
    assert signature.output_fields["answer"].annotation == list[str]
    assert all(_role_of(signature, name) == "plain" for name in ("question", "hint", "answer"))


def test_annotated_namespace_spelling_still_works():
    signature = dspy.Signature(
        "question -> answer: Annotated[str, citations]",
        custom_types={"Annotated": Annotated, "citations": R.citations},
    )
    assert _role_of(signature, "answer") == "citations"


def test_shorthand_and_annotated_produce_the_same_field():
    via_shorthand = dspy.Signature("question -> answer: str @citations")
    via_namespace = dspy.Signature(
        "question -> answer: Annotated[str, citations]",
        custom_types={"Annotated": Annotated, "citations": R.citations},
    )
    left = via_shorthand.output_fields["answer"]
    right = via_namespace.output_fields["answer"]
    assert left.annotation == right.annotation
    assert list(left.metadata or []) == list(right.metadata or [])
