"""End-to-end journeys through the adapter system.

These tests drive `dspy.Predict` with stub LMs and assert user-visible
semantics: typed predictions out, role declarations honored, the sacred
signature contract, legacy-type compatibility, and codec-visible rendering.
Byte-level parity is the golden corpus's job, not this file's.

The `test_epic_d_*` tests are strict xfails encoding Epic D's ratified
surface (roadmap/epic-D-adapter-serializer.md): they fail today and flip
loudly when the serializer epic lands.
"""

import warnings
from enum import Enum
from typing import Annotated

import pydantic
import pytest

import dspy
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.signatures import roles as R
from dspy.utils.dummies import DummyLM


class Mood(Enum):
    HAPPY = "happy"
    SAD = "sad"


class Item(pydantic.BaseModel):
    name: str
    score: float


class RichSignature(dspy.Signature):
    """Extract structured facts."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    count: int = dspy.OutputField()
    facts: list[Item] = dspy.OutputField()
    mood: Mood = dspy.OutputField()
    note: str | None = dspy.OutputField()


RICH_ANSWER = {
    "answer": "blue",
    "count": 3,
    "facts": [Item(name="a", score=0.5), Item(name="b", score=1.0)],
    "mood": "happy",
    "note": "fine",
}


def _predict(adapter, signature, answers, **inputs):
    lm = DummyLM(answers, adapter=adapter)
    with dspy.context(lm=lm, adapter=adapter):
        return dspy.Predict(signature)(**inputs)


# ---------------------------------------------------------------------------
# Every adapter, signature in -> typed prediction out
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "adapter",
    [
        dspy.ChatAdapter(use_json_adapter_fallback=False),
        dspy.JSONAdapter(),
        dspy.XMLAdapter(),
        BAMLAdapter(),
    ],
    ids=["chat", "json", "xml", "baml"],
)
def test_adapter_returns_typed_prediction(adapter):
    result = _predict(adapter, RichSignature, [dict(RICH_ANSWER)], question="q")

    assert result.answer == "blue"
    assert result.count == 3
    assert isinstance(result.count, int)
    assert result.facts == [Item(name="a", score=0.5), Item(name="b", score=1.0)]
    assert all(isinstance(item, Item) for item in result.facts)
    assert result.mood is Mood.HAPPY
    assert result.note == "fine"
    assert set(result.keys()) == {"answer", "count", "facts", "mood", "note"}


def test_two_step_adapter_end_to_end():
    extractor = DummyLM([{"answer": "blue", "count": 3}])
    main_lm = DummyLM([{"text": "The answer is blue and the count is three."}])
    adapter = dspy.TwoStepAdapter(extraction_model=extractor)
    with dspy.context(lm=main_lm, adapter=adapter):
        result = dspy.Predict("question -> answer, count: int")(question="q")

    assert result.answer == "blue"
    assert result.count == 3


# ---------------------------------------------------------------------------
# Role declarations on real predictions
# ---------------------------------------------------------------------------


def _role_of(signature, field_name):
    from dspy.adapters._engine.roles import resolve_semantic_role

    return resolve_semantic_role(signature.output_fields[field_name])


def test_annotated_role_in_class_signature_predicts_normally():
    class Grounded(dspy.Signature):
        question: str = dspy.InputField()
        answer: Annotated[str, R.citations] = dspy.OutputField()

    result = _predict(dspy.ChatAdapter(), Grounded, [{"answer": "cited"}], question="q")
    assert result.answer == "cited"
    assert _role_of(Grounded, "answer") == "citations"


def test_marker_sugar_role_predicts_normally():
    class Grounded(dspy.Signature):
        question: str = dspy.InputField()
        answer: R.citations[str] = dspy.OutputField()

    result = _predict(dspy.ChatAdapter(), Grounded, [{"answer": "cited"}], question="q")
    assert result.answer == "cited"
    assert _role_of(Grounded, "answer") == "citations"


def test_role_kwarg_predicts_normally():
    class Grounded(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField(role="citations")

    result = _predict(dspy.ChatAdapter(), Grounded, [{"answer": "cited"}], question="q")
    assert result.answer == "cited"
    assert _role_of(Grounded, "answer") == "citations"


def test_annotated_role_in_string_signature_predicts_normally():
    signature = dspy.Signature(
        "question -> answer: Annotated[str, citations]",
        custom_types={"Annotated": Annotated, "citations": R.citations},
    )
    result = _predict(dspy.ChatAdapter(), signature, [{"answer": "cited"}], question="q")
    assert result.answer == "cited"
    assert _role_of(signature, "answer") == "citations"


def test_unknown_role_kwarg_raises_at_declaration():
    with pytest.raises(ValueError, match="Unknown semantic role"):
        dspy.OutputField(role="vibes")


def test_conflicting_role_spellings_refuse_loudly():
    class Conflicted(dspy.Signature):
        question: str = dspy.InputField()
        answer: Annotated[str, R.citations] = dspy.OutputField(role="reasoning")

    with pytest.raises(ValueError):
        _predict(dspy.ChatAdapter(), Conflicted, [{"answer": "x"}], question="q")


# ---------------------------------------------------------------------------
# The sacred signature: exhaust vs contract
# ---------------------------------------------------------------------------


def test_undeclared_reasoning_is_exhaust_not_output():
    lm = DummyLM([{"reasoning": "step by step", "answer": "42"}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.ChainOfThought("question -> answer")(question="q")

    assert result.answer == "42"
    assert "reasoning" not in result.keys()
    assert "reasoning" not in result.toDict()
    with pytest.raises(KeyError):
        result["reasoning"]
    assert result._trajectory["reasoning"] == "step by step"


def test_exhaust_dot_access_shims_with_warning():
    lm = DummyLM([{"reasoning": "step by step", "answer": "42"}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.ChainOfThought("question -> answer")(question="q")

    with pytest.warns(DeprecationWarning, match="_trajectory"):
        assert result.reasoning == "step by step"


def test_declared_reasoning_stays_contractual():
    lm = DummyLM([{"reasoning": "step by step", "answer": "42"}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.ChainOfThought("question -> reasoning, answer")(question="q")

    assert "reasoning" in result.keys()
    assert result["reasoning"] == "step by step"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert result.reasoning == "step by step"


# ---------------------------------------------------------------------------
# Legacy semantic types: the compat guarantee
# ---------------------------------------------------------------------------


def test_legacy_reasoning_type_textual_path():
    class WithReasoning(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    lm = DummyLM([{"reasoning": "because", "answer": "42"}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.Predict(WithReasoning)(question="q")

    assert result.answer == "42"
    assert str(result.reasoning) == "because"
    assert _role_of(WithReasoning, "reasoning") == "reasoning"


def test_legacy_image_input_renders_content_block():
    class Describe(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        caption: str = dspy.OutputField()

    lm = DummyLM([{"caption": "a red square"}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.Predict(Describe)(image=dspy.Image("https://example.com/x.png"))

    assert result.caption == "a red square"
    user_message = lm.history[-1]["messages"][-1]
    blocks = user_message["content"]
    assert any(
        isinstance(block, dict) and block.get("type") == "image_url" for block in blocks
    ), f"no image content block in {blocks!r}"


def test_legacy_tool_input_renders_textually():
    def search(query: str) -> str:
        """Search the web."""
        return "results"

    class WithTools(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        answer: str = dspy.OutputField()

    lm = DummyLM([{"answer": "used the tool"}])
    with dspy.context(lm=lm, adapter=dspy.ChatAdapter()):
        result = dspy.Predict(WithTools)(question="q", tools=[dspy.Tool(search)])

    assert result.answer == "used the tool"
    rendered = str(lm.history[-1]["messages"])
    assert "search" in rendered
    assert "Search the web" in rendered


# ---------------------------------------------------------------------------
# Codec-visible behavior
# ---------------------------------------------------------------------------


def test_baml_renders_pydantic_inputs_indented_chat_renders_compact():
    class Summarize(dspy.Signature):
        item: Item = dspy.InputField()
        summary: str = dspy.OutputField()

    value = Item(name="widget", score=0.9)
    chat_messages = dspy.ChatAdapter().format(Summarize, [], {"item": value})
    baml_messages = BAMLAdapter().format(Summarize, [], {"item": value})

    chat_user = next(m["content"] for m in chat_messages if m["role"] == "user")
    baml_user = next(m["content"] for m in baml_messages if m["role"] == "user")

    assert '{\n  "name": "widget"' in baml_user
    assert '{\n  "name": "widget"' not in chat_user
    assert '"widget"' in chat_user


# ---------------------------------------------------------------------------
# Epic D contracts (strict xfail: these flip when the serializer lands)
# ---------------------------------------------------------------------------

LITERAL_TABLE_KEYS = {
    "input_field_render",
    "output_field_render",
    "field_separator",
    "output_structure",
    "completed_marker",
    "output_requirement",
    "parse_pattern",
}


def test_epic_d_public_roles_import():
    assert dspy.roles.citations is R.citations


def test_epic_d_role_string_shorthand():
    signature = dspy.Signature("question -> answer: str @citations")
    assert _role_of(signature, "answer") == "citations"


def test_epic_d_strategies_binding_surface():
    adapter = dspy.ChatAdapter(strategies={"reasoning": "textual_field"})
    result = _predict(adapter, "question -> answer", [{"answer": "x"}], question="q")
    assert result.answer == "x"


def test_strategies_binding_forces_the_textual_lane():
    """An explicit textual binding keeps the reasoning field in the prompt
    and out of the provider channel, whatever the LM supports."""

    class Thoughtful(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = dspy.ChatAdapter(strategies={"reasoning": "textual_field"})
    lm = DummyLM([{"reasoning": "because", "answer": "42"}], adapter=adapter)
    with dspy.context(lm=lm, adapter=adapter):
        result = dspy.Predict(Thoughtful)(question="q")

    assert result.answer == "42"
    assert str(result.reasoning) == "because"
    rendered = str(lm.history[-1]["messages"])
    assert "reasoning" in rendered  # the field is textually hosted


def test_strategies_bindings_survive_the_json_adapter_fallback():
    """The retry adapter carries the declared bindings — a forced-textual
    channel must not silently go native on fallback."""
    adapter = dspy.ChatAdapter(strategies={"reasoning": "textual_field"})
    fallback = adapter._make_json_adapter_fallback()
    assert fallback.strategies == {"reasoning": "textual_field"}
    assert dspy.ChatAdapter()._make_json_adapter_fallback().strategies == {}


def test_strategies_binding_refuses_unknown_names_eagerly():
    with pytest.raises(ValueError, match="bindable roles"):
        dspy.ChatAdapter(strategies={"vibes": "auto"})
    with pytest.raises(ValueError, match="valid reasoning strategies"):
        dspy.ChatAdapter(strategies={"reasoning": "quantum"})
    with pytest.raises(ValueError, match="not implemented"):
        dspy.ChatAdapter(strategies={"reasoning": "prefill"})
    with pytest.raises(ValueError, match="must agree"):
        dspy.ChatAdapter(strategies={"tools": "native_fc"})  # use_native_function_calling defaults False
    with pytest.raises(ValueError, match="must agree"):
        dspy.JSONAdapter(strategies={"tools": "textual_json"})  # JSONAdapter defaults native FC on


def test_epic_d_literal_table_export():
    # Post-rescope (D-018): the preset template is the literal table's full
    # form; literal_table() survives as the DERIVED 7-key summary view —
    # never authored, always computed from the template.
    table = dspy.ChatAdapter().literal_table()
    assert set(table) <= LITERAL_TABLE_KEYS
    assert table["output_structure"] == "markers"
    assert table["completed_marker"] == "[[ ## completed ## ]]"


def test_epic_d_adapter_serialize_load_roundtrip():
    adapter = dspy.ChatAdapter()
    entry = adapter.dump_entry()
    assert isinstance(entry, dict)

    reloaded = dspy.adapters.load_entry(entry)
    signature = dspy.Signature("question -> answer")
    assert reloaded.format(signature, [], {"question": "q"}) == adapter.format(
        signature, [], {"question": "q"}
    )
