"""Stage A1 tests: signature authoring, Example, Prediction, typed errors."""

import importlib.util
from pathlib import Path

import pytest

import dspy
from dspy.signatures import roles as roles_module
from dspy.signatures.roles import register_role_derivation, resolve_semantic_role, semantic_role_for

# ---------------------------------------------------------------------------
# Signature authoring — class-based
# ---------------------------------------------------------------------------


class TestClassSignatures:
    def test_fields_and_direction(self):
        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        assert list(QA.input_fields) == ["question"]
        assert list(QA.output_fields) == ["answer"]
        assert QA.signature == "question -> answer"

    def test_docstring_becomes_instructions(self):
        class QA(dspy.Signature):
            """Answer briefly."""

            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        assert QA.instructions == "Answer briefly."

    def test_default_instructions(self):
        class QA(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        assert "`question`" in QA.instructions
        assert "`answer`" in QA.instructions

    def test_prefix_and_desc_defaults(self):
        class Sig(dspy.Signature):
            user_query: str = dspy.InputField()
            final_answer: str = dspy.OutputField(desc="the answer")

        assert Sig.fields["user_query"].json_schema_extra["prefix"] == "User Query:"
        assert Sig.fields["final_answer"].json_schema_extra["desc"] == "the answer"

    def test_plain_field_refused(self):
        import pydantic

        with pytest.raises(TypeError, match="InputField or OutputField"):

            class Bad(dspy.Signature):
                question: str = pydantic.Field()
                answer: str = dspy.OutputField()


# ---------------------------------------------------------------------------
# Signature authoring — string-based
# ---------------------------------------------------------------------------


class TestStringSignatures:
    def test_basic(self):
        sig = dspy.Signature("question -> answer")
        assert list(sig.input_fields) == ["question"]
        assert list(sig.output_fields) == ["answer"]
        # Untyped fields default to str.
        assert sig.fields["question"].annotation is str

    def test_typed(self):
        sig = dspy.Signature("x: int, y: list[str] -> z: dict[str, float]")
        assert sig.fields["x"].annotation is int
        assert sig.fields["y"].annotation == list[str]
        assert sig.fields["z"].annotation == dict[str, float]

    def test_instructions_argument(self):
        sig = dspy.Signature("question -> answer", "Answer in French.")
        assert sig.instructions == "Answer in French."

    def test_custom_types(self):
        import pydantic

        class Payload(pydantic.BaseModel):
            text: str

        sig = dspy.Signature("data: Payload -> summary", custom_types={"Payload": Payload})
        assert sig.fields["data"].annotation is Payload

    def test_missing_arrow_refused(self):
        with pytest.raises(ValueError, match="->"):
            dspy.Signature("question, answer")

    def test_duplicate_names_refused(self):
        with pytest.raises(ValueError, match="distinct"):
            dspy.Signature("question -> question")

    def test_ensure_signature(self):
        sig = dspy.ensure_signature("question -> answer")
        assert sig.signature == "question -> answer"
        assert dspy.ensure_signature(sig) is sig
        with pytest.raises(ValueError):
            dspy.ensure_signature(sig, "new instructions")


class TestSignatureManipulation:
    def test_with_instructions(self):
        sig = dspy.Signature("question -> answer")
        new = sig.with_instructions("Be terse.")
        assert new is not sig
        assert new.instructions == "Be terse."
        assert new.signature == sig.signature

    def test_append_instructions(self):
        sig = dspy.Signature("question -> answer", "Base.")
        new = sig.append_instructions("More.")
        assert "Base." in new.instructions and "More." in new.instructions

    def test_insert_delete(self):
        sig = dspy.Signature("question -> answer")
        with_ctx = sig.prepend("context", dspy.InputField())
        assert list(with_ctx.input_fields) == ["context", "question"]
        back = with_ctx.delete("context")
        assert list(back.input_fields) == ["question"]

    def test_equals(self):
        a = dspy.Signature("question -> answer", "Hi.")
        b = dspy.Signature("question -> answer", "Hi.")
        c = dspy.Signature("question -> answer", "Other.")
        assert a.equals(b)
        assert not a.equals(c)


# ---------------------------------------------------------------------------
# Semantic roles
# ---------------------------------------------------------------------------


@pytest.fixture()
def clean_role_registry():
    saved = dict(roles_module._DERIVED)
    yield
    roles_module._DERIVED.clear()
    roles_module._DERIVED.update(saved)


class TestSemanticRoles:
    def test_string_shorthand(self):
        sig = dspy.Signature("question -> answer: str @citations")
        assert resolve_semantic_role(sig.fields["answer"]) == "citations"

    def test_kwarg_spelling(self):
        class Sig(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField(role="reasoning")

        assert resolve_semantic_role(Sig.fields["answer"]) == "reasoning"

    def test_marker_subscript_sugar(self):
        class Sig(dspy.Signature):
            question: str = dspy.InputField()
            answer: dspy.roles.citations[str] = dspy.OutputField()

        assert resolve_semantic_role(Sig.fields["answer"]) == "citations"

    def test_unknown_role_refused(self):
        with pytest.raises(ValueError, match="Unknown semantic role"):
            dspy.Signature("question -> answer @sonnets")

    def test_conflicting_declarations_refuse_at_construction(self):
        with pytest.raises(ValueError, match="conflicting semantic roles"):

            class Bad(dspy.Signature):
                question: str = dspy.InputField()
                answer: dspy.roles.citations[str] = dspy.OutputField(role="reasoning")

    def test_registry_derivation(self, clean_role_registry):
        class FancyImage:
            pass

        register_role_derivation(FancyImage, "media")
        assert semantic_role_for(FancyImage) == "media"
        assert semantic_role_for(list[FancyImage]) == "media"
        # Subclasses derive the registered role too.

        class Fancier(FancyImage):
            pass

        assert semantic_role_for(Fancier) == "media"

    def test_registry_starts_without_adapter_imports(self):
        # The signature layer never imports the adapter world: plain shapes
        # derive "plain" with an empty registry.
        assert semantic_role_for(str) == "plain"

    def test_registry_refuses_conflicts(self, clean_role_registry):
        class T:
            pass

        register_role_derivation(T, "media")
        with pytest.raises(ValueError, match="already registered"):
            register_role_derivation(T, "code")
        with pytest.raises(ValueError, match="Unknown semantic role"):
            register_role_derivation(T, "sonnets")


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------


class TestExample:
    def test_build_and_access(self):
        ex = dspy.Example(question="Why?", answer="Because.")
        assert ex.question == "Why?"
        assert ex["answer"] == "Because."
        assert "question" in ex
        assert ex.get("missing", "d") == "d"

    def test_base_dict(self):
        ex = dspy.Example({"a": 1}, b=2)
        assert ex.a == 1 and ex.b == 2

    def test_inputs_labels(self):
        ex = dspy.Example(question="Why?", answer="Because.").with_inputs("question")
        assert ex.inputs().keys() == ["question"]
        assert ex.labels().keys() == ["answer"]

    def test_inputs_requires_declaration(self):
        with pytest.raises(ValueError, match="with_inputs"):
            dspy.Example(a=1).inputs()

    def test_copy_and_without(self):
        ex = dspy.Example(a=1, b=2)
        assert ex.copy(b=3).b == 3
        assert ex.without("b").keys() == ["a"]

    def test_to_dict_nested(self):
        inner = dspy.Example(x=1)
        ex = dspy.Example(inner=inner, items=[inner])
        assert ex.toDict() == {"inner": {"x": 1}, "items": [{"x": 1}]}

    def test_equality(self):
        assert dspy.Example(a=1) == dspy.Example(a=1)
        assert dspy.Example(a=1) != dspy.Example(a=2)


# ---------------------------------------------------------------------------
# Prediction and the trajectory channel
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_declared_outputs(self):
        pred = dspy.Prediction(answer="4")
        assert pred.answer == "4"

    def test_trajectory_channel(self):
        pred = dspy.Prediction(answer="4")
        assert pred._trajectory == {}
        pred._trajectory["reasoning"] = "2+2"
        assert pred._trajectory["reasoning"] == "2+2"
        # Exhaust never leaks into the declared fields.
        assert "reasoning" not in pred._store

    def test_trajectory_attribute_read_warns(self):
        pred = dspy.Prediction(answer="4")
        pred._trajectory["trajectory"] = [{"tool": "search"}]
        with pytest.warns(DeprecationWarning, match="mechanism\\s+exhaust|exhaust"):
            assert pred.trajectory == [{"tool": "search"}]

    def test_missing_attribute_raises(self):
        pred = dspy.Prediction(answer="4")
        with pytest.raises(AttributeError):
            _ = pred.nonexistent
        with pytest.raises(AttributeError):
            _ = pred._private_thing

    def test_score_arithmetic(self):
        a = dspy.Prediction(score=0.5)
        b = dspy.Prediction(score=0.25)
        assert float(a) == 0.5
        assert a + b == 0.75
        assert a > b
        assert b <= a

    def test_from_completions(self):
        pred = dspy.Prediction.from_completions([{"answer": "4"}, {"answer": "5"}])
        assert pred.answer == "4"
        assert len(pred.completions) == 2
        assert pred.completions[1].answer == "5"


# ---------------------------------------------------------------------------
# The typed-error table
# ---------------------------------------------------------------------------


class TestTypedErrors:
    def test_raiseable_is_exactly_the_table(self):
        assert set(dspy.RAISEABLE) == {"ToolError", "InterpreterError", "AdapterParseError", "LMError"}

    def test_catchable_names(self):
        assert dspy.CATCHABLE_NAMES == set(dspy.RAISEABLE) | {
            "InterpreterTypeError",
            "InterpreterKeyError",
            "InterpreterArithmeticError",
        }
        assert dspy.HANDLER_NAMES == dspy.CATCHABLE_NAMES | {"Exception"}

    def test_channels(self):
        assert issubclass(dspy.LMError, dspy.CatchableError)
        assert issubclass(dspy.LoopCapError, dspy.UncatchableError)
        assert not issubclass(dspy.LoopCapError, dspy.CatchableError)
        assert issubclass(dspy.CatchableError, dspy.PirError)

    def test_handler_matching_is_exact_name(self):
        assert dspy.handler_matches(dspy.ToolError("x"), "ToolError")
        assert not dspy.handler_matches(dspy.ToolError("x"), "LMError")
        assert dspy.handler_matches(dspy.ToolError("x"), "Exception")
        # Harness guards are invisible to program handlers, even `Exception`.
        assert not dspy.handler_matches(dspy.LoopCapError("cap"), "Exception")
        assert not dspy.handler_matches(dspy.LoopCapError("cap"), "LoopCapError")

    def test_engine_mirror_shares_class_identity(self):
        # Load the engine's errors module by file path (its package __init__
        # is dormant until stage A3) and check it re-exports the same classes.
        path = Path(dspy.__file__).parent / "programir" / "engine" / "errors.py"
        spec = importlib.util.spec_from_file_location("_engine_errors_probe", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.LMError is dspy.LMError
        assert module.RAISEABLE == dspy.RAISEABLE
        assert module.CATCHABLE_NAMES == dspy.CATCHABLE_NAMES

    def test_interpreter_errors_join_the_table(self):
        from dspy.primitives import CodeExecutionError, CodeInterpreterError

        assert issubclass(CodeInterpreterError, dspy.InterpreterError)
        assert issubclass(CodeExecutionError, CodeInterpreterError)


def test_no_ambient_state_surface():
    """The settings machinery is gone: no context, no settings, no thread-locals."""
    assert not hasattr(dspy, "settings")
    assert not hasattr(dspy, "context")
    assert not hasattr(dspy, "load_settings")
