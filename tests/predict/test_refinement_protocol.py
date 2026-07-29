"""The refinement-unit protocol: declared flexible leaves, not runtime traversal.

A refinement unit is a `Parameter` leaf reported by `Module.named_refinement_units()`.
The traversal treats every `Parameter` as an atomic boundary, so a leaf that
constructs predictors from its own state (the `dspy.Flex` shape) exposes itself
as the unit and keeps its derived runtime predictors private. Units decline
call-time hints unless they opt in; `dspy.Predict` opts in and interprets hints
into its effective call before the resolved adapter formats it.
"""

import pytest

import dspy
from dspy.predict.parameter import Hint, Parameter
from dspy.predict.predict import Predict
from dspy.primitives.prediction import Prediction
from dspy.utils.dummies import DummyLM


class CodeLeaf(dspy.Module, Parameter):
    """A Flex-shaped leaf: its atomic state is `module_src`; the predictor it
    constructs from that state is derived runtime machinery, not a parameter."""

    def __init__(self, signature):
        super().__init__()
        self.module_src = "class Impl(dspy.Module): ..."  # the atomic update unit
        self.derived = Predict(signature)  # stands in for a sandbox-constructed predictor

    def forward(self, **kwargs):
        return self.derived(**kwargs)


class Program(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = Predict("question -> answer")
        self.leaf = CodeLeaf("question -> answer")

    def forward(self, question):
        return self.leaf(question=self.qa(question=question).answer)


def test_named_refinement_units_matches_predictor_names_for_plain_programs():
    class Plain(dspy.Module):
        def __init__(self):
            super().__init__()
            self.first = Predict("a -> b")
            self.inner = dspy.ChainOfThought("b -> c")

        def forward(self, **kwargs):
            return self.inner(b=self.first(**kwargs).b)

    program = Plain()
    unit_names = [name for name, _ in program.named_refinement_units()]
    predictor_names = [name for name, _ in program.named_predictors()]
    assert unit_names == predictor_names == ["first", "inner.predict"]


def test_parameter_leaf_is_an_atomic_boundary():
    program = Program()

    unit_names = [name for name, _ in program.named_refinement_units()]
    # The leaf itself is the unit; its derived predictor is not an independent target.
    assert unit_names == ["qa", "leaf"]
    assert all(not name.startswith("leaf.") for name in unit_names)

    # The runtime graph is still visible *from inside* the leaf (execution needs it),
    # just never surfaced as an independently refinable unit of the parent.
    assert [name for name, _ in program.leaf.named_predictors()] == ["derived"]


def test_bare_predict_is_its_own_unit():
    predict = Predict("q -> a")
    assert [name for name, _ in predict.named_refinement_units()] == ["self"]


def test_parameters_decline_hints_by_default():
    leaf = CodeLeaf("q -> a")
    hint = Hint(target="leaf", content="try harder")
    assert not leaf.accepts_hint(hint)
    with pytest.raises(NotImplementedError):
        leaf.apply_hint(hint)


def test_hint_defaults_are_non_authoritative():
    hint = Hint(target="qa", content="answer in one word")
    assert hint.authority == "non_authoritative"
    assert hint.scope == "next_candidate"
    assert hint.provenance is None


def test_predict_interprets_hint_into_effective_call_only():
    lm = DummyLM([{"a": "out"}, {"a": "out again"}])
    dspy.configure(lm=lm, adapter=None)

    seen = []

    class TrackingAdapter(dspy.ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            seen.append((signature, dict(inputs)))
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    predict = Predict("q -> a")
    predict.adapter = TrackingAdapter()
    predict.apply_hint(Hint(target="self", content="be brief"))

    with dspy.context(trace=[]):
        predict(q="hello")
        trace = dspy.settings.trace.copy()

    # The adapter receives an already-defined call: extended signature + hint input.
    signature, inputs = seen[0]
    assert "hint_" in signature.input_fields
    assert inputs["hint_"] == "be brief"

    # The persistent signature, the trace, and the serialized state stay hint-free.
    assert "hint_" not in predict.signature.fields
    assert all("hint_" not in traced_inputs for _, traced_inputs, _ in trace)
    assert "_hints" not in predict.dump_state()

    predict.clear_hints()
    predict(q="hello again")
    assert "hint_" not in seen[-1][1]


def test_hints_do_not_survive_into_copies_of_the_original():
    predict = Predict("q -> a")
    copy_before = predict.deepcopy()
    predict.apply_hint(Hint(target="self", content="be brief"))
    copy_after = predict.deepcopy()

    assert copy_before._hints == []
    assert len(copy_after._hints) == 1
    copy_after.clear_hints()
    assert len(predict._hints) == 1  # clearing the copy leaves the original alone


def test_opt_in_leaf_receives_explicit_hint_from_refine():
    received = []

    class HintableLeaf(dspy.Module, Parameter):
        def __init__(self):
            super().__init__()
            self.state = "v0"

        def accepts_hint(self, hint=None):
            return True

        def apply_hint(self, hint):
            received.append(hint)

        def forward(self, question):
            return Prediction(answer="always-this")

    lm = DummyLM(
        [
            {"discussion": "The leaf was wrong.", "advice": {"leaf": "Say something else."}},
        ]
    )
    dspy.configure(lm=lm, adapter=None)

    class Wrapper(dspy.Module):
        def __init__(self):
            super().__init__()
            self.leaf = HintableLeaf()

        def forward(self, **kwargs):
            return self.leaf(**kwargs)

    refine = dspy.Refine(module=Wrapper(), N=2, reward_fn=lambda _, __: 0.0, threshold=1.0)
    refine(question="q")

    # The second attempt's candidate leaf received an explicit, provenance-carrying hint.
    assert len(received) == 1
    hint = received[0]
    assert isinstance(hint, Hint)
    assert hint.target == "leaf"
    assert hint.content == "Say something else."
    assert hint.authority == "non_authoritative"
    assert hint.provenance["source"] == "dspy.Refine/OfferFeedback"
    assert hint.provenance["reward"] == 0.0
