"""Stage A8 tests: the BestOfN/Refine IR macros.

Per macro: the compile-clean assert (manifest structure incl. the Try/If
skeleton), engine runs with a scripted DummyLM, ENGINE-vs-NATIVE
equivalence driven through BOTH the early-exit and the exhausted-N
branches on BOTH arms (the A7 fix-wave lesson), save -> load -> run,
argmax correctness, reward-leaf call counts, and the construction
refusals — including Refine's feedback teaching error.
"""

import json

import pytest

import dspy
from dspy.core.errors import ToolError
from dspy.lm import BINDINGS


@pytest.fixture(autouse=True)
def clean_bindings():
    saved = dict(BINDINGS)
    BINDINGS.clear()
    yield
    BINDINGS.clear()
    BINDINGS.update(saved)


def chat_completion(**fields: str) -> str:
    """Render one scripted completion in the chat adapter's convention."""
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


def assert_identical_calls(engine_lm, native_lm):
    """The equivalence core: same call count, messages, and kwargs."""
    assert len(engine_lm.calls) == len(native_lm.calls)
    for engine_call, native_call in zip(engine_lm.calls, native_lm.calls, strict=True):
        assert engine_call["messages"] == native_call["messages"]
        assert engine_call["kwargs"] == native_call["kwargs"]


def try_handler_types(value) -> list[str]:
    """Every Try handler's caught type, in document order."""
    found: list[str] = []
    if isinstance(value, dict):
        if value.get("node") == "Try":
            found.extend(handler["type"] for handler in value["handlers"])
        for sub in value.values():
            found.extend(try_handler_types(sub))
    elif isinstance(value, list):
        for sub in value:
            found.extend(try_handler_types(sub))
    return found


def table_reward(outputs) -> float:
    """Reward from a fixed answer table (unknown answers score 0.0)."""
    scores = {"a1": 0.2, "a2": 0.9, "a3": 0.4}
    return scores.get(outputs["answer"], 0.0)


def capital_reward(outputs) -> float:
    """Score 1.0 for a known capital answer."""
    return 1.0 if outputs["answer"] in ("Paris", "Ottawa") else 0.0


def flat_reward(outputs) -> float:
    """Score every attempt the same: the tie-semantics probe."""
    return 0.5


def digit_reward(outputs) -> float:
    """Score 1.0 for a digit answer."""
    return 1.0 if outputs["answer"].isdigit() else 0.0


def answers(*values: str) -> list[str]:
    return [chat_completion(answer=value) for value in values]


def count_reward_calls(program):
    """Wrap the materialized reward tool with a counter; return it.

    The engine resolves tools from the executable's table, so wrapping
    there counts exactly the reward-LEAF invocations of subsequent
    engine runs.
    """
    executable = program._executable()
    original = executable.tools["reward"]
    counter = {"calls": 0}

    def counting(**kwargs):
        counter["calls"] += 1
        return original(**kwargs)

    executable.tools["reward"] = counting
    return executable, counter


# ---------------------------------------------------------------------------
# BestOfN: structure
# ---------------------------------------------------------------------------


class TestBestOfNStructure:
    def test_compiles_clean_with_macro_skeleton(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 3, table_reward, threshold=0.5)

        manifest = best.to_manifest()
        tree = manifest["components"]["1_module_tree"]
        assert tree["kind"] == "BestOfN"
        assert [child["kind"] for child in tree["children"]] == ["Predict"]
        assert tree["tools"] == ["reward"]

        forward = manifest["components"]["5_forward"]["self"]
        assert forward["args"] == ["question"]
        loop = forward["body"][4]
        assert loop["node"] == "For"
        assert loop["range"] == 3  # the attempt cap is a LITERAL
        # The attempt skeleton: typed skip handlers, best-so-far If,
        # threshold If with the Break.
        assert try_handler_types(forward) == ["AdapterParseError", "ToolError"]
        best_if = loop["body"][2]
        assert best_if["node"] == "If"
        assert best_if["test"]["node"] == "BoolOp"
        threshold_if = loop["body"][3]
        assert threshold_if["node"] == "If"
        assert threshold_if["test"] == {
            "node": "Compare",
            "op": "ge",
            "left": {"node": "Var", "name": "best_score"},
            "right": {"node": "Const", "value": 0.5},
        }
        assert threshold_if["body"] == [{"node": "Break"}]
        # The all-failed refusal is a typed Raise, not a silent record.
        refusal = forward["body"][5]
        assert refusal["node"] == "If"
        assert refusal["body"][0]["node"] == "Raise"
        assert refusal["body"][0]["exc"] == "ToolError"

        # The reward leaf carries the user function's source verbatim.
        assert manifest["components"]["6_tools"]["reward"]["source"] == "tools/reward.py"
        sidecar = best.compile_ir().sidecars["tools/reward.py"].decode("utf-8")
        assert "def table_reward(outputs) -> float:" in sidecar
        # Owner-independent on purpose: the same user reward must yield
        # a byte-identical wrapper under BestOfN and Refine, so nested
        # macros sharing one reward dedupe in the tool pool.
        assert '"""Declared reward leaf: score one attempt' in sidecar

    def test_without_threshold_has_no_break(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 3, table_reward)
        forward = best.to_manifest()["components"]["5_forward"]["self"]
        loop = forward["body"][4]
        assert len(loop["body"]) == 3  # attempts, Try, best-so-far If — no threshold If
        assert json.dumps(forward).count('"Break"') == 0

    def test_manifest_forward_is_the_built_tree(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 2, table_reward)
        manifest_forward = best.to_manifest()["components"]["5_forward"]["self"]
        assert json.dumps(manifest_forward) == json.dumps(best._forward_tree())

    def test_construction_refusals(self):
        predictor = dspy.Predict("question -> answer")
        with pytest.raises(TypeError, match="wraps a dspy.Module"):
            dspy.BestOfN(lambda: None, 3, table_reward)
        with pytest.raises(ValueError, match="int >= 1"):
            dspy.BestOfN(predictor, 0, table_reward)
        with pytest.raises(ValueError, match="threshold must be a number"):
            dspy.BestOfN(predictor, 3, table_reward, threshold="high")
        with pytest.raises(ValueError, match="plain named function"):
            dspy.BestOfN(predictor, 3, lambda outputs: 1.0)
        with pytest.raises(ValueError, match="ONE argument"):

            def two_args(example, prediction) -> float:
                """Metrics take two arguments; rewards take one."""
                return 1.0

            dspy.BestOfN(predictor, 3, two_args)
        with pytest.raises(ValueError, match="reserves the name"):
            dspy.BestOfN(dspy.Predict("score -> answer"), 3, table_reward)
        with pytest.raises(ValueError, match="would be overwritten"):
            dspy.BestOfN(dspy.Predict("question -> answer, attempts"), 3, table_reward)

    def test_unexportable_reward_refuses_at_construction(self):
        bound = "closed-over"

        def closure_reward(outputs) -> float:
            """Close over a local: not a declarable leaf."""
            return float(len(bound))

        # The embedded source reads a name the wrapper cannot resolve —
        # refused where the reward is DECLARED, not at the first compile.
        with pytest.raises(ValueError, match="reads globals"):
            dspy.BestOfN(dspy.Predict("question -> answer"), 3, closure_reward)


# ---------------------------------------------------------------------------
# BestOfN: behavior, both branches through both arms
# ---------------------------------------------------------------------------


class TestBestOfNBehavior:
    def test_argmax_returns_attempt_two(self):
        # Scripted rewards 0.2 / 0.9 / 0.4: attempt 2's outputs win.
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 3, table_reward)
        lm = dspy.DummyLM(answers("a1", "a2", "a3"))
        dspy.configure(lm=lm)

        prediction = best(question="pick one")

        assert prediction.answer == "a2"
        assert prediction.best_score == 0.9
        assert prediction.attempts == 3
        assert len(lm.calls) == 3

    def test_tie_keeps_the_earliest_attempt(self):
        # Equal rewards (0.5 / 0.5): strict `>` keeps attempt 1's outputs
        # on BOTH arms — the docstring's tie promise. A `gt` -> `ge` slip
        # flips both arms to t2, and every OTHER scripted reward in this
        # battery is distinct, so only this test can see it.
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 2, flat_reward)
        script = answers("t1", "t2")

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = best(question="pick one")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = best.forward_native(question="pick one")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == dict(native_record)
        assert engine_prediction.answer == "t1"  # the EARLIEST tied attempt
        assert engine_prediction.best_score == 0.5
        assert engine_prediction.attempts == 2  # no early exit: both ran

    def test_exhausted_n_equivalence(self):
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 3, table_reward)
        script = answers("a1", "a2", "a3")

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = best(question="pick one")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = best.forward_native(question="pick one")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == native_record
        assert engine_prediction.attempts == 3

    def test_early_exit_equivalence(self):
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 5, table_reward, threshold=0.5)
        script = answers("a1", "a2", "a3")  # a3 must never be consumed

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = best(question="pick one")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = best.forward_native(question="pick one")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == native_record
        assert engine_prediction.answer == "a2"
        assert engine_prediction.attempts == 2
        assert len(engine_lm.calls) == 2  # the loop stopped at attempt 2 of 5

    def test_early_exit_makes_exactly_two_reward_calls(self):
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 5, table_reward, threshold=0.5)
        dspy.configure(lm=dspy.DummyLM(answers("a1", "a2", "a3")))
        executable, counter = count_reward_calls(best)

        prediction = executable(question="pick one")

        assert prediction["attempts"] == 2
        assert counter["calls"] == 2  # one reward call per consumed attempt

    def test_failed_attempts_skip_and_count(self):
        # Attempt 1 fails to parse; attempt 2 succeeds. Both arms must
        # skip the failure via the typed handler and still count it.
        # (A two-output signature: one output with no labels would
        # degenerate to full_text and parse anything.)
        best = dspy.BestOfN(dspy.Predict("question -> answer, note"), 3, table_reward)
        script = [
            "not a parseable completion",
            chat_completion(answer="a2", note="x"),
            chat_completion(answer="a1", note="y"),
        ]

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = best(question="pick one")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = best.forward_native(question="pick one")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == native_record
        assert engine_prediction.answer == "a2"
        assert engine_prediction.attempts == 3  # the failed attempt is counted

    def test_all_attempts_failed_raises_typed(self):
        best = dspy.BestOfN(dspy.Predict("question -> answer, note"), 2, table_reward)
        dspy.configure(lm=dspy.DummyLM(["nope", "still nope"]))
        with pytest.raises(ToolError, match="all 2 attempt"):
            best(question="pick one")

        dspy.configure(lm=dspy.DummyLM(["nope", "still nope"]))
        with pytest.raises(ToolError, match="all 2 attempt"):
            best.forward_native(question="pick one")

    def test_save_load_run(self, tmp_path):
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 3, table_reward)
        dspy.configure(lm=dspy.DummyLM(answers("a1", "a2", "a3")))
        direct = best(question="pick one")

        target = tmp_path / "artifact"
        best.save(target)
        loaded = dspy.load(target, bindings={"lm": {"dummy": dspy.DummyLM(answers("a1", "a2", "a3"))}})
        replayed = loaded(question="pick one")

        assert replayed.toDict() == direct.toDict()
        # The reward leaf rebuilt from its sidecar and really scored:
        # attempt 2 won again on the receiver side.
        assert replayed.answer == "a2"
        assert replayed.best_score == 0.9

    def test_edited_literals_recompile(self):
        best = dspy.BestOfN(dspy.Predict("question -> answer"), 3, table_reward)
        dspy.configure(lm=dspy.DummyLM(answers("a1", "a2", "a3")))
        assert best(question="q").attempts == 3

        best.n = 1
        dspy.configure(lm=dspy.DummyLM(answers("a1")))
        prediction = best(question="q")
        assert prediction.attempts == 1
        assert prediction.answer == "a1"
        assert best.to_manifest()["components"]["5_forward"]["self"]["body"][4]["range"] == 1


# ---------------------------------------------------------------------------
# BestOfN wraps ReAct: the macro is structure around ANY module leaf
# ---------------------------------------------------------------------------


def lookup(country: str) -> str:
    """Return the capital of a country from a fixed table."""
    capitals = {"france": "Paris", "canada": "Ottawa"}
    return capitals.get(country.lower(), "unknown")


def react_script(answer: str) -> list[str]:
    return [
        chat_completion(
            next_thought="Look it up.",
            next_tool_name="lookup",
            next_tool_args=json.dumps({"country": "france"}),
        ),
        chat_completion(next_thought="Done.", next_tool_name="finish", next_tool_args="{}"),
        chat_completion(reasoning="From the observation.", answer=answer),
    ]


class TestBestOfNWrapsReAct:
    def test_wrapped_react_runs_and_exits_early(self):
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=3)
        best = dspy.BestOfN(react, 2, capital_reward, threshold=1.0)
        lm = dspy.DummyLM(react_script("Paris"))
        dspy.configure(lm=lm)

        manifest = best.to_manifest()
        tree = manifest["components"]["1_module_tree"]
        assert tree["kind"] == "BestOfN"
        assert [child["kind"] for child in tree["children"]] == ["ReAct"]
        assert set(manifest["components"]["5_forward"]) == {"self", "inner"}
        assert set(manifest["components"]["6_tools"]) == {"lookup", "reward"}

        prediction = best(question="What is the capital of France?")
        assert prediction.answer == "Paris"
        assert prediction.best_score == 1.0
        assert prediction.attempts == 1  # threshold hit on the first attempt
        assert len(lm.calls) == 3  # one full ReAct run, no second attempt

    def test_wrapped_react_equivalence(self):
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=3)
        best = dspy.BestOfN(react, 2, capital_reward, threshold=1.0)
        script = react_script("Paris")

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = best(question="What is the capital of France?")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = best.forward_native(question="What is the capital of France?")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == dict(native_record)

    def test_wrapped_react_save_load_run(self, tmp_path):
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=3)
        best = dspy.BestOfN(react, 2, capital_reward, threshold=1.0)
        dspy.configure(lm=dspy.DummyLM(react_script("Paris")))
        direct = best(question="What is the capital of France?")

        target = tmp_path / "artifact"
        best.save(target)
        loaded = dspy.load(target, bindings={"lm": {"dummy": dspy.DummyLM(react_script("Paris"))}})
        replayed = loaded(question="What is the capital of France?")

        assert replayed.toDict() == direct.toDict()
        calls = replayed._trajectory["predictor_calls"]
        assert [call["predictor"] for call in calls] == ["inner.react", "inner.react", "inner.extract"]


# ---------------------------------------------------------------------------
# Macro tool leaves: fixed pool names, collision teaching errors
# ---------------------------------------------------------------------------


def reward(query: str) -> str:
    """A user tool that happens to wear the macro's reward leaf name."""
    return query


def describe_attempt(text: str) -> str:
    """A user tool that happens to wear the feedback leaf's name."""
    return text


class TestMacroToolLeafCollisions:
    def test_nested_macros_on_one_reward_share_one_leaf(self):
        # Refine(BestOfN) and BestOfN(BestOfN) on the SAME reward
        # function dedupe into ONE pooled reward leaf and compile clean —
        # the wrapper source carries no owner-class text.
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        outer = dspy.Refine(
            dspy.BestOfN(dspy.Predict("question -> answer"), 2, table_reward), 2, table_reward, threshold=1.0
        )
        manifest = outer.to_manifest()
        assert list(manifest["components"]["6_tools"]) == ["reward"]

        doubled = dspy.BestOfN(dspy.BestOfN(dspy.Predict("question -> answer"), 2, table_reward), 2, table_reward)
        assert list(doubled.to_manifest()["components"]["6_tools"]) == ["reward"]

    def test_nested_macros_with_different_rewards_refuse_at_construction(self):
        inner = dspy.BestOfN(dspy.Predict("question -> answer"), 2, table_reward)
        with pytest.raises(ValueError, match="already owns a different tool"):
            dspy.BestOfN(inner, 2, capital_reward)

    def test_inner_tool_named_reward_refuses_at_construction(self):
        react = dspy.ReAct("question -> answer", tools=[reward], max_iters=2)
        with pytest.raises(ValueError, match="tool leaf named 'reward'.*the inner ReAct"):
            dspy.BestOfN(react, 2, capital_reward)

    def test_collision_error_names_the_owning_submodule(self):
        class Agent(dspy.Module):
            def __init__(self):
                self.react = dspy.ReAct("question -> answer", tools=[reward], max_iters=2)

            def forward(self, question):
                result = self.react(question=question)
                return result

        with pytest.raises(ValueError, match="sub-module at path 'react'"):
            dspy.BestOfN(Agent(), 2, capital_reward)

    def test_inner_describe_attempt_refuses_under_feedback_only(self):
        # feedback=True claims the describe_attempt leaf; without
        # feedback the macro claims no such name and the tool passes.
        react = dspy.ReAct("question, hints -> answer", tools=[describe_attempt], max_iters=2)
        with pytest.raises(ValueError, match="tool leaf named 'describe_attempt'"):
            dspy.Refine(react, 2, digit_reward, threshold=1.0, feedback=True)

        dspy.configure(lm=dspy.DummyLM(["unused"]))
        plain = dspy.ReAct("question -> answer", tools=[describe_attempt], max_iters=2)
        best = dspy.BestOfN(plain, 2, digit_reward)  # no feedback claim, no collision
        assert set(best.to_manifest()["components"]["6_tools"]) == {"describe_attempt", "reward"}


# ---------------------------------------------------------------------------
# Refine: the sibling face
# ---------------------------------------------------------------------------


class TestRefine:
    def test_threshold_is_mandatory(self):
        with pytest.raises(ValueError, match="requires a threshold"):
            dspy.Refine(dspy.Predict("question -> answer"), 3, digit_reward, threshold=None)

    def test_threshold_hit_stops_the_loop(self):
        refine = dspy.Refine(dspy.Predict("question -> answer"), 5, digit_reward, threshold=1.0)
        lm = dspy.DummyLM(answers("maybe two", "2", "unreached"))
        dspy.configure(lm=lm)
        executable, counter = count_reward_calls(refine)

        prediction = executable(question="How many moons does Mars have?")

        assert prediction["answer"] == "2"
        assert prediction["attempts"] == 2
        assert len(lm.calls) == 2  # the third script entry never fired
        assert counter["calls"] == 2

    def test_feedback_threads_the_prior_attempt(self):
        refine = dspy.Refine(dspy.Predict("question, hints -> answer"), 5, digit_reward, threshold=1.0, feedback=True)
        script = answers("maybe two", "2")

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = refine(question="How many moons does Mars have?")

        # The second attempt's rendered prompt carries the first
        # attempt's output and score through the hints input.
        second_prompt = engine_lm.calls[1]["messages"][-1]["content"]
        assert 'Previous attempt (score 0.0): {"answer": "maybe two"}' in second_prompt
        # The first attempt saw empty hints.
        first_prompt = engine_lm.calls[0]["messages"][-1]["content"]
        assert "Previous attempt" not in first_prompt

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = refine.forward_native(question="How many moons does Mars have?")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == dict(native_record)
        assert engine_prediction.answer == "2"
        assert engine_prediction.attempts == 2

    def test_feedback_structure_in_manifest(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        refine = dspy.Refine(dspy.Predict("question, hints -> answer"), 3, digit_reward, threshold=1.0, feedback=True)

        manifest = refine.to_manifest()
        tree = manifest["components"]["1_module_tree"]
        assert tree["kind"] == "Refine"
        assert sorted(tree["tools"]) == ["describe_attempt", "reward"]
        forward = manifest["components"]["5_forward"]["self"]
        # hints is threaded internally: NOT a macro argument.
        assert forward["args"] == ["question"]
        assert try_handler_types(forward) == ["AdapterParseError", "ToolError"]

    def test_feedback_without_hints_refuses_at_construction(self):
        with pytest.raises(ValueError, match="declares no `hints` input"):
            dspy.Refine(dspy.Predict("question -> answer"), 3, digit_reward, threshold=1.0, feedback=True)

    def test_refine_save_load_run(self, tmp_path):
        refine = dspy.Refine(dspy.Predict("question, hints -> answer"), 5, digit_reward, threshold=1.0, feedback=True)
        dspy.configure(lm=dspy.DummyLM(answers("maybe two", "2")))
        direct = refine(question="How many moons does Mars have?")

        target = tmp_path / "artifact"
        refine.save(target)
        loaded = dspy.load(target, bindings={"lm": {"dummy": dspy.DummyLM(answers("maybe two", "2"))}})
        replayed = loaded(question="How many moons does Mars have?")

        assert replayed.toDict() == direct.toDict()
        assert replayed.attempts == 2

    def test_exhausted_without_threshold_hit_returns_best(self):
        # The exhausted branch under Refine: nothing clears the bar, the
        # best-so-far still comes back — on both arms.
        refine = dspy.Refine(dspy.Predict("question -> answer"), 2, table_reward, threshold=1.0)
        script = answers("a1", "a3")

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = refine(question="pick one")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = refine.forward_native(question="pick one")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == dict(native_record)
        assert engine_prediction.answer == "a3"  # 0.4 beats 0.2
        assert engine_prediction.best_score == 0.4
        assert engine_prediction.attempts == 2
