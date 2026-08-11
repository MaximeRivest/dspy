"""Stage A4 tests: the module library, subset-clean by construction.

Per module: the compile-clean assert (the forward lowers into node-set
v0.3, no refusals), an engine run with a scripted DummyLM, and for ReAct
and RLM the equivalence test — engine path and native path, identically
scripted, byte-identical LM calls and identical outputs. ReAct also
proves save -> load(bindings) -> run.

Plus the mechanism these modules stand on: the declared-literal loop cap
(`ir_literals`) bakes into the artifact and refreshes on edit.
"""

import json

import pytest

import dspy
from dspy.core.errors import InterpreterError, ToolError
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


def lookup(country: str) -> str:
    """Return the capital of a country from a fixed table."""
    capitals = {"france": "Paris", "canada": "Ottawa"}
    return capitals.get(country.lower(), "unknown")


def react_step(thought: str, tool: str, args: dict) -> str:
    return chat_completion(
        next_thought=thought,
        next_tool_name=tool,
        next_tool_args=json.dumps(args),
    )


# ---------------------------------------------------------------------------
# ChainOfThought: a leaf with the reasoning role engaged
# ---------------------------------------------------------------------------


class TestChainOfThought:
    def test_cot_is_a_leaf_and_compiles_clean(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        cot = dspy.ChainOfThought("question -> answer")

        manifest = cot.to_manifest()
        tree = manifest["components"]["1_module_tree"]
        assert tree["kind"] == "Predict"  # a leaf, not a module node
        assert tree["children"] == []
        assert set(manifest["components"]["5_forward"]) == {"self"}

        fields = {record["name"]: record for record in manifest["components"]["2_signature"]["self"]["fields"]}
        assert list(fields) == ["question", "reasoning", "answer"]
        assert fields["reasoning"]["semantic_role"] == "reasoning"

    def test_cot_answers_with_reasoning_via_prefix_cot(self):
        # DummyLM declares instruct=True, so the auto binding resolves to
        # prefix_cot: the fragment rides the user message, the field parses.
        lm = dspy.DummyLM([chat_completion(reasoning="17 * 4 = 68.", answer="68")])
        cot = dspy.ChainOfThought("question -> answer", lm=lm)

        prediction = cot(question="What is 17 * 4?")

        assert prediction.answer == "68"
        assert prediction.reasoning == "17 * 4 = 68."
        assert "Reason step by step in the `reasoning` field" in lm.calls[0]["messages"][-1]["content"]

    def test_cot_keeps_a_declared_reasoning_field(self):
        cot = dspy.ChainOfThought("question -> reasoning, answer")
        assert list(cot.signature.output_fields) == ["reasoning", "answer"]

    def test_cot_strategies_passthrough(self):
        cot = dspy.ChainOfThought("question -> answer", strategies={"reasoning": "prefix_cot"})
        assert cot.resolve_adapter().strategies["reasoning"] == "prefix_cot"

    def test_cot_engine_equivalence(self):
        # The role must survive the manifest round trip: the rebuilt
        # predictor renders the same fragment bytes.
        from dspy.programir import materialize
        from dspy.programir._dspy import compile_with_live

        cot = dspy.ChainOfThought("question -> answer")
        script = [chat_completion(reasoning="Because.", answer="Blue.")]

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        ir, live = compile_with_live(cot)
        engine_prediction = materialize(ir, live)(question="Why is the sky blue?")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_prediction = cot(question="Why is the sky blue?")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == native_prediction.toDict()
        assert engine_prediction.reasoning == "Because."


# ---------------------------------------------------------------------------
# ReAct
# ---------------------------------------------------------------------------


def react_script(answer: str) -> list[str]:
    """One tool call, one finish, one extract."""
    return [
        react_step("I should look up the capital.", "lookup", {"country": "france"}),
        react_step("The trajectory has what I need.", "finish", {}),
        chat_completion(reasoning="The observation names the capital.", answer=answer),
    ]


class TestReAct:
    def test_react_compiles_clean_with_literal_cap_and_tool_leaves(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=5)

        manifest = react.to_manifest()
        forward = manifest["components"]["5_forward"]["self"]
        assert forward["args"] == ["question"]
        loop = forward["body"][1]
        assert loop["node"] == "For"
        assert loop["range"] == 5  # the cap is a LITERAL in the artifact

        assert set(manifest["components"]["6_tools"]) == {"lookup"}
        assert manifest["components"]["6_tools"]["lookup"]["language"] == "python"
        assert {"react", "extract"} <= set(manifest["components"]["2_signature"])

    def test_react_engine_run_tool_call_then_finish(self):
        lm = dspy.DummyLM(react_script("Paris"))
        dspy.configure(lm=lm)
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=5)

        prediction = react(question="What is the capital of France?")

        assert prediction.answer == "Paris"
        assert len(lm.calls) == 3
        calls = prediction._trajectory["predictor_calls"]
        assert [call["predictor"] for call in calls] == ["react", "react", "extract"]
        # The tool's observation reached the second react step's prompt.
        assert '"observation_0": "Paris"' in lm.calls[1]["messages"][-1]["content"]
        # And the extract prompt carries the whole trajectory.
        assert '"tool_name_1": "finish"' in lm.calls[2]["messages"][-1]["content"]

    def test_react_tool_failure_is_a_typed_observation(self):
        script = [
            react_step("Try bad args.", "lookup", {"nation": "france"}),  # wrong kwarg -> ToolError
            react_step("Try an unknown tool.", "search", {"q": "france"}),  # unknown -> ToolError
            react_step("Give up gathering.", "finish", {}),
            chat_completion(reasoning="Nothing was gathered.", answer="unknown"),
        ]
        lm = dspy.DummyLM(script)
        dspy.configure(lm=lm)
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=5)

        prediction = react(question="What is the capital of France?")

        assert prediction.answer == "unknown"
        assert "The tool call failed" in lm.calls[1]["messages"][-1]["content"]
        assert "The tool call failed" in lm.calls[2]["messages"][-1]["content"]

    def test_react_equivalence(self):
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=5)

        engine_lm = dspy.DummyLM(react_script("Paris"))
        dspy.configure(lm=engine_lm)
        engine_prediction = react(question="What is the capital of France?")

        native_lm = dspy.DummyLM(react_script("Paris"))
        dspy.configure(lm=native_lm)
        native_prediction = react.forward_native(question="What is the capital of France?")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == native_prediction.toDict()
        assert engine_prediction.answer == "Paris"

    def test_react_save_load_run(self, tmp_path):
        dspy.configure(lm=dspy.DummyLM(react_script("Paris")))
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=5)
        direct = react(question="What is the capital of France?")

        target = tmp_path / "artifact"
        react.save(target)
        assert (target / "manifest.json").is_file()

        loaded = dspy.load(target, bindings={"lm": {"dummy": dspy.DummyLM(react_script("Paris"))}})
        replayed = loaded(question="What is the capital of France?")

        assert replayed.toDict() == direct.toDict()
        # The tool leaf rebuilt from its source sidecar and really ran:
        # the observation in the replayed trajectory came from executing it.
        calls = replayed._trajectory["predictor_calls"]
        assert [call["predictor"] for call in calls] == ["react", "react", "extract"]
        assert calls[1]["inputs"]["trajectory"]["observation_0"] == "Paris"

    def test_react_cap_is_baked_and_refreshes_on_edit(self):
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=2)

        # Two tool rounds, never finish: the literal cap ends the loop.
        lm = dspy.DummyLM(
            [
                react_step("Round one.", "lookup", {"country": "france"}),
                react_step("Round two.", "lookup", {"country": "canada"}),
                chat_completion(reasoning="Cap reached.", answer="Paris"),
            ]
        )
        dspy.configure(lm=lm)
        assert react(question="capital?").answer == "Paris"
        assert len(lm.calls) == 3

        # Editing the declared literal recompiles: one round only now.
        react.max_iters = 1
        lm = dspy.DummyLM(
            [
                react_step("Only round.", "lookup", {"country": "france"}),
                chat_completion(reasoning="Cap reached.", answer="Paris"),
            ]
        )
        dspy.configure(lm=lm)
        assert react(question="capital?").answer == "Paris"
        assert len(lm.calls) == 2
        assert react.to_manifest()["components"]["5_forward"]["self"]["body"][1]["range"] == 1

    def test_react_refusals_are_loud(self):
        with pytest.raises(ValueError, match="at least one tool"):
            dspy.ReAct("question -> answer", tools=[])
        with pytest.raises(ValueError, match="reserves the field name"):
            dspy.ReAct("question, trajectory -> answer", tools=[lookup])
        with pytest.raises(ValueError, match="reserves the tool name"):

            def finish() -> str:
                """Collide with the termination move."""
                return "done"

            dspy.ReAct("question -> answer", tools=[lookup, finish])


# ---------------------------------------------------------------------------
# ReActV2
# ---------------------------------------------------------------------------


def v2_step(thought: str, calls: list[dict]) -> str:
    return chat_completion(next_thought=thought, tool_calls=json.dumps(calls))


class TestReActV2:
    def test_v2_compiles_clean_and_runs_submit(self):
        agent = dspy.ReActV2("question -> answer", tools=[lookup], max_iters=4)
        lm = dspy.DummyLM(
            [
                v2_step("Look it up.", [{"name": "lookup", "args": {"country": "france"}}]),
                v2_step("Submit.", [{"name": "submit", "args": {"answer": "Paris"}}]),
            ]
        )
        dspy.configure(lm=lm)

        manifest = agent.to_manifest()
        assert set(manifest["components"]["6_tools"]) == {"lookup"}

        prediction = agent(question="What is the capital of France?")
        assert prediction.answer == "Paris"
        assert prediction.termination_reason == "submit"
        assert len(prediction.history) == 2
        assert prediction.history[0]["results"] == [{"name": "lookup", "value": "Paris"}]

    def test_v2_empty_batch_terminates(self):
        agent = dspy.ReActV2("question -> answer", max_iters=3)
        dspy.configure(lm=dspy.DummyLM([v2_step("Nothing to do.", [])]))

        prediction = agent(question="anything")
        assert prediction.termination_reason == "empty_tool_calls"
        assert prediction.answer == ""
        assert prediction.history == []

    def test_v2_equivalence(self):
        agent = dspy.ReActV2("question -> answer", tools=[lookup], max_iters=4)
        script = [
            v2_step("Look it up.", [{"name": "lookup", "args": {"country": "france"}}]),
            v2_step("Submit.", [{"name": "submit", "args": {"answer": "Paris"}}]),
        ]

        engine_lm = dspy.DummyLM(script)
        dspy.configure(lm=engine_lm)
        engine_prediction = agent(question="capital?")

        native_lm = dspy.DummyLM(script)
        dspy.configure(lm=native_lm)
        native_record = agent.forward_native(question="capital?")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == native_record


# ---------------------------------------------------------------------------
# RLM
# ---------------------------------------------------------------------------


def rlm_script(answer: str) -> list[str]:
    """One failing interpreter round, one success, one extract."""
    return [
        chat_completion(code="result = undefined_name"),
        chat_completion(code="result = 6 * 7"),
        chat_completion(answer=answer),
    ]


class TestRLM:
    def test_rlm_compiles_clean_with_a_declared_interpreter_leaf(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        rlm = dspy.RLM("question -> answer", max_iters=3)

        manifest = rlm.to_manifest()
        assert set(manifest["components"]["7_interpreter"]) == {"interp"}
        profile = manifest["components"]["7_interpreter"]["interp"]
        assert profile["language"] == "python"
        assert profile["placement"]["rung"] == "in_process"
        assert manifest["components"]["1_module_tree"]["uses_interpreter"] is True

        forward = manifest["components"]["5_forward"]["self"]
        assert forward["args"] == ["question"]
        assert any(node["node"] == "While" for node in forward["body"])

    def test_rlm_engine_run_with_interpreter_rounds(self):
        lm = dspy.DummyLM(rlm_script("42"))
        dspy.configure(lm=lm)
        rlm = dspy.RLM("question -> answer", max_iters=3)

        prediction = rlm(question="What is 6 * 7?")

        assert prediction.answer == "42"
        calls = prediction._trajectory["predictor_calls"]
        assert [call["predictor"] for call in calls] == ["gen", "gen", "extract"]
        # The failed round left its typed-error note in the transcript.
        second_prompt = lm.calls[1]["messages"][-1]["content"]
        assert "(error: the code failed; fix it and try again)" in second_prompt
        # The successful round's output reached the extract prompt.
        assert "42" in lm.calls[2]["messages"][-1]["content"]

    def test_rlm_equivalence(self):
        rlm = dspy.RLM("question -> answer", max_iters=3)

        engine_lm = dspy.DummyLM(rlm_script("42"))
        dspy.configure(lm=engine_lm)
        engine_prediction = rlm(question="What is 6 * 7?")

        native_lm = dspy.DummyLM(rlm_script("42"))
        dspy.configure(lm=native_lm)
        native_prediction = rlm.forward_native(question="What is 6 * 7?")

        assert_identical_calls(engine_lm, native_lm)
        assert engine_prediction.toDict() == native_prediction.toDict()

    def test_rlm_cap_ends_a_run_that_never_computes(self):
        rlm = dspy.RLM("question -> answer", max_iters=1)
        lm = dspy.DummyLM(
            [
                chat_completion(code="oops ="),  # syntax error round
                chat_completion(answer="no result"),
            ]
        )
        dspy.configure(lm=lm)

        prediction = rlm(question="anything")
        assert prediction.answer == "no result"
        # extract saw the empty result channel.
        assert "[[ ## result ## ]]" in lm.calls[1]["messages"][-1]["content"]

    def test_rlm_interpreter_contract(self):
        interpreter = dspy.InProcessInterpreter()
        assert interpreter(code="result = 2 + 3") == "5"
        with pytest.raises(InterpreterError, match="assign the final value"):
            interpreter(code="x = 1")
        with pytest.raises(InterpreterError, match="code execution failed"):
            interpreter(code="result = open('x')")  # no builtins in the namespace


# ---------------------------------------------------------------------------
# The dispatch wrapper and the tool table (the module library's floor)
# ---------------------------------------------------------------------------


class TestDispatchMachinery:
    def test_unknown_tool_raises_the_typed_error_natively(self):
        react = dspy.ReAct("question -> answer", tools=[lookup])
        with pytest.raises(ToolError, match="unknown tool"):
            react.tools["nope"](args={})

    def test_dispatch_wrapper_converts_failures_to_tool_error(self):
        react = dspy.ReAct("question -> answer", tools=[lookup])
        assert react.tools["lookup"](args={"country": "France"}) == "Paris"
        with pytest.raises(ToolError, match="Execution error in lookup"):
            react.tools["lookup"](args={"nation": "France"})

    def test_lambda_tools_refuse_loudly(self):
        with pytest.raises(ValueError, match="plain named function"):
            dspy.ReAct("question -> answer", tools=[dspy.Tool(lambda x: x, name="echo")])
