"""Stage A3 tests: the execution spine — the program IS the IR.

The battery the charter names:

(a) a branchy 2-predictor module runs THROUGH the engine with a DummyLM;
(b) THE EQUIVALENCE TEST — the engine path and the native-Python path,
    driven by identically scripted DummyLMs, produce identical
    predictor-call sequences (messages and kwargs) and identical
    outputs, on both branches;
(c) save -> load(bindings) -> run works and matches the direct run;
(d) an unresolved binding refuses loudly, naming itself;
(e) explain / lint / cost run over the compiled program.

Plus the spine's edges: the teaching error at first call, per-predictor
bindings (live object and table name), and the compile cache following
binding changes.
"""

import pytest

import dspy
from dspy.lm import BINDINGS
from dspy.programir.errors import ProgramIRRefusal


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


class Branchy(dspy.Module):
    """Two predictors, one branch: the route decides the solve hint."""

    def __init__(self):
        self.classify = dspy.Predict("question -> category")
        self.solve = dspy.Predict("question, hint -> answer")

    def forward(self, question):
        route = self.classify(question=question)
        if route.category == "math":
            answer = self.solve(question=question, hint="use arithmetic")
        else:
            answer = self.solve(question=question, hint="answer plainly")
        return answer


def branchy_script(category: str, answer: str) -> list[str]:
    return [chat_completion(category=category), chat_completion(answer=answer)]


# ---------------------------------------------------------------------------
# (a) The engine path runs a branchy module
# ---------------------------------------------------------------------------


class TestEngineExecution:
    def test_branchy_module_runs_through_the_engine(self):
        lm = dspy.DummyLM(branchy_script("math", "4"))
        dspy.configure(lm=lm)
        program = Branchy()

        prediction = program(question="2+2?")

        assert prediction.answer == "4"
        assert len(lm.calls) == 2
        # The run's exhaust names the predictor-call sequence.
        calls = prediction._trajectory["predictor_calls"]
        assert [call["predictor"] for call in calls] == ["classify", "solve"]
        assert calls[1]["inputs"] == {"question": "2+2?", "hint": "use arithmetic"}

    def test_the_other_branch_takes_the_other_hint(self):
        lm = dspy.DummyLM(branchy_script("history", "1789"))
        dspy.configure(lm=lm)
        program = Branchy()

        prediction = program(question="When did the French Revolution start?")

        assert prediction.answer == "1789"
        calls = prediction._trajectory["predictor_calls"]
        assert calls[1]["inputs"]["hint"] == "answer plainly"

    def test_predict_leaf_runs_directly_with_completion_exhaust(self):
        lm = dspy.DummyLM([chat_completion(answer="Paris")])
        predictor = dspy.Predict("question -> answer", lm=lm)

        prediction = predictor(question="Capital of France?")

        assert prediction.answer == "Paris"
        assert "Paris" in prediction._trajectory["completion"]

    def test_predict_refuses_unknown_inputs(self):
        predictor = dspy.Predict("question -> answer", lm=dspy.DummyLM(["x"]))
        with pytest.raises(ValueError, match="unknown input"):
            predictor(quesiton="typo")


# ---------------------------------------------------------------------------
# (b) THE EQUIVALENCE TEST
# ---------------------------------------------------------------------------


class TestEquivalence:
    @pytest.mark.parametrize(
        ("category", "answer", "hint"),
        [("math", "4", "use arithmetic"), ("poetry", "a rose", "answer plainly")],
    )
    def test_engine_and_native_paths_are_identical(self, category, answer, hint):
        question = "the question"
        program = Branchy()

        engine_lm = dspy.DummyLM(branchy_script(category, answer))
        dspy.configure(lm=engine_lm)
        engine_prediction = program(question=question)

        native_lm = dspy.DummyLM(branchy_script(category, answer))
        dspy.configure(lm=native_lm)
        native_prediction = program.forward_native(question=question)

        # Identical predictor-call sequences: same messages, same kwargs,
        # in the same order — the branch included.
        assert len(engine_lm.calls) == len(native_lm.calls) == 2
        for engine_call, native_call in zip(engine_lm.calls, native_lm.calls, strict=True):
            assert engine_call["messages"] == native_call["messages"]
            assert engine_call["kwargs"] == native_call["kwargs"]
        assert hint in engine_lm.calls[1]["messages"][-1]["content"]

        # Identical outputs.
        assert engine_prediction.toDict() == native_prediction.toDict() == {"answer": answer}

    def test_bare_predict_equivalence(self):
        predictor = dspy.Predict("question -> answer")

        from dspy.programir import materialize
        from dspy.programir._dspy import compile_with_live

        engine_lm = dspy.DummyLM([chat_completion(answer="Paris")])
        dspy.configure(lm=engine_lm)
        ir, live = compile_with_live(predictor)
        engine_prediction = materialize(ir, live)(question="Capital of France?")

        native_lm = dspy.DummyLM([chat_completion(answer="Paris")])
        dspy.configure(lm=native_lm)
        native_prediction = predictor(question="Capital of France?")

        assert engine_lm.calls[0]["messages"] == native_lm.calls[0]["messages"]
        assert engine_prediction.toDict() == native_prediction.toDict() == {"answer": "Paris"}


# ---------------------------------------------------------------------------
# (c) save -> load(bindings) -> run
# ---------------------------------------------------------------------------


class TestSaveLoad:
    def test_save_load_run_matches_the_direct_run(self, tmp_path):
        dspy.configure(lm=dspy.DummyLM(branchy_script("math", "4")))
        program = Branchy()
        direct_prediction = program(question="2+2?")

        target = tmp_path / "artifact"
        program.save(target)
        assert (target / "manifest.json").is_file()

        loaded = dspy.load(target, bindings={"lm": {"dummy": dspy.DummyLM(branchy_script("math", "4"))}})
        loaded_prediction = loaded(question="2+2?")

        assert loaded_prediction.toDict() == direct_prediction.toDict() == {"answer": "4"}
        assert [call["predictor"] for call in loaded_prediction._trajectory["predictor_calls"]] == [
            "classify",
            "solve",
        ]

    def test_saved_artifact_reads_and_links(self, tmp_path):
        from dspy.programir import link, read

        dspy.configure(lm=dspy.DummyLM(["unused"]))
        program = Branchy()
        program.save(tmp_path / "artifact")

        ir = read(tmp_path / "artifact")
        table = link(ir)
        assert set(table) == {"classify", "solve"}
        assert table["classify"] == {"adapter": "chat", "lm": "dummy", "delta": None}


# ---------------------------------------------------------------------------
# (d) Unresolved bindings refuse loudly
# ---------------------------------------------------------------------------


class TestLoudRefusals:
    def test_load_without_lm_binding_names_the_entry(self, tmp_path):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        program = Branchy()
        program.save(tmp_path / "artifact")

        with pytest.raises(ValueError, match="LM pool entry 'dummy'"):
            dspy.load(tmp_path / "artifact")

    def test_load_with_a_dangling_binding_name_refuses(self, tmp_path):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        program = Branchy()
        program.save(tmp_path / "artifact")

        with pytest.raises(ValueError, match="name no pool entry"):
            dspy.load(tmp_path / "artifact", bindings={"lm": {"wrong-name": dspy.DummyLM(["x"])}})

    def test_call_without_any_lm_binding_refuses(self):
        program = Branchy()
        with pytest.raises(dspy.BindingError, match="No 'lm' binding is configured"):
            program(question="anything")

    def test_predictor_bound_to_a_missing_table_name_refuses(self):
        predictor = dspy.Predict("question -> answer", lm="fast_lm")
        with pytest.raises(dspy.BindingError, match="'fast_lm'"):
            predictor(question="anything")

    def test_forward_outside_the_subset_teaches_at_first_call(self):
        class Reaches(dspy.Module):
            def __init__(self):
                self.threshold = 2
                self.answer = dspy.Predict("question -> answer")

            def forward(self, question):
                if len(question) > self.threshold:
                    result = self.answer(question=question)
                    return result
                result = self.answer(question=question)
                return result

        dspy.configure(lm=dspy.DummyLM(["unused"]))
        program = Reaches()
        with pytest.raises(ProgramIRRefusal, match="zero-reach-back"):
            program(question="hello")
        with pytest.raises(ProgramIRRefusal, match="zero-reach-back"):
            program.compile_ir()


# ---------------------------------------------------------------------------
# Per-predictor bindings and the compile cache
# ---------------------------------------------------------------------------


class TestBindings:
    def test_per_predictor_lm_beats_the_table(self):
        table_lm = dspy.DummyLM([chat_completion(answer="from-table")])
        own_lm = dspy.DummyLM([chat_completion(answer="from-own")])
        dspy.configure(lm=table_lm)

        predictor = dspy.Predict("question -> answer")
        predictor.set_lm(own_lm)

        assert predictor(question="who answers?").answer == "from-own"
        assert table_lm.calls == []

    def test_a_string_binding_resolves_through_the_table(self):
        fast = dspy.DummyLM([chat_completion(answer="fast")])
        dspy.configure(lm=dspy.DummyLM(["unused"]), fast_lm=fast)

        predictor = dspy.Predict("question -> answer", lm="fast_lm")

        assert predictor(question="which lm?").answer == "fast"

    def test_reconfiguring_the_table_reaches_the_engine_path(self):
        program = Branchy()
        first = dspy.DummyLM(branchy_script("math", "4"))
        dspy.configure(lm=first)
        assert program(question="2+2?").answer == "4"

        second = dspy.DummyLM(branchy_script("math", "9"))
        dspy.configure(lm=second)
        assert program(question="3+6?").answer == "9"
        assert len(first.calls) == 2
        assert len(second.calls) == 2

    def test_the_compiled_program_is_cached_between_calls(self):
        program = Branchy()
        lm = dspy.DummyLM(branchy_script("math", "4") + branchy_script("math", "9"))
        dspy.configure(lm=lm)
        program(question="2+2?")
        executable = program._ir_cache[1]
        program(question="3+6?")
        assert program._ir_cache[1] is executable

    def test_new_demos_invalidate_the_cache(self):
        program = Branchy()
        lm = dspy.DummyLM(branchy_script("math", "4") + branchy_script("math", "9"))
        dspy.configure(lm=lm)
        program(question="2+2?")
        executable = program._ir_cache[1]

        program.classify.demos.append(dspy.Example(question="1+1?", category="math").with_inputs("question"))
        prediction = program(question="3+6?")
        assert program._ir_cache[1] is not executable
        assert prediction.answer == "9"
        # The demo reached the recompiled prompt.
        assert "1+1?" in lm.calls[2]["messages"][1]["content"]


# ---------------------------------------------------------------------------
# (e) explain / lint / cost / diff over the compiled program
# ---------------------------------------------------------------------------


class TestObservability:
    def test_explain_renders_the_manifest_views(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        text = Branchy().explain()
        assert "ProgramIR explain" in text
        assert "5_forward" in text
        assert "classify" in text and "solve" in text

    def test_lint_is_quiet_on_a_clean_program_and_names_dead_branches(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        assert Branchy().lint() == []

        class DeadBranch(dspy.Module):
            def __init__(self):
                self.answer = dspy.Predict("question -> answer")

            def forward(self, question):
                if "a" == "b":
                    result = self.answer(question=question)
                    return result
                result = self.answer(question=question)
                return result

        findings = DeadBranch().lint()
        assert any(finding.code == "PIR-L-DEADBRANCH" for finding in findings)

    def test_cost_bounds_the_branchy_program(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        estimate = Branchy().cost()
        assert estimate["total_calls"].minimum == 2.0
        assert estimate["total_calls"].maximum == 2.0

    def test_diff_names_an_instruction_change(self):
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        before = Branchy()
        after = Branchy()
        after.solve.signature = after.solve.signature.with_instructions("Solve it very carefully.")

        assert dspy.diff(before, before) == []
        lines = "\n".join(dspy.diff(before, after))
        assert "solve" in lines
        assert "very carefully" in lines
