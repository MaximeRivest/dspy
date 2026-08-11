"""Stage A8 tests: FlexIR, the reflection optimizer.

The loop under test: render the current program (explain view + score +
failing examples + prior refusals) for a scripted reflection DummyLM,
parse its closed-vocabulary edit proposals, refuse the bad ones LOUDLY
(recorded, and fed back into the next reflection call), apply the valid
ones to a candidate, keep iff strictly better, checkpoint every accepted
candidate as a loadable artifact, and record a dspy.diff-renderable
trajectory. Deterministic end to end.
"""

import json

import pytest

import dspy
from dspy import optim
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


def reflection_reply(ops: list) -> str:
    """One scripted reflection completion carrying an edit-proposal array."""
    return chat_completion(proposals=json.dumps(ops))


CAPITALS = {"France": "Paris", "Canada": "Ottawa"}

#: The task LM answers correctly ONLY under these exact instructions.
MAGIC = "Answer with the exact capital city name."
#: And this instruction edit makes it answer nonsense — the regression.
REGRESS = "Answer in French."


def task_pilot(messages):
    """A deterministic task LM keyed off the rendered prompt content."""
    rendered = "\n".join(str(message["content"]) for message in messages)
    country = next((c for c in CAPITALS if f"capital of {c}?" in rendered), None)
    if REGRESS in rendered:
        return chat_completion(answer="je ne sais pas")
    if MAGIC in rendered and country:
        return chat_completion(answer=CAPITALS[country])
    return chat_completion(answer="unknown")


def exact_answer(example, prediction):
    return example.answer == prediction.answer


def capital_reward(outputs) -> float:
    """Score 1.0 for a known capital answer."""
    return 1.0 if outputs["answer"] in ("Paris", "Ottawa") else 0.0


def make_example(country: str) -> dspy.Example:
    return dspy.Example(question=f"What is the capital of {country}?", answer=CAPITALS[country]).with_inputs("question")


def devset_of(*countries: str) -> list[dspy.Example]:
    return [make_example(country) for country in countries]


class Solo(dspy.Module):
    """A one-predictor composite: the wrap target lives at path `solver`."""

    def __init__(self):
        self.solver = dspy.Predict("question -> answer")

    def forward(self, question):
        result = self.solver(question=question)
        return result


# ---------------------------------------------------------------------------
# (i) A valid improving edit is kept
# ---------------------------------------------------------------------------


class TestImprovingEdit:
    def run_once(self):
        program = dspy.Predict("question -> answer")
        reflection = dspy.DummyLM([reflection_reply([{"op": "set_instructions", "path": "self", "text": MAGIC}])])
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1)
        dspy.configure(lm=dspy.DummyLM(task_pilot))
        optimizer.compile(program, trainset=devset_of("France", "Canada"))
        return program, optimizer, reflection

    def test_improving_instruction_edit_is_kept(self):
        program, optimizer, reflection = self.run_once()

        assert program.signature.instructions == MAGIC
        baseline, first = optimizer.trajectory
        assert (baseline["label"], baseline["score"], baseline["accepted"]) == ("baseline", 0.0, True)
        assert (first["label"], first["score"], first["accepted"]) == ("iteration-0", 1.0, True)
        assert first["applied"] == [{"op": "set_instructions", "path": "self", "text": MAGIC}]
        assert first["refusals"] == []

        # The reflection LM saw the report: explain view, score, and the
        # baseline's failing examples with inputs/expected/got.
        report = reflection.calls[0]["messages"][-1]["content"]
        assert "== current program ==" in report
        assert "0.0 over 2 example(s)" in report
        assert '"question": "What is the capital of France?"' in report
        assert '"answer": "Paris"' in report  # expected
        assert '"answer": "unknown"' in report  # got

    def test_trajectory_is_diff_renderable(self):
        _, optimizer, _ = self.run_once()
        baseline, first = optimizer.trajectory
        lines = dspy.diff(baseline["manifest"], first["manifest"])
        assert any("instructions (3a)" in line for line in lines)

    def test_determinism_under_the_same_scripts(self):
        def shape(optimizer):
            return [
                (entry["label"], entry["score"], entry["accepted"], tuple(entry["refusals"]))
                for entry in optimizer.trajectory
            ]

        _, first_run, _ = self.run_once()
        _, second_run, _ = self.run_once()
        assert shape(first_run) == shape(second_run)


# ---------------------------------------------------------------------------
# (ii) Invalid proposals: refused loudly, recorded, fed back
# ---------------------------------------------------------------------------


class TestRefusalFeedbackLoop:
    def test_invalid_op_and_bad_path_are_refused_and_fed_back(self):
        program = dspy.Predict("question -> answer")
        before = program.signature.instructions
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {"op": "improve_vibes", "path": "self"},
                        {"op": "set_instructions", "path": "nope", "text": "x"},
                    ]
                ),
                reflection_reply([]),
            ]
        )
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=2)
        task_lm = dspy.DummyLM(task_pilot)
        dspy.configure(lm=task_lm)

        optimizer.compile(program, trainset=devset_of("France"))

        entry = optimizer.trajectory[1]
        assert len(entry["refusals"]) == 2
        assert "unknown op 'improve_vibes'" in entry["refusals"][0]
        assert "closed vocabulary" in entry["refusals"][0]
        assert "no predictor at path 'nope'" in entry["refusals"][1]
        # NEVER partially applied: nothing ran, nothing changed.
        assert entry["applied"] == []
        assert entry["score"] is None
        assert entry["accepted"] is False
        assert program.signature.instructions == before

        # The refusals are IN the next reflection call's rendered input.
        second_report = reflection.calls[1]["messages"][-1]["content"]
        assert "improve_vibes" in second_report
        assert "no predictor at path 'nope'" in second_report
        # And the first report had none.
        first_report = reflection.calls[0]["messages"][-1]["content"]
        assert "== refused proposals from your previous reply ==\n(none)" in first_report

        # Only the baseline evaluation ever ran the task LM.
        assert len(task_lm.calls) == 1

    def test_bad_shape_and_bad_index_refuse(self):
        program = dspy.Predict("question -> answer")
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {"op": "set_instructions", "path": "self"},  # missing text
                        {"op": "add_demo", "path": "self", "inputs": {"q": "x"}, "labels": {"answer": "y"}},
                        {"op": "remove_demo", "path": "self", "index": 0},
                        {"op": "wrap_best_of_n", "path": "solver", "N": 2},  # no reward configured
                        "not an object",
                    ]
                )
            ]
        )
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1)
        dspy.configure(lm=dspy.DummyLM(task_pilot))

        optimizer.compile(program, trainset=devset_of("France"))

        refusals = optimizer.trajectory[1]["refusals"]
        assert len(refusals) == 5
        assert "missing ['text']" in refusals[0]
        assert "unknown field(s) ['q']" in refusals[1]
        assert "remove_demo index 0 is invalid" in refusals[2]
        assert "needs FlexIR(reward=...)" in refusals[3]
        assert "a proposal is a JSON object" in refusals[4]
        assert optimizer.trajectory[1]["applied"] == []

    def test_non_string_paths_refuse_and_valid_siblings_apply(self):
        # A JSON array/object/number where the path string belongs is a
        # REFUSAL, never a TypeError out of compile — and the valid
        # sibling in the same batch still applies and scores.
        program = Solo()
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {"op": "set_instructions", "path": "solver", "text": MAGIC},
                        {"op": "set_instructions", "path": ["solver"], "text": "x"},
                        {
                            "op": "add_demo",
                            "path": {"p": "solver"},
                            "inputs": {"question": "q"},
                            "labels": {"answer": "a"},
                        },
                        {"op": "remove_demo", "path": 3, "index": 0},
                    ]
                )
            ]
        )
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1)
        dspy.configure(lm=dspy.DummyLM(task_pilot))

        optimizer.compile(program, trainset=devset_of("France"))

        entry = optimizer.trajectory[1]
        assert len(entry["refusals"]) == 3
        for refusal, got in zip(entry["refusals"], ["list", "dict", "int"], strict=True):
            assert "path must be a STRING predictor path" in refusal
            assert got in refusal
        assert entry["applied"] == [{"op": "set_instructions", "path": "solver", "text": MAGIC}]
        assert entry["score"] == 1.0
        assert program.solver.signature.instructions == MAGIC


class TestMalformedReflectionReply:
    @pytest.mark.parametrize(
        "reply",
        [
            chat_completion(proposals=json.dumps({"op": "set_instructions"})),  # a JSON object
            chat_completion(proposals=json.dumps("hello")),  # a quoted string
            chat_completion(proposals=""),  # an empty value
            "bare prose with no proposals marker at all",  # no field marker
        ],
        ids=["object", "string", "empty", "no-marker"],
    )
    def test_unparseable_reply_refuses_and_the_loop_continues(self, reply):
        # A reply whose `proposals` field does not parse as a JSON array
        # used to raise AdapterParseError out of compile, killing the
        # loop. The contract is refuse-loudly-and-feed-back: record the
        # refusal, show it to the next reflection call, keep going.
        program = dspy.Predict("question -> answer")
        reflection = dspy.DummyLM(
            [reply, reflection_reply([{"op": "set_instructions", "path": "self", "text": MAGIC}])]
        )
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=2)
        dspy.configure(lm=dspy.DummyLM(task_pilot))

        optimizer.compile(program, trainset=devset_of("France"))

        first = optimizer.trajectory[1]
        assert len(first["refusals"]) == 1
        assert "could not parse `proposals` as a JSON array" in first["refusals"][0]
        assert first["applied"] == []
        assert first["score"] is None  # nothing applied, nothing evaluated
        # The refusal is IN the next reflection call's rendered input...
        second_report = reflection.calls[1]["messages"][-1]["content"]
        assert "could not parse `proposals`" in second_report
        # ...and the loop stayed alive: iteration 2's valid edit landed.
        assert program.signature.instructions == MAGIC
        assert optimizer.trajectory[2]["accepted"] is True


class TestDemoValueValidation:
    def test_unrenderable_demo_values_refuse_and_the_loop_survives(self):
        # Well-formed shape, poison VALUES (a list holding null): the
        # dry-run render converts what used to be a raw TypeError deep
        # in the adapter — killing compile and stranding the demo — into
        # a recorded refusal; the valid sibling still applies, scores,
        # and reverts cleanly (0.0 is not strictly better).
        program = Solo()
        poison = {
            "op": "add_demo",
            "path": "solver",
            "inputs": {"question": {"nested": ["deep", {"x": 1}]}},
            "labels": {"answer": ["Paris", None, 3.5]},
        }
        valid = {
            "op": "add_demo",
            "path": "solver",
            "inputs": {"question": "What is the capital of France?"},
            "labels": {"answer": "Paris"},
        }
        reflection = dspy.DummyLM([reflection_reply([poison, valid])])
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1)
        dspy.configure(lm=dspy.DummyLM(task_pilot))

        optimizer.compile(program, trainset=devset_of("France"))

        entry = optimizer.trajectory[1]
        assert len(entry["refusals"]) == 1
        assert "add_demo values do not render" in entry["refusals"][0]
        assert entry["applied"] == [valid]
        assert entry["score"] == 0.0  # the candidate really evaluated — no crash
        assert entry["accepted"] is False
        assert program.solver.demos == []  # nothing stranded: poison refused, valid reverted


class TestExceptionSafeUnwind:
    def test_evaluate_crash_reverts_applied_edits(self):
        # A NON-catchable crash inside the candidate evaluation (an engine
        # error, not a metric raise — metric raises now score 0.0 and
        # continue, brief 1.f) kills compile loudly. The applied instruction
        # edit AND the structural wrap must both be unwound: an exception
        # path may not strand unscored candidate state. Here the task LM
        # raises RuntimeError the moment the poisoned instruction reaches it,
        # so the crash originates on the candidate run, not the baseline.
        program = Solo()
        before = program.solver.signature.instructions
        original_solver = program.solver
        poison = "POISONED UNSCORED STATE"

        def exploding_task(messages):
            rendered = "\n".join(str(message["content"]) for message in messages)
            if poison in rendered:
                raise RuntimeError("engine exploded on the poisoned candidate")
            return task_pilot(messages)

        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {"op": "set_instructions", "path": "solver", "text": poison},
                        {"op": "wrap_best_of_n", "path": "solver", "N": 2},
                    ]
                )
            ]
        )
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1, reward=capital_reward)
        dspy.configure(lm=dspy.DummyLM(exploding_task))

        with pytest.raises(RuntimeError, match="engine exploded"):
            optimizer.compile(program, trainset=devset_of("France"))

        assert program.solver is original_solver  # the wrap is unwound
        assert program.solver.signature.instructions == before  # no poisoned text
        tree = program.to_manifest()["components"]["1_module_tree"]
        assert [child["kind"] for child in tree["children"]] == ["Predict"]


# ---------------------------------------------------------------------------
# (iii) A valid but regressing edit: applied to the candidate, then rejected
# ---------------------------------------------------------------------------


class TestRegressingEdit:
    def test_regressing_edit_is_scored_and_reverted(self):
        program = dspy.Predict("question -> answer")
        program.signature = program.signature.with_instructions(MAGIC)  # start at 1.0
        proposal = {"op": "set_instructions", "path": "self", "text": REGRESS}
        reflection = dspy.DummyLM([reflection_reply([proposal])])
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1)
        dspy.configure(lm=dspy.DummyLM(task_pilot))

        optimizer.compile(program, trainset=devset_of("France", "Canada"))

        entry = optimizer.trajectory[1]
        # The edit WAS applied to a candidate and really scored...
        assert entry["applied"] == [proposal]
        assert entry["score"] == 0.0
        # ...and rejected by the strict-improvement rule: absent from the final.
        assert entry["accepted"] is False
        assert entry["manifest"] is None
        assert program.signature.instructions == MAGIC
        assert program.to_manifest()["components"]["3a_instructions"]["self"] == MAGIC
        assert optimizer.trajectory[0]["score"] == 1.0
        assert entry["best_score"] == 1.0


# ---------------------------------------------------------------------------
# (iv) wrap_best_of_n: a structural edit through the builder
# ---------------------------------------------------------------------------


class TestWrapBestOfN:
    def run_wrap(self, tmp_path):
        program = Solo()
        devset = devset_of("France")
        # Positional task script: baseline (wrong), then the wrapped
        # candidate's two attempts (wrong, right) — BestOfN's reward
        # picks the right one, so the wrap IMPROVES the score.
        task_lm = dspy.DummyLM(
            [
                chat_completion(answer="unknown"),
                chat_completion(answer="unknown"),
                chat_completion(answer="Paris"),
            ]
        )
        dspy.configure(lm=task_lm)
        reflection = dspy.DummyLM([reflection_reply([{"op": "wrap_best_of_n", "path": "solver", "N": 2}])])
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1, reward=capital_reward)
        optimizer.compile(program, trainset=devset, checkpoint_dir=tmp_path / "run")
        return program, optimizer

    def test_wrap_lands_in_the_manifest_and_improves(self, tmp_path):
        program, optimizer = self.run_wrap(tmp_path)

        assert isinstance(program.solver, dspy.BestOfN)
        manifest = program.to_manifest()  # compiles clean
        solver_node = manifest["components"]["1_module_tree"]["children"][0]
        assert solver_node["kind"] == "BestOfN"
        assert solver_node["name"] == "solver"
        assert [child["kind"] for child in solver_node["children"]] == ["Predict"]
        assert solver_node["tools"] == ["reward"]
        # The macro's forward is a pooled entry with the literal cap.
        assert manifest["components"]["5_forward"]["solver"]["body"][4]["range"] == 2
        assert "reward" in manifest["components"]["6_tools"]

        baseline, first = optimizer.trajectory
        assert (baseline["score"], first["score"]) == (0.0, 1.0)
        assert first["accepted"] is True
        assert first["manifest"] is not None
        # The trajectory diff names the structural change: the solver
        # leaf became a module with its own forward and reward tool.
        lines = dspy.diff(baseline["manifest"], first["manifest"])
        assert any("predictor 'solver.inner' added" in line for line in lines)
        assert any("forward 'solver' added" in line for line in lines)
        assert any("tool 'reward' added" in line for line in lines)

    def test_wrap_checkpoints_load_and_run(self, tmp_path):
        self.run_wrap(tmp_path)

        scores = json.loads((tmp_path / "run" / "scores.json").read_text())
        assert [record["candidate"] for record in scores] == ["candidate-000", "candidate-001"]
        assert [record["label"] for record in scores] == ["baseline", "iteration-0"]

        loaded = dspy.load(
            tmp_path / "run" / "candidate-001",
            bindings={
                "lm": {"dummy": dspy.DummyLM([chat_completion(answer="unknown"), chat_completion(answer="Paris")])}
            },
        )
        replayed = loaded(question="What is the capital of France?")
        assert replayed.answer == "Paris"
        assert replayed.best_score == 1.0
        assert replayed.attempts == 2

    def test_rejected_wrap_is_unwound(self):
        # The wrap applies to the candidate, cannot improve a perfect
        # baseline, and the structural edit is fully reverted.
        program = Solo()
        program.solver.signature = program.solver.signature.with_instructions(MAGIC)
        original_solver = program.solver
        reflection = dspy.DummyLM([reflection_reply([{"op": "wrap_best_of_n", "path": "solver", "N": 2}])])
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1, reward=capital_reward)
        dspy.configure(lm=dspy.DummyLM(task_pilot))

        optimizer.compile(program, trainset=devset_of("France"))

        entry = optimizer.trajectory[1]
        assert entry["score"] == 1.0  # the wrapped candidate really ran
        assert entry["accepted"] is False  # 1.0 is not STRICTLY better
        assert program.solver is original_solver  # structure restored
        tree = program.to_manifest()["components"]["1_module_tree"]
        assert [child["kind"] for child in tree["children"]] == ["Predict"]
