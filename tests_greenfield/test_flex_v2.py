"""Stage A10 tests: FlexIR v2 — the leaf-implementation rewrites.

v2 adds the code ops: the reflection LM writes ordinary Python that
becomes a declared authored tool leaf, and the forward tree swaps a
Predict call for that tool — an LM call replaced by code. The tests here
pin, end to end and deterministically under scripted DummyLMs:

- THE CHEAPNESS ASSERT: a genuine predict->code swap where the scripted
  code truly matches the LM's deterministic behavior, so `lm_calls` drops
  while the holdout score holds, the manifest shows a tool where a predict
  was, and save->load->run works;
- the partial fast-path (code on in-distribution input, LM fallback on a
  decline), asserted via lm_calls on two inputs;
- the reward-hacking guard: an edit that overfits the dev split but
  regresses on the holdout is REJECTED and absent from the final program;
- generated-code safety: closures / global reads / disallowed imports
  refuse loudly and are fed back; injection-shaped source is admitted only
  as carried data and NEVER executed;
- determinism and v1-parity refusals (unknown op / bad path).
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


def _input_value(messages, field: str) -> str:
    """The last rendered value of one input field, from the chat prompt."""
    rendered = "\n".join(str(message["content"]) for message in messages)
    marker = f"[[ ## {field} ## ]]\n"
    value = ""
    index = rendered.find(marker)
    while index != -1:
        start = index + len(marker)
        end = rendered.find("\n", start)
        value = rendered[start:] if end == -1 else rendered[start:end]
        index = rendered.find(marker, start)
    return value.strip()


# ---------------------------------------------------------------------------
# A deterministic uppercasing task the LM can express exactly as code.
# ---------------------------------------------------------------------------


class Tagger(dspy.Module):
    """One predictor whose output is a deterministic transform of its input."""

    def __init__(self):
        self.tagger = dspy.Predict("text -> tag")

    def forward(self, text):
        result = self.tagger(text=text)
        return result


def upper_task(messages):
    """The task LM: it uppercases the `text` input — a pure transform."""
    return chat_completion(tag=_input_value(messages, "text").upper())


def exact_tag(example, prediction):
    return example.tag == prediction.tag


UPPER_CODE = 'def tag_code(text: str) -> dict:\n    return {"tag": text.upper()}\n'


def tag_devset(*words: str):
    return [dspy.Example(text=word, tag=word.upper()).with_inputs("text") for word in words]


# ---------------------------------------------------------------------------
# (1) replace_predict_with_code — THE CHEAPNESS ASSERT
# ---------------------------------------------------------------------------


class TestReplacePredictWithCode:
    def run(self, tmp_path=None):
        program = Tagger()
        devset = tag_devset("cat", "dog", "fox")
        holdout = tag_devset("owl", "bee")
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code",
                            "path": "tagger",
                            "tool_name": "tag_code",
                            "python_source": UPPER_CODE,
                        },
                        {"op": "delete_dead_leaf", "path": "tagger"},
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=holdout)
        kwargs = {"checkpoint_dir": tmp_path / "run"} if tmp_path else {}
        optimizer.compile(program, trainset=devset, **kwargs)
        return program, optimizer

    def test_cheapness_lm_calls_drop_while_score_holds(self):
        _program, optimizer = self.run()
        baseline, first = optimizer.trajectory

        # Baseline: the LM answers all three correctly, one call each.
        assert baseline["score"] == 1.0
        assert baseline["lm_calls"] == 3
        # After the swap: SAME score, ZERO LM calls — the cheapness thesis.
        assert first["accepted"] is True
        assert first["score"] == 1.0
        assert first["lm_calls"] == 0
        assert first["holdout_score"] == 1.0
        assert [proposal["op"] for proposal in first["applied"]] == [
            "replace_predict_with_code",
            "delete_dead_leaf",
        ]

    def test_manifest_shows_a_tool_where_a_predict_was(self):
        program, _ = self.run()
        manifest = program.to_manifest()  # compiles clean
        # The predict leaf is gone; a tool leaf stands in its place.
        assert [child["kind"] for child in manifest["components"]["1_module_tree"]["children"]] == []
        assert list(manifest["components"]["6_tools"]) == ["tag_code"]
        assert manifest["components"]["6_tools"]["tag_code"]["authored_by"] == "optimizer"
        # The forward's call site is now a tool call, not a predict call.
        call = manifest["components"]["5_forward"]["self"]["body"][0]["value"]
        assert call["leaf"] == {"kind": "tool", "ref": "tag_code"}
        # The cost view reflects the removed LM leaf: zero calls reachable.
        assert program.cost()["total_calls"].maximum == 0
        # The explain view names the authored tool.
        assert "tag_code" in program.explain()

    def test_swap_runs_through_the_engine_with_zero_lm_calls(self):
        program, _ = self.run()
        prediction = program(text="hello")
        assert prediction.tag == "HELLO"
        assert (prediction._trajectory.get("predictor_calls") or []) == []

    def test_checkpoint_loads_and_runs_without_an_lm(self, tmp_path):
        self.run(tmp_path)
        scores = json.loads((tmp_path / "run" / "scores.json").read_text())
        assert [record["label"] for record in scores] == ["baseline", "iteration-0"]
        # The accepted candidate loads with an EMPTY LM binding — it needs
        # no model at all — and runs the pure-Python leaf.
        loaded = dspy.load(tmp_path / "run" / "candidate-001", bindings={"lm": {}})
        assert loaded(text="world").tag == "WORLD"

    def test_authored_by_provenance_survives_the_artifact(self, tmp_path):
        # PIR-014: the optimizer provenance must survive into the on-disk
        # artifact so a receiver can audit or re-place the leaf.
        self.run(tmp_path)
        entry = json.loads((tmp_path / "run" / "candidate-001" / "manifest.json").read_text())
        assert entry["components"]["6_tools"]["tag_code"]["authored_by"] == "optimizer"

    def test_diff_names_the_swap(self):
        _, optimizer = self.run()
        baseline, first = optimizer.trajectory
        lines = dspy.diff(baseline["manifest"], first["manifest"])
        assert any("tool 'tag_code' added" in line for line in lines)

    def test_determinism_under_the_same_script(self):
        def shape(optimizer):
            return [
                (entry["label"], entry["score"], entry["lm_calls"], entry["accepted"]) for entry in optimizer.trajectory
            ]

        _, first_run = self.run()
        _, second_run = self.run()
        assert shape(first_run) == shape(second_run)


# ---------------------------------------------------------------------------
# (2) replace_predict_with_code_partial — fast path + LM fallback
# ---------------------------------------------------------------------------


CAPITALS = {"France": "Paris", "Canada": "Ottawa", "Japan": "Tokyo", "Peru": "Lima"}


class Solver(dspy.Module):
    def __init__(self):
        self.solver = dspy.Predict("country -> capital")

    def forward(self, country):
        result = self.solver(country=country)
        return result


def capital_task(messages):
    country = _input_value(messages, "country")
    return chat_completion(capital=CAPITALS.get(country, "unknown"))


def exact_capital(example, prediction):
    return example.capital == prediction.capital


PARTIAL_CODE = (
    "def solve_partial(country: str) -> dict | None:\n"
    '    table = {"France": "Paris", "Canada": "Ottawa"}\n'
    "    if country in table:\n"
    '        return {"capital": table[country]}\n'
    "    return None\n"
)


class TestPartialReplace:
    def run(self):
        program = Solver()
        # France/Canada go through code; Japan/Peru decline to the LM.
        devset = [
            dspy.Example(country=c, capital=CAPITALS[c]).with_inputs("country")
            for c in ("France", "Canada", "Japan", "Peru")
        ]
        holdout = [dspy.Example(country="France", capital="Paris").with_inputs("country")]
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code_partial",
                            "path": "solver",
                            "tool_name": "solve_partial",
                            "python_source": PARTIAL_CODE,
                        }
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(capital_task))
        optimizer = optim.FlexIR(reflection, exact_capital, iterations=1, holdout=holdout)
        optimizer.compile(program, trainset=devset)
        return program, optimizer

    def test_code_path_and_fallback_both_score_correct(self):
        _program, optimizer = self.run()
        baseline, first = optimizer.trajectory
        # Baseline: four LM calls, all correct.
        assert (baseline["score"], baseline["lm_calls"]) == (1.0, 4)
        # After the partial swap: still all correct, but only the two
        # declining inputs reach the LM — N < lm_calls < 2N, cheapness held.
        assert first["accepted"] is True
        assert first["score"] == 1.0
        assert first["lm_calls"] == 2

    def test_the_two_paths_assert_via_lm_calls(self):
        program, _ = self.run()
        # In-distribution input: the fast path, ZERO LM calls for the leaf.
        program.invalidate_ir()
        in_dist = program(country="France")
        assert in_dist.capital == "Paris"
        assert (in_dist._trajectory.get("predictor_calls") or []) == []
        # A declined input: the fallback ARM runs the predict, one LM call,
        # and the answer comes through the CallPredict path.
        program.invalidate_ir()
        declined = program(country="Japan")
        assert declined.capital == "Tokyo"
        assert len(declined._trajectory.get("predictor_calls") or []) == 1

    def test_partial_tree_is_the_spelled_try_if_shape(self):
        program, _ = self.run()
        body = program.to_manifest()["components"]["5_forward"]["self"]["body"]
        # Statement 0: Try(fast=CallTool; except ToolError -> Const None).
        try_node = body[0]
        assert try_node["node"] == "Try"
        fast = try_node["body"][0]
        assert fast["node"] == "Assign" and fast["target"] == "_flex_fast_0"
        assert fast["value"]["leaf"] == {"kind": "tool", "ref": "solve_partial"}
        assert try_node["handlers"][0]["type"] == "ToolError"
        assert try_node["handlers"][0]["body"][0]["value"] == {"node": "Const", "value": None}
        # Statement 1: If(_flex_fast_0 != None) -> use it, else CallPredict.
        if_node = body[1]
        assert if_node["node"] == "If"
        assert if_node["test"]["op"] == "ne"
        assert if_node["orelse"][0]["value"]["leaf"] == {"kind": "predict", "ref": "solver"}

    def test_partial_checkpoint_round_trips(self, tmp_path):
        program = Solver()
        devset = [dspy.Example(country=c, capital=CAPITALS[c]).with_inputs("country") for c in ("France", "Japan")]
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code_partial",
                            "path": "solver",
                            "tool_name": "solve_partial",
                            "python_source": PARTIAL_CODE,
                        }
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(capital_task))
        optimizer = optim.FlexIR(
            reflection,
            exact_capital,
            iterations=1,
            holdout=[dspy.Example(country="France", capital="Paris").with_inputs("country")],
        )
        optimizer.compile(program, trainset=devset, checkpoint_dir=tmp_path / "run")
        # Load the candidate; bind an LM for the fallback arm.
        loaded = dspy.load(
            tmp_path / "run" / "candidate-001",
            bindings={"lm": {"dummy": dspy.DummyLM([chat_completion(capital="Tokyo")])}},
        )
        assert loaded(country="France").capital == "Paris"  # code arm, no LM
        assert loaded(country="Japan").capital == "Tokyo"  # fallback arm


# ---------------------------------------------------------------------------
# (3) The reward-hacking guard: overfit dev, regress holdout -> rejected
# ---------------------------------------------------------------------------


class TestRewardHackingGuard:
    def test_devset_memorizing_code_is_rejected_on_the_holdout(self):
        # The devset is {cat, dog}; the holdout is {fox}. A memorizing tool
        # hardcodes the dev answers and returns "" otherwise — it aces the
        # dev split (cheaper too) but regresses on the holdout the LM never
        # saw. The two-channel + holdout gate must REJECT it.
        program = Tagger()
        devset = tag_devset("cat", "dog")
        holdout = tag_devset("fox")
        memorizer = (
            "def tag_code(text: str) -> dict:\n"
            '    memo = {"cat": "CAT", "dog": "DOG"}\n'
            '    return {"tag": memo.get(text, "")}\n'
        )
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code",
                            "path": "tagger",
                            "tool_name": "tag_code",
                            "python_source": memorizer,
                        }
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=holdout)
        optimizer.compile(program, trainset=devset)

        entry = optimizer.trajectory[1]
        # The edit really applied and scored the dev split (perfect, cheaper)...
        assert [proposal["op"] for proposal in entry["applied"]] == ["replace_predict_with_code"]
        assert entry["score"] == 1.0
        assert entry["lm_calls"] == 0
        # ...but the holdout regressed, so it was REJECTED and fed back.
        assert entry["holdout_score"] == 0.0
        assert entry["accepted"] is False
        assert entry["manifest"] is None
        assert "reward-hacking guard" in entry["rejection"]
        assert "holdout" in entry["rejection"]
        # The final program EXCLUDES the memorizing tool — the predict stands.
        manifest = program.to_manifest()
        assert [child["kind"] for child in manifest["components"]["1_module_tree"]["children"]] == ["Predict"]
        assert manifest["components"]["6_tools"] == {}
        # And the guard is IN the next reflection call had there been one.
        assert any("holdout" in refusal for refusal in optimizer.trajectory[1]["refusals"]) or entry["rejection"]


# ---------------------------------------------------------------------------
# (4) Generated-code safety: refuse loudly, record, never execute
# ---------------------------------------------------------------------------


class TestGeneratedCodeSafety:
    @pytest.mark.parametrize(
        "source, needle",
        [
            (
                # a global read
                'def tag_code(text: str) -> dict:\n    return {"tag": SECRET}\n',
                "reads globals",
            ),
            (
                # a closure-shaped nested def reaching outward is caught as a
                # global read of the outer name; a plain disallowed import:
                'def tag_code(text: str) -> dict:\n    import os\n    return {"tag": os.getcwd()}\n',
                "outside the optimizer allowlist",
            ),
            (
                # a second def
                'def helper():\n    pass\ndef tag_code(text: str) -> dict:\n    return {"tag": text}\n',
                "EXACTLY one",
            ),
            (
                # wrong parameters (not the signature's input fields)
                'def tag_code(other: str) -> dict:\n    return {"tag": other}\n',
                "must be exactly the replaced signature",
            ),
        ],
        ids=["global-read", "disallowed-import", "second-def", "bad-io-contract"],
    )
    def test_unsafe_source_refuses_loudly_and_is_recorded(self, source, needle):
        program = Tagger()
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code",
                            "path": "tagger",
                            "tool_name": "tag_code",
                            "python_source": source,
                        }
                    ]
                ),
                reflection_reply([]),
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=2, holdout=tag_devset("owl"))
        optimizer.compile(program, trainset=tag_devset("cat"))

        entry = optimizer.trajectory[1]
        assert len(entry["refusals"]) == 1
        assert "admission" in entry["refusals"][0]
        assert needle in entry["refusals"][0]
        assert entry["applied"] == []
        # The predict is untouched (never partially applied) and no tool landed.
        manifest = program.to_manifest()
        assert manifest["components"]["6_tools"] == {}
        # The refusal is fed to the NEXT reflection call.
        second_report = reflection.calls[1]["messages"][-1]["content"]
        assert needle in second_report

    def test_injection_shaped_source_is_data_never_executed(self, tmp_path):
        # A source that LOOKS like an attack (os.system) is DATA to the
        # applier: it passes through admission (which refuses the import),
        # and the optimizer never execs it. Assert no side effect.
        sentinel = tmp_path / "pwned"
        injection = (
            "def tag_code(text: str) -> dict:\n"
            "    import os\n"
            f'    os.system("touch {sentinel}")\n'
            '    return {"tag": text}\n'
        )
        program = Tagger()
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code",
                            "path": "tagger",
                            "tool_name": "tag_code",
                            "python_source": injection,
                        }
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=tag_devset("owl"))
        optimizer.compile(program, trainset=tag_devset("cat"))

        entry = optimizer.trajectory[1]
        assert entry["applied"] == []
        assert "outside the optimizer allowlist" in entry["refusals"][0]
        # THE SAFETY ASSERT: nothing was ever executed.
        assert not sentinel.exists()


# ---------------------------------------------------------------------------
# (5) v1-parity refusals + the 0.4-splat guard
# ---------------------------------------------------------------------------


class TestVocabularyParity:
    def test_unknown_op_and_bad_path_still_refuse(self):
        program = Tagger()
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {"op": "transmute", "path": "tagger"},
                        {
                            "op": "replace_predict_with_code",
                            "path": "nope",
                            "tool_name": "x",
                            "python_source": UPPER_CODE,
                        },
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1)
        optimizer.compile(program, trainset=tag_devset("cat"))

        refusals = optimizer.trajectory[1]["refusals"]
        assert len(refusals) == 2
        assert "unknown op 'transmute'" in refusals[0]
        assert "closed vocabulary" in refusals[0]
        assert "no predictor at path 'nope'" in refusals[1]
        assert optimizer.trajectory[1]["applied"] == []

    def test_delete_with_live_sites_refuses(self):
        # delete_dead_leaf must count sites first: a predict with a live
        # call site cannot be deleted.
        program = Tagger()
        reflection = dspy.DummyLM([reflection_reply([{"op": "delete_dead_leaf", "path": "tagger"}])])
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1)
        optimizer.compile(program, trainset=tag_devset("cat"))

        refusals = optimizer.trajectory[1]["refusals"]
        assert len(refusals) == 1
        assert "still has 1 live call site" in refusals[0]
        # The predict survives.
        assert [c["kind"] for c in program.to_manifest()["components"]["1_module_tree"]["children"]] == ["Predict"]

    def test_missing_code_field_refuses_by_shape(self):
        program = Tagger()
        reflection = dspy.DummyLM(
            [reflection_reply([{"op": "replace_predict_with_code", "path": "tagger", "tool_name": "x"}])]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1)
        optimizer.compile(program, trainset=tag_devset("cat"))

        refusals = optimizer.trajectory[1]["refusals"]
        assert len(refusals) == 1
        assert "missing ['python_source']" in refusals[0]

    def test_the_catalog_lists_exactly_the_seven_ops(self):
        from dspy.optim.flex import _VOCABULARY

        assert set(_VOCABULARY) == {
            "set_instructions",
            "add_demo",
            "remove_demo",
            "wrap_best_of_n",
            "replace_predict_with_code",
            "replace_predict_with_code_partial",
            "delete_dead_leaf",
        }

    def test_splat_bearing_call_site_refuses_with_a_teaching_error(self):
        # A v0.4 record-splat call site (ReAct's `self.react(**inputs, ...)`)
        # is not yet rewritable; the applier keys on call-site nodes and
        # refuses with a teaching message rather than mis-rewriting it.
        def look(query: str) -> str:
            """Look a fact up."""
            return query

        dspy.configure(
            lm=dspy.DummyLM(
                [
                    chat_completion(next_thought="t", next_tool_name="finish", next_tool_args="{}"),
                    chat_completion(answer="a"),
                ]
            )
        )
        react = dspy.ReAct("question -> answer", tools=[look], max_iters=2)
        code = "def rc(question: str) -> dict:\n    return {}\n"
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [{"op": "replace_predict_with_code", "path": "react", "tool_name": "rc", "python_source": code}]
                )
            ]
        )
        optimizer = optim.FlexIR(reflection, lambda example, prediction: 1.0, iterations=1)
        optimizer.compile(react, trainset=[dspy.Example(question="q", answer="a").with_inputs("question")])

        refusals = optimizer.trajectory[1]["refusals"]
        assert len(refusals) == 1
        assert "0.4 record-splat site" in refusals[0]
        assert "rc" not in react.to_manifest()["components"]["6_tools"]
