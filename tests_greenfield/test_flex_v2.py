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

    def test_checkpoint_loads_only_when_the_authored_leaf_is_granted(self, tmp_path):
        self.run(tmp_path)
        scores = json.loads((tmp_path / "run" / "scores.json").read_text())
        assert [record["label"] for record in scores] == ["baseline", "iteration-0"]
        # The trust pairing rule (spec/trust.md): an optimizer-authored code
        # leaf is placed at the `isolation_required` rung, so `materialize`
        # FAILS CLOSED — a receiver cannot silently run it in-process from
        # its sidecar. Loading with only an (empty) LM binding refuses,
        # naming the tool and teaching the grant.
        candidate = tmp_path / "run" / "candidate-001"
        with pytest.raises(ValueError, match="requires isolation"):
            dspy.load(candidate, bindings={"lm": {}})

        # The receiver GRANTS the reviewed leaf by binding a callable under
        # its pool name (the binding IS the grant). Then it loads and runs
        # the pure-Python leaf with no model at all.
        def tag_code(text: str) -> dict:
            return {"tag": text.upper()}

        loaded = dspy.load(candidate, bindings={"lm": {}, "tool": {"tag_code": tag_code}})
        assert loaded(text="world").tag == "WORLD"

    def test_authored_by_provenance_survives_the_artifact(self, tmp_path):
        # PIR-014: the optimizer provenance must survive into the on-disk
        # artifact so a receiver can audit or re-place the leaf.
        self.run(tmp_path)
        entry = json.loads((tmp_path / "run" / "candidate-001" / "manifest.json").read_text())
        assert entry["components"]["6_tools"]["tag_code"]["authored_by"] == "optimizer"

    def test_optimizer_authored_leaf_carries_the_isolation_required_rung(self, tmp_path):
        # Critical 1(b), the trust pairing rule (spec/trust.md): an
        # optimizer-authored code leaf is NOT placed in-process like a
        # builtin tool — its placement records that it must run under
        # isolation the receiver supplies. The fact is recorded honestly
        # (recording is passive); `materialize` enforces the deliverable
        # half (the consent gate) by failing closed on an ungranted
        # sidecar load — pinned by the checkpoint-grant test above.
        program, _ = self.run(tmp_path)
        placement = program.to_manifest()["components"]["6_tools"]["tag_code"]["placement"]
        assert placement["rung"] == "isolation_required"
        assert placement["rung"] != "in_process"
        assert placement["isolation"] == "required"
        # It survives to the on-disk artifact too — a receiver/auditor sees
        # the isolation requirement without executing anything.
        entry = json.loads((tmp_path / "run" / "candidate-001" / "manifest.json").read_text())
        assert entry["components"]["6_tools"]["tag_code"]["placement"]["rung"] == "isolation_required"

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


CAPITALS = {"France": "Paris", "Canada": "Ottawa", "Japan": "Tokyo", "Peru": "Lima", "Germany": "Berlin"}


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
        # Holdout is DISJOINT from the trainset (Germany, not in the devset);
        # it declines to the LM, which answers it correctly, so the candidate
        # holds on the split it never saw.
        holdout = [dspy.Example(country="Germany", capital="Berlin").with_inputs("country")]
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
            holdout=[dspy.Example(country="Germany", capital="Berlin").with_inputs("country")],
        )
        optimizer.compile(program, trainset=devset, checkpoint_dir=tmp_path / "run")
        candidate = tmp_path / "run" / "candidate-001"
        # The authored fast-path leaf requires an explicit grant even when
        # the LM fallback is bound — binding the LM alone still refuses.
        with pytest.raises(ValueError, match="requires isolation"):
            dspy.load(candidate, bindings={"lm": {"dummy": dspy.DummyLM([chat_completion(capital="Tokyo")])}})

        # Grant the reviewed fast-path code AND bind an LM for the fallback arm.
        def solve_partial(country: str) -> dict | None:
            table = {"France": "Paris", "Canada": "Ottawa"}
            if country in table:
                return {"capital": table[country]}
            return None

        loaded = dspy.load(
            candidate,
            bindings={
                "lm": {"dummy": dspy.DummyLM([chat_completion(capital="Tokyo")])},
                "tool": {"solve_partial": solve_partial},
            },
        )
        assert loaded(country="France").capital == "Paris"  # code arm, no LM
        assert loaded(country="Japan").capital == "Tokyo"  # fallback arm

    def test_partial_on_a_return_site_refuses_instead_of_no_oping(self):
        # The partial Try/If shape only wraps a top-level
        # `result = self.<leaf>(...)` Assign. On a `return self.<leaf>(...)`
        # site the rewrite matched nothing yet the op was reported as
        # applied and no cheapness landed (lm_calls stayed at the pre-swap
        # count). It must REFUSE the site with a teaching error, never claim
        # a silent no-op as success.
        class SolverReturn(dspy.Module):
            def __init__(self):
                self.solver = dspy.Predict("country -> capital")

            def forward(self, country):
                return self.solver(country=country)  # one-line return, not an Assign

        program = SolverReturn()
        devset = [
            dspy.Example(country=c, capital=CAPITALS[c]).with_inputs("country")
            for c in ("France", "Canada", "Japan", "Peru")
        ]
        holdout = [dspy.Example(country="Germany", capital="Berlin").with_inputs("country")]
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

        entry = optimizer.trajectory[1]
        # The op did NOT silently claim success; it refused with teaching text.
        assert entry["applied"] == []
        assert len(entry["refusals"]) == 1
        assert "not a top-level" in entry["refusals"][0]
        assert "replace_predict_with_code" in entry["refusals"][0]
        # The forward is untouched: still the bare Return, still an LM call,
        # and no tool leaked onto the module.
        body = program.to_manifest()["components"]["5_forward"]["self"]["body"]
        assert [statement.get("node") for statement in body] == ["Return"]
        assert program.to_manifest()["components"]["6_tools"] == {}
        assert not hasattr(program, "solve_partial")


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

    def test_poisoning_add_demo_is_gated_on_the_holdout(self):
        # The guard must cover DATA ops, not only code ops: demos and
        # instructions are the classic overfit surface. A demo that poisons
        # the LM to win the rigged dev split while collapsing the truly-
        # held-out split must be REJECTED exactly like a memorizing code
        # leaf — the has_code condition on the gate was the blind spot.
        program = Tagger()
        # The dev split is rigged to want "POISON"; the holdout wants the
        # true uppercase answer. The poison demo flips the LM to "POISON".
        devset = [
            dspy.Example(text="cat", tag="POISON").with_inputs("text"),
            dspy.Example(text="dog", tag="POISON").with_inputs("text"),
        ]
        holdout = [dspy.Example(text="fox", tag="FOX").with_inputs("text")]

        def poisonable_task(messages):
            rendered = "\n".join(str(message["content"]) for message in messages)
            text = _input_value(messages, "text")
            return chat_completion(tag="POISON" if "POISON_DEMO" in rendered else text.upper())

        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "add_demo",
                            "path": "tagger",
                            "inputs": {"text": "POISON_DEMO"},
                            "labels": {"tag": "POISON"},
                        }
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(poisonable_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=holdout)
        optimizer.compile(program, trainset=devset)

        entry = optimizer.trajectory[1]
        # The demo really applied and won the dev split...
        assert [proposal["op"] for proposal in entry["applied"]] == ["add_demo"]
        assert entry["score"] == 1.0
        # ...but the holdout regressed, so the (non-code) candidate is REJECTED.
        assert entry["holdout_score"] == 0.0
        assert entry["accepted"] is False
        assert "reward-hacking guard" in entry["rejection"]
        # The poison demo is not carried into the final program.
        for _path, predictor in program.named_predictors():
            assert predictor.demos == []

    def test_a_rejected_data_op_never_lowers_the_holdout_ceiling(self):
        # Chained defeat: iter0 poisons via add_demo (dev up, holdout down);
        # iter1 ships a pure memorizer code op. If an un-gated accept had
        # lowered best_holdout to 0.0, the memorizer would pass the gate
        # trivially. With the gate on EVERY candidate and best_holdout a
        # monotonic ceiling, iter0 is rejected (ceiling stays 1.0) and iter1
        # is rejected too — neither ships.
        program = Tagger()
        devset = [
            dspy.Example(text="cat", tag="POISON").with_inputs("text"),
            dspy.Example(text="dog", tag="POISON").with_inputs("text"),
        ]
        holdout = [dspy.Example(text="fox", tag="FOX").with_inputs("text")]
        memorizer = (
            "def tag_code(text: str) -> dict:\n"
            '    memo = {"cat": "POISON", "dog": "POISON"}\n'
            '    return {"tag": memo.get(text, "")}\n'
        )

        def poisonable_task(messages):
            rendered = "\n".join(str(message["content"]) for message in messages)
            text = _input_value(messages, "text")
            return chat_completion(tag="POISON" if "POISON_DEMO" in rendered else text.upper())

        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "add_demo",
                            "path": "tagger",
                            "inputs": {"text": "POISON_DEMO"},
                            "labels": {"tag": "POISON"},
                        }
                    ]
                ),
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code",
                            "path": "tagger",
                            "tool_name": "tag_code",
                            "python_source": memorizer,
                        }
                    ]
                ),
            ]
        )
        dspy.configure(lm=dspy.DummyLM(poisonable_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=2, holdout=holdout)
        optimizer.compile(program, trainset=devset)

        _baseline, iter0, iter1 = optimizer.trajectory
        # iter0 poison demo: dev up, holdout down -> rejected, ceiling held.
        assert iter0["accepted"] is False
        assert iter0["holdout_score"] == 0.0
        # iter1 memorizer: holdout 0.0 is still gated against the 1.0 ceiling.
        assert iter1["accepted"] is False
        assert iter1["holdout_score"] == 0.0
        # Nothing shipped: no tool, the predict stands, an unseen input still
        # reaches the LM (which answers it correctly).
        assert program.to_manifest()["components"]["6_tools"] == {}
        assert program(text="wolf").tag == "WOLF"

    def test_always_true_partial_fast_path_is_refused(self):
        # A _partial fast path that NEVER declines (returns a dict for every
        # input) is a full replacement wearing partial clothing. Even when it
        # is correct on dev AND the holdout, it permanently bypasses the LM
        # on whatever the fixed split does not exercise — cheapness at an
        # invisible quality cost. If it never declined on the holdout (0 LM
        # calls there), refuse and teach the honest alternatives.
        program = Solver()
        capitals = {"France": "Paris", "Canada": "Ottawa", "Germany": "Berlin"}
        devset = [dspy.Example(country=c, capital=capitals[c]).with_inputs("country") for c in ("France", "Canada")]
        holdout = [dspy.Example(country="Germany", capital="Berlin").with_inputs("country")]
        never_declines = (
            "def solve_partial(country: str) -> dict | None:\n"
            '    table = {"France": "Paris", "Canada": "Ottawa", "Germany": "Berlin"}\n'
            '    return {"capital": table.get(country, "GUESS-" + country)}\n'  # always returns a dict
        )
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code_partial",
                            "path": "solver",
                            "tool_name": "solve_partial",
                            "python_source": never_declines,
                        }
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(capital_task))
        optimizer = optim.FlexIR(reflection, exact_capital, iterations=1, holdout=holdout)
        optimizer.compile(program, trainset=devset)

        entry = optimizer.trajectory[1]
        # It aced dev (1.0), aced the holdout (1.0), and dropped LM calls...
        assert entry["score"] == 1.0
        assert entry["holdout_score"] == 1.0
        assert entry["lm_calls"] == 0
        # ...but the never-declining fast path is REJECTED as a disguised full swap.
        assert entry["accepted"] is False
        assert "NEVER declined" in entry["rejection"]
        assert "replace_predict_with_code" in entry["rejection"]
        # The final program keeps the predict; no tool shipped.
        assert program.to_manifest()["components"]["6_tools"] == {}

    def test_overlapping_holdout_is_refused_at_compile(self):
        # A holdout that overlaps the trainset silently disables the guard
        # (a memorizer aces both). Refuse it loudly at compile time.
        program = Tagger()
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
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=tag_devset("cat", "dog"))
        with pytest.raises(ValueError, match="holdout overlaps the trainset"):
            optimizer.compile(program, trainset=tag_devset("cat", "dog"))


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

    @pytest.mark.parametrize(
        "body, needle",
        [
            ('    __import__("os").system("x")\n', "builtin '__import__'"),
            ('    eval("1 + 1")\n', "builtin 'eval'"),
            ("    exec('x = 1')\n", "builtin 'exec'"),
            ('    compile("1", "x", "eval")\n', "builtin 'compile'"),
            ('    open("/etc/hostname")\n', "builtin 'open'"),
            ("    globals()\n", "builtin 'globals'"),
            ('    getattr(text, "__class__")\n', "builtin 'getattr'"),
            ("    x = ().__class__.__bases__[0].__subclasses__()\n", "reflection attribute"),
            ("    g = (lambda: 0).__globals__\n", "reflection attribute"),
        ],
        ids=[
            "dynamic-import",
            "eval",
            "exec",
            "compile",
            "open",
            "globals",
            "getattr-dunder",
            "class-bases-chain",
            "globals-attr",
        ],
    )
    def test_builtin_escape_doors_refuse_at_admission(self, tmp_path, body, needle):
        # The import allowlist walks only import NODES; `__import__`, `eval`,
        # `exec`, `compile`, `open` reach import/exec/filesystem machinery
        # with NO import node, and `dir(builtins)` self-containment admits
        # them. Admission must refuse every builtin-mediated escape (risk 2)
        # — and prove it end to end: a leaf carrying `__import__(...).system`
        # as a stand-in for a hidden HTTP LM call must NEVER fire a side
        # effect during compile.
        sentinel = tmp_path / "escaped"
        source = f'def tag_code(text: str) -> dict:\n{body}    return {{"tag": text}}\n'.replace(
            'system("x")', f'system("touch {sentinel}")'
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
        assert entry["applied"] == []
        assert len(entry["refusals"]) == 1
        assert "admission" in entry["refusals"][0]
        assert needle in entry["refusals"][0]
        # No tool landed; the predict is untouched.
        assert program.to_manifest()["components"]["6_tools"] == {}
        # THE SAFETY ASSERT: the builtin-mediated side effect never fired.
        assert not sentinel.exists()
        # The refusal is fed to the next reflection call.
        assert needle in reflection.calls[1]["messages"][-1]["content"]

    def test_unparseable_source_is_a_refusal_not_a_crash(self):
        # A `def` missing its colon raises SyntaxError, which is NOT a
        # ValueError; without a guard it escapes the applier's
        # `except ValueError` and aborts the whole compile() run. Admission
        # must turn it into a teaching refusal and keep the loop alive.
        program = Tagger()
        broken = 'def tag_code(text: str) -> dict\n    return {"tag": text}\n'  # missing ':'
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code",
                            "path": "tagger",
                            "tool_name": "tag_code",
                            "python_source": broken,
                        }
                    ]
                ),
                reflection_reply([]),
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=2, holdout=tag_devset("owl"))
        # The run COMPLETES — the malformed proposal did not abort it.
        optimizer.compile(program, trainset=tag_devset("cat"))

        entry = optimizer.trajectory[1]
        assert entry["applied"] == []
        assert len(entry["refusals"]) == 1
        assert "does not parse as Python" in entry["refusals"][0]
        # The loop ran both iterations (the crash would have killed it at 1).
        assert [record["label"] for record in optimizer.trajectory] == [
            "baseline",
            "iteration-0",
            "iteration-1",
        ]
        # The refusal was fed to the next reflection call.
        assert "does not parse as Python" in reflection.calls[1]["messages"][-1]["content"]

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
