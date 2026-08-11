"""FlexIR v3 tests: the general structure ops, gated by the same wisdom.

v3 lets the reflection LM do ANYTHING to the ProgramIR — bind new predict
and tool leaves (`add_predict`, `add_tool`) and replace a module's whole
forward (`rewrite_forward`, in the printer's dialect) — while every path
stays inside the v2 invariants: the closed dispatch table, the admission
chain, apply-on-copy + whole-batch unwind, the holdout reward-hacking
gate, the lm_calls cheapness channel, and the teaching-refusal ledger.
Security is tunable per instance (`code_trust`, `extra_imports`), never
silently. Everything here is deterministic under scripted DummyLMs.
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
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


def reflection_reply(ops: list) -> str:
    return chat_completion(proposals=json.dumps(ops))


def _input_value(messages, field: str) -> str:
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
# Shared fixtures: the one-predict tagger (v2's) and a decomposable task
# ---------------------------------------------------------------------------


class Tagger(dspy.Module):
    def __init__(self):
        self.tagger = dspy.Predict("text -> tag")

    def forward(self, text):
        result = self.tagger(text=text)
        return result


def upper_task(messages):
    return chat_completion(tag=_input_value(messages, "text").upper())


def exact_tag(example, prediction):
    return example.tag == prediction.tag


def tag_devset(*words: str):
    return [dspy.Example(text=word, tag=word.upper()).with_inputs("text") for word in words]


UPPER_TOOL = 'def tag_code(text: str) -> dict:\n    return {"tag": text.upper()}\n'

REWIRE_TO_TOOL = "def forward(self, text):\n    result = self.tag_code(text=text)\n    return result\n"


class Combo(dspy.Module):
    """One predict whose task has TWO separable parts (uppercase + '!')."""

    def __init__(self):
        self.solver = dspy.Predict("text -> answer")

    def forward(self, text):
        result = self.solver(text=text)
        return result


def decomposable_task(messages):
    """The task LM: correct only when the work is split into two leaves.

    Asked for `loud` (the first sub-step): it uppercases. Asked for
    `answer` with a `loud` input (the second sub-step): it appends "!".
    Asked for `answer` straight from `text` (the undecomposed leaf): it
    echoes the text — wrong on every example.
    """
    rendered = "\n".join(str(message["content"]) for message in messages)
    if "[[ ## loud ## ]]" in rendered and "[[ ## answer ## ]]" not in rendered:
        return chat_completion(loud=_input_value(messages, "text").upper())
    loud = _input_value(messages, "loud")
    if loud:
        return chat_completion(answer=loud + "!")
    return chat_completion(answer=_input_value(messages, "text"))


def exact_answer(example, prediction):
    return example.answer == prediction.answer


def combo_devset(*words: str):
    return [dspy.Example(text=word, answer=word.upper() + "!").with_inputs("text") for word in words]


DECOMPOSED_FORWARD = (
    "def forward(self, text):\n"
    "    step = self.shout(text=text)\n"
    "    result = self.finisher(loud=step.loud)\n"
    "    return result\n"
)

DECOMPOSE_OPS = [
    {
        "op": "add_predict",
        "path": "self",
        "name": "shout",
        "signature": "text -> loud",
        "instructions": "Uppercase the text.",
    },
    {
        "op": "add_predict",
        "path": "self",
        "name": "finisher",
        "signature": "loud -> answer",
        "instructions": "Append the finishing mark to the loud text.",
    },
    {"op": "rewrite_forward", "path": "self", "python_source": DECOMPOSED_FORWARD},
    {"op": "delete_dead_leaf", "path": "solver"},
]


# ---------------------------------------------------------------------------
# (1) THE DECOMPOSITION ASSERT — add_predict + rewrite_forward, accepted
# ---------------------------------------------------------------------------


class TestDecomposition:
    def run(self, tmp_path=None):
        program = Combo()
        devset = combo_devset("cat", "dog", "fox")
        holdout = combo_devset("owl", "bee")
        reflection = dspy.DummyLM([reflection_reply(DECOMPOSE_OPS)])
        dspy.configure(lm=dspy.DummyLM(decomposable_task))
        optimizer = optim.FlexIR(reflection, exact_answer, iterations=1, holdout=holdout)
        kwargs = {"checkpoint_dir": tmp_path / "run"} if tmp_path else {}
        optimizer.compile(program, trainset=devset, **kwargs)
        return program, optimizer

    def test_decomposed_program_scores_strictly_higher_and_is_accepted(self):
        _program, optimizer = self.run()
        baseline, first = optimizer.trajectory
        # Baseline: the undecomposed predict is wrong on every example.
        assert baseline["score"] == 0.0
        assert baseline["lm_calls"] == 3
        # The decomposition PAYS: it doubles lm_calls (2 per example), so
        # only the score channel can accept it — and it does, strictly.
        assert first["accepted"] is True
        assert first["score"] == 1.0
        assert first["lm_calls"] == 6
        assert first["holdout_score"] == 1.0
        assert [proposal["op"] for proposal in first["applied"]] == [
            "add_predict",
            "add_predict",
            "rewrite_forward",
            "delete_dead_leaf",
        ]

    def test_explain_shows_both_new_leaves_and_the_dead_one_is_gone(self):
        program, _ = self.run()
        view = program.explain()
        assert "shout" in view
        assert "finisher" in view
        manifest = program.to_manifest()
        kinds = [child["name"] for child in manifest["components"]["1_module_tree"]["children"]]
        assert sorted(kinds) == ["finisher", "shout"]

    def test_it_runs_end_to_end(self):
        program, _ = self.run()
        assert program(text="new").answer == "NEW!"

    def test_checkpoint_loads_via_dspy_load(self, tmp_path):
        self.run(tmp_path)
        scores = json.loads((tmp_path / "run" / "scores.json").read_text())
        assert [record["label"] for record in scores] == ["baseline", "iteration-0"]
        loaded = dspy.load(
            tmp_path / "run" / "candidate-001",
            bindings={"lm": {"dummy": dspy.DummyLM(decomposable_task)}},
        )
        assert loaded(text="sun").answer == "SUN!"

    def test_rewrite_round_trips_through_the_printer(self):
        from dspy.programir._dspy import leaf_table
        from dspy.programir.forward import compile_forward
        from dspy.programir.printer import to_function

        program, _ = self.run()
        tree = program.to_manifest()["components"]["5_forward"]["self"]
        recompiled = compile_forward(to_function(tree), leaf_table(program))
        assert recompiled == tree

    def test_determinism_under_the_same_script(self):
        def shape(optimizer):
            return [
                (entry["label"], entry["score"], entry["lm_calls"], entry["accepted"]) for entry in optimizer.trajectory
            ]

        _, first_run = self.run()
        _, second_run = self.run()
        assert shape(first_run) == shape(second_run)


# ---------------------------------------------------------------------------
# (2) The cheapness channel still prices structure: equal score at MORE
# lm_calls is refused
# ---------------------------------------------------------------------------


class TestDecompositionMustPay:
    def test_equal_score_at_more_lm_calls_is_refused(self):
        # The tagger already scores 1.0 at one call per example. A rewrite
        # that adds a second predict and keeps the score equal doubles the
        # calls — neither channel improves, so the candidate is refused.
        program = Tagger()
        ops = [
            {
                "op": "add_predict",
                "path": "self",
                "name": "checker",
                "signature": "text -> tag",
                "instructions": "Tag the text.",
            },
            {
                "op": "rewrite_forward",
                "path": "self",
                "python_source": (
                    "def forward(self, text):\n"
                    "    draft = self.tagger(text=text)\n"
                    "    result = self.checker(text=text)\n"
                    "    return result\n"
                ),
            },
        ]
        reflection = dspy.DummyLM([reflection_reply(ops)])
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=tag_devset("owl"))
        optimizer.compile(program, trainset=tag_devset("cat", "dog"))

        entry = optimizer.trajectory[1]
        assert entry["applied"] != []
        assert entry["score"] == 1.0
        assert entry["lm_calls"] == 4  # doubled from 2
        assert entry["accepted"] is False
        assert "neither channel improved" in entry["rejection"]
        # Unwound: the extra predict is gone and the forward is the original.
        assert not hasattr(program, "checker")
        assert len(program.to_manifest()["components"]["1_module_tree"]["children"]) == 1


# ---------------------------------------------------------------------------
# (3) add_predict — refusal classes
# ---------------------------------------------------------------------------


def run_one(program, ops, *, metric=exact_tag, task=upper_task, devset=None, holdout=None, **flex_kwargs):
    reflection = dspy.DummyLM([reflection_reply(ops)])
    dspy.configure(lm=dspy.DummyLM(task))
    optimizer = optim.FlexIR(reflection, metric, iterations=1, holdout=holdout, **flex_kwargs)
    optimizer.compile(program, trainset=devset or tag_devset("cat"))
    return optimizer


class TestAddPredictRefusals:
    def test_non_identifier_name_refuses(self):
        program = Tagger()
        optimizer = run_one(
            program,
            [{"op": "add_predict", "path": "self", "name": "2bad", "signature": "a -> b", "instructions": "x"}],
        )
        refusals = optimizer.trajectory[1]["refusals"]
        assert len(refusals) == 1
        assert "must be a Python identifier" in refusals[0]
        assert not hasattr(program, "2bad".strip())

    def test_existing_attribute_refuses(self):
        program = Tagger()
        optimizer = run_one(
            program,
            [{"op": "add_predict", "path": "self", "name": "tagger", "signature": "a -> b", "instructions": "x"}],
        )
        assert "already exists" in optimizer.trajectory[1]["refusals"][0]

    def test_bad_signature_string_refuses_with_teaching(self):
        program = Tagger()
        optimizer = run_one(
            program,
            [{"op": "add_predict", "path": "self", "name": "extra", "signature": "no arrow here", "instructions": "x"}],
        )
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "does not build" in refusal
        assert "input_a, input_b -> output_c" in refusal
        assert not hasattr(program, "extra")

    def test_empty_instructions_refuse(self):
        program = Tagger()
        optimizer = run_one(
            program,
            [{"op": "add_predict", "path": "self", "name": "extra", "signature": "a -> b", "instructions": ""}],
        )
        assert "non-empty string" in optimizer.trajectory[1]["refusals"][0]

    def test_bare_predict_root_refuses(self):
        program = dspy.Predict("text -> tag")
        optimizer = run_one(
            program,
            [{"op": "add_predict", "path": "self", "name": "extra", "signature": "a -> b", "instructions": "x"}],
        )
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "bare Predict" in refusal
        assert "composite Module" in refusal

    def test_dead_weight_is_not_refused_but_priced(self):
        # An added predict with zero call sites mid-run is legal; at
        # evaluate time nothing improved, so the CANDIDATE is refused by
        # the channels, not the proposal.
        program = Tagger()
        optimizer = run_one(
            program,
            [{"op": "add_predict", "path": "self", "name": "extra", "signature": "a -> b", "instructions": "x"}],
            devset=tag_devset("cat", "dog"),
            holdout=tag_devset("owl"),
        )
        entry = optimizer.trajectory[1]
        assert entry["refusals"] == []
        assert [proposal["op"] for proposal in entry["applied"]] == ["add_predict"]
        assert entry["accepted"] is False
        assert "neither channel improved" in entry["rejection"]
        assert not hasattr(program, "extra")  # unwound


# ---------------------------------------------------------------------------
# (4) add_tool — happy path (wired via rewrite_forward) + refusal classes
# ---------------------------------------------------------------------------


ADD_TOOL_OPS = [
    {"op": "add_tool", "path": "self", "name": "tag_code", "python_source": UPPER_TOOL},
    {"op": "rewrite_forward", "path": "self", "python_source": REWIRE_TO_TOOL},
    {"op": "delete_dead_leaf", "path": "tagger"},
]


class TestAddTool:
    def run(self, **flex_kwargs):
        program = Tagger()
        optimizer = run_one(
            program,
            ADD_TOOL_OPS,
            devset=tag_devset("cat", "dog", "fox"),
            holdout=tag_devset("owl", "bee"),
            **flex_kwargs,
        )
        return program, optimizer

    def test_tool_plus_rewrite_is_the_cheapness_path(self):
        program, optimizer = self.run()
        baseline, first = optimizer.trajectory
        assert (baseline["score"], baseline["lm_calls"]) == (1.0, 3)
        assert first["accepted"] is True
        assert (first["score"], first["lm_calls"]) == (1.0, 0)
        manifest = program.to_manifest()
        assert list(manifest["components"]["6_tools"]) == ["tag_code"]
        assert manifest["components"]["6_tools"]["tag_code"]["authored_by"] == "optimizer"
        call = manifest["components"]["5_forward"]["self"]["body"][0]["value"]
        assert call["leaf"] == {"kind": "tool", "ref": "tag_code"}

    def test_default_trust_is_isolation_required(self):
        program, _ = self.run()
        placement = program.to_manifest()["components"]["6_tools"]["tag_code"]["placement"]
        assert placement["rung"] == "isolation_required"
        assert placement["isolation"] == "required"

    def test_in_process_trust_changes_the_rung_and_keeps_provenance(self):
        program, optimizer = self.run(code_trust="in_process")
        assert optimizer.trajectory[1]["accepted"] is True
        entry = program.to_manifest()["components"]["6_tools"]["tag_code"]
        assert entry["placement"]["rung"] == "in_process"
        assert entry["placement"]["isolation"] == "none"
        # The provenance stamp MUST survive so a receiver can audit or
        # re-place the leaf even when the user chose in-process trust.
        assert entry["authored_by"] == "optimizer"

    def test_in_process_artifact_loads_without_a_grant_ceremony(self, tmp_path):
        program = Tagger()
        reflection = dspy.DummyLM([reflection_reply(ADD_TOOL_OPS)])
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(
            reflection, exact_tag, iterations=1, holdout=tag_devset("owl"), code_trust="in_process"
        )
        optimizer.compile(program, trainset=tag_devset("cat"), checkpoint_dir=tmp_path / "run")
        loaded = dspy.load(tmp_path / "run" / "candidate-001", bindings={"lm": {}})
        assert loaded(text="world").tag == "WORLD"

    def test_isolated_artifact_still_fails_closed(self, tmp_path):
        program = Tagger()
        reflection = dspy.DummyLM([reflection_reply(ADD_TOOL_OPS)])
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=tag_devset("owl"))
        optimizer.compile(program, trainset=tag_devset("cat"), checkpoint_dir=tmp_path / "run")
        with pytest.raises(ValueError, match="requires isolation"):
            dspy.load(tmp_path / "run" / "candidate-001", bindings={"lm": {}})


class TestAddToolRefusals:
    def test_missing_type_hint_refuses(self):
        program = Tagger()
        source = 'def helper(text) -> dict:\n    return {"tag": text}\n'
        optimizer = run_one(program, [{"op": "add_tool", "path": "self", "name": "helper", "python_source": source}])
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "admission" in refusal
        assert "need type hints" in refusal
        assert not hasattr(program, "helper")

    def test_missing_return_annotation_refuses(self):
        program = Tagger()
        source = 'def helper(text: str):\n    return {"tag": text}\n'
        optimizer = run_one(program, [{"op": "add_tool", "path": "self", "name": "helper", "python_source": source}])
        assert "return annotation" in optimizer.trajectory[1]["refusals"][0]

    def test_existing_attribute_refuses(self):
        program = Tagger()
        source = 'def tagger(text: str) -> dict:\n    return {"tag": text}\n'
        optimizer = run_one(program, [{"op": "add_tool", "path": "self", "name": "tagger", "python_source": source}])
        assert "already exists" in optimizer.trajectory[1]["refusals"][0]

    def test_disallowed_import_refuses_like_v2(self):
        program = Tagger()
        source = 'def helper(text: str) -> dict:\n    import os\n    return {"tag": os.getcwd()}\n'
        optimizer = run_one(program, [{"op": "add_tool", "path": "self", "name": "helper", "python_source": source}])
        assert "outside the optimizer allowlist" in optimizer.trajectory[1]["refusals"][0]


# ---------------------------------------------------------------------------
# (5) extra_imports — a per-instance widening, never global
# ---------------------------------------------------------------------------


BASE64_TOOL = (
    "def tag_code(text: str) -> dict:\n"
    "    import base64\n"
    "    base64.b64encode(text.encode())\n"
    '    return {"tag": text.upper()}\n'
)


class TestExtraImports:
    def ops(self):
        return [
            {"op": "add_tool", "path": "self", "name": "tag_code", "python_source": BASE64_TOOL},
            {"op": "rewrite_forward", "path": "self", "python_source": REWIRE_TO_TOOL},
            {"op": "delete_dead_leaf", "path": "tagger"},
        ]

    def test_default_instance_refuses_the_module(self):
        program = Tagger()
        optimizer = run_one(program, self.ops(), holdout=tag_devset("owl"))
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "['base64']" in refusal
        assert "outside the optimizer allowlist" in refusal

    def test_widened_instance_admits_it(self):
        program = Tagger()
        optimizer = run_one(
            program,
            self.ops(),
            devset=tag_devset("cat", "dog"),
            holdout=tag_devset("owl"),
            extra_imports=frozenset({"base64"}),
        )
        entry = optimizer.trajectory[1]
        assert entry["refusals"] == []
        assert entry["accepted"] is True
        assert list(program.to_manifest()["components"]["6_tools"]) == ["tag_code"]

    def test_the_module_level_allowlist_is_never_mutated(self):
        from dspy.optim.code_leaf import ADMITTED_IMPORTS

        before = frozenset(ADMITTED_IMPORTS)
        program = Tagger()
        run_one(
            program,
            self.ops(),
            devset=tag_devset("cat", "dog"),
            holdout=tag_devset("owl"),
            extra_imports=frozenset({"base64"}),
        )
        assert ADMITTED_IMPORTS == before
        assert "base64" not in ADMITTED_IMPORTS

    def test_bad_code_trust_and_extra_imports_refuse_at_construction(self):
        with pytest.raises(ValueError, match="code_trust"):
            optim.FlexIR(dspy.DummyLM([]), exact_tag, code_trust="sandboxed")
        with pytest.raises(ValueError, match="extra_imports"):
            optim.FlexIR(dspy.DummyLM([]), exact_tag, extra_imports="base64")


# ---------------------------------------------------------------------------
# (6) rewrite_forward — refusal classes and the ledger's teaching errors
# ---------------------------------------------------------------------------


class TestRewriteForwardRefusals:
    def test_unknown_leaf_ref_surfaces_the_compiler_refusal_verbatim(self):
        program = Tagger()
        source = "def forward(self, text):\n    result = self.ghost(text=text)\n    return result\n"
        optimizer = run_one(program, [{"op": "rewrite_forward", "path": "self", "python_source": source}])
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "rejected by the forward compiler" in refusal
        assert "PIR-E-NODE-002" in refusal
        assert "ghost" in refusal

    def test_two_defs_refuse(self):
        program = Tagger()
        source = "def helper():\n    pass\ndef forward(self, text):\n    return self.tagger(text=text)\n"
        optimizer = run_one(program, [{"op": "rewrite_forward", "path": "self", "python_source": source}])
        assert "EXACTLY one function definition" in optimizer.trajectory[1]["refusals"][0]

    def test_unparseable_source_is_a_refusal_not_a_crash(self):
        program = Tagger()
        source = "def forward(self, text)\n    return 1\n"  # missing ':'
        optimizer = run_one(program, [{"op": "rewrite_forward", "path": "self", "python_source": source}])
        assert "does not parse as Python" in optimizer.trajectory[1]["refusals"][0]

    def test_wrong_def_name_refuses(self):
        program = Tagger()
        source = "def main(self, text):\n    return self.tagger(text=text)\n"
        optimizer = run_one(program, [{"op": "rewrite_forward", "path": "self", "python_source": source}])
        assert "must be named 'forward'" in optimizer.trajectory[1]["refusals"][0]

    def test_unsupported_python_refuses_through_the_closed_node_set(self):
        program = Tagger()
        source = (
            "def forward(self, text):\n    with open(text) as f:\n        pass\n    return self.tagger(text=text)\n"
        )
        optimizer = run_one(program, [{"op": "rewrite_forward", "path": "self", "python_source": source}])
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "rejected by the forward compiler" in refusal or "rejected" in refusal

    def test_bare_predict_root_refuses(self):
        program = dspy.Predict("text -> tag")
        source = "def forward(self, text):\n    return self.tagger(text=text)\n"
        optimizer = run_one(program, [{"op": "rewrite_forward", "path": "self", "python_source": source}])
        assert "bare Predict" in optimizer.trajectory[1]["refusals"][0]

    def test_predict_leaf_path_refuses(self):
        program = Tagger()
        source = "def forward(self, text):\n    return self.tagger(text=text)\n"
        optimizer = run_one(program, [{"op": "rewrite_forward", "path": "tagger", "python_source": source}])
        assert "Predict leaf, not a composite module" in optimizer.trajectory[1]["refusals"][0]

    def test_unknown_op_field_still_refuses_the_closed_list(self):
        program = Tagger()
        optimizer = run_one(
            program,
            [
                {"op": "rewrite_forward", "path": "self", "python_source": "x", "extra": 1},
                {"op": "grow_new_module", "path": "self"},
            ],
        )
        refusals = optimizer.trajectory[1]["refusals"]
        assert len(refusals) == 2
        assert "unexpected ['extra']" in refusals[0]
        assert "unknown op 'grow_new_module'" in refusals[1]
        assert "closed vocabulary" in refusals[1]


# ---------------------------------------------------------------------------
# (7) delete_dead_leaf generalized to authored tools
# ---------------------------------------------------------------------------


class TestDeleteToolLeaf:
    def test_live_tool_site_refuses_with_the_count(self):
        # Iteration 0 wires the tool (accepted); iteration 1 tries to
        # delete it while its call site is live — refused.
        program = Tagger()
        reflection = dspy.DummyLM(
            [
                reflection_reply(ADD_TOOL_OPS),
                reflection_reply([{"op": "delete_dead_leaf", "path": "tag_code"}]),
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=2, holdout=tag_devset("owl"))
        optimizer.compile(program, trainset=tag_devset("cat", "dog"))
        assert optimizer.trajectory[1]["accepted"] is True
        refusal = optimizer.trajectory[2]["refusals"][0]
        assert "still has 1 live call site" in refusal
        # The tool survives.
        assert list(program.to_manifest()["components"]["6_tools"]) == ["tag_code"]

    def test_dead_tool_deletes_cleanly(self):
        # A tool added but never wired has zero sites; deleting it in the
        # same batch applies.
        program = Tagger()
        optimizer = run_one(
            program,
            [
                {
                    "op": "add_tool",
                    "path": "self",
                    "name": "helper",
                    "python_source": UPPER_TOOL.replace("tag_code", "helper"),
                },
                {"op": "delete_dead_leaf", "path": "helper"},
            ],
            devset=tag_devset("cat"),
        )
        entry = optimizer.trajectory[1]
        assert entry["refusals"] == []
        assert [proposal["op"] for proposal in entry["applied"]] == ["add_tool", "delete_dead_leaf"]
        assert not hasattr(program, "helper")

    def test_unknown_leaf_path_refuses(self):
        program = Tagger()
        optimizer = run_one(program, [{"op": "delete_dead_leaf", "path": "phantom"}])
        assert "no predictor or tool leaf at path 'phantom'" in optimizer.trajectory[1]["refusals"][0]


# ---------------------------------------------------------------------------
# (8) The unwind law: a worse-scoring structure batch restores the champion
# bit-identically
# ---------------------------------------------------------------------------


class TestUnwind:
    def test_worse_batch_unwinds_to_a_bit_identical_champion(self):
        program = Tagger()
        before_ops = [
            {
                "op": "add_predict",
                "path": "self",
                "name": "extra",
                "signature": "text -> tag",
                "instructions": "Tag the text.",
            },
            {
                "op": "add_tool",
                "path": "self",
                "name": "mangle",
                "python_source": 'def mangle(text: str) -> dict:\n    return {"tag": "WRONG"}\n',
            },
            {
                "op": "rewrite_forward",
                "path": "self",
                "python_source": "def forward(self, text):\n    result = self.mangle(text=text)\n    return result\n",
            },
        ]
        reflection = dspy.DummyLM([reflection_reply(before_ops)])
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=tag_devset("owl"))

        baseline_manifest = None
        optimizer.compile(program, trainset=tag_devset("cat", "dog"))
        baseline_manifest = optimizer.trajectory[0]["manifest"]

        entry = optimizer.trajectory[1]
        # The whole batch applied and scored 0.0 (the mangler is wrong)...
        assert [proposal["op"] for proposal in entry["applied"]] == [
            "add_predict",
            "add_tool",
            "rewrite_forward",
        ]
        assert entry["score"] == 0.0
        assert entry["accepted"] is False
        # ...and the unwind restored the champion BIT-IDENTICALLY.
        assert not hasattr(program, "extra")
        assert not hasattr(program, "mangle")
        assert "build_forward_ir" not in program.__dict__
        after = program.to_manifest()
        assert json.dumps(after, sort_keys=True, default=str) == json.dumps(
            baseline_manifest, sort_keys=True, default=str
        )
        # And the champion still runs as before.
        assert program(text="sky").tag == "SKY"


# ---------------------------------------------------------------------------
# (9) The holdout gate covers rewrite_forward candidates
# ---------------------------------------------------------------------------


class TestHoldoutGateOnRewrites:
    def test_memorizing_rewrite_is_refused_by_the_reward_hacking_guard(self):
        # The tool memorizes the dev answers; the rewrite wires it in. Dev
        # goes 1.0 at zero calls (both channels look great) — but the
        # holdout the reflection LM never saw collapses, so the candidate
        # is refused and unwound.
        program = Tagger()
        memorizer = (
            "def tag_code(text: str) -> dict:\n"
            '    memo = {"cat": "CAT", "dog": "DOG"}\n'
            '    return {"tag": memo.get(text, "")}\n'
        )
        ops = [
            {"op": "add_tool", "path": "self", "name": "tag_code", "python_source": memorizer},
            {"op": "rewrite_forward", "path": "self", "python_source": REWIRE_TO_TOOL},
            {"op": "delete_dead_leaf", "path": "tagger"},
        ]
        optimizer = run_one(program, ops, devset=tag_devset("cat", "dog"), holdout=tag_devset("fox"))
        entry = optimizer.trajectory[1]
        assert entry["score"] == 1.0
        assert entry["lm_calls"] == 0
        assert entry["holdout_score"] == 0.0
        assert entry["accepted"] is False
        assert "reward-hacking guard" in entry["rejection"]
        # Unwound: the predict stands, no tool shipped.
        assert program.to_manifest()["components"]["6_tools"] == {}
        assert program(text="elk").tag == "ELK"


# ---------------------------------------------------------------------------
# (10) allowed_deps — user-gated third-party dependencies on authored leaves
# ---------------------------------------------------------------------------


DEP_TOOL = 'def tag_code(text: str) -> dict:\n    # deps: beautifulsoup4\n    return {"tag": text.upper()}\n'

DEP_OPS = [
    {"op": "add_tool", "path": "self", "name": "tag_code", "python_source": DEP_TOOL},
    {"op": "rewrite_forward", "path": "self", "python_source": REWIRE_TO_TOOL},
    {"op": "delete_dead_leaf", "path": "tagger"},
]


class TestAllowedDeps:
    def test_default_refuses_any_dep_verbatim(self):
        # The default pins today's law exactly: no deps, with the same
        # teaching message as before allowed_deps existed.
        program = Tagger()
        optimizer = run_one(program, DEP_OPS, holdout=tag_devset("owl"))
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "declares `# deps: beautifulsoup4`" in refusal
        assert "optimizer-authored code carries no third-party dependencies" in refusal
        assert "use the stdlib allowlist only" in refusal
        assert program.to_manifest()["components"]["6_tools"] == {}

    def test_allowed_dep_admits_and_rides_the_pool_entry_and_environment(self):
        program = Tagger()
        optimizer = run_one(
            program,
            DEP_OPS,
            devset=tag_devset("cat", "dog"),
            holdout=tag_devset("owl"),
            allowed_deps=frozenset({"beautifulsoup4"}),
        )
        entry = optimizer.trajectory[1]
        assert entry["refusals"] == []
        assert entry["accepted"] is True
        manifest = program.to_manifest()
        # The dep lands in the pool entry exactly as parse_deps produces it...
        assert manifest["components"]["6_tools"]["tag_code"]["deps"] == ["beautifulsoup4"]
        # ...and the export env-union (PEP 723) picks it up with zero new
        # plumbing.
        assert "beautifulsoup4" in manifest["components"]["9_environment"]["python"]["dependencies"]

    def test_disallowed_dep_refuses_naming_the_allowed_set(self):
        program = Tagger()
        optimizer = run_one(
            program,
            DEP_OPS,
            holdout=tag_devset("owl"),
            allowed_deps=frozenset({"numpy"}),
        )
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "['beautifulsoup4']" in refusal
        assert "outside the allowed set ['numpy']" in refusal
        assert "allowed_deps" in refusal

    def test_dep_name_never_admits_the_import_name(self):
        # beautifulsoup4 imports as bs4: granting the DEP must not admit
        # the IMPORT. The refusal teaches the extra_imports pairing.
        program = Tagger()
        source = (
            "def tag_code(text: str) -> dict:\n"
            "    # deps: beautifulsoup4\n"
            "    import bs4\n"
            '    return {"tag": str(bs4) + text}\n'
        )
        ops = [{"op": "add_tool", "path": "self", "name": "tag_code", "python_source": source}]
        optimizer = run_one(program, ops, holdout=tag_devset("owl"), allowed_deps=frozenset({"beautifulsoup4"}))
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "['bs4']" in refusal
        assert "outside the optimizer allowlist" in refusal
        assert "extra_imports" in refusal
        assert "does NOT admit its import name" in refusal

    def test_bad_allowed_deps_refuses_at_construction(self):
        with pytest.raises(ValueError, match="allowed_deps"):
            optim.FlexIR(dspy.DummyLM([]), exact_tag, allowed_deps="beautifulsoup4")

    def test_the_reflection_prompt_spells_the_pairing_rule(self):
        optimizer = optim.FlexIR(
            dspy.DummyLM([]),
            exact_tag,
            allowed_deps=frozenset({"beautifulsoup4"}),
            extra_imports=frozenset({"bs4"}),
        )
        instructions = optimizer.reflect.signature.instructions
        assert "ONLY for these packages: beautifulsoup4" in instructions
        assert "IMPORT name must ALSO be in the import allowlist" in instructions
        # A default instance says nothing about deps at all.
        plain = optim.FlexIR(dspy.DummyLM([]), exact_tag)
        assert "# deps:" not in plain.reflect.signature.instructions
