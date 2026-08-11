"""Tests for the Tier 1 workbench tools (lint, diff, cost)."""

import copy
import math

import dspy
from dspy.programir import compile
from dspy.programir.tools import cost, diff, lint


class TwoStage(dspy.Module):
    def __init__(self):
        self.draft = dspy.Predict("question -> draft", max_tokens=64)
        self.finish = dspy.Predict("question, draft -> answer", max_tokens=64)

    def forward(self, question):
        draft = self.draft(question=question)
        answer = self.finish(question=question, draft=draft.draft)
        return answer


class ThrownAway(dspy.Module):
    """The second predictor's result is computed and never read."""

    def __init__(self):
        self.main = dspy.Predict("text -> sentiment")
        self.side = dspy.Predict("text -> lang")

    def forward(self, text):
        side = self.side(text=text)
        main = self.main(text=text)
        return main


class NeverCalled(dspy.Module):
    def __init__(self):
        self.used = dspy.Predict("question -> answer")
        self.orphan = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.used(question=question)


class DeadBranch(dspy.Module):
    def __init__(self):
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        if "x" == "y":
            result = self.answer(question=question)
        else:
            result = self.answer(question=question)
        return result


class StuckWhile(dspy.Module):
    def __init__(self):
        self.answer = dspy.Predict("question -> answer")

    def forward(self, question):
        while question == "":
            result = self.answer(question=question)
        result = self.answer(question=question)
        return result


class CappedWhile(dspy.Module):
    def __init__(self):
        self.gen = dspy.Predict("question -> code")

    def forward(self, question):
        result = ""
        attempts = 0
        while result == "":
            if attempts == 2:
                break
            pred = self.gen(question=question)
            result = pred.code
            attempts = attempts + 1
        return result


class LoopWithBreak(dspy.Module):
    def __init__(self):
        self.react = dspy.Predict("question -> thought")
        self.extract = dspy.Predict("question -> answer")

    def forward(self, question):
        thought = ""
        for step in range(3):
            pred = self.react(question=question)
            if pred.thought == "finish":
                break
            thought = pred.thought
        final = self.extract(question=question)
        return final


def _manifest(program_class):
    program = program_class()
    program.set_lm(dspy.LM("openai/test-model"))
    return compile(program).to_manifest()


def _codes(findings):
    return {finding.code for finding in findings}


# ─── lint ────────────────────────────────────────────────────────────


def test_lint_clean_program_has_no_findings():
    assert lint.lint(_manifest(TwoStage)) == []


def test_lint_unused_input_field():
    manifest = _manifest(TwoStage)
    manifest["components"]["2_signature"]["finish"]["fields"].insert(
        0, {"name": "context", "direction": "input", "prefix": "Context:",
            "desc": None, "shape": {"type": "string"}, "semantic_role": None},
    )
    findings = lint.lint(manifest)
    assert _codes(findings) == {"PIR-L-FIELD-IN"}
    assert findings[0].path == "2_signature/finish/fields[0]"
    assert "'context'" in findings[0].message


def test_lint_unused_output_field():
    findings = lint.lint(_manifest(ThrownAway))
    assert _codes(findings) == {"PIR-L-FIELD-OUT"}
    assert findings[0].path.startswith("2_signature/side/")
    assert "'lang'" in findings[0].message


def test_lint_unreachable_predictor_and_unbound_pools():
    findings = lint.lint(_manifest(NeverCalled))
    unreachable = [f for f in findings if f.code == "PIR-L-UNREACH"]
    assert any("predictor 'orphan'" in f.message and f.severity == "error"
               for f in unreachable)


def test_lint_unreachable_tool_entry():
    manifest = _manifest(TwoStage)
    manifest["components"]["6_tools"]["helper"] = {
        "language": "python", "source": "tools/helper.py",
    }
    findings = [f for f in lint.lint(manifest) if f.code == "PIR-L-UNREACH"]
    assert any(f.path == "6_tools/helper" for f in findings)


def test_lint_dead_branch():
    findings = lint.lint(_manifest(DeadBranch))
    dead = [f for f in findings if f.code == "PIR-L-DEADBRANCH"]
    assert len(dead) == 1
    assert dead[0].severity == "error"
    assert dead[0].path == "5_forward/self/body[0]/test"
    assert "never runs" in dead[0].message


def test_lint_while_without_state_change():
    findings = lint.lint(_manifest(StuckWhile))
    stuck = [f for f in findings if f.code == "PIR-L-WHILE"]
    assert len(stuck) == 1
    assert "cannot terminate" in stuck[0].message


def test_lint_capped_while_is_clean():
    assert lint.lint(_manifest(CappedWhile)) == []


def test_lint_empty_except_handler():
    manifest = _manifest(TwoStage)
    manifest["components"]["5_forward"]["self"]["body"].insert(0, {
        "node": "Try",
        "body": [{"node": "Assign", "target": "probe",
                  "value": {"node": "Const", "value": "x"}}],
        "handlers": [{"type": "ToolError", "body": []}],
    })
    findings = [f for f in lint.lint(manifest) if f.code == "PIR-L-EXCEPT"]
    assert len(findings) == 1
    assert findings[0].path == "5_forward/self/body[0]/handlers[0]"


def test_lint_demo_key_mismatch():
    manifest = _manifest(TwoStage)
    manifest["components"]["3b_demos"]["draft"] = [
        {"input_keys": ["question"], "question": "q?", "bogus": "label"},
    ]
    findings = [f for f in lint.lint(manifest) if f.code == "PIR-L-DEMO"]
    assert len(findings) == 1
    assert findings[0].path == "3b_demos/draft[0]/bogus"
    assert findings[0].severity == "error"


def test_lint_report_renders():
    text = lint.build_text(_manifest(DeadBranch))
    assert "PIR-L-DEADBRANCH" in text
    assert "SUMMARY" in text


# ─── diff ────────────────────────────────────────────────────────────


def test_diff_equal_manifests_report_no_differences():
    manifest = _manifest(TwoStage)
    assert diff.diff(manifest, copy.deepcopy(manifest)) == []
    assert "no differences" in diff.build_text(manifest, copy.deepcopy(manifest))


def test_diff_reports_predictor_and_forward_deltas():
    old = _manifest(TwoStage)
    new = copy.deepcopy(old)
    components = new["components"]
    components["3a_instructions"]["draft"] = "Draft a careful answer."
    components["3b_demos"]["draft"] = [
        {"input_keys": ["question"], "question": "q?", "draft": "d."},
    ]
    components["3c_predictor_config"]["finish"]["max_tokens"] = 128
    components["5_forward"]["self"]["body"].insert(1, {
        "node": "Assign", "target": "note", "value": {"node": "Const", "value": "hi"},
    })
    text = diff.build_text(old, new)
    assert "predictor 'draft'" in text
    assert "instructions (3a)" in text
    assert "demos (3b): 0 -> 1 baked" in text
    assert "config (3c) max_tokens: 64 -> 128" in text
    assert '+ 5_forward/self/body[1]: note = "hi"' in text


def test_diff_reports_pool_changes():
    old = _manifest(TwoStage)
    new = copy.deepcopy(old)
    (lm_name,) = new["components"]["8_lm"]
    new["components"]["8_lm"][lm_name]["forward_contract"] = "chat_completions_v2"
    text = diff.build_text(old, new)
    assert f"~ lm '{lm_name}': forward_contract changed" in text


# ─── cost ────────────────────────────────────────────────────────────


def test_cost_straight_line_pipeline():
    result = cost.estimate(_manifest(TwoStage))
    assert result["calls"]["draft"] == cost.Bounds(1, 1, 1)
    assert result["calls"]["finish"] == cost.Bounds(1, 1, 1)
    assert result["total_calls"] == cost.Bounds(2, 2, 2)
    per_call = result["tokens_per_call"]["draft"]
    assert per_call["prompt_tokens"] > 0
    assert per_call["output_cap"] == 64
    assert "chars/4" in per_call["basis"]


def test_cost_for_loop_with_break():
    result = cost.estimate(_manifest(LoopWithBreak))
    react = result["calls"]["react"]
    assert react.minimum == 1
    assert react.maximum == 3
    assert result["calls"]["extract"] == cost.Bounds(1, 1, 1)


def test_cost_while_reads_break_guard_cap():
    result = cost.estimate(_manifest(CappedWhile))
    gen = result["calls"]["gen"]
    assert gen.minimum == 0
    assert gen.maximum == 3  # guard at attempts == 2 fires on the third pass
    assert not math.isinf(result["total_calls"].maximum)


def test_cost_unguarded_while_is_unbounded():
    result = cost.estimate(_manifest(StuckWhile))
    assert math.isinf(result["calls"]["answer"].maximum)
    assert "unbounded" in cost.build_text(_manifest(StuckWhile))


def test_cost_report_renders():
    text = cost.build_text(_manifest(TwoStage))
    assert "LM CALLS PER PREDICTOR" in text
    assert "chars/4 heuristic" in text
