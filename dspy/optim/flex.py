"""FlexIR: a reflection optimizer that REWRITES THE IR ITSELF.

v1 optimized a program's DATA — instructions and demos — plus one
structural macro (wrap_best_of_n). v2 adds the LEAF-IMPLEMENTATION
rewrites: the reflection LM writes ordinary Python that BECOMES a
declared authored tool leaf, and the forward tree swaps a Predict call
for that tool — an LM call replaced by code, making the program cheaper.
The unified-leaf law: same signature, different implementation.

The loop, per iteration:

1. RENDER the champion for the reflection LM (five slots, section 4 of
   the brief): the signature spec, the explain view + printed forwards,
   the closed edit catalog, the cost view (static bounds PLUS the last
   run's measured `lm_calls`), and up to three failing examples with the
   prior refusals ledger.
2. CALL one reflection Predict whose typed `proposals` output the adapter
   parses — never a regex.
3. VALIDATE each proposal against a candidate. Data ops apply through the
   predictor state surface; code ops pass the FULL admission gate
   (`code_leaf.admit_tool_source` — one undecorated def, self-contained,
   the stdlib import allowlist, the io-contract from the replaced
   signature) before the tree is rewritten with `build.py` constructors.
   Any refusal is RECORDED in the ledger, fed to the next proposal, and
   NEVER partially applied.
4. SCORE the candidate by engine replay. Acceptance is TWO-CHANNEL: a
   candidate is kept only if the dev metric holds (or rises) AND
   `lm_calls` strictly drops (cheapness), OR the score strictly rises;
   and — for EVERY candidate, not only code-bearing ones — the holdout
   split the reflection LM never sees must not regress (the reward-hacking
   guard; demos and instructions overfit exactly like a memorizing code
   leaf). Every accepted candidate is checkpointed as a loadable artifact.

Generated code is DATA to the applier: the dispatch is a closed op
table, no string templating exists anywhere, and the one code field runs
only through admission and then the engine — never exec'd by the
optimizer itself. An injection-shaped source is admitted only as a tool
leaf's carried source (and refuses if it is not self-contained or reaches
a disallowed import); it is never executed here.

Everything is sequential and deterministic under a scripted reflection
DummyLM; the optimizer performs exactly `iterations` reflection calls.
"""

from __future__ import annotations

import copy
import json
import types
from typing import Any, Callable

from dspy.core.errors import AdapterParseError
from dspy.core.example import Example
from dspy.lm.lm import LM
from dspy.modules.best_of_n import BestOfN
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.optim.base import (
    Checkpointer,
    Optimizer,
    apply_state,
    check_trainset,
    evaluate,
    snapshot_state,
)
from dspy.optim.code_leaf import ADMITTED_IMPORTS, admit_tool_source
from dspy.programir.build import (
    Assign,
    CallPredict,
    CallTool,
    Compare,
    Const,
    Except,
    If,
    Try,
    Var,
)
from dspy.programir.tools.cost import build_text as cost_build_text
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import make_signature

__all__ = ["FlexIR"]

#: The closed edit vocabulary: op -> the exact non-op keys it takes.
#: Any other op name, any missing or unexpected key, refuses with a
#: teaching error into the ledger. Edits are DATA; the applier dispatches
#: on this table and never execs or splices proposal text.
_VOCABULARY = {
    # v1 data + macro ops (byte-compatible).
    "set_instructions": frozenset({"path", "text"}),
    "add_demo": frozenset({"path", "inputs", "labels"}),
    "remove_demo": frozenset({"path", "index"}),
    "wrap_best_of_n": frozenset({"path", "N"}),
    # v2 leaf-implementation rewrites.
    "replace_predict_with_code": frozenset({"path", "tool_name", "python_source"}),
    "replace_predict_with_code_partial": frozenset({"path", "tool_name", "python_source"}),
    "delete_dead_leaf": frozenset({"path"}),
}

_CODE_OPS = frozenset({"replace_predict_with_code", "replace_predict_with_code_partial"})

_REFLECTION_INSTRUCTIONS = """You are improving a dspy program by REWRITING ITS IR. Read the program report and propose edits.
Reply in `proposals` with a JSON array. Each element must be EXACTLY one of:
- {"op": "set_instructions", "path": "<predictor path>", "text": "<new instructions>"}
- {"op": "add_demo", "path": "<predictor path>", "inputs": {"<input field>": "..."}, "labels": {"<output field>": "..."}}
- {"op": "remove_demo", "path": "<predictor path>", "index": <demo index>}
- {"op": "wrap_best_of_n", "path": "<sub-module path>", "N": <attempts>}
- {"op": "replace_predict_with_code", "path": "<predictor path>", "tool_name": "<new tool name>", "python_source": "def <fn>(<input fields>) -> dict: ..."}
- {"op": "replace_predict_with_code_partial", "path": "<predictor path>", "tool_name": "<name>", "python_source": "def <fn>(<input fields>) -> dict | None: ..."}
- {"op": "delete_dead_leaf", "path": "<predictor path with zero call sites>"}
Replace a predict with code when the failures and the cost view show the step needs NO LM judgment
(a deterministic transform, a lookup). The code's parameters must be EXACTLY the predict's input fields,
each type-hinted; it returns a dict with one key per output field. Use the _partial op when most inputs
are mechanical and a few need the model: return None to decline to the LM. Fix instructions when a
failure is about WHAT the model should do or know; change the structure when it is about HOW steps are
wired. Generated code is self-contained (no closures, no global reads) and may import only the pure
stdlib allowlist: ALLOWLIST. The vocabulary is CLOSED: any other op, unknown path, extra or missing
key, or a source that fails admission is refused, and the refusal is shown to you in the next report.
Propose an empty array to change nothing."""


def _reflection_signature():
    fields = {
        "program_report": (
            str,
            InputField(desc="the current program, its cost view, failing examples, and refused proposals"),
        ),
        "proposals": (list, OutputField(desc="a JSON array of edit operations from the closed vocabulary")),
    }
    instructions = _REFLECTION_INSTRUCTIONS.replace("ALLOWLIST", ", ".join(sorted(ADMITTED_IMPORTS)))
    return make_signature(fields, instructions, signature_name="FlexReflection")


class FlexIR(Optimizer):
    """Reflect, propose closed-vocabulary IR edits, keep what scores better and cheaper.

    Args:
        reflection_lm: The LM the reflection predictor runs with (any
            live `dspy.LM`; a scripted `DummyLM` makes runs replayable).
        metric: `metric(example, prediction) -> bool | float`; a result
            below 1.0 counts as a failing example in the report.
        iterations: Reflection rounds; the optimizer makes exactly this
            many reflection calls.
        reward: Optional plain reward function `reward(outputs) -> float`
            for `wrap_best_of_n` edits (the macro requires a declared
            reward leaf). Without it, wrap proposals are refused.
        holdout: Optional `dspy.Example` values, disjoint from the
            trainset, that the reflection LM NEVER sees. EVERY candidate
            (code, demo, or instruction) is accepted only if it also holds
            on this split — the reward-hacking guard (a candidate that
            overfits the trainset but regresses here is refused). The
            holdout MUST be disjoint from the trainset: an overlapping
            holdout is refused at compile time, since it would silently
            disable the guard. Without a holdout, candidates carry no
            held-quality guarantee and the trajectory records that.
        eps: Score tolerance for the two-channel acceptance rule.

    Attributes:
        trajectory: After `compile`, one record per round (plus the
            baseline): `{"iteration", "label", "proposals", "refusals",
            "applied", "score", "holdout_score", "lm_calls", "best_score",
            "best_lm_calls", "accepted", "checkpoint", "manifest"}`.
            Accepted records carry the candidate's full manifest, so any
            two trajectory points render under `dspy.diff`.

    Examples:
        ```python
        optimizer = dspy.optim.FlexIR(reflection_lm, exact_match, iterations=4, holdout=holdout)
        compiled = optimizer.compile(program, trainset=trainset)
        ```
    """

    def __init__(
        self,
        reflection_lm: LM,
        metric: Callable[[Example, Any], Any],
        iterations: int = 8,
        *,
        reward: Callable[[Any], float] | None = None,
        holdout: Any = None,
        eps: float = 1e-9,
    ):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError(f"FlexIR iterations must be an int >= 1, got {iterations!r}")
        self.metric = metric
        self.iterations = iterations
        self.reward = reward
        self.holdout = holdout
        self.eps = eps
        self.reflect = Predict(_reflection_signature(), lm=reflection_lm)
        self.trajectory: list[dict[str, Any]] = []

    def compile(self, program: Module, *, trainset: Any, checkpoint_dir: Any = None) -> Module:
        """Run the reflection loop; return the program with the best state.

        Args:
            program: The program to optimize (mutated in place; code and
                wrap edits rebind children and attach forward rewrites).
            trainset: `dspy.Example` values with declared inputs; also the
                devset the reflection LM sees and every candidate is
                scored on.
            checkpoint_dir: When given, the baseline and every accepted
                candidate are saved under it as loadable artifacts.
        """
        devset = check_trainset(trainset)
        holdout = check_trainset(self.holdout, name="holdout") if self.holdout is not None else None
        if holdout:
            _refuse_overlapping_holdout(devset, holdout)
        checkpointer = Checkpointer(checkpoint_dir) if checkpoint_dir is not None else None
        self.trajectory = []

        baseline = evaluate(program, devset, self.metric)
        best_score = baseline.score
        best_lm_calls = baseline.lm_calls
        best_results = baseline.results
        best_holdout = evaluate(program, holdout, self.metric).score if holdout else None
        self.trajectory.append(
            {
                "iteration": -1,
                "label": "baseline",
                "proposals": [],
                "refusals": [],
                "applied": [],
                "score": baseline.score,
                "holdout_score": best_holdout,
                "lm_calls": baseline.lm_calls,
                "best_score": best_score,
                "best_lm_calls": best_lm_calls,
                "accepted": True,
                "checkpoint": checkpointer.accept(program, score=baseline.score, label="baseline")
                if checkpointer
                else None,
                "manifest": program.to_manifest(),
            }
        )

        pending_refusals: list[str] = []
        for iteration in range(self.iterations):
            report = self._render_report(program, best_score, best_lm_calls, best_results, pending_refusals)
            refusals: list[str] = []
            try:
                proposals = self.reflect(program_report=report).proposals
            except AdapterParseError as error:
                # A whole-reply malformation (bare prose, a JSON object, a
                # quoted string, an empty value where the array belongs) is
                # the reflection LM misspeaking, not optimizer
                # misconfiguration: refuse it loudly, feed it back, keep the
                # loop alive. Program state is untouched (the crash precedes
                # any edit).
                refusals.append(f"refused reply: could not parse `proposals` as a JSON array of edit objects — {error}")
                proposals = []

            applied: list[dict[str, Any]] = []
            undo_structural: list[tuple[Module, str, Any]] = []
            snapshot = snapshot_state(program)
            has_partial = False
            if not isinstance(proposals, list):
                refusals.append(
                    f"refused reply: proposals must be a JSON array of edit objects, got {type(proposals).__name__}"
                )
                proposals = []
            try:
                for proposal in proposals:
                    refusal = self._apply_one(program, proposal, undo_structural)
                    if refusal is None:
                        applied.append(proposal)
                        if isinstance(proposal, dict) and proposal.get("op") == "replace_predict_with_code_partial":
                            has_partial = True
                    else:
                        refusals.append(refusal)
                result = evaluate(program, devset, self.metric) if applied else None
                holdout_result = evaluate(program, holdout, self.metric) if (result is not None and holdout) else None
                holdout_score = holdout_result.score if holdout_result is not None else None
            except Exception:
                # Any crash between the first applied edit and the score
                # comparison (inside apply or evaluate) would otherwise
                # strand unscored, unreverted candidate state on the live
                # program. Unwind first, then let the error propagate.
                self._unwind(program, undo_structural, snapshot)
                raise

            record: dict[str, Any] = {
                "iteration": iteration,
                "label": f"iteration-{iteration}",
                "proposals": list(proposals),
                "refusals": refusals,
                "applied": applied,
                "score": None if result is None else result.score,
                "holdout_score": holdout_score,
                "lm_calls": None if result is None else result.lm_calls,
                "best_score": best_score,
                "best_lm_calls": best_lm_calls,
                "accepted": False,
                "checkpoint": None,
                "manifest": None,
            }
            record["rejection"] = None
            if result is not None:
                verdict = self._accept(
                    result, holdout_result, best_score, best_lm_calls, best_holdout, has_partial=has_partial
                )
                if verdict is None:
                    best_score = result.score
                    best_lm_calls = result.lm_calls
                    best_results = result.results
                    # The champion holdout baseline is a MONOTONIC CEILING:
                    # every accept passed the holdout gate above, so it
                    # never sinks below the prior champion. Raising it (max)
                    # keeps a later candidate from chaining through a
                    # lowered baseline (an accepted data op must not neuter
                    # the guard for a subsequent memorizing code leaf).
                    if holdout_score is not None:
                        best_holdout = holdout_score if best_holdout is None else max(best_holdout, holdout_score)
                    record["accepted"] = True
                    record["manifest"] = program.to_manifest()
                    if checkpointer:
                        record["checkpoint"] = checkpointer.accept(
                            program, score=result.score, label=f"iteration-{iteration}"
                        )
                else:
                    # A rejected candidate is not a PROPOSAL refusal (the
                    # proposals were valid and applied); it is a scoring
                    # verdict. Keep `refusals` for proposal-level feedback
                    # (v1 parity) and record the verdict separately — both
                    # feed the next reflection call, so the LM sees why its
                    # applied edit did not win (the two-channel numbers, or
                    # the holdout regression).
                    record["rejection"] = verdict
                    self._unwind(program, undo_structural, snapshot)
            record["best_score"] = best_score
            record["best_lm_calls"] = best_lm_calls
            self.trajectory.append(record)
            pending_refusals = refusals + ([record["rejection"]] if record["rejection"] else [])
        return program

    def _accept(
        self,
        result: Any,
        holdout_result: Any,
        best_score: float,
        best_lm_calls: int,
        best_holdout: float | None,
        *,
        has_partial: bool = False,
    ) -> str | None:
        """Return None to accept, or the ledger refusal spelling the reason.

        Two-channel acceptance (brief section 3), plus the holdout gate:
          - accept if the dev score strictly rises; OR
          - accept if the dev score holds (within eps) AND `lm_calls`
            strictly drops (cheapness with held quality);
          - AND, for EVERY candidate (not only code-bearing ones), the
            holdout score must not regress below the champion's (the
            reward-hacking guard). Demos and instructions are the classic
            overfit surface: a poisoning `add_demo` that wins the dev
            split while collapsing the truly-held-out split must be
            refused exactly like a memorizing code leaf.
          - AND a `_partial` op whose fast path NEVER declined on the
            holdout split (the LM was bypassed on every held-out input) is
            refused: an always-true fast path is a full replacement wearing
            partial clothing, and its "cheapness" hides whatever the fixed
            split does not exercise. The honest op for a no-decline fast
            path is `replace_predict_with_code`, gated identically.
        """
        eps = self.eps
        holdout_score = holdout_result.score if holdout_result is not None else None
        score_rises = result.score > best_score + eps
        held_cheaper = result.score >= best_score - eps and result.lm_calls < best_lm_calls
        if not (score_rises or held_cheaper):
            return (
                f"refused candidate: neither channel improved — dev score {result.score} vs best {best_score}, "
                f"lm_calls {result.lm_calls} vs best {best_lm_calls}. Accept needs a strictly higher score, "
                "or an equal score at strictly fewer LM calls."
            )
        if best_holdout is not None:
            if holdout_score is None or holdout_score < best_holdout - eps:
                return (
                    f"refused candidate: it improved the dev split (score {result.score}, lm_calls "
                    f"{result.lm_calls}) but REGRESSED on the holdout split the reflection LM never sees "
                    f"(holdout {holdout_score} vs best {best_holdout}) — the reward-hacking guard. The edit "
                    "overfits the dev examples; write an edit that GENERALIZES, or leave the step in place."
                )
        if has_partial and holdout_result is not None and holdout_result.lm_calls == 0:
            return (
                f"refused candidate: the _partial fast path NEVER declined to the LM on the holdout split "
                f"({len(holdout_result.results)} example(s), 0 LM calls) — an always-true fast path is a FULL "
                "replacement, not a partial one, and it permanently bypasses the LM on every input the fixed "
                "split does not exercise (measured, not proven). Make the fast path return None on inputs it "
                "cannot answer, or use `replace_predict_with_code` for an honest full swap."
            )
        return None

    def _unwind(self, program: Module, undo_structural: list[tuple[Module, str, Any]], snapshot: Any) -> None:
        """Rewind one candidate fully: structure in reverse, then state.

        Every path that abandons a candidate — a rejected score, a holdout
        regression, or any exception raised while applying or evaluating —
        runs this, so the live program never carries unscored residue. A
        structural undo entry restores an attribute to its prior value, or
        DELETES it when the prior value is the `_ABSENT` sentinel (an
        injected tool leaf that had no attribute before).
        """
        for parent, name, previous in reversed(undo_structural):
            if previous is _ABSENT:
                if name in parent.__dict__:
                    delattr(parent, name)
            else:
                setattr(parent, name, previous)
        apply_state(program, snapshot)
        if undo_structural:
            program.invalidate_ir()

    # ------------------------------------------------------------------
    # Render: what the reflection LM sees (five slots, section 4)
    # ------------------------------------------------------------------

    def _render_report(self, program: Module, score: float, lm_calls: int, results: list, refusals: list[str]) -> str:
        manifest = program.to_manifest()
        lines = ["== current program =="]
        lines.append(program.explain())
        lines.append("== cost view ==")
        lines.append(cost_build_text(manifest))
        lines.append(f"measured last run: {lm_calls} LM call(s) over {len(results)} example(s)")
        lines.append("== current score ==")
        lines.append(f"{score} over {len(results)} example(s)")
        lines.append("== failing examples (up to 3) ==")
        failing = [(example, prediction, value) for example, prediction, value in results if value < 1.0][:3]
        if not failing:
            lines.append("(none)")
        for example, prediction, _ in failing:
            inputs = json.dumps(example.inputs().toDict(), sort_keys=True, ensure_ascii=False)
            expected = json.dumps(example.labels().toDict(), sort_keys=True, ensure_ascii=False)
            if prediction is None:
                got = "(the run raised a typed error)"
            else:
                got = json.dumps(prediction.toDict(), sort_keys=True, ensure_ascii=False, default=str)
            lines.append(f"- inputs {inputs} expected {expected} got {got}")
        lines.append("== refused proposals from your previous reply ==")
        if refusals:
            lines.extend(f"- {refusal}" for refusal in refusals)
        else:
            lines.append("(none)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Validate + apply: one proposal, atomically — refusal or effect
    # ------------------------------------------------------------------

    def _apply_one(self, program: Module, proposal: Any, undo_structural: list) -> str | None:
        """Apply one proposal to the candidate, or return its refusal.

        A proposal is atomic: it passes every check and applies, or it is
        refused whole — never partially applied. Validation runs against
        the candidate's CURRENT state, so proposals in one batch see the
        effects of earlier valid ones.
        """
        if not isinstance(proposal, dict):
            return f"refused {proposal!r}: a proposal is a JSON object with an 'op' key"
        tag = json.dumps(proposal, sort_keys=True, ensure_ascii=False, default=str)
        op = proposal.get("op")
        if op not in _VOCABULARY:
            return f"refused {tag}: unknown op {op!r} — the closed vocabulary is {sorted(_VOCABULARY)}"
        required = _VOCABULARY[op]
        missing = sorted(required - set(proposal))
        extra = sorted(set(proposal) - required - {"op"})
        if missing or extra:
            detail = f"; missing {missing}" if missing else ""
            detail += f"; unexpected {extra}" if extra else ""
            return f"refused {tag}: {op} takes exactly the keys {sorted(required | {'op'})}{detail}"
        if op == "wrap_best_of_n":
            return self._apply_wrap(program, proposal, tag, undo_structural)
        if op in _CODE_OPS:
            return self._apply_code(program, proposal, tag, undo_structural, partial=op.endswith("partial"))
        if op == "delete_dead_leaf":
            return self._apply_delete(program, proposal, tag, undo_structural)

        path = proposal["path"]
        if not isinstance(path, str):
            return (
                f"refused {tag}: path must be a STRING predictor path (a dotted attribute path like "
                f"'solver'), got {type(path).__name__}"
            )
        predictors = dict(program.named_predictors())
        if path not in predictors:
            return f"refused {tag}: no predictor at path {path!r} (predictor paths: {sorted(predictors)})"
        predictor = predictors[path]

        if op == "set_instructions":
            text = proposal["text"]
            if not isinstance(text, str) or not text:
                return f"refused {tag}: set_instructions text must be a non-empty string"
            predictor.signature = predictor.signature.with_instructions(text)
            return None

        if op == "add_demo":
            inputs, labels = proposal["inputs"], proposal["labels"]
            if not isinstance(inputs, dict) or not inputs or not isinstance(labels, dict) or not labels:
                return f"refused {tag}: add_demo inputs and labels must be non-empty JSON objects"
            unknown = sorted(set(inputs) - set(predictor.signature.input_fields))
            unknown += sorted(set(labels) - set(predictor.signature.output_fields))
            if unknown:
                return (
                    f"refused {tag}: add_demo names unknown field(s) {unknown} — {path} declares "
                    f"inputs {sorted(predictor.signature.input_fields)} and outputs "
                    f"{sorted(predictor.signature.output_fields)}"
                )
            candidate = Example(**inputs, **labels).with_inputs(*inputs)
            # Field NAMES are checked above; VALUES are checked by a dry-run
            # render. A demo the adapter cannot format would otherwise apply
            # cleanly and crash the next evaluate deep in prompt rendering.
            demos = [demo.toDict() if hasattr(demo, "toDict") else dict(demo) for demo in predictor.demos]
            try:
                predictor.resolve_adapter().format(
                    predictor.signature, inputs=dict(inputs), demos=[*demos, candidate.toDict()]
                )
            except Exception as error:
                return (
                    f"refused {tag}: add_demo values do not render — the adapter could not format the "
                    f"demo ({type(error).__name__}: {error}); use values the declared fields can carry"
                )
            predictor.demos = [*predictor.demos, candidate]
            return None

        # remove_demo
        index = proposal["index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(predictor.demos):
            return f"refused {tag}: remove_demo index {index!r} is invalid — {path} has {len(predictor.demos)} demo(s)"
        predictor.demos = [demo for position, demo in enumerate(predictor.demos) if position != index]
        return None

    # ------------------------------------------------------------------
    # v2 code ops: the leaf-implementation rewrites
    # ------------------------------------------------------------------

    def _apply_code(
        self, program: Module, proposal: dict, tag: str, undo_structural: list, *, partial: bool
    ) -> str | None:
        """Swap a predict call for admitted code (full or partial fallback).

        The reflection LM's `python_source` becomes a DECLARED tool leaf on
        the module that owns the predict; the call site in that module's
        forward is rewritten with `build.py` constructors — full replace
        swaps `CallPredict -> CallTool`; partial wraps a Try/If fallback
        around the fast path (a decline or a `ToolError` uses the LM).
        """
        path = proposal["path"]
        tool_name = proposal["tool_name"]
        source = proposal["python_source"]
        if not isinstance(path, str):
            return f"refused {tag}: path must be a STRING predictor path, got {type(path).__name__}"
        if not isinstance(tool_name, str) or not tool_name:
            return f"refused {tag}: tool_name must be a non-empty string"
        predictors = dict(program.named_predictors())
        if path not in predictors:
            return f"refused {tag}: no predictor at path {path!r} (predictor paths: {sorted(predictors)})"
        predictor = predictors[path]

        owner, attribute = self._owner_of(program, path)
        if owner is None:
            return (
                f"refused {tag}: predictor {path!r} is the root program leaf; a bare Predict has no "
                "enclosing forward to rewrite — wrap it in a composite Module first"
            )
        if tool_name in owner.__dict__:
            return f"refused {tag}: attribute {tool_name!r} already exists on the module owning {path!r}"

        # Locate the owning module's forward tree; refuse a 0.4 splat site
        # (the applier keys on call-site nodes; a splat-bearing site is not
        # yet rewritable here — a teaching refusal, never a mis-rewrite).
        try:
            tree = self._forward_tree_of(program, owner)
        except _SplatSiteError:
            return (
                f"refused {tag}: {path!r}'s call site is a 0.4 record-splat site; the code rewrite is not "
                "yet supported for splat-bearing calls (node-set 0.4 propagation pending)"
            )
        sites = _find_predict_sites(tree, attribute)
        if not sites:
            return f"refused {tag}: found no CallPredict site for {path!r} in its owning forward"

        try:
            admitted = admit_tool_source(tool_name, source, predictor.signature, partial=partial)
        except ValueError as error:
            return f"refused {tag}: python_source rejected by admission — {error}"

        new_tree = copy.deepcopy(tree)
        counter = _counter_start(new_tree)
        if partial:
            rewritten = _rewrite_partial(new_tree, attribute, tool_name, counter)
            if rewritten == 0:
                # A predict site EXISTS (checked above) but none is a
                # top-level `Assign(target, CallPredict)` — the only shape
                # the partial Try/If wraps. A `return self.leaf(...)` or a
                # nested site would silently no-op while `applied` claimed
                # success and no cheapness landed. Refuse with a teaching
                # error so the reflection LM sees the mismatch and can
                # reshape the call site or use the full replace op instead.
                return (
                    f"refused {tag}: {path!r}'s predict call site is not a top-level "
                    "`result = self.<leaf>(...)` assignment (e.g. it is `return self.<leaf>(...)` or a nested "
                    "call), which the partial fast-path/fallback shape cannot wrap. Rewrite the forward to "
                    "assign the call to a variable first, or use `replace_predict_with_code` for a full swap."
                )
        else:
            _rewrite_full(new_tree, attribute, tool_name)

        # Bind the tool leaf on the owner and attach the rewritten forward.
        previous_attr = owner.__dict__.get(tool_name, _ABSENT)
        previous_builder = _bound_builder(owner)
        setattr(owner, tool_name, admitted.function)
        _attach_forward(owner, new_tree)
        program.invalidate_ir()
        undo_structural.append((owner, tool_name, previous_attr))
        undo_structural.append((owner, "build_forward_ir", previous_builder))
        return None

    def _apply_delete(self, program: Module, proposal: dict, tag: str, undo_structural: list) -> str | None:
        """Remove a predictor leaf with ZERO call sites across all forwards.

        The clean converse of the code ops: after a full replace, the old
        predict is dead. Count its call sites across every module's forward
        first; refuse (with the site list) if any remain. Link would refuse
        a dangling ref anyway; this keeps accepted artifacts orphan-free.
        """
        path = proposal["path"]
        if not isinstance(path, str):
            return f"refused {tag}: path must be a STRING predictor path, got {type(path).__name__}"
        predictors = dict(program.named_predictors())
        if path not in predictors:
            return f"refused {tag}: no predictor at path {path!r} (predictor paths: {sorted(predictors)})"
        owner, attribute = self._owner_of(program, path)
        if owner is None:
            return f"refused {tag}: cannot delete the root program leaf {path!r}"

        remaining = self._count_sites(program, attribute)
        if remaining:
            return (
                f"refused {tag}: predictor {path!r} still has {remaining} live call site(s); replace them "
                "with code (or remove the calls) before deleting the leaf"
            )
        previous = owner.__dict__.get(attribute, _ABSENT)
        if previous is _ABSENT:
            return f"refused {tag}: the module owning {path!r} has no attribute {attribute!r} to delete"
        delattr(owner, attribute)
        program.invalidate_ir()
        undo_structural.append((owner, attribute, previous))
        return None

    def _count_sites(self, program: Module, attribute: str) -> int:
        """Count live CallPredict sites naming `attribute` across all forwards."""
        manifest = program.to_manifest()
        total = 0
        for forward in manifest["components"]["5_forward"].values():
            total += len(_find_predict_sites(forward, attribute))
        return total

    def _owner_of(self, program: Module, path: str) -> tuple[Module | None, str]:
        """The module owning a dotted predictor path, and the attribute name.

        `self.solver` -> (root, "solver"); `inner.react` -> (the module at
        `inner`, "react"). The root leaf path (`self`, a bare Predict
        program) owns no enclosing forward and returns (None, "self").
        """
        if path == "self":
            return None, "self"
        parts = path.split(".")
        attribute = parts[-1]
        parent_path = "self" if len(parts) == 1 else ".".join(parts[:-1])
        modules = dict(program._named_modules())
        return modules.get(parent_path), attribute

    def _forward_tree_of(self, program: Module, owner: Module) -> dict:
        """The owning module's CURRENT compiled forward tree (a fresh copy).

        Compiles the champion and reads the forward keyed by the owner's
        path; a prior code edit's `build_forward_ir` is already reflected
        because it ran at that compile. Raises `_SplatSiteError` when the tree
        carries a record-splat leaf call (the 0.4 guard).
        """
        owner_path = next(path for path, module in program._named_modules() if module is owner)
        manifest = program.to_manifest()
        tree = manifest["components"]["5_forward"][owner_path]
        for _path, call in _iter_leaf_calls(tree):
            if "splat" in call:
                raise _SplatSiteError(owner_path)
        return tree

    def _apply_wrap(self, program: Module, proposal: dict, tag: str, undo_structural: list) -> str | None:
        if self.reward is None:
            return (
                f"refused {tag}: wrap_best_of_n needs FlexIR(reward=...) — the BestOfN macro requires "
                "a declared reward leaf"
            )
        n = proposal["N"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 2:
            return f"refused {tag}: wrap_best_of_n N must be an int >= 2, got {n!r}"
        path = proposal["path"]
        if not isinstance(path, str) or not path or path == "self":
            return (
                f"refused {tag}: wrap_best_of_n path names a sub-module attribute (dotted), never the "
                "root — wrap the whole program by constructing BestOfN around it yourself"
            )
        parent: Module = program
        parts = path.split(".")
        for part in parts[:-1]:
            child = parent.__dict__.get(part)
            if not isinstance(child, Module):
                return f"refused {tag}: path {path!r} does not resolve — {part!r} is not a sub-module"
            parent = child
        name = parts[-1]
        target = parent.__dict__.get(name)
        if not isinstance(target, Module):
            module_paths = sorted(module_path for module_path, _ in program._named_modules() if module_path != "self")
            return f"refused {tag}: no sub-module at path {path!r} (sub-module paths: {module_paths})"
        try:
            wrapped = BestOfN(target, n, self.reward)
        except (TypeError, ValueError) as error:
            return f"refused {tag}: BestOfN construction refused — {error}"
        setattr(parent, name, wrapped)
        program.invalidate_ir()
        undo_structural.append((parent, name, target))
        return None


# ---------------------------------------------------------------------------
# Holdout integrity: the reward-hacking guard is only real if the holdout
# is genuinely unseen
# ---------------------------------------------------------------------------


def _refuse_overlapping_holdout(devset: list[Example], holdout: list[Example]) -> None:
    """Refuse when any holdout input also appears in the trainset.

    The reward-hacking guard measures a candidate on a split the reflection
    LM never sees. A holdout that overlaps (or equals) the trainset is not
    held out at all: a pure memorizer then aces both and ships. A natural
    caller mistake (passing the same list, or a random split that duplicates
    rows) would SILENTLY disable the primary defense, so fail closed with a
    teaching error rather than trust it.
    """
    train_inputs = {_input_key(example) for example in devset}
    overlap = sorted({_input_key(example) for example in holdout} & train_inputs)
    if overlap:
        raise ValueError(
            f"FlexIR holdout overlaps the trainset on {len(overlap)} input(s) (e.g. {overlap[:3]}); the "
            "reward-hacking guard needs a holdout the reflection LM never sees. Pass a holdout DISJOINT "
            "from trainset, or split one labeled set so disjointness is guaranteed."
        )


def _input_key(example: Example) -> str:
    """A stable identity for an example's inputs (order-independent)."""
    return json.dumps(example.inputs().toDict(), sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Structural helpers: locate and rewrite call sites with build.py constructors
# ---------------------------------------------------------------------------

#: Sentinel: an undo entry whose "previous value" is "the attribute did
#: not exist" (an injected tool leaf, deleted on unwind).
_ABSENT = object()


class _SplatSiteError(Exception):
    """Raised when a forward carries a 0.4 record-splat leaf call."""


def _iter_leaf_calls(tree: dict):
    """Yield `(path, call_node)` for every leaf Call in a forward tree."""
    from dspy.programir.contract_validate import iter_calls

    return iter_calls(tree)


def _find_predict_sites(tree: dict, attribute: str) -> list[tuple[tuple, dict]]:
    """Every `CallPredict(attribute)` site in a forward tree, with its path."""
    sites = []
    for path, call in _iter_leaf_calls(tree):
        leaf = call.get("leaf", {})
        if leaf.get("kind") == "predict" and leaf.get("ref") == attribute:
            sites.append((path, call))
    return sites


def _rewrite_full(tree: dict, attribute: str, tool_name: str) -> None:
    """Swap every `CallPredict(attribute)` leaf to `CallTool(tool_name)`.

    Downstream `Attr(target, field)` reads keep working: a tool returning
    the output record as a dict is field-addressable exactly where a
    prediction was (SEM-7). The predict leaf usually goes dead — clean it
    with delete_dead_leaf.
    """
    for _path, call in _find_predict_sites(tree, attribute):
        call["leaf"] = {"kind": "tool", "ref": tool_name}


def _rewrite_partial(tree: dict, attribute: str, tool_name: str, counter: list[int]) -> int:
    """Wrap each `target = CallPredict(attribute)(**kw)` in a Try/If fallback.

    The exact tree shape (brief section 2):

        Try(
            body=[Assign(F, CallTool(tool_name, **kw))],
            handlers=[Except("ToolError", body=[Assign(F, Const(None))])],
        ),
        If(
            Compare("ne", Var(F), Const(None)),
            body=[Assign(target, Var(F))],
            orelse=[Assign(target, CallPredict(path, **kw))],
        )

    `F` is a fresh hygienic name (`_flex_fast_<n>`). Both arms bind
    `target` to a field-addressable record, so downstream reads are
    unchanged. Rewriting statement lists in place preserves order. Returns
    the number of sites actually wrapped, so the applier can refuse a
    proposal whose predict site is not a rewritable top-level Assign
    (rather than silently no-op while reporting the op as applied).
    """
    return _rewrite_partial_body(tree["body"], attribute, tool_name, counter)


def _rewrite_partial_body(body: list, attribute: str, tool_name: str, counter: list[int]) -> int:
    rewritten = 0
    index = 0
    while index < len(body):
        statement = body[index]
        replacement = _partial_replacement(statement, attribute, tool_name, counter)
        if replacement is not None:
            body[index : index + 1] = replacement
            index += len(replacement)
            rewritten += 1
            continue
        for key in ("body", "orelse"):
            if isinstance(statement.get(key), list):
                rewritten += _rewrite_partial_body(statement[key], attribute, tool_name, counter)
        for handler in statement.get("handlers", []) or []:
            rewritten += _rewrite_partial_body(handler["body"], attribute, tool_name, counter)
        index += 1
    return rewritten


def _partial_replacement(statement: dict, attribute: str, tool_name: str, counter: list[int]) -> list | None:
    """The Try/If fallback for `target = CallPredict(attribute)(**kw)`, else None.

    Only a top-level `Assign(target, CallPredict(attribute))` is rewritten
    — the mechanical call-site shape a predict occupies. A predict call
    buried inside another expression is left for the full op.
    """
    if statement.get("node") != "Assign":
        return None
    value = statement.get("value", {})
    leaf = value.get("leaf", {}) if isinstance(value, dict) else {}
    if not (value.get("node") == "Call" and leaf.get("kind") == "predict" and leaf.get("ref") == attribute):
        return None
    target = statement["target"]
    kwargs = {key: _as_node(node) for key, node in value.get("kwargs", {}).items()}
    fresh = f"_flex_fast_{counter[0]}"
    counter[0] += 1
    return [
        Try(
            [Assign(fresh, CallTool(tool_name, **kwargs))],
            [Except("ToolError", [Assign(fresh, Const(None))])],
        ),
        If(
            Compare("ne", Var(fresh), Const(None)),
            [Assign(target, Var(fresh))],
            orelse=[Assign(target, CallPredict(attribute, **kwargs))],
        ),
    ]


def _as_node(node: Any) -> Any:
    """Pass an already-built node dict straight through the constructors."""
    return node


def _counter_start(tree: dict) -> list[int]:
    """Next free `_flex_fast_<n>` index, so repeated partials never collide."""
    used = -1
    for name in _assigned_names(tree["body"]):
        if name.startswith("_flex_fast_"):
            suffix = name[len("_flex_fast_") :]
            if suffix.isdigit():
                used = max(used, int(suffix))
    return [used + 1]


def _assigned_names(body: list):
    for statement in body:
        if not isinstance(statement, dict):
            continue
        if statement.get("node") == "Assign":
            yield statement["target"]
        for key in ("body", "orelse"):
            if isinstance(statement.get(key), list):
                yield from _assigned_names(statement[key])
        for handler in statement.get("handlers", []) or []:
            yield from _assigned_names(handler["body"])


def _attach_forward(module: Module, tree: dict) -> None:
    """Attach a `build_forward_ir` returning the rewritten tree (macro door).

    Re-runs the shared admission each compile (`Forward` -> `admit_forward`)
    with the module's current leaf table; the printer's round-trip law
    makes the printed source the reviewable artifact form.
    """

    def build_forward_ir(self, _tree=tree):
        return _tree

    module.build_forward_ir = types.MethodType(build_forward_ir, module)


def _bound_builder(module: Module) -> Any:
    """The module's instance-bound `build_forward_ir`, or `_ABSENT`.

    A class-level `build_forward_ir` (a macro like BestOfN) is left alone —
    only an instance binding this optimizer attached is captured for undo.
    """
    return module.__dict__.get("build_forward_ir", _ABSENT)
