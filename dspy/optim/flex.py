"""FlexIR: a reflection optimizer with a closed, validated edit vocabulary.

The loop, per iteration:

1. RENDER the current program for the reflection LM: the explain-view
   text, the current score, up to three failing examples (inputs,
   expected, got) — plus every refusal from the PREVIOUS iteration's
   proposals. The refusal feedback loop is the point: the reflection LM
   is taught the vocabulary by the same teaching errors a human gets.
2. CALL a reflection Predict whose signature outputs `proposals`, a JSON
   array of edit operations in a CLOSED vocabulary:

   - ``{"op": "set_instructions", "path", "text"}``
   - ``{"op": "add_demo", "path", "inputs", "labels"}``
   - ``{"op": "remove_demo", "path", "index"}``
   - ``{"op": "wrap_best_of_n", "path", "N"}`` — a STRUCTURAL edit: the
     named sub-module is wrapped in the `BestOfN` macro (built through
     the builder, declared reward leaf and all).

3. VALIDATE each proposal against the current candidate: an unknown op,
   a bad path, or a bad shape is refused loudly — the refusal is
   RECORDED (trajectory + next report), and the bad proposal is never
   partially applied. Valid proposals apply in order.
4. SCORE the candidate by engine replay over the devset; KEEP it iff the
   score strictly improves; every accepted candidate is checkpointed
   through the existing `Checkpointer` (an ordinary loadable artifact)
   and its manifest recorded, so the trajectory renders under
   `dspy.diff`.

Everything is sequential and deterministic: a scripted reflection
DummyLM makes the whole run replayable, and the optimizer performs
exactly `iterations` reflection calls.
"""

from __future__ import annotations

import json
from typing import Any, Callable

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
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import make_signature

__all__ = ["FlexIR"]

#: The closed edit vocabulary: op -> the exact non-op keys it takes.
_VOCABULARY = {
    "set_instructions": frozenset({"path", "text"}),
    "add_demo": frozenset({"path", "inputs", "labels"}),
    "remove_demo": frozenset({"path", "index"}),
    "wrap_best_of_n": frozenset({"path", "N"}),
}

_REFLECTION_INSTRUCTIONS = """You are improving a dspy program. Read the program report and propose edits.
Reply in `proposals` with a JSON array. Each element must be EXACTLY one of:
- {"op": "set_instructions", "path": "<predictor path>", "text": "<new instructions>"}
- {"op": "add_demo", "path": "<predictor path>", "inputs": {"<input field>": "..."}, "labels": {"<output field>": "..."}}
- {"op": "remove_demo", "path": "<predictor path>", "index": <demo index>}
- {"op": "wrap_best_of_n", "path": "<sub-module path>", "N": <attempts>}
The vocabulary is CLOSED: any other op, an unknown path, or an extra or missing key is refused,
and the refusal is shown to you in the next report. Propose an empty array to change nothing."""


def _reflection_signature():
    fields = {
        "program_report": (
            str,
            InputField(desc="the current program, its score, failing examples, and refused proposals"),
        ),
        "proposals": (list, OutputField(desc="a JSON array of edit operations from the closed vocabulary")),
    }
    return make_signature(fields, _REFLECTION_INSTRUCTIONS, signature_name="FlexReflection")


class FlexIR(Optimizer):
    """Reflect, propose closed-vocabulary edits, keep what scores better.

    Args:
        reflection_lm: The LM the reflection predictor runs with (any
            live `dspy.LM`; a scripted `DummyLM` makes runs replayable).
        metric: `metric(example, prediction) -> bool | float`; a result
            below 1.0 counts as a failing example in the report.
        iterations: Reflection rounds; the optimizer makes exactly this
            many reflection calls.
        reward: Optional plain reward function `reward(outputs) ->
            float` for `wrap_best_of_n` edits (the macro requires a
            declared reward leaf). Without it, wrap proposals are
            refused with a teaching error.

    Attributes:
        trajectory: After `compile`, one record per round (plus the
            baseline): `{"iteration", "label", "proposals", "refusals",
            "applied", "score", "best_score", "accepted", "checkpoint",
            "manifest"}`. Accepted records carry the candidate's full
            manifest, so any two trajectory points render under
            `dspy.diff`.

    Examples:
        ```python
        optimizer = dspy.optim.FlexIR(reflection_lm, exact_match, iterations=4)
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
    ):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError(f"FlexIR iterations must be an int >= 1, got {iterations!r}")
        self.metric = metric
        self.iterations = iterations
        self.reward = reward
        self.reflect = Predict(_reflection_signature(), lm=reflection_lm)
        self.trajectory: list[dict[str, Any]] = []

    def compile(self, program: Module, *, trainset: Any, checkpoint_dir: Any = None) -> Module:
        """Run the reflection loop; return the program with the best state.

        Args:
            program: The program to optimize (mutated in place; a
                `wrap_best_of_n` acceptance rebinds the named child).
            trainset: `dspy.Example` values with declared inputs; also
                the devset every candidate is scored on.
            checkpoint_dir: When given, the baseline and every accepted
                candidate are saved under it as loadable artifacts.
        """
        devset = check_trainset(trainset)
        checkpointer = Checkpointer(checkpoint_dir) if checkpoint_dir is not None else None
        self.trajectory = []

        baseline = evaluate(program, devset, self.metric)
        best_score = baseline.score
        best_results = baseline.results
        self.trajectory.append(
            {
                "iteration": -1,
                "label": "baseline",
                "proposals": [],
                "refusals": [],
                "applied": [],
                "score": baseline.score,
                "best_score": best_score,
                "accepted": True,
                "checkpoint": checkpointer.accept(program, score=baseline.score, label="baseline")
                if checkpointer
                else None,
                "manifest": program.to_manifest(),
            }
        )

        pending_refusals: list[str] = []
        for iteration in range(self.iterations):
            report = self._render_report(program, best_score, best_results, pending_refusals)
            proposals = self.reflect(program_report=report).proposals

            refusals: list[str] = []
            applied: list[dict[str, Any]] = []
            undo_structural: list[tuple[Module, str, Module]] = []
            snapshot = snapshot_state(program)
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
                    else:
                        refusals.append(refusal)
                result = evaluate(program, devset, self.metric) if applied else None
            except Exception:
                # An exception between the first applied edit and the
                # score comparison (a crash inside apply or evaluate)
                # would otherwise strand unscored, unreverted candidate
                # state on the live program. Unwind first, then let the
                # error propagate loudly.
                self._unwind(program, undo_structural, snapshot)
                raise

            record: dict[str, Any] = {
                "iteration": iteration,
                "label": f"iteration-{iteration}",
                "proposals": list(proposals),
                "refusals": refusals,
                "applied": applied,
                "score": None if result is None else result.score,
                "best_score": best_score,
                "accepted": False,
                "checkpoint": None,
                "manifest": None,
            }
            if result is not None:
                if result.score > best_score:  # keep iff STRICTLY better
                    best_score = result.score
                    best_results = result.results
                    record["accepted"] = True
                    record["manifest"] = program.to_manifest()
                    if checkpointer:
                        record["checkpoint"] = checkpointer.accept(
                            program, score=result.score, label=f"iteration-{iteration}"
                        )
                else:
                    self._unwind(program, undo_structural, snapshot)
            record["best_score"] = best_score
            self.trajectory.append(record)
            pending_refusals = refusals
        return program

    def _unwind(self, program: Module, undo_structural: list[tuple[Module, str, Module]], snapshot: Any) -> None:
        """Rewind one candidate fully: structure in reverse, then state.

        Every path that abandons a candidate — a rejected score AND any
        exception raised while applying or evaluating it — runs this, so
        the live program never carries unscored residue.
        """
        for parent, name, child in reversed(undo_structural):
            setattr(parent, name, child)
        apply_state(program, snapshot)
        if undo_structural:
            program.invalidate_ir()

    # ------------------------------------------------------------------
    # Render: what the reflection LM sees
    # ------------------------------------------------------------------

    def _render_report(self, program: Module, score: float, results: list, refusals: list[str]) -> str:
        lines = ["== current program =="]
        lines.append(program.explain())
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

        A proposal is atomic: it either passes every check and applies,
        or it is refused whole — never partially applied. Validation
        runs against the candidate's CURRENT state, so proposals in one
        batch see the effects of earlier valid ones.
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
            predictor.demos = [*predictor.demos, Example(**inputs, **labels).with_inputs(*inputs)]
            return None

        # remove_demo
        index = proposal["index"]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(predictor.demos):
            return f"refused {tag}: remove_demo index {index!r} is invalid — {path} has {len(predictor.demos)} demo(s)"
        predictor.demos = [demo for position, demo in enumerate(predictor.demos) if position != index]
        return None

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
