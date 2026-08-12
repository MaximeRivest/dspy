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

v3 generalizes the closed vocabulary from "swap one leaf" to "restructure
the program": `add_predict` and `add_tool` bind NEW leaves on an owner
module, and `rewrite_forward` replaces a module's whole forward with an
authored source in the printer's dialect, lowered through the SAME
`compile_forward` admission the normal compile path runs. The reflection
LM can now do anything to the ProgramIR — but every path stays gated by
the same wisdom: the dispatch table stays closed, code and forwards enter
only through admission, every batch applies-on-copy and unwinds whole,
the holdout gate covers every candidate, and cheapness is priced by
`lm_calls`. Security is tunable, not silent: `code_trust` chooses whether
authored leaves fail closed at load (`"isolated"`, the default) or run
in-process for the optimizing user's own loop; `extra_imports` widens the
import allowlist per instance, never globally.

Everything is sequential and deterministic under a scripted reflection
DummyLM; the optimizer performs exactly `iterations` reflection calls.
"""

from __future__ import annotations

import ast
import copy
import json
import os
import re
import subprocess
import tempfile
import types
from pathlib import Path
from typing import Any, Callable

from dspy.core.errors import AdapterParseError
from dspy.core.example import Example
from dspy.core.prediction import Prediction
from dspy.lm.lm import LM
from dspy.modules.best_of_n import BestOfN
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.optim.base import (
    Checkpointer,
    EvaluationResult,
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
from dspy.programir.engine.isolation import IsolationLevel as _IsolationLevel
from dspy.programir.errors import ProgramIRRefusal
from dspy.programir.forward import compile_forward
from dspy.programir.printer import render_forward
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
    # v3 general structure edits: the reflection LM can add leaves and
    # rewrite whole forwards — every path still gated by the same wisdom
    # (closed dispatch, admission, apply-on-copy + unwind, holdout gate,
    # cheapness channel, the teaching-refusal ledger).
    "add_predict": frozenset({"path", "name", "signature", "instructions"}),
    "add_tool": frozenset({"path", "name", "python_source"}),
    "rewrite_forward": frozenset({"path", "python_source"}),
}

_CODE_OPS = frozenset({"replace_predict_with_code", "replace_predict_with_code_partial"})

#: Optional keys an op ACCEPTS beyond its required set (still closed — a
#: key outside (required + optional) refuses). PIR-021: add_tool may carry
#: the invocation discriminant `kind` and the static effect row `grants`.
_OPTIONAL_KEYS = {
    "add_tool": frozenset({"kind", "grants"}),
}

_REFLECTION_INSTRUCTIONS = """You are improving a dspy program by REWRITING ITS IR. Read the program report and propose edits.
Reply in `proposals` with a JSON array. Each element must be EXACTLY one of:
- {"op": "set_instructions", "path": "<predictor path>", "text": "<new instructions>"}
- {"op": "add_demo", "path": "<predictor path>", "inputs": {"<input field>": "..."}, "labels": {"<output field>": "..."}}
- {"op": "remove_demo", "path": "<predictor path>", "index": <demo index>}
- {"op": "wrap_best_of_n", "path": "<sub-module path>", "N": <attempts>}
- {"op": "replace_predict_with_code", "path": "<predictor path>", "tool_name": "<new tool name>", "python_source": "def <fn>(<input fields>) -> dict: ..."}
- {"op": "replace_predict_with_code_partial", "path": "<predictor path>", "tool_name": "<name>", "python_source": "def <fn>(<input fields>) -> dict | None: ..."}
- {"op": "delete_dead_leaf", "path": "<predictor or authored-tool path with zero call sites>"}
- {"op": "add_predict", "path": "<owner module path, 'self' for the root>", "name": "<new attribute name>", "signature": "<'input_a, input_b -> output_c'>", "instructions": "<the new leaf's prompt — required>"}
- {"op": "add_tool", "path": "<owner module path, 'self' for the root>", "name": "<new attribute name>", "python_source": "def <fn>(<typed params>) -> dict: ..."}
- {"op": "rewrite_forward", "path": "<owner module path, 'self' for the root>", "python_source": "def forward(self, <args>): ..."}
For rewrite_forward, write the forward EXACTLY in the dialect shown in the program report; paths and leaf
names come from the report; the node set is CLOSED — unsupported Python refuses. A new predict or tool is
only reachable after a rewrite_forward in the SAME proposal list wires a call to it.
add_tool also takes optional "kind" ("call" default, or "session" for a stateful leaf) and "grants" (a list
of existing leaf names a session leaf may call back into through its bridge — the session function then takes
the grant bridge as its FIRST parameter and reaches only those leaves).
Use add_predict + rewrite_forward to DECOMPOSE a step; use add_tool + rewrite_forward to insert
deterministic code; small data edits first when the failure is about WHAT, structure edits when it is
about HOW.
Replace a predict with code when the failures and the cost view show the step needs NO LM judgment
(a deterministic transform, a lookup). The code's parameters must be EXACTLY the predict's input fields,
each type-hinted; it returns a dict with one key per output field. Use the _partial op when most inputs
are mechanical and a few need the model: return None to decline to the LM. Fix instructions when a
failure is about WHAT the model should do or know; change the structure when it is about HOW steps are
wired. Generated code is self-contained (no closures, no global reads) and may import only the pure
stdlib allowlist: ALLOWLIST. The vocabulary is CLOSED: any other op, unknown path, extra or missing
key, or a source that fails admission is refused, and the refusal is shown to you in the next report.
Propose an empty array to change nothing."""


def _reflection_signature(extra_imports: frozenset[str] | None = None, allowed_deps: frozenset[str] | None = None):
    fields = {
        "program_report": (
            str,
            InputField(desc="the current program, its cost view, failing examples, and refused proposals"),
        ),
        "proposals": (list, OutputField(desc="a JSON array of edit operations from the closed vocabulary")),
    }
    allowlist = ADMITTED_IMPORTS | (extra_imports or frozenset())
    instructions = _REFLECTION_INSTRUCTIONS.replace("ALLOWLIST", ", ".join(sorted(allowlist)))
    if allowed_deps:
        instructions += (
            " Generated code may declare `# deps: ...` ONLY for these packages: "
            + ", ".join(sorted(allowed_deps))
            + ". A dep's IMPORT name must ALSO be in the import allowlist above "
            "(e.g. beautifulsoup4 imports as bs4); otherwise the import refuses."
        )
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
        code_trust: `"isolated"` (the default) keeps the trust pairing
            rule: every optimizer-authored code leaf carries the
            isolation-required rung and FAILS CLOSED at load unless the
            receiver grants it a reviewed callable. `"in_process"` gives
            authored leaves the in-process placement, so the optimizing
            user's own loop and saved artifact run WITHOUT a grant
            ceremony. This is NOT a sandbox and claims none: in-process
            optimizer-authored code runs with FULL AMBIENT AUTHORITY (your
            filesystem, your process); it is a choice you make for your
            own loop. The `authored_by: "optimizer"` provenance survives
            in the pool entry either way, so a receiver can audit or
            re-place the leaf.
        extra_imports: Additional module names appended to the stdlib
            import allowlist for THIS optimizer instance only (the
            module-level `ADMITTED_IMPORTS` is never mutated). Widening
            the allowlist widens what generated code can do; the
            denylisted builtins and dunder walks stay denied regardless.
        allowed_deps: Package names (as they appear on `# deps:` lines,
            e.g. "beautifulsoup4") that optimizer-authored code may
            declare as third-party dependencies. The default (None) keeps
            today's law: any `# deps:` refuses. Admitted deps ride the
            pool entry's `deps[]` into the artifact's environment block
            (PEP 723 union) with no extra plumbing. Two deliberate-act
            notes: (1) a dep typically implies IMPURE code, so pairing
            this with `code_trust` and `extra_imports` is the user's own
            choice, made knowingly; (2) a dep name does NOT admit its
            import name — beautifulsoup4 imports as bs4, and the import
            allowlist stays governed by `extra_imports` alone, so grant
            the IMPORT name there as well.
        auto_install: When True (default False — a deliberate door,
            default closed), a dep-carrying leaf that passes admission
            has its missing granted packages installed BEFORE the
            candidate is evaluated, via `uv pip install` into the
            CURRENT interpreter's environment. Know what that means:
            installing a package executes third-party build code; a
            REJECTED candidate's installs are NOT unwound (the env
            drifts from the lock until the next export re-locks); and it
            needs network (or a warm uv cache). An install failure is a
            teaching refusal into the ledger — the proposal's fault, not
            an infra abort. Only meaningful alongside `allowed_deps`.
        eval_mode: WHERE candidates are scored — never what is accepted
            (acceptance, holdout gate, ledger, and unwind are identical
            in both modes). `"in_process"` (default) scores through the
            live engine as always. `"artifact"` exports every candidate
            (baseline included) to a temp artifact and scores it in a
            SUBPROCESS under that artifact's own environment — scoring
            semantics == deployment semantics, environment included.
            Environments are cached by lockfile hash (under
            `<checkpoint_dir>/.envs`, else `~/.cache/dspy-flexir`), so
            only candidates that change deps pay a resolve. Artifact
            mode requires a SELF-CONTAINED metric function (its source
            travels to the child; refused at compile otherwise) and
            serializes scripted DummyLM state to the child; real-LM
            child binding is not implemented yet — use in_process for
            live providers.
        eval_env_overrides: Artifact-mode only: distribution names
            mapped to local paths installed EDITABLE into the child env
            in place of the locked release. Defaults to `{"dspy":
            <this repo>}` — the manifest pins `dspy==<greenfield
            version>`, which is not the PyPI dspy, so the child must get
            the running tree.
        _eval_same_env: TEST-ONLY escape hatch: artifact-mode children
            run under the current interpreter instead of a freshly built
            uv environment, so the export/harness/protocol/caching logic
            is testable offline. Never set this outside tests.

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
        isolation_floor: str = "fork_ratchet",
        code_trust: str | None = None,
        extra_imports: frozenset[str] | None = None,
        allowed_deps: frozenset[str] | None = None,
        auto_install: bool = False,
        eval_mode: str = "in_process",
        eval_env_overrides: dict[str, Any] | None = None,
        scoring_isolation: str = "none",
        broker_egress: frozenset[str] | None = None,
        _eval_same_env: bool = False,
        _isolation_backend: Any = None,
    ):
        if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
            raise ValueError(f"FlexIR iterations must be an int >= 1, got {iterations!r}")
        from dspy.programir.engine.isolation import parse_level

        # code_trust migrated to isolation_floor over the D-042 vocabulary.
        # Old values stay as documented aliases (no deprecation warning —
        # Maxime's call later): "isolated" -> "fork_ratchet" (authored code
        # walled off, the old isolation-required rung), "in_process" ->
        # "none" (runs in the loop's own process, the old placement).
        if code_trust is not None:
            aliases = {"isolated": "fork_ratchet", "in_process": "none"}
            if code_trust not in aliases:
                raise ValueError(
                    f"FlexIR code_trust must be 'isolated' or 'in_process' (aliases of isolation_floor), "
                    f"got {code_trust!r}"
                )
            isolation_floor = aliases[code_trust]
        self._floor_level = parse_level(isolation_floor)
        if extra_imports is not None and (
            isinstance(extra_imports, str) or not all(isinstance(name, str) for name in extra_imports)
        ):
            raise ValueError(f"FlexIR extra_imports must be an iterable of module-name strings, got {extra_imports!r}")
        if allowed_deps is not None and (
            isinstance(allowed_deps, str) or not all(isinstance(name, str) for name in allowed_deps)
        ):
            raise ValueError(f"FlexIR allowed_deps must be an iterable of package-name strings, got {allowed_deps!r}")
        if not isinstance(auto_install, bool):
            raise ValueError(f"FlexIR auto_install must be a bool, got {auto_install!r}")
        if eval_mode not in ("in_process", "artifact"):
            raise ValueError(f"FlexIR eval_mode must be 'in_process' or 'artifact', got {eval_mode!r}")
        self._scoring_level = parse_level(scoring_isolation)
        if broker_egress is not None and (
            isinstance(broker_egress, str) or not all(isinstance(name, str) for name in broker_egress)
        ):
            raise ValueError(f"FlexIR broker_egress must be an iterable of hostname strings, got {broker_egress!r}")
        self.metric = metric
        self.iterations = iterations
        self.reward = reward
        self.holdout = holdout
        self.eps = eps
        self.isolation_floor = self._floor_level.name
        # The authored-leaf placement stamp derives from the floor: floor
        # `none` runs in-process (the old "in_process" trust), any wall at
        # or above `fork` stamps the isolation-required rung (the old
        # "isolated"). `admit_tool_source` still speaks the two-value
        # code_trust; the floor is the source of truth now.
        self.code_trust = "in_process" if self._floor_level == _NONE_LEVEL else "isolated"
        self.extra_imports = frozenset(extra_imports) if extra_imports is not None else None
        self.allowed_deps = frozenset(allowed_deps) if allowed_deps is not None else None
        self.auto_install = auto_install
        self.eval_mode = eval_mode
        self.eval_env_overrides = eval_env_overrides
        self._eval_same_env = _eval_same_env
        self.scoring_isolation = self._scoring_level.name
        self.broker_egress = frozenset(broker_egress) if broker_egress is not None else frozenset()
        self._isolation_backend = _isolation_backend
        self._last_envelope: dict[str, Any] | None = None
        self._env_cache: Any = None
        self._metric_source: str | None = None
        self._script_lms: dict[str, Any] = {}
        self.reflect = Predict(_reflection_signature(self.extra_imports, self.allowed_deps), lm=reflection_lm)
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
        if self.eval_mode == "artifact":
            self._prepare_artifact_mode(checkpoint_dir)

        baseline = self._evaluate(program, devset)
        best_score = baseline.score
        best_lm_calls = baseline.lm_calls
        best_results = baseline.results
        self._best_attribution = baseline.attribution
        best_holdout = self._evaluate(program, holdout).score if holdout else None
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
                "envelope": self._last_envelope,
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
                result = self._evaluate(program, devset) if applied else None
                holdout_result = self._evaluate(program, holdout) if (result is not None and holdout) else None
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
                "envelope": self._last_envelope if result is not None else None,
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
                    self._best_attribution = result.attribution
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

    def _ensure_leaf_deps(self, admitted: Any, tag: str) -> str | None:
        """Rung 2: install an admitted leaf's granted deps, or refuse.

        Runs only under `auto_install=True` and only for a leaf that
        actually carries deps; the default is exactly the prior
        behavior (nothing checked, nothing installed). Refusal happens
        BEFORE any structural change, so the proposal stays atomic —
        refused whole, never half-applied. The installs themselves are
        NOT unwound on a later rejection (see the auto_install docs).
        """
        deps = admitted.leaf.entry.get("deps") or []
        if not deps or not self.auto_install:
            return None
        from dspy.optim import env_prepare

        problem = env_prepare.ensure_deps(deps)
        if problem is not None:
            return f"refused {tag}: {problem}"
        return None

    # ------------------------------------------------------------------
    # Score: in-process, or as an exported artifact in its own env
    # ------------------------------------------------------------------

    def _evaluate(self, program: Module, dataset: Any):
        """Score one candidate: WHERE depends on eval_mode, never WHAT."""
        if self.eval_mode == "in_process":
            return evaluate(program, dataset, self.metric)
        return self._evaluate_as_artifact(program, dataset)

    def _prepare_artifact_mode(self, checkpoint_dir: Any) -> None:
        """Set up artifact scoring: the metric source and the env cache.

        The metric's source travels to the scoring child, so it must be
        a named, self-contained function (the same law authored leaves
        obey) — refused HERE, at compile time, with a teaching error,
        not deep inside the first child run.
        """
        from dspy.optim.env_prepare import REPO_ROOT, EnvCache
        from dspy.programir.leaves import _check_self_contained, _source

        subject = "FlexIR artifact-mode metric"
        try:
            source = _source(self.metric, subject=subject)
            _check_self_contained(self.metric, source, subject=subject)
        except ValueError as error:
            raise ValueError(
                f"FlexIR eval_mode='artifact' sends the metric's SOURCE to the scoring child, so the metric "
                f"must be a named, self-contained function — {error}"
            ) from error
        self._metric_source = source
        cache_root = (
            Path(checkpoint_dir) / ".envs" if checkpoint_dir is not None else Path.home() / ".cache/dspy-flexir"
        )
        self._env_cache = EnvCache(
            cache_root,
            overrides=self.eval_env_overrides or {"dspy": REPO_ROOT},
            same_env=self._eval_same_env,
        )
        self._prepare_isolation()

    def _prepare_isolation(self) -> None:
        """Build the scoring envelope; refuse loudly if the host under-runs it.

        `scoring_isolation="none"` is byte-identical to before (no
        backend, no preexec, no envelope record). Any higher level asks
        the Linux backend for the best-effort wall; a request the host
        cannot honestly establish is a loud `IsolationDowngrade`, never a
        silent weaker wall.
        """
        from dspy.programir.engine.isolation import (
            IsolationLevel,
            IsolationPolicy,
            LinuxIsolationBackend,
        )

        if self._scoring_level == IsolationLevel.none:
            self._isolation_policy = IsolationPolicy(level=IsolationLevel.none)
            return
        backend = self._isolation_backend or LinuxIsolationBackend()
        self._isolation_backend = backend
        reached = backend.best_effort_level(self._scoring_level)  # raises IsolationDowngrade if under-floor
        self._isolation_policy = IsolationPolicy(level=reached, broker_egress=self.broker_egress)

    def _evaluate_as_artifact(self, program: Module, dataset: Any):
        """Export the candidate; score it in a child under its own env.

        The child runs the SAME `evaluate` this class runs in-process,
        so the sacred distinction survives the boundary: catchable
        program errors score 0.0 per example inside the child; anything
        that escapes (an unloadable artifact, an engine guard) exits the
        child non-zero and raises HERE — infrastructure is never blamed
        on the candidate.
        """
        with tempfile.TemporaryDirectory(prefix="flexir-artifact-") as tmp:
            artifact = Path(tmp) / "artifact"
            program.save(artifact)
            interpreter = self._env_cache.interpreter_for(artifact)
            specs, extra_env = self._serialize_lms(program)
            child_env, broker, self._last_envelope = self._child_channel(extra_env, tmp)
            job = {
                "artifact": str(artifact),
                "examples": [
                    {"values": example.toDict(), "input_keys": list(example.inputs().keys())} for example in dataset
                ],
                "metric_source": self._metric_source,
                "lm": specs,
            }
            payload_text = json.dumps(job, default=str)
            # Belt and suspenders: NOTHING credential-shaped touches disk.
            # The serialization above already routes secrets into env-var
            # NAMES; this scan proves it held, for the job payload AND the
            # exported artifact, before either is used.
            self._assert_secret_free(payload_text.encode("utf-8"), "the scoring job payload")
            for file in sorted(artifact.rglob("*")):
                if file.is_file():
                    self._assert_secret_free(file.read_bytes(), f"the exported artifact file {file.name!r}")
            job_path = Path(tmp) / "job.json"
            job_path.write_text(payload_text)
            preexec = None
            if self._scoring_level != _NONE_LEVEL and self._isolation_backend is not None:
                preexec = self._isolation_backend.child_preexec(self._isolation_policy)
            try:
                child = subprocess.run(
                    [interpreter, "-m", "dspy.optim._score_harness", str(job_path)],
                    capture_output=True,
                    text=True,
                    env=child_env,
                    preexec_fn=preexec,
                )
            finally:
                if broker is not None:
                    broker.stop()
        if child.returncode != 0:
            tail = "\n".join((child.stderr or "").strip().splitlines()[-6:])
            raise RuntimeError(
                f"FlexIR artifact-mode scoring failed in the child process (infrastructure, never scored "
                f"against the candidate): exit {child.returncode}\n{tail}"
            )
        payload = json.loads(child.stdout.strip().splitlines()[-1])
        self._advance_scripts(payload.get("consumed", {}))
        results = []
        for example, record in zip(dataset, payload["results"], strict=True):
            prediction = None if record["prediction"] is None else Prediction(**record["prediction"])
            results.append((example, prediction, record["value"]))
        return EvaluationResult(score=payload["score"], results=results, lm_calls=payload["lm_calls"])

    def check_isolation_invariance(self, program: Module, dataset: Any) -> str | None:
        """Refuse a candidate whose behavior CHANGES with the envelope.

        The D-042 invariance law (Q9 point 4): behavior must be
        isolation-invariant — a leaf that scores differently under the
        envelope than in-process is relying on an undeclared side channel,
        and the divergence indicts the LEAF, not the level. This hook
        scores the same candidate both ways on the same scripted run and
        returns a teaching refusal on any divergence, or None when the
        law holds. It is only checkable where a scripted run makes both
        passes deterministic (the tests), so the loop wires it but does
        not force it; the law it enforces is the whole reason raising the
        wall is a binding and not an authorship act.
        """
        in_process = evaluate(program, dataset, self.metric)
        under_envelope = self._evaluate_as_artifact(program, dataset)
        if abs(in_process.score - under_envelope.score) > self.eps or in_process.lm_calls != under_envelope.lm_calls:
            return (
                f"refused candidate: behavior DIVERGED under isolation — in_process scored "
                f"{in_process.score} at {in_process.lm_calls} LM call(s), the envelope scored "
                f"{under_envelope.score} at {under_envelope.lm_calls}. Behavior must be isolation-invariant "
                "(D-042); a divergence indicts the leaf's undeclared side channel, not the wall."
            )
        return None

    def _child_channel(
        self, extra_env: dict[str, str], scratch: str
    ) -> tuple[dict[str, str], Any, dict[str, Any] | None]:
        """Assemble the child env, start the broker (if any), record the envelope.

        Three cases, documented so it is always clear WHICH credential
        channel is live:

        - No broker: the child inherits the parent env plus the fallback
          secret vars (`extra_env`) — the env-NAME channel (back-compat,
          the `scoring_isolation="none"` default too).
        - Broker active (`broker_egress` non-empty AND some real LM hit
          it): the child gets `HTTPS_PROXY`/`HTTP_PROXY` and NO credential
          vars; the broker injects the header on egress. `extra_env` is
          NOT added — the secret never enters the child.
        """
        from dspy.programir.engine.isolation import IsolationPolicy

        policy = getattr(self, "_isolation_policy", IsolationPolicy())
        envelope = policy.describe_envelope() if self._scoring_level != _NONE_LEVEL else None
        if self.broker_egress and self._broker_inject:
            from dspy.optim.broker import EgressBroker

            broker = EgressBroker(self.broker_egress, inject=self._broker_inject).start()
            proxy = broker.proxy_url
            child_env = {**os.environ, "HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "NO_PROXY": ""}
            if envelope is not None:
                envelope = {**envelope, "broker": sorted(self.broker_egress)}
            return child_env, broker, envelope
        return {**os.environ, **extra_env}, None, envelope

    def _serialize_lms(self, program: Module) -> tuple[dict[str, Any], dict[str, str]]:
        """Serialize the program's LM pool for the child, or refuse.

        Returns `(specs, extra_env)`. Scripted DummyLMs cross the
        boundary as data: a reply list is sent from its cursor onward
        (the parent's cursor advances by the child's consumption
        afterward, so successive child runs read the script exactly as
        one in-process run would); a function-scripted DummyLM sends its
        self-contained source. A plain `dspy.LM` crosses as a RECEIVER
        BINDING — model identity, capability facts, non-secret kwargs —
        with every credential replaced by the NAME of an environment
        variable (`_real_lm_spec`); `extra_env` carries the fallback
        vars set on the child process only. An LM subclass with no
        reconstruction contract keeps a teaching refusal naming exactly
        what is missing — never a silent misscore, never a silent drop.
        """
        from dspy.lm.dummy import DummyLM
        from dspy.programir._dspy import compile_with_live
        from dspy.programir.leaves import _check_self_contained, _source

        _, live = compile_with_live(program)
        specs: dict[str, Any] = {}
        extra_env: dict[str, str] = {}
        self._script_lms = {}
        self._secret_values = set()
        self._broker_inject: dict[str, dict[str, str]] = {}
        for name, lm in live["lm"].items():
            if isinstance(lm, DummyLM):
                if lm._script is not None:
                    specs[name] = {"script": lm._script[lm._cursor :]}
                    self._script_lms[name] = lm
                else:
                    subject = f"FlexIR artifact-mode DummyLM {name!r}"
                    source = _source(lm._fn, subject=subject)
                    _check_self_contained(lm._fn, source, subject=subject)
                    specs[name] = {"function_source": source}
            elif type(lm) is LM:
                specs[name] = self._real_lm_spec(name, lm, extra_env)
            else:
                raise ValueError(
                    f"FlexIR eval_mode='artifact' cannot rebind LM pool entry {name!r} in the scoring child: "
                    f"{type(lm).__name__} is neither a DummyLM nor a plain dspy.LM, and it declares no "
                    "child-side construction contract (model identity + capability facts + JSON-safe kwargs). "
                    "Use eval_mode='in_process' for this LM, or bind a plain dspy.LM."
                )
        return specs, extra_env

    def _real_lm_spec(self, name: str, lm: LM, extra_env: dict[str, str]) -> dict[str, Any]:
        """One plain `dspy.LM` as a child receiver binding — secrets as names.

        The spec reuses the LM's own constructor contract (`model`, the
        capability facts, the default request kwargs) — no parallel
        serialization. Credential-shaped kwargs NEVER enter the spec:
        each is replaced by `{"env": <var name>}`. The var name is
        recovered from the parent's environment when a variable holds
        that exact value; otherwise (the FALLBACK path — a raw key whose
        env origin is unknown) a private `DSPY_FLEX_LM_*` var is set on
        the child process only, so the secret moves process-to-process
        and still never touches disk or argv. The two audit failure
        classes both refuse loudly: an unserializable kwarg (silent
        loss) and an opaque header bag (secret leak) are errors, not
        best-effort guesses.
        """
        kwargs: dict[str, Any] = {}
        credentials: dict[str, dict[str, str]] = {}
        for key, value in lm.kwargs.items():
            if key in _OPAQUE_SECRET_KWARGS:
                raise ValueError(
                    f"FlexIR eval_mode='artifact' cannot serialize LM {name!r}: kwarg {key!r} is an opaque "
                    "header/secret bag the credential channel cannot audit — remove it or use "
                    "eval_mode='in_process' (refusing beats leaking)"
                )
            if _is_secret_kwarg(key):
                if not isinstance(value, str) or not value:
                    raise ValueError(
                        f"FlexIR eval_mode='artifact' cannot serialize LM {name!r}: credential kwarg {key!r} "
                        f"is {type(value).__name__}, not a string"
                    )
                self._secret_values.add(value)
                host = _lm_host(lm)
                if self.broker_egress and key == "api_key" and host in self.broker_egress:
                    # Broker channel (Q7): the credential NEVER enters the
                    # child — not even as an env var. The broker attaches
                    # `Authorization: Bearer <key>` on egress to the
                    # allowlisted host. The child gets the proxy vars plus
                    # a NON-SECRET placeholder key: modern provider clients
                    # (openai >= 2.x) refuse to send a request with no key
                    # at all, so the child must hold something — and the
                    # broker REPLACES the Authorization header on egress,
                    # so the placeholder never reaches the provider.
                    self._broker_inject[host] = {"header": "Authorization", "value": f"Bearer {value}"}
                    kwargs[key] = _BROKER_PLACEHOLDER_KEY
                    continue
                # Env-name channel (scoring_isolation without a broker, and
                # the back-compat default): the credential rides an env-var
                # NAME the child reads from its inherited environment.
                env_name = _env_var_holding(value)
                if env_name is None:
                    env_name = "DSPY_FLEX_LM_" + re.sub(r"[^A-Z0-9]+", "_", f"{name}_{key}".upper()).strip("_")
                    extra_env[env_name] = value
                credentials[key] = {"env": env_name}
                continue
            try:
                json.dumps(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"FlexIR eval_mode='artifact' cannot serialize LM {name!r}: kwarg {key!r} is not "
                    f"JSON-serializable ({error}); dropping it silently would change the child's sampling"
                ) from error
            kwargs[key] = value
        return {
            "class": "LM",
            "model": lm.model,
            "capabilities": lm.capabilities.to_dict(),
            "kwargs": kwargs,
            "credentials": credentials,
        }

    def _assert_secret_free(self, data: bytes, where: str) -> None:
        """Refuse loudly if any known credential value appears in `data`."""
        for secret in getattr(self, "_secret_values", ()):
            if secret.encode("utf-8") in data:
                raise RuntimeError(
                    f"FlexIR refused to proceed: a credential value leaked into {where}. Secrets ride "
                    "env-var NAMES only; this is the belt-and-suspenders scan catching a leak before disk."
                )

    def _advance_scripts(self, consumed: dict[str, int]) -> None:
        for name, count in consumed.items():
            lm = self._script_lms.get(name)
            if lm is not None:
                lm._cursor += count

    # ------------------------------------------------------------------
    # Render: what the reflection LM sees (five slots, section 4)
    # ------------------------------------------------------------------

    def _render_report(self, program: Module, score: float, lm_calls: int, results: list, refusals: list[str]) -> str:
        manifest = program.to_manifest()
        lines = ["== current program =="]
        lines.append(program.explain())
        lines.append("== forwards (the rewrite_forward dialect — write yours exactly like these) ==")
        for path in sorted(manifest["components"]["5_forward"]):
            lines.append(f"# forward of {path}")
            try:
                lines.append(render_forward(manifest["components"]["5_forward"][path]).rstrip())
            except ProgramIRRefusal:
                lines.append("(this forward has no printed spelling)")
        lines.append("== cost view ==")
        lines.append(cost_build_text(manifest))
        lines.append(f"measured last run: {lm_calls} LM call(s) over {len(results)} example(s)")
        # PIR-021 per-leaf attributed counts: a call through a session
        # leaf's bridge shows under BOTH names (the total counts it once).
        attribution = getattr(self, "_best_attribution", None)
        if attribution:
            attributed = "  ".join(f"{name}:{count}" for name, count in sorted(attribution.items()))
            lines.append(f"measured per-leaf attribution: {attributed}")
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
        optional = _OPTIONAL_KEYS.get(op, frozenset())
        missing = sorted(required - set(proposal))
        extra = sorted(set(proposal) - required - optional - {"op"})
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
        if op == "add_predict":
            return self._apply_add_predict(program, proposal, tag, undo_structural)
        if op == "add_tool":
            return self._apply_add_tool(program, proposal, tag, undo_structural)
        if op == "rewrite_forward":
            return self._apply_rewrite_forward(program, proposal, tag, undo_structural)

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
            admitted = admit_tool_source(
                tool_name,
                source,
                predictor.signature,
                partial=partial,
                extra_imports=self.extra_imports,
                code_trust=self.code_trust,
                allowed_deps=self.allowed_deps,
            )
        except ValueError as error:
            return f"refused {tag}: python_source rejected by admission — {error}"
        install_refusal = self._ensure_leaf_deps(admitted, tag)
        if install_refusal is not None:
            return install_refusal

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
        """Remove a predict or authored-tool leaf with ZERO call sites.

        The clean converse of the code and structure ops: after a full
        replace or a rewrite, the old leaf is dead. Count its call sites
        (CallPredict for a predict, CallTool for a tool) across every
        module's forward first; refuse if any remain. Link would refuse a
        dangling ref anyway; this keeps accepted artifacts orphan-free.
        """
        path = proposal["path"]
        if not isinstance(path, str):
            return f"refused {tag}: path must be a STRING leaf path, got {type(path).__name__}"
        predictors = dict(program.named_predictors())
        owner, attribute = self._owner_of(program, path)
        if path in predictors:
            kind, noun = "predict", "predictor"
        elif (
            owner is not None
            and callable(owner.__dict__.get(attribute))
            and not isinstance(owner.__dict__.get(attribute), Module)
        ):
            kind, noun = "tool", "tool"
        else:
            return f"refused {tag}: no predictor or tool leaf at path {path!r} (predictor paths: {sorted(predictors)})"
        if owner is None:
            return f"refused {tag}: cannot delete the root program leaf {path!r}"

        remaining = self._count_sites(program, attribute, kind=kind)
        if remaining:
            return (
                f"refused {tag}: {noun} {path!r} still has {remaining} live call site(s); replace them "
                "with code (or rewrite the forward to remove the calls) before deleting the leaf"
            )
        # PIR-021: a grant reference is a live site too. A session leaf
        # that grants this leaf would be left with a dangling bridge —
        # refuse, naming the granting leaf, before the delete.
        granting = self._grant_holders(program, attribute)
        if granting:
            return (
                f"refused {tag}: {noun} {path!r} is granted by session leaf(s) {sorted(granting)}; a grant is a "
                "live site — remove the grant (or delete the granting session leaf) before deleting this leaf"
            )
        previous = owner.__dict__.get(attribute, _ABSENT)
        if previous is _ABSENT:
            return f"refused {tag}: the module owning {path!r} has no attribute {attribute!r} to delete"
        delattr(owner, attribute)
        program.invalidate_ir()
        undo_structural.append((owner, attribute, previous))
        return None

    def _count_sites(self, program: Module, attribute: str, *, kind: str = "predict") -> int:
        """Count live Call sites of one leaf kind naming `attribute`."""
        manifest = program.to_manifest()
        total = 0
        for forward in manifest["components"]["5_forward"].values():
            total += len(_find_leaf_sites(forward, attribute, kind=kind))
        return total

    def _grant_holders(self, program: Module, attribute: str) -> set[str]:
        """Names of tool leaves whose grants bridge into `attribute` (PIR-021)."""
        from dspy.programir.leaves import granted_leaf_name

        holders: set[str] = set()
        for tool_name, entry in program.to_manifest()["components"]["6_tools"].items():
            for grant in entry.get("grants", []) or []:
                if isinstance(grant, dict) and granted_leaf_name(grant) == attribute:
                    holders.add(tool_name)
        return holders

    # ------------------------------------------------------------------
    # v3 structure ops: add leaves, rewrite whole forwards
    # ------------------------------------------------------------------

    def _module_at(self, program: Module, path: Any, tag: str) -> tuple[Module | None, str | None]:
        """Resolve an OWNER-module path for the v3 ops, or refuse teaching.

        The v3 ops name the module that OWNS the edit ('self' for the
        root composite), never a predictor. A bare-Predict root has no
        forward of its own and no attribute table worth editing, so it
        refuses; so does a path that lands on a Predict leaf.
        """
        if not isinstance(path, str):
            return None, (
                f"refused {tag}: path must be a STRING module path ('self' for the root), got {type(path).__name__}"
            )
        modules = dict(program._named_modules())
        owner = modules.get(path)
        if owner is None:
            return None, f"refused {tag}: no module at path {path!r} (module paths: {sorted(modules)})"
        if isinstance(owner, Predict):
            if path == "self":
                return None, (
                    f"refused {tag}: the root program is a bare Predict — it has no forward of its own to "
                    "rewrite and no attribute table to extend; wrap it in a composite Module first"
                )
            return None, (
                f"refused {tag}: {path!r} is a Predict leaf, not a composite module — name the module that "
                "OWNS the forward ('self' for the root)"
            )
        return owner, None

    def _apply_add_predict(self, program: Module, proposal: dict, tag: str, undo_structural: list) -> str | None:
        """Bind a NEW Predict leaf on an owner module.

        The leaf starts with zero call sites — a later `rewrite_forward`
        in the SAME proposal list wires it. If it still has zero sites at
        evaluate time it is dead weight, not a refusal: the cheapness
        channel prices it (an accepted candidate must still win a
        channel), and `delete_dead_leaf` cleans it.
        """
        owner, refusal = self._module_at(program, proposal["path"], tag)
        if refusal is not None:
            return refusal
        name = proposal["name"]
        if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
            return f"refused {tag}: add_predict name must be a Python identifier, got {name!r}"
        if name in owner.__dict__:
            return f"refused {tag}: attribute {name!r} already exists on the module at {proposal['path']!r}"
        instructions = proposal["instructions"]
        if not isinstance(instructions, str) or not instructions:
            return (
                f"refused {tag}: add_predict instructions must be a non-empty string — the instructions ARE "
                "the new leaf's prompt; an uninstructed predict is an unteachable one"
            )
        signature = proposal["signature"]
        if not isinstance(signature, str) or not signature:
            return f"refused {tag}: add_predict signature must be an 'input_a, input_b -> output_c' string"
        try:
            predictor = Predict(signature)
        except Exception as error:
            return (
                f"refused {tag}: signature {signature!r} does not build — {type(error).__name__}: {error}; "
                "write it as 'input_a, input_b -> output_c'"
            )
        predictor.signature = predictor.signature.with_instructions(instructions)
        setattr(owner, name, predictor)
        program.invalidate_ir()
        undo_structural.append((owner, name, _ABSENT))
        return None

    def _apply_add_tool(self, program: Module, proposal: dict, tag: str, undo_structural: list) -> str | None:
        """Bind a NEW admitted code leaf on an owner module (a free tool).

        Same admission chain as the replace ops, WITHOUT a replaced
        signature: the io-contract derives from the source's own type
        hints (every parameter hinted; return annotation `dict`). Wire it
        with a `rewrite_forward` in the same proposal list; the cheapness
        channel prices dead weight.
        """
        owner, refusal = self._module_at(program, proposal["path"], tag)
        if refusal is not None:
            return refusal
        name = proposal["name"]
        if not isinstance(name, str) or not name.isidentifier() or not name.isascii():
            return f"refused {tag}: add_tool name must be a Python identifier, got {name!r}"
        if name in owner.__dict__:
            return f"refused {tag}: attribute {name!r} already exists on the module at {proposal['path']!r}"
        # PIR-021: the optional `kind` (call | session) and `grants[]`
        # (closed static effect row). Absent kind = "call" — exactly the
        # prior free tool. Grants name existing pool leaves this leaf may
        # bridge into; a session leaf reaches them through the bridge.
        kind = proposal.get("kind", "call")
        if kind not in ("call", "session"):
            return f"refused {tag}: add_tool kind must be 'call' or 'session', got {kind!r}"
        grants_spec = proposal.get("grants", [])
        grant_refusal, grant_entries = self._validate_grants(program, grants_spec, kind, tag)
        if grant_refusal is not None:
            return grant_refusal
        try:
            admitted = admit_tool_source(
                name,
                proposal["python_source"],
                None,
                partial=False,
                extra_imports=self.extra_imports,
                code_trust=self.code_trust,
                allowed_deps=self.allowed_deps,
                session=(kind == "session"),
            )
        except ValueError as error:
            return f"refused {tag}: python_source rejected by admission — {error}"
        install_refusal = self._ensure_leaf_deps(admitted, tag)
        if install_refusal is not None:
            return install_refusal
        function = admitted.function
        if kind == "session":
            function._dspy_leaf_kind = "session"
        if grant_entries:
            function._dspy_leaf_grants = grant_entries
        setattr(owner, name, function)
        program.invalidate_ir()
        undo_structural.append((owner, name, _ABSENT))
        return None

    def _validate_grants(
        self, program: Module, grants_spec: Any, kind: str, tag: str
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Validate add_tool grants: a list naming EXISTING pool leaves.

        Each grant is either a bare leaf-name string or `{"leaf": name}`.
        A grant must resolve to a live predictor or tool leaf, or it
        refuses (teaching). Grants require kind="session" — a call-kind
        leaf with grants is a mis-shape (only a session holds a bridge).
        Returns `(refusal, entries)` where entries are contract-valid
        `{kind, name}` bridge grants via `leaves.leaf_grant`.
        """
        from dspy.programir.leaves import leaf_grant

        if not grants_spec:
            return None, []
        if kind != "session":
            return (
                f"refused {tag}: add_tool grants require kind='session' — only a session leaf holds a grant "
                "bridge (a call-kind leaf is a pure request/response with no callbacks)",
                [],
            )
        if not isinstance(grants_spec, list):
            return f"refused {tag}: add_tool grants must be a JSON array of leaf names", []
        predictor_paths = {path for path, _ in program.named_predictors()}
        tool_names = set(program.to_manifest()["components"]["6_tools"])
        entries: list[dict[str, str]] = []
        for grant in grants_spec:
            leaf = grant if isinstance(grant, str) else (grant.get("leaf") if isinstance(grant, dict) else None)
            if not isinstance(leaf, str) or not leaf:
                return (
                    f"refused {tag}: each add_tool grant names a leaf (a string or {{'leaf': name}}), got {grant!r}",
                    [],
                )
            if leaf not in predictor_paths and leaf not in tool_names:
                return (
                    f"refused {tag}: grant names leaf {leaf!r}, which resolves to no predictor or tool "
                    f"(predictors: {sorted(predictor_paths)}, tools: {sorted(tool_names)})",
                    [],
                )
            entries.append(leaf_grant(leaf))
        return None, entries

    def _apply_rewrite_forward(self, program: Module, proposal: dict, tag: str, undo_structural: list) -> str | None:
        """Replace an owner module's forward with an authored source — gated.

        THE general op. The source is one `def forward(self, ...)` in the
        printer's dialect; it becomes a live function only via the
        `to_function` linecache convention (register + exec the single
        def), then `compile_forward` lowers it against the module's OWN
        leaf table, declared literals, and declared signature — exactly
        the mapping the normal compile path uses. Any compiler refusal
        (an unknown leaf ref, an unsupported construct) surfaces verbatim
        into the ledger. On success the tree attaches through the same
        macro door the code ops use, with a correct undo entry.
        """
        from dspy.modules._generate import load_generated
        from dspy.programir._dspy import _declared_literals, _declared_signature, leaf_table

        owner, refusal = self._module_at(program, proposal["path"], tag)
        if refusal is not None:
            return refusal
        source = proposal["python_source"]
        if not isinstance(source, str) or not source.strip():
            return f"refused {tag}: python_source must be a non-empty string holding one `def forward(self, ...)`"
        try:
            parsed = ast.parse(source)
        except SyntaxError as error:
            return (
                f"refused {tag}: python_source does not parse as Python ({error.msg} at line {error.lineno}); "
                "write one valid `def forward(self, ...)` in the report's dialect"
            )
        if len(parsed.body) != 1 or not isinstance(parsed.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
            return (
                f"refused {tag}: python_source must be EXACTLY one function definition "
                "(no second def, no decorators, no statements around it)"
            )
        definition = parsed.body[0]
        if definition.decorator_list:
            return f"refused {tag}: rewrite_forward source uses decorators; write the undecorated def"
        if definition.name != "forward":
            return f"refused {tag}: the def must be named 'forward', got {definition.name!r}"
        function = load_generated(source, tag=f"flex-rewrite-{proposal['path']}", name=definition.name)
        try:
            tree = compile_forward(
                function,
                leaf_table(owner),
                literals=_declared_literals(owner),
                signature=_declared_signature(owner),
            )
        except ProgramIRRefusal as error:
            return f"refused {tag}: rewrite_forward rejected by the forward compiler — {error.code}: {error}"
        except ValueError as error:
            return f"refused {tag}: rewrite_forward rejected — {error}"
        previous_builder = _bound_builder(owner)
        _attach_forward(owner, tree)
        program.invalidate_ir()
        undo_structural.append((owner, "build_forward_ir", previous_builder))
        return None

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


# ---------------------------------------------------------------------------
# Credential channel: secrets ride env-var NAMES to the scoring child
# ---------------------------------------------------------------------------

#: Kwarg-name markers that make an LM kwarg credential material. Wider
#: than the exact `api_key` on purpose: the serialization audit's leak
#: class was sibling credentials (`azure_ad_token`, `aws_secret_access_key`)
#: escaping an exact-name filter.
_SECRET_KWARG_MARKERS = ("api_key", "secret", "password")

#: Cached `IsolationLevel.none` — the byte-identical-to-before pole. A
#: module constant keeps the hot path free of a repeated enum lookup.
_NONE_LEVEL = _IsolationLevel.none

#: Kwargs that can smuggle credentials inside an opaque structure (an
#: `Authorization` header in a dict). No per-value audit is attempted —
#: they refuse whole (refusing beats leaking).
_OPAQUE_SECRET_KWARGS = frozenset({"extra_headers", "default_headers", "headers"})

#: The NON-SECRET key the broker-channel child holds so its provider
#: client will send the request at all (openai >= 2.x refuses an empty
#: key client-side, so the request would die before the broker could
#: inject). The broker replaces the Authorization header on egress; this
#: value never reaches the provider and is safe in specs and logs.
_BROKER_PLACEHOLDER_KEY = "sk-broker-injected"


def _is_secret_kwarg(key: str) -> bool:
    lowered = key.lower()
    if any(marker in lowered for marker in _SECRET_KWARG_MARKERS):
        return True
    # `_token` matches the sibling-credential class (azure_ad_token) while
    # sparing counters like `max_tokens` (plural, not a credential).
    return lowered == "token" or lowered.endswith("_token")


def _lm_host(lm: LM) -> str | None:
    """The hostname of an LM's api_base, for broker allowlist matching."""
    from urllib.parse import urlsplit

    base = lm.kwargs.get("api_base") or lm.kwargs.get("base_url")
    if not isinstance(base, str) or not base:
        return None
    return urlsplit(base).hostname


def _env_var_holding(value: str) -> str | None:
    """The first environment variable whose value IS `value`, or None.

    Recovering the NAME lets the binding carry a reference instead of
    the secret; the child reads the same variable from its inherited
    environment.
    """
    for name, held in os.environ.items():
        if held == value:
            return name
    return None


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


def _find_leaf_sites(tree: dict, attribute: str, *, kind: str) -> list[tuple[tuple, dict]]:
    """Every `Call({kind}, attribute)` site in a forward tree, with its path."""
    sites = []
    for path, call in _iter_leaf_calls(tree):
        leaf = call.get("leaf", {})
        if leaf.get("kind") == kind and leaf.get("ref") == attribute:
            sites.append((path, call))
    return sites


def _find_predict_sites(tree: dict, attribute: str) -> list[tuple[tuple, dict]]:
    """Every `CallPredict(attribute)` site in a forward tree, with its path."""
    return _find_leaf_sites(tree, attribute, kind="predict")


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
