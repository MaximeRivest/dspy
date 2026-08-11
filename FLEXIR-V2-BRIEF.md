All substrate claims confirmed against the greenfield worktree. One deviation found and noted in the brief: `dspy/optim/flex.py` (FlexIR v1) is not yet present in `/home/maxime/Projects/dspy-greenfield` — the A8 stage is in flight (uncommitted `dspy/modules/best_of_n.py` and `refine.py` are there; `dspy/optim/` holds only base/bootstrap/labeled_fewshot/random_search). The brief carries this as a hard precondition.

FLEXIR-V2 DESIGN BRIEF — LEAF-IMPLEMENTATION REWRITES ON THE PROGRAMIR
=======================================================================
Status: hand to the builder agent verbatim. One stage. Greenfield worktree
(/home/maxime/Projects/dspy-greenfield, branch greenfield-ir).

Maxime's directive, which this brief serves: flex must REWRITE THE IR ITSELF —
including generating tools that run normal Python in the forward to REPLACE LM
calls, making programs cheaper — with the reflection LM as the rewriter. The
old string substrate dies.

SUBSTRATE (confirmed on disk, cite before you change anything)
--------------------------------------------------------------
- Typed IR node builders: dspy/programir/build.py — one constructor per node
  kind, node set v0.3; names validated at construction; `Forward(args, body,
  leaves=...)` runs the SAME `admit_forward` the parse compiler runs (build.py:639-656).
  Closed operator sets at build.py:143-146 (`eq/ne/lt/le/gt/ge/in/not_in`,
  `add/sub/mult/div`, `and/or`, `not/neg`). `Try`/`Except` handler types include
  `ToolError` (build.py:147-149).
- Printer: dspy/programir/printer.py — THE ROUND-TRIP LAW:
  `compile_forward(to_function(tree), leaves) == tree`, deterministic bytes;
  `to_function` yields a genuine function via linecache.
- Authored leaves: dspy/programir/leaves.py — `extract_tool` produces a pool
  entry {name, description, parameters, return_schema, source: "tools/<name>.py",
  deps, language, placement} with a carried source sidecar; `_source` requires
  one undecorated function def; `_check_self_contained` refuses closures and
  global reads (leaves.py:99-151).
- Optimizer loop: dspy/optim/base.py — "propose, score, keep"; `run_engine`
  replays through `program._executable()`; `evaluate` returns
  `EvaluationResult(score, results, lm_calls)` with `_count_lm_calls` reading
  the `_trajectory` exhaust (base.py:105-176); `Checkpointer.accept` saves each
  accepted candidate as a loadable artifact plus `scores.json` (base.py:179-208).
- Tools over manifests: dspy/programir/tools/cost.py (`estimate`, per-predictor
  call Bounds + prompt-token estimates, `build_text`), diff.py, lint.py.
- Views: dspy/programir/explain_view.py — deterministic View-1 text of a
  ratified manifest.
- Engine fact that sizes this stage: `Attr` is key lookup on a Prediction OR a
  plain dict (engine/interpret.py:553-566, SEM-7). A tool that returns the
  output record as a dict is field-addressable exactly where a prediction was.
  Therefore NO ENGINE CHANGES are required for any op below.
- Load is link: `dspy.load(path, bindings=...)` (module.py docstring); dangling
  refs refuse loudly. The §16 dangling-tool-name failure of old flex is
  structurally impossible here.
- D-041 (fork roadmap/05-decisions.md:86): node-set 0.4 inputs-bag + signature-
  record splat is RATIFIED but NOT yet propagated to this tree (GREENFIELD.md
  amendment in the working diff). See the 0.4 note at the end of section 2.

PRECONDITION (hard): the A8 stage — dspy/modules/best_of_n.py, refine.py, and
FlexIR v1 in dspy/optim/flex.py (edit vocabulary: instructions / demos /
wrap_best_of_n) — must be landed in your worktree before you start. At brief
time best_of_n.py and refine.py sit uncommitted and dspy/optim/flex.py does not
exist yet. If it is still absent when you start, STOP and report; do not
reinvent v1.

1. WHAT OLD FLEX GOT RIGHT — MUST SURVIVE
------------------------------------------
From the study reports on the string-substrate flex (fork:
dspy/predict/flex/, dspy/teleprompt/gepa/gepa_flex_utils.py, docs
flex.md). Keep the behavior; kill the substrate.

a. Code, not prompt, as the unit of optimization. "The prompt is no longer the
   unit of optimization; the program code is" (flex.md). v2 keeps this, but the
   unit is the IR tree, never a source string.
b. LM→plain-Python replacement for cheapness was already the intent
   (primitives_doc.py:119-123: "When no step needs an LM, write a pure-Python
   forward and define no predictors"). v2 makes it a first-class edit.
c. Broken candidates score as failures; they never crash the run. Bind failure
   → whole-batch failure score; per-example crash → failure IN PLACE with score
   alignment preserved; "the crash IS the feedback" flows into reflection.
   v2 equivalent: a refused edit consumes a proposal slot and feeds the
   refusal ledger; a running candidate that errs per-example scores 0.0 there
   (optim/base.evaluate already does this via CatchableError).
d. Infrastructure errors are NEVER scored as candidate failures. Old flex built
   a whole tag-correlation protocol to keep LMError typed across its sandbox.
   v2 gets this for free — no boundary — but preserve the rule: typed LM/infra
   errors propagate out of evaluate; only catchable program errors score 0.0.
e. Reflection sees the WHOLE program, not per-leaf traces, "because the
   predictors are part of what's being rewritten — they may not exist in the
   next candidate". Keep whole-program inputs/outputs/feedback as the failure
   view.
f. Metric exceptions during scoring → failure score for that example only; the
   run continues.
g. Proposal failure handling, split by cause: reflection-LM infra error
   propagates (never burn budget on a fake success); a malformed proposal keeps
   the current champion and records feedback.
h. Re-optimization seeds from the CURRENT program state, manual edits included.
   v2: the seed is the live program's IR — snapshot it, never a baseline
   template.
i. Instruction optimization stays first-class inside the same loop, with the
   working guidance kept verbatim: "Fix instructions when a failure is about
   WHAT the model should do or know; change the structure when it is about HOW
   steps are wired."
j. The signature is the invariant seam. The optimized program stays
   type-substitutable for the original: same module signature, outputs
   validated, missing required output = failure, never a silent stringify.
k. Scoring semantics == deployment semantics. Old flex re-instantiated per
   forward to guarantee this; the engine gives it natively (the artifact IS
   what runs).
l. Cost as a scoring channel. Old flex exposed `program_trace` so a metric
   could penalize LM-call count. v2 promotes this: `EvaluationResult.lm_calls`
   is part of the acceptance rule, not a metric trick.
m. The five-slot prompt scaffold worked (task description / available context /
   capability catalog with hard rules and RIGHT/WRONG worked patterns / current
   artifact / formatted failures). Keep the shape; replace every slot's content
   with IR views (section 4).
n. Budgets. Old flex capped predictor calls per forward. The engine's bounded
   loops cover the runaway case; keep an explicit eval budget (max metric
   calls) on the loop.

What dies with the substrate: `module_src` as one opaque string leaf; the
sandbox shim fake-dspy and its six string-crossing protocols; fence stripping;
string-rendered signatures; name-dangling unserialized tools;
`named_predictors() == []` opacity; mandatory Deno in the hot path.

2. THE FLEXIR-V2 EDIT VOCABULARY — A CLOSED LIST
-------------------------------------------------
An edit is one JSON object `{"op": <name>, ...}`. The applier is a dispatch
table over EXACTLY these op names. Any other op name, any unknown field, any
wrong field type → refusal with a teaching error into the ledger. Edits are
DATA. The applier never execs, formats, or splices proposal text into source.
The only code-bearing field is `python_source`, and it passes through tool-leaf
admission (section 3) before anything touches it.

v1 ops (carried forward from dspy/optim/flex.py unchanged):
  1. set_instructions {path, instructions}
     Component 3a via the ProgramState/apply_state surface (optim/base.py).
  2. set_demos {path, demos}
     Component 3b, same surface. Demos are JSON records matching the leaf
     signature.
  3. wrap_best_of_n {path, n, ...}
     The BestOfN IR macro (dspy/modules/best_of_n.py) applied around the leaf
     at `path`, exactly as v1 landed it.

v2 ops (this stage):
  4. replace_predict_with_code {path, tool_name, python_source}
     Intent: this LM call needs no LM. The reflection LM writes ONE
     self-contained Python function. Its io-contract DERIVES from the predict
     leaf's signature — the unified-leaf law (a leaf's interface is a
     signature; Predict↔code swaps are implementation choice):
       - parameters: exactly the signature's input fields, with type hints;
       - return annotation: `dict` — the output record, one key per declared
         output field.
     After admission the function becomes a DECLARED AUTHORED TOOL LEAF in the
     tools pool: carried source sidecar `tools/<tool_name>.py`, deps, schemas,
     `authored_by: "optimizer"` provenance, placement per trust rules
     (in_process by default in the optimizing user's own loop; a receiver may
     re-place it to a sandbox rung — placement is the receiver's choice, never
     baked in).
     Tree rewrite, via build.py constructors only: every call site
     `Assign(target, CallPredict(path, **kw))` becomes
     `Assign(target, CallTool(tool_name, **kw))`. Downstream `Attr(target, f)`
     keeps working because Attr is dict key lookup (interpret.py:553-566). The
     predict leaf usually goes dead; clean it with op 7.

  5. replace_predict_with_code_partial {path, tool_name, python_source}
     Intent: code handles the common case; the LM stays as fallback. Same
     admission as op 4 with one contract change: return annotation
     `dict | None`; returning None means "decline — use the LM". A raise inside
     the fast path counts as a decline, never a program failure. The tree
     shape, spelled exactly (per call site `Assign(target, CallPredict(path,
     **kw))`; `F` is a fresh hygienic name, e.g. `_flex_fast_<n>`):

       Try(
           body=[Assign(F, CallTool(tool_name, **kw))],
           handlers=[Except("ToolError", body=[Assign(F, Const(None))])],
       ),
       If(
           Compare("ne", Var(F), Const(None)),
           body=[Assign(target, Var(F))],
           orelse=[Assign(target, CallPredict(path, **kw))],
       )

     Both arms bind `target` to a field-addressable record (dict or
     prediction), so every downstream read is unchanged. The rewritten forward
     re-admits through `Forward(...)` and must round-trip through the printer.

  6. inline_constant {path_or_site, value}
     Replace a leaf call site (or a named intermediate) whose observed output
     is constant across the devset with `Const(value)` / `Dict({...})` (for a
     record, a Dict of Consts). `value` must be JSON data. This is the
     degenerate cheapness edit; it exists so the LM does not smuggle constants
     through op 4 as code.
  7. delete_dead_leaf {path}
     Remove a pool entry (predict or tool) that has ZERO call sites across all
     forwards. Count sites first; refuse with the site list if any remain.
     Link would refuse a dangling ref anyway; this op is the clean converse —
     it keeps accepted artifacts free of orphaned leaves after ops 4/5.

Node-set 0.4 guard: the builders in this worktree are v0.3. D-041's inputs-bag
and record-splat call sites (`CallPredict(ref, **inputs-record)`) are ratified
but not yet propagated here. Until the 0.4 propagation lands, the applier must
REFUSE to rewrite a splat-bearing call site with a teaching refusal ("0.4 splat
site; rewrite not yet supported") rather than mis-rewrite it. The applier keys
on call-site nodes, not envelopes, so nothing else changes at 0.4.

3. THE LOOP: PROPOSE → VALIDATE → SCORE → KEEP
-----------------------------------------------
The loop extends the A8 FlexIR v1 loop in dspy/optim/flex.py; base machinery is
optim/base.py (Optimizer, evaluate, Checkpointer).

PROPOSE. One reflection-LM call per step, through a greenfield `dspy.Predict`
with a TYPED output field (`edits: list[dict]` or a pydantic Edit list) so the
adapter — not a regex — owns parsing. This kills the fence-stripping fragility
class outright. A parse failure is a refusal, not a crash.

VALIDATE (admission — this is where generated code is admitted).
Apply the edit list to a COPY of the champion. Per edit:
  - Data ops (1, 2, 6): type-check the values; apply via ProgramState /
    builder Const. No code path exists.
  - Code ops (4, 5): run `admit_tool_source(name, python_source, signature)`
    (new, dspy/optim/code_leaf.py) which enforces, in order:
      1. `ast.parse` succeeds; the module body is EXACTLY one undecorated
         `def` (reuse the checks in leaves._source, source-first).
      2. Self-containment: no closures possible (source-born), no global reads
         (reuse leaves._check_self_contained on the AST).
      3. Import allowlist for optimizer-authored code: a closed stdlib set
         (re, json, math, string, textwrap, datetime, collections, itertools,
         functools, unicodedata, decimal, fractions, statistics, heapq,
         bisect). socket/http/urllib/requests/subprocess/os/sys/importlib/
         ctypes etc. refuse. `# deps:` must be empty for authored_by:
         optimizer. (Rationale: risk 2, section 5.)
      4. io-contract from the signature: parameter names/hints match the
         leaf's input fields; return annotation matches the op's contract
         (`dict` for op 4, `dict | None` for op 5).
      5. Build the pool entry the leaves.py way (parameters, return_schema,
         source sidecar, placement) plus `authored_by: "optimizer"`; compile
         the function `to_function`-style (linecache + exec of the single def
         in a clean namespace — defines the function, runs no body code).
    Then rewrite the forward tree with build.py constructors; `Forward(...)`
    re-runs `admit_forward`; render with `printer.render_forward` (the
    round-trip law makes the printed source the reviewable artifact form);
    run tools/lint.py over the candidate manifest; link resolves every ref.
  - ANY refusal (ProgramIRRefusal or admission ValueError) → record
    {edit, code, message} in the refusal ledger, feed it to the next proposal,
    keep the champion, spend no eval budget.

SCORE (behavioral safety is MEASURED, never proven).
`evaluate(candidate, devset, metric)` through engine replay. Two numbers per
candidate: `score` and `lm_calls`. Acceptance rule, explicit:
  - accept if `dev_score > best_score + eps`, OR
  - accept if `dev_score >= best_score - eps` AND `lm_calls < best_lm_calls`
    (cheapness with held quality),
  - AND, for any candidate carrying a code op (4/5/6): `holdout_score >=
    best_holdout_score - eps` on the holdout split the reflection LM has never
    seen (risk 1). A holdout drop refuses the candidate and the refusal
    (with the two scores) goes into the ledger.
There is no proof obligation on generated code. The metric on devset + holdout
IS the safety measure; admission only guarantees shape and containment.

KEEP / ROLLBACK / TRAJECTORY.
On accept: apply the candidate to the live program and
`Checkpointer.accept(program, score=..., label=<op summary>)` — every
checkpoint directory is a loadable ProgramIR artifact; `scores.json` is the
ordered trajectory. On reject: nothing to undo — all edits were applied to a
copy; the champion object is untouched. Rollback to ANY earlier point is
`dspy.load(<checkpoint dir>, bindings=...)`. Record per-checkpoint: score,
holdout score, lm_calls, edit list applied.

4. THE PROMPTS — WHAT THE REFLECTION LM SEES
---------------------------------------------
Keep old flex's five-slot scaffold (it worked; report section 4). New content:

  Slot 1 — task_description: the module signature spec — name, docstring
  objective, typed input/output fields. Same content as old
  `render_signature_spec`, sourced from the manifest.

  Slot 2 — current_program (replaces current_source): TWO views of the
  champion. (a) `explain_view` View-1 text of the manifest — pools, leaves,
  placements, provenance, versions. (b) `printer.render_forward` output per
  forward — readable Python the LM can reason over. The printer's determinism
  makes prompts reproducible.

  Slot 3 — edit_catalog (replaces PRIMITIVES_CATALOG): the closed op list of
  section 2 with exact JSON shapes; the code contract (params = input fields,
  return = output record dict, None-decline for partial); the import
  allowlist; hard rules ("output ONLY the edit list", "never invent op names
  or leaf paths — paths come from the views above"); and RIGHT/WRONG worked
  patterns, one per op, in the old catalog's contrast style. Keep the WHAT/HOW
  line from section 1.i and add: "Replace a predict with code when the
  failures and the cost view show the step needs no LM judgment. Use the
  partial op when most inputs are mechanical and a few need the model."

  Slot 4 — cost_view: `tools/cost.py build_text` over the champion manifest
  (per-predictor call bounds, prompt-token estimates) PLUS measured reality:
  total `lm_calls` from the last EvaluationResult and, where the exhaust's
  `predictor_calls` entries name leaves, a per-leaf measured call count. This
  slot is what makes cheapness proposable rather than accidental.

  Slot 5 — failures_and_refusals: (a) the lowest-scoring devset examples as
  whole-program records — inputs, produced outputs or the typed error text
  ("the crash IS the feedback" carries over), metric feedback where the metric
  provides it; NEVER holdout examples. (b) The last N refusal-ledger entries:
  the offending edit, the refusal code, the teaching message — so the LM stops
  re-proposing the same illegal edit.

Output contract: the typed `edits` field only. No prose. The adapter parses;
a malformed reply is a ledger entry, not a crash.

5. RISKS AND MITIGATIONS
-------------------------
R1. Reward hacking via metric overfit (code that memorizes devset answers —
    old flex forbade "hardcoded example outputs" by prompt only).
    → The holdout split (section 3) is the primary gate: never shown to the
    LM, checked at acceptance for every code-bearing candidate. Secondary:
    `inline_constant` exists so honest constants have a cheaper legal spelling,
    and lint flags Const-heavy authored tools (large literal tables) in the
    run report for human review. Measured, not proven — by design.
R2. A code leaf hiding LM-call-shaped behavior (an authored tool that calls a
    hosted model over HTTP fakes cheapness and evades `lm_calls`).
    → The import allowlist refuses every network/process module at admission;
    `# deps:` must be empty for optimizer-authored code; `authored_by:
    "optimizer"` provenance survives into the artifact so any receiver can
    audit or re-place the leaf to a sandbox rung before trusting it.
R3. Injection via proposals (a proposal that carries instructions or code
    outside the vocabulary).
    → Data-only application: dispatch on the closed op table; unknown ops and
    fields refuse; no string templating exists anywhere in the applier (build.py
    constructors validate names at construction); the single code field passes
    the full admission and nothing else; exec happens only on the one admitted
    def and the body runs only through the engine at score time. Devset text
    inside prompts is delimited as data; a prompt-injected proposal still faces
    the same admission and scoring wall.
R4. Behavioral drift under partial replacement (plausible-but-wrong fast-path
    records that dodge the metric on devset).
    → Same holdout gate; plus the acceptance report prints per-example score
    deltas and the manifest diff (tools/diff.py) so drift is visible.
R5. wrap_best_of_n inflating cost silently.
    → `lm_calls` sits inside the acceptance rule; a quality gain must pay for
    its calls explicitly.
R6. Infra/candidate misattribution (the failure class old flex fought with tag
    correlation).
    → Preserved rule: only CatchableError scores 0.0 in evaluate; typed
    LM/provider errors propagate and abort the step, never the trajectory.
R7. Dead-leaf deletion breaking a sibling forward.
    → delete_dead_leaf counts call sites across ALL forwards before acting;
    link is the backstop.

6. IMPLEMENTATION PLAN — ONE BUILDER-AGENT STAGE
-------------------------------------------------
Precondition: A8 landed (dspy/optim/flex.py v1, modules/best_of_n.py,
refine.py committed). Verify, else stop and report.

Files:
  - dspy/optim/flex.py — extend: the v2 op table and applier, the acceptance
    rule with holdout split, the refusal ledger, prompt assembly (five slots),
    the 0.4-splat refusal guard. Keep v1 ops byte-compatible.
  - dspy/optim/code_leaf.py (new) — `admit_tool_source(name, source,
    signature) -> AuthoredLeaf`: the section-3 admission chain. Reuse
    leaves.py internals (`_check_self_contained`, the one-def check, deps
    parsing); add the source-first entry point and the import allowlist. Do
    not fork the checks — factor, or call the private helpers and pin them
    with tests.
  - dspy/optim/flex_prompts.py (new, or a module-level block in flex.py if
    small) — the edit catalog text and slot assembly. Deterministic output for
    a given champion + ledger.
  - No engine changes. No printer changes. If a rewrite needs a node the
    builders refuse, the edit refuses — that is the design working, not a gap.

Tests (tests_greenfield/, DummyLM only, no network):
  - test_flex_v2_vocabulary.py — closed-list law (unknown op / unknown field
    refuses with a teaching error); each op's applier unit-tested; the partial
    op's Try/Except-ToolError + If/Compare-ne tree pinned as exact node JSON
    and round-tripped through the printer; delete_dead_leaf refuses while
    sites remain and succeeds after op 4 empties them; inline_constant record
    spelling.
  - test_flex_v2_admission.py — a good function admits and lands in the pool
    with authored_by: "optimizer" and a source sidecar; each refusal class
    (multi-def, decorator, closure-shaped/global read, missing hint, io-
    contract mismatch against the signature, disallowed import, non-empty
    deps) produces a distinct ledger record.
  - test_flex_v2_loop.py — deterministic SCRIPTED-REFLECTION vectors: a fake
    reflection LM returning a fixed edit-JSON sequence. Vector 1: illegal op →
    ledger entry appears in the NEXT prompt. Vector 2: valid
    replace_predict_with_code → accepted, checkpoint dir loads via dspy.load,
    scores.json trajectory grows. Vector 3: score-tanking edit → rejected,
    champion state bit-identical (applied-on-copy proven). Vector 4: devset-
    memorizing dict tool → devset score up, holdout score down → refused,
    ledger names the holdout gate.
  - THE CHEAPNESS ASSERT (genuine, not mocked-in): a two-predict program where
    step two is mechanical (e.g. count/uppercase over step one's field) under
    DummyLM. Scripted reflection proposes the correct op-4 function. Assert:
    `EvaluationResult.lm_calls` drops from 2N to N over the N-example devset
    while `score` holds exactly; explain view shows the authored tool; cost
    build_text reflects the removed LM leaf; diff shows the edit. For op 5 on
    a mixed devset (some inputs decline): assert N < lm_calls < 2N and that
    declined examples produced their answers through the CallPredict arm.
  - test_flex_v2_prompts.py — slot assembly is deterministic; holdout examples
    never appear in any slot; the catalog names exactly the seven ops.

Definition of done: the vocabulary is closed and pinned; generated code enters
only through admission; every accepted candidate is a loadable artifact; the
cheapness assert passes; the fork's string-substrate flex is untouched (it dies
by replacement, not by edit — upstream isolation rule stands).