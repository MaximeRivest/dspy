# Epic I — the exporter: `dspy.export` emits the ProgramIR

**Status:** v4 IN PROGRESS (2026-08-07) — contract readiness pass complete;
canonical `ProgramIR`, compile/write/read/link foundations, bare-Predict
DSPy compilation, predictor-level `set_adapter`, and the v0.1 composite-module
forward compiler plus authored tool/metric/devset and structural interpreter
extraction, portable Python environment locking, and the phase-1 public
`dspy.export` composition landed; the dspy grade-1 conformance row is green.
Weights baking and corpus regeneration remain.
Letter I because D–H are claimed (`03-campaign.md`); the
proposed slot is **between D and E** (dependencies: Epic D only), which
amends the D-016 ratified order — that amendment is this doc's first
ratification ask.

**Charter.** Epic D made the adapter data; `programir-contract` governs
what an artifact *is*; three grade-1 implementations hold artifacts. The
missing edge is the canonical save pipeline. This epic builds it as four
composable operations:

```
compile(live frontend program) -> ProgramIR
write(ProgramIR, path)         -> self-contained artifact directory
read(path)                       -> validated + link-checked ProgramIR
link(ProgramIR)                  -> resolved-binding table   # grade 1, pure
materialize(ProgramIR, bindings) -> executable program       # Epic F
```

**`dspy.export(program, path, metric=None, devset=None)` is only the first
public composition: `compile + write`.** It never owns a second walker,
manifest builder, or filesystem implementation. A future ProgramIR-backed
`Module.save(..., format="program_ir")`, optimizer checkpoint, and
`dspy.load(path, bindings=...)` delegate to these SAME operations. This is
L4 (`checkpoint = save`) applied to API design: there is one artifact path,
not an export path that later gets folded into a different save path.

D-028 frames the stakes: the IR is the product, dspy-Python is frontend #1
— this epic builds frontend #1's save path. It also unblocks the dspy
conformance row, which still reads "shim TODO (lands with the exporter)"
in `programir-contract/IMPLEMENTATIONS.md`. The contract's
`schema/manifest.schema.json` now exists; exporter-regenerated artifacts
are the evidence pass that replaces its provisional migrated fixtures and
sharpens the remaining open field-level rulings.

## Scope conflicts found (for the coordinator, then Maxime)

1. **06-orchestration.md's D-γ bullet** claimed "server examples 01–04
   regenerate from the real exporter." As built (epic-D v5), D-γ regenerated
   only the **`4_adapter` components** through `Adapter.dump_entry()`; the
   full-artifact exporter never existed and no epic owns it. The residue —
   whole manifests through a real `dspy.export` — moves here. 06 should be
   edited to say so when this epic is adopted.
2. **03-campaign Epic F** claims "`ast.parse` + whitelist compile" inside
   the engine consolidation. The exporter cannot exist without that compile
   step, so this epic builds it first, dspy-side, as a compile-only module;
   **F-α's consolidation list gains it as a seed, not a rival.** Boundary:
   this epic compiles (forward source → node JSON, or refusal); Epic F
   executes. F-α adopts `dspy/programir/compile.py` the way it adopts the
   server examples' compilers.
3. **The provisional fixtures are stale below the manifest level**: the
   `cases/artifacts-provisional/` `4_adapter` entries carry the
   pre-D-024 short shape (`versions: {adapter_ir: "0.1", …}`, authored
   `literal_table`, `type`/`format_identity` keys) — not the landed
   canonical `ENTRY_KEYS` entry. Regeneration through the real exporter
   fixes this as a side effect; the corpus commit must be planned knowing
   the diff is not cosmetic. The manifest schema itself has since landed;
   `programir-contract/spec/manifest.md` still calling it TODO is stale
   contract prose to fix spec-first before regeneration.
4. **Today's save/load names have three incompatible meanings.**
   `Module.save(path.json)` writes state into an existing architecture;
   `Module.save(path, save_program=True)` cloudpickles a live architecture;
   `dspy.load(path, allow_pickle=True)` reconstructs that pickle. ProgramIR
   reconstructs its own architecture and therefore folds naturally into
   `Module.save(..., format="program_ir")` plus top-level `dspy.load`, NOT
   into the mutating `module.load(state_path)` contract. The pickle mode is
   not silently redefined: representability refusals make that a breaking
   change. It remains a compatibility backend until a separate deprecation.

## Save/load architecture (the anti-fork contract)

The implementation is layered so folding is delegation, not migration:

- **Frontend assembly, no frontend IR.** `ProgramIR` is the only in-memory
  program representation. A dspy-specific bridge reads live `Module` /
  `Predict` / settings objects and feeds plain component fragments directly
  to the canonical builder; no public or internal `FrontendProgram` value
  crosses the compile boundary. FunctAI's graph dialect feeds the SAME
  builder without importing dspy or reading ambient state.
- **Compile.** `compile(live frontend program) -> ProgramIR`: frontend
  resolution plus pure canonical assembly. Signature metaclasses and Module
  subclass hooks MAY cache immutable fragments (field schemas/roles and a
  validated forward AST), but the complete IR assembles at instance/export
  time because children, demos, and bindings are instance state. These
  caches are never a second representation. No path, filesystem, clock,
  environment lock, or credential value enters compile.
- **Write.** Materializes one ProgramIR as the directory serialization,
  including sidecars, env lock, canonical JSON, and the positive secret
  scan. Packaging policy lives here, never in compile.
- **Read.** Grade-1 operation shipped in this epic: parse, schema/version
  validate, and link-check into a ProgramIR value. It executes no authored
  code and resolves no receiver binding.
- **Link.** Grade-1 operation shipped here: resolve every internal pool
  reference and return the pure resolved-binding table defined by the
  contract. `read()` invokes this same operation after schema/version
  validation. It performs no I/O and materializes nothing.
- **Materialize.** Grade-2 operation owned by Epic F: resolve receiver
  bindings, materialize lawful leaves, verify declared capabilities, and
  return an executable module. No ambient fallback.

The public composition is consequently fixed:

```python
ir = dspy.programir.compile(program, metric=metric, devset=devset)
dspy.programir.write(ir, path)
ir = dspy.programir.read(path)                         # validates + grade-1 links
binding_table = dspy.programir.link(ir)                 # this epic, pure
program = dspy.programir.materialize(ir, bindings=...) # Epic F

# convenience spelling, this epic
dspy.export(program, path, metric=metric, devset=devset)

# later delegating spellings; no new implementation
program.save(path, format="program_ir", metric=metric, devset=devset)
program = dspy.load(path, bindings=...)  # later: read + materialize
```

`dspy.export` remains as the explicit portability API even after `save`
gains the backend. Optimizer checkpoints call `compile + write` directly;
a second checkpoint serializer is forbidden. The existing state-only
`module.load()` continues to mutate an already-authored architecture and
is not the ProgramIR reconstruction door.

## The acceptance program

`roadmap/exemplar-program.py`, minus its flagged-speculative parts. In
scope: deps-comment tools, the ReAct tool pool, nested modules
(`PolicyCheck`), per-predictor `set_lm`/`set_adapter`, demos with
`input_keys`, metric + devset, credentials as declared names, the shared
weight-owning LM (one pool entry, two bindings — phase 2), and every
interpreter object that supplies D-033's structural identity profile. Out,
each named where it falls below: `dspy.BashInterpreter` (no such class; an
authored Bash interpreter is representable when it supplies the profile);
the exemplar's forward *body* (v0.2 constructs — see the boundary);
`BootstrapFinetune`/LoRA-delta plumbing (non-goal); `PythonInterpreter(scope=)`
(the current dspy class is the Deno sandbox; the scope-declared in-process
namespace interpreter is ex-15's reference form — the exporter emits the
identity profile of what the program actually holds).

The implementer authors **`roadmap/exemplar-program-v01.py`** in I-3: the
exemplar with its forward rewritten in v0.1 constructs (dict writes become
predictor outputs threaded by `Assign`; the f-string interpreter call is
dropped or becomes a tool). That file is the E2E stimulus; the pristine
exemplar stays as the aspiration and must **refuse with teaching errors,
never miscompile**.

## The v0.1 boundary (stated once; consequences everywhere)

The contract's `spec/node-set.md` ratified the v0.1 whitelist + SEM-1..8
provisional rulings; the **v0.2 ergonomics batch (`Format`/f-strings,
`Dict`, `Index`/`SetIndex`, `AssignTuple`, orderings, membership,
arithmetic beyond `add`, `For` over lists) is PROPOSED, not ratified.** The
compiler in this epic emits v0.1 ONLY. Consequence, stated plainly: the
exemplar's forward does not compile at this epic's end — its f-string
(`code = f"result = round(...)"`), dict literals (`findings = {...}`),
subscript reads and writes (`findings["tier"] = account["tier"]`,
`step["name"]`), `len()`, `>=`, `in`, and the `**step["args"]` splat are
all v0.2-or-later territory. Refusals name construct + line, and for
constructs in the proposed batch add "proposed in node-set v0.2, not
ratified" — teaching, and honest about time. When v0.2 ratifies
contract-side, the compiler grows in its own PR under this epic's refusal
tests flipped; nothing in this epic pre-implements the proposal.

## Resolution rule — effective LM and adapter per predictor

One rule, both leaf kinds, identical to runtime resolution:

1. predictor-level override — `predictor.lm` (set by `set_lm`);
   `predictor.adapter` (set by `set_adapter`, PR I-2);
2. ambient — `dspy.settings.lm` / `dspy.settings.adapter` at export time;
3. LM absent everywhere → export **refuses naming the predictor path**
   (L5: never a partial binding). Adapter absent → the `ChatAdapter`
   default, exactly as the runtime resolves it.

**Pooling is identity dedup by object.** Two predictors bound to the same
LM *object* share one pool entry (the exemplar's `tiny`: one weights blob,
two bindings). Distinct objects with equal config are distinct entries —
the serializer never guesses equality; merging is an optimizer-view
concern. Entry names derive deterministically under the contract's pool-name
rule: identity dedup first, then base normalization and `-2`, `-3`, ...
collision suffixes allocated in source walk order. Adapters pool through
`Adapter.dump_entry()`; the entry's `name` keys the pool; a non-engine
adapter (legacy override bodies) refuses loudly — the D-γ behavior proven
on server example 04, kept verbatim. The ambient snapshot
(`max_errors`, `async_max_workers`, …) lands in `11_ambient_policy`.

### The `set_adapter` decision: IN, minimal (justified)

The task could assume `set_adapter` lands elsewhere; this epic scopes the
minimal version in, as its own flagged public-surface PR (I-2). Why:

- **The acceptance program needs it to exercise the machinery.** Without a
  per-predictor adapter, every artifact has a one-entry adapter pool and
  the pool/binding walk for component 4 ships untested against any real
  program (all nine provisional fixtures are single-adapter). The exemplar
  (`draft` on ChatAdapter, ambient JSONAdapter) is the first two-entry pool.
- **One resolution rule, not two.** An export-only override channel would
  create a second adapter-resolution path that Epic H would then have to
  kill; the predictor-level attribute keeps export and runtime resolving
  identically.
- **The cost is a mirror.** `Predict` gains `adapter = None`;
  `Module.set_adapter` mirrors `set_lm` line for line; call-time
  resolution checks the attribute before ambient in `forward` and
  `aforward` identically (04-process: sync and async change together).

Bounds: no context-manager form, no per-call override changes, no new
semantics — ambient adapters already resolve at call time; this only adds
the override slot ahead of them. Name ratified by Maxime at the I-β
checkpoint alongside `dspy.export`.

## Component emission (live object → manifest form → refusal)

Target shapes are the D-029-migrated forms in
`cases/artifacts-provisional/` (manifest level) and the landed serde
(component 4). Where fixture and spec diverge, spec-first, before code.

| # | source in the live program | emitted | refuses when |
|---|---|---|---|
| 1 | `named_sub_modules()` walk | `{kind, name, children, forward_ref}`; Predict leaves carry `bindings: {adapter, lm, delta: null}` by pool name | unresolvable LM (rule above) |
| 2 | each predictor's signature | per field: `{name, direction, prefix, desc, shape (JSON Schema), semantic_role}`; role via `_engine/roles.resolve_semantic_role` (derivation table). Legacy fused types split into shape ⊕ role — `dspy.Citations` ⇒ its serialized shape + `semantic_role: "citations"`; no Python identity leaks into `shape` | a field whose shape has no JSON Schema |
| 3a | `signature.instructions` | single-homed here — never restated in template or config | — |
| 3b | `predictor.demos` | values + `input_keys[]` (from `Example._input_keys`) | non-serializable demo value, naming predictor + field |
| 3c | `predictor.config` | as-is | non-JSON value |
| 4 | effective adapters | `dump_entry()` — canonical `ENTRY_KEYS` shape, own versions block | non-engine adapter (loud, names the adapter) |
| 5 | `forward` per module | `ast.parse` + v0.1 whitelist → node JSON; leaf `Call`s resolve against the walked tree (`self.x(...)` → predict/module ref; dict/`self.tools` dispatch → the dynamic-tool convention with `name` expr; interpreter attr → interpreter leaf **with `ref`, D-029**) | any non-whitelist node, naming construct + line; unresolvable leaf |
| 6 | plain-function tools | `{name, parameters, return_schema, source, deps, language: "python", placement}`; source baked to `tools/<name>.py`; `# deps:` comment scan (grammar below); schemas via the `dspy.Tool` machinery | closure/global-reading function (not self-contained); missing hints |
| 7 | interpreter objects | **named pool** (default `main`), D-033 structural identity profile + placement; builtins extracted explicitly, custom objects via `programir_profile()`; stamp `versions.interpreter_profile = "1.0"` | absent/malformed profile, or placement inward of its declared isolation floor |
| 8 | effective LMs | one entry per LM (D-029): identity + placement; `engine` only on baked entries; `served_aliases` optional, `weights_identity` required; nested `weights` only in phase 2 | phase 1: weight-owning in-process LM (see phase split) |
| 9 | aggregated deps | per-language env blocks; Python = PEP 723 entry + `uv lock --script` (defect gate below) | lock failure |
| 10 | api-key-bearing configs | **declared names only** (PIR-005), `credential_ref` in placement blocks | the byte-absence scan hits (below) |
| 11 | `dspy.settings` snapshot | the annotated-IOU flat block | — |
| 12 | `metric=`, `devset=` kwargs | optional evaluation block `{metrics, devset}`: metric introspected exactly like a tool into the named `metrics` pool; ordered devset examples use the flat demo record with required `input_keys[]` | non-introspectable metric (lambda, closure) or non-serializable devset value |
| versions | `_engine/versions.py` + programir constants | **stamped FIRST (PIR-001)**: `ir_version`, `node_set`, `roles`, `strategies`, `codecs`, `adapter_ir`, `lm15` — numbers never restated by hand (anti-drift test below) | — |
| provenance | exporter identity | `{source: "dspy.export", dspy_sha, date, evidence}` (PIR-014) | — |

**`# deps:` grammar (pinned in I-4):** the first comment line inside the
function body matching `# deps: <name>[, <name>…]` — comma-separated
distribution names; absent means stdlib-only. Same scan for tools, metric,
and authored LM class sources.

**Determinism:** exporting the same program twice is byte-identical. No
implicit clock read is permitted. An optional provenance date is explicit
writer input (or derived from `SOURCE_DATE_EPOCH`) and therefore part of
the caller-selected build inputs; absent by default. Pool-name allocation
order is walk order; canonical JSON recursively sorts object keys, so
host-map insertion order never affects emitted bytes.

### LM identity extraction policy (I-1)

Component 8 is produced by a dedicated dspy frontend extractor over live LM
objects. `BaseLM.dump_state()` remains reconstruction state and is NEVER
reinterpreted as ProgramIR representation. The extractor derives:

- `weights_identity` from the model identity. For closed provider weights,
  the provider-scoped model id is the honest available identity. For a
  `dspy.LM` routed through `_OpenAICompatLM`, the engine block's model id is
  the identity rather than the served endpoint alias.
- Endpoint and credential slots are role-named `LM_ENDPOINT` and
  `LM_API_KEY` (with deterministic pool-name disambiguation when multiple LM
  entries need distinct slots), never provider-flavored. Endpoint URLs and
  credential values are bindings, not identity.
- `served_aliases` is optional evidence only.
- Non-builtin LM classes resolve `packaged` versus `authored` origin under
  IR-spec §e0-class; an unresolved origin refuses rather than emitting a
  dangling import path.

The extractor may read reconstruction state as evidence about a known live LM,
but emits only fields governed by the component-8 policy above.

## Credentials: names plus the positive byte-absence scan

Deriving `credential_ref` names is the easy half. The gate is positive:
after writing, the exporter scans **every emitted byte** (manifest, baked
sources, env entry, lock, weights-directory text files) for the *values* of
every credential visible at export time — process env vars matching key
patterns plus any `api_key` strings found on the LM objects — and refuses
on a hit naming file + credential *name*, never the value. The 2026-07
serialization audit found silent secret-leak defects on the state-dict
path; this scan is why `dspy.export` cannot regress the same way.

## The env-block gate (a named, on-server-proven defect)

Every on-server example env block pins
`dspy = { path = "/home/maxime/docmaker/examples-build/dspy-src", editable = true }`
— an absolute editable path that resolves only on the authoring box. The
exporter must emit a **portable dspy requirement** (pinned release; git
URL + full SHA while unreleased), and the check is mechanical and off-box:
the emitted entry locks and import-probes in a clean environment that does
not contain the authoring checkout (runs on the 48-core server from a
scratch directory). An artifact whose env block cannot resolve off-box
fails the epic, full stop.

## Weights baking (8b) — the honest phase split

- **Phase 1 (waves I-α/I-β): declared-tier artifacts only.** Every LM
  entry endpoint-bound; `profile_check` reports in-profile for the
  declared examples (the 09/10/11 shapes). A weight-owning in-process LM
  **refuses** in phase 1 ("weights baking lands in I-γ") — it is never
  silently re-declared as remote, which would misstate placement.
- **Phase 2 (wave I-γ): baking.** `weights/model.safetensors` +
  `rebuild_config.json` + `tokenizer/` + `tying.json` + `device.json`,
  nested inside the LM's one pool entry (D-029 ruling 1), `frozen`/
  weight-ref tag, `engine` present, shared-object dedup (one blob, N
  bindings). **The mechanism is proven** — on-server examples 01–08 and 12
  write and reload exactly this sidecar family, and 12 proves shared-entry
  + checkpoint-=-save through the content store. **The exporter
  integration is new**: a detection seam on the LM object (a declared
  protocol for weight-owning `BaseLM` subclasses — which attributes own
  model/tokenizer — never `isinstance` against an example class), authored
  LM class source baked with its deps comment, and all of it from inside
  `dspy.export` rather than a hand-written build script. Phase 2's risk is
  the seam design, not the bytes.

## Definition of done (mechanical)

1. **SHIPPED:** `dspy.export(program, path, metric=None, devset=None)` is
   public (sync-only — export is not a hot path; **flagged public-surface
   addition**, ratified with `set_adapter` at the I-β checkpoint). `path`
   names the artifact **directory** (the contract ships directories; the
   exemplar's `"ticket_assistant.ir"` is a directory name). Its body composes
   `compile + write`; a test monkeypatches those operations and proves
   delegation. The wrapper additionally performs the live credential-value
   census required by the writer's positive byte scan; component traversal
   and manifest construction remain exclusively `compile`'s job. The internal
   operations are independently testable, no wrapper mints a manifest, and
   `dspy.export` returns the finalized ProgramIR produced by `write`.
2. The emitted artifact is the self-contained directory form:
   `manifest.json`, `tools/`, `metric/`, `lm/`, `weights/` (phase 2),
   `env_entry.py` + lock. Deterministic per the rule above.
3. Every emitted artifact passes `load_manifest` + `check_versions` +
   `link` + `profile_check` **GREEN through all four shims** at the pinned
   contract SHA: reference, Go, TypeScript, and dspy.
4. The nine provisional fixtures (`ap-01..03, 09, 10, 12, 13, 14, 15`)
   are replaced by exporter-emitted equivalents in a **dedicated
   contract-repo corpus commit — zero spec or source changes in that
   commit (corpus rule L8)**; mutated-manifest families (`link-errors`,
   `manifest-errors`, `versions`, `profile`) re-derive from the new bytes
   in the same dedicated commit; provenance blocks flip
   `migrated_by: tools/migrate_v0.py` → exported-by-dspy-at-SHA. If any
   shim's validation disagrees with the regenerated bytes, that is a
   contract bug fixed **spec-first, before** the corpus commit — never
   papered over in fixtures.
5. dspy gains **`CONTRACT_PIN`** (one 40-hex SHA of programir-contract
   main + newline) and a grade-1 **`pir` shim**; the contract's
   `harness/shims.json` registers it and IMPLEMENTATIONS.md's dspy row
   goes green (contract-side commits — Maxime checkpoint, cross-repo);
   `dspy-ci` gains a contract-harness lane that checks out the pinned SHA
   and runs the corpus against the dspy shim. An anti-drift test reads the
   pinned checkout's version constants against what the exporter stamps —
   numbers live in one place per repo, compared mechanically. `read(path)`
   invokes the same validator/link-check implementation as the shim; there
   is no private loader beside the conformance loader.
6. `set_adapter` shipped per the decision above; sync/async parity tests;
   export and runtime resolve through the same rule.
7. Compiler oracle: the forwards of on-server examples 13–15 compile to
   node JSON **equal to the fixture bodies** (the strongest available
   oracle — those bodies were hand-derived from these exact modules);
   refusal tests cover every v0.2-proposed construct and every named
   refusal (`With`, `Yield`, lambda, import, nested def), each naming
   construct + line; the pristine exemplar forward refuses, never
   miscompiles.
8. E2E: `exemplar-program-v01.py` exports; its artifact is green through
   all three shims; the byte-absence scan passes with credentials present
   in the authoring environment (the test plants a fake key and asserts
   refusal when a tool source embeds it).
9. Standing gates: dspy golden adapter corpus **zero drift** (the exporter
   only reads adapters); full `dspy-ci` matrix green (stage new files
   first); stacked commits; no push, no PRs without Maxime's word.

## PR stack

**Wave I-α — the artifact builder.**

- **I-1 — direct ProgramIR compile/write/read skeleton.** New package
  `dspy/programir/` (`model.py`, `compile.py`, `write.py`, `read.py`) plus
  the dspy bridge (`_dspy.py`). `ProgramIR` is the only program value; the
  shared plain-component builder performs no framework introspection while
  frontend resolution remains isolated in the bridge. The engine boundary
  test additionally pins that `_engine` never imports `programir`.
  Components 1/2/3a/3b/3c/4/8-declared/10/11/12 + versions stamped first
  + provenance; dspy resolution rule (ambient + `set_lm`; adapter
  ambient-only until I-2); canonical writer; deterministic naming; the
  byte-absence scan. `read()` validates + link-checks through the same
  grade-1 implementation exposed by the shim. DoD: a two-predictor
  program's artifact passes the four grade-1 ops through the **reference**
  shim; `read(write(compile(program)))` returns the same ProgramIR;
  refusal tests: no-LM, non-engine adapter, non-serializable demo.
- **I-2 — `set_adapter` (public surface, flagged) — SHIPPED.** `Predict`
  owns an adapter slot reset alongside its LM; `Module.set_adapter()` applies
  it recursively; sync/async execution and ProgramIR compile call the same
  predictor resolution method (predictor → ambient → Chat default). The
  effective adapter is installed in the call context so streaming and nested
  machinery observe the same binding. DoD proven for runtime precedence and
  bare-Predict export; the two-entry pool roundtrip rides I-3's composite
  module compiler.
- **I-3 — the forward compiler, v0.1 node set — CORE SHIPPED.**
  `inspect.getsource` + `ast.parse` + the contract's closed encoding;
  Predict/sub-module leaf resolution; deterministic nested module trees and
  object-identity pools; D-029 interpreter `ref`; static and dynamic tool-call
  encodings; construct+source-line refusals with v0.2 teaching hints; authored
  `exemplar-program-v01.py`. Composite two-adapter/one-shared-LM artifacts pass
  load/version/link/profile/node-compile/explain through the reference shim.
  Ex-14/15's tool/interpreter object census and fixture-equality check complete
  in I-4 when those live objects enter the bridge; the node encodings already
  pass the reference `node_compile` operation.

**Wave I-β — leaves, environment, conformance.** (Checkpoint: Maxime
ratifies `dspy.export`/`set_adapter` names and the cross-repo shim
registration.)

- **I-4 — tools, interpreters, metric — STRUCTURAL CORE SHIPPED.** The
  pinned deps-comment scan, closure/global-read self-containment gate,
  `dspy.Tool` parameter schema reuse, return schema, static and dynamic
  dispatch identities, and `tools/*.py` sidecars landed. Named metrics use
  the same extractor into component 12's ratified `{metrics, devset}` block;
  devset records are ordered flat values plus required `input_keys[]`.
  Generated tool/eval artifacts pass reference load/version/link/node-compile;
  the component-12 `explain` defect found by that artifact was fixed in the
  contract reference (`c85c6bf`) rather than worked around in the exporter.
  D-033 then replaced the fused interpreter kind with an open structural
  profile; builtins use explicit extractors and custom objects declare plain
  data via `programir_profile()`. Grade-1 readers hold arbitrary runtimes while
  grade 2 remains free to refuse unsupported execution. The current
  Deno/Pyodide `PythonInterpreter` refuses honestly because its live object
  pins neither runtime version; emitting an exact profile before fixing that
  declaration would violate L6. Structural interpreter artifacts pass every
  reference grade-1 operation. Remaining I-4 evidence item: ex-14's on-server
  tool bodies compared fixture-equal before the corpus flip.
- **I-5 — env blocks (writer policy) — PYTHON CORE SHIPPED.** Compile
  unions tool + metric deps, adds the portable `dspy==<release>` pin, and
  emits a deterministic PEP 723 entry plus component-9 declarations without
  running a resolver. `write` alone runs `uv lock --script` in the staged
  artifact, requires the declared lock to appear, scans the finalized bytes,
  then publishes atomically. It returns the finalized `ProgramIR` (including
  generated locks), so `read(path)` equals the value actually written rather
  than the pre-package input. A clean `/tmp` `uv run --script` import probe
  passed; two independent writes were byte-identical; emitted entry/lock bytes
  contain no editable path or author checkout. Authored LM-class deps join the
  same aggregation in I-7 when that source extractor lands. (Locks are
  *emitted artifacts* here — the known repo `uv.lock` noise rule is unrelated.)
- **I-6 — the `pir` shim + CONTRACT_PIN + CI — CONFORMANCE SHIPPED.**
  Grade-1 ops (`load_manifest`, `check_versions`, `link`, `profile_check`,
  `node_compile`, `diff`, `explain`) use the same bundled contract schemas
  and validator the reader invokes; no second permissive loader exists. The
  contract registers `python -m dspy.programir.shim`, and the dspy row is
  green **33/33** at its durable `CONTRACT_PIN`. Bundled schemas are pinned
  contract inputs; version support remains single-sourced in
  `dspy/programir/versions.py`. The external `dspy-ci` wrapper remains the
  execution lane for this machine; contract harness conformance is also
  exercised directly before every pin bump.

**Wave I-γ — weights and the corpus flip.** (Checkpoint: Maxime approves
the dedicated corpus commit and any push.)

- **I-7 — weights baking (phase 2).** The detection seam (declared
  protocol), sidecar family, shared-entry dedup, frozen/weight-ref tags,
  `engine` on baked entries, authored LM source baking. DoD: re-exporting
  ex-12's authoring program reproduces the proven sidecar layout; one
  blob, two bindings; scan gate covers the weights directory.
- **I-8 — regeneration + corpus commit.** The nine examples re-exported
  **through `dspy.export`** on-server (build scripts become authoring
  program + one export call); includes re-authoring the ex-04/ex-12
  TerseAdapter as a `dspy.TemplateAdapter` (byte-parity with its legacy
  renders proven before regeneration — otherwise 04 stays refused and 12
  cannot regenerate); the dedicated contract-repo fixture commit per DoD
  item 4; IMPLEMENTATIONS.md updated.

## Non-goals (do not start)

- **Content-addressed store emission.** Directory form only; the store's
  manifest schema is a contract open item; ex-12's `store/` stays
  example-side evidence.
- **Ship-object compile-down** (dropping metric/devset for shipping) — a
  later projection over the same artifact (IR-spec §e).
- **Optimizer integration.** No `BootstrapFinetune` plumbing, no LoRA
  deltas (`delta` is emitted `null` always), no checkpoint loops — the
  exporter *is* the save those need; wiring them is F/G territory.
- **v0.2 node set.** PROPOSED; the compiler grows only after contract-side
  ratification, in its own PR.
- **Sidecar wire contracts (D-027).** Placement blocks are emitted;
  kernel-protocol/MCP text is E/F territory.
- **Executing anything.** The shim is grade 1 (hold); `node_execute` and
  `verify` arrive with Epic F, upgrading the same shim to grade 2.
- **`set_adapter` beyond the minimal twin.** No context-manager form, no
  per-call override surface changes.
- **Inventing interpreter implementations.** D-033 makes every interpreter
  structurally representable, but this epic adds no `BashInterpreter` or new
  sandbox. Grade-2 execution support remains Epic F.

## Addendum (2026-08-07) — FunctAI-first consequence

Maxime's direction: the recommended authoring surface (the graph dialect,
`roadmap/exemplar-program-graph.py`; conventions in
`roadmap/frontend-contract.md`) ships in **FunctAI first**. Consequence
for this epic: **I-1's canonical plain-component builder and I-3's forward
compiler land extraction-ready** behind a mechanical import boundary.
Framework introspection lives in frontend bridges: dspy's bridge may import
settings/modules/clients to RESOLVE a compat program directly into ProgramIR;
FunctAI's bridge has no ambient imports and refuses unbound leaves. Immutable
fragments may be cached at Signature/Module class creation, but no
`FrontendProgram` or other second IR exists. In this epic the builder remains
under `dspy/programir/`, extraction-ready behind the same mechanical boundary
used by D-021. FunctAI consumes it without importing dspy runtime objects only
AFTER that neutral core is extracted. After extraction, dspy's class dialect
and FunctAI's graph dialect remain two skins over one save pipeline. The
boundary test feeds equivalent plain fragments from fixture and dspy bridges
and demands identical ProgramIR; the strict FunctAI bridge replaces the
fixture after extraction.

## Open questions for Maxime

1. The epic letter/slot (amends D-016's order) and whether I-α may run
   parallel to E-α (file-disjoint except `predict.py`; default: serial).
2. **Resolved for this draft:** `dspy.export` is the explicit artifact API
   and first caller of the canonical save pipeline. Existing `Module.save`
   modes stay unchanged in this epic; later
   `Module.save(..., format="program_ir")` delegates to `compile + write`.
   Program reconstruction belongs to top-level `dspy.load`, while the
   existing mutating `module.load` remains state-only.
3. **Resolved for this draft:** no implicit clock. Canonical export is
   byte-deterministic; optional provenance time is explicit writer input or
   `SOURCE_DATE_EPOCH`.
4. **Resolved for v3:** component 12 is the optional `{metrics, devset}`
   evaluation block; devset examples use the same flat values + required
   `input_keys[]` record as demos. Contract landed spec-first.
