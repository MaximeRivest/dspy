# Evals, datasets, and the objective's own lifecycle

**Status:** design draft. Companion to `roadmap/IR-program-spec.md`
(component 12, §e0-binding, §e1, §e2 fixed points, §c1) and
`roadmap/DESIGN-trainable-lm.md` (the reward is the metric; RL raises the
stakes on shipping it).

## 1. The devset gets its own slot and ladder

Today the spec gives the **metric** the full placement treatment (rung 0
baked source → judge-LM → scoring service) but the **devset** rides
implicitly inside component 12. Promotion: the devset becomes an explicit
sub-slot with its own placement block and identity field.

| rung | form | identity | binding |
|---|---|---|---|
| 0 — bake | content-addressed JSONL blob in the artifact | the blob hash | none |
| middle — reference local | a database / file store the receiver operates | pinned snapshot hash | `endpoint_ref` |
| outer — named remote | HF dataset id, warehouse query | dataset id + **pinned revision/hash** | `endpoint_ref` + `credential_ref` |

The §e0-binding split transfers verbatim: **identity bakes, location binds.**
The honest rule that follows: **a score is a claim about a dataset identity;
an unpinned (living) dataset yields an unwarranted score by construction** —
the evaluation analogue of `cost = None`, stated, never guessed. Reasons not
to bake are the genuinely-external axes only: size, privacy, licensing.

Each example record carries `input_keys[]` (the 3b fix) and, when mined from
production (§3), a `provenance` block (source call, feedback signal).

## 2. The ETL pipeline — a closed verb vocabulary (dataset → signature shape)

An external dataset almost never matches the program's signature: fields need
renaming, mapping, filtering, splitting. That mapping must travel in the
artifact — otherwise "pinned dataset identity" pins the wrong thing (the raw
source, not what the metric actually consumed). The house pattern applies: a
**closed, versioned operator vocabulary** (the forward-AST whitelist and the
adapter field transforms, at the data layer), a dplyr-shaped pipeline as
plain data:

| verb | meaning | notes |
|---|---|---|
| `rename` | source column → signature field | the common case |
| `select` | keep only named fields | |
| `filter` | keep rows matching a predicate | closed predicate grammar (compare/boolop over fields — the §d expression discipline) |
| `map_field` | coerce one field's representation | named codec/format coercions, never arbitrary code |
| `derive` | new field from existing ones | restricted expression grammar, same admission style as the node set |
| `sample` / `shuffle` | seeded subset / order | seed is data |
| `split` | train/dev/test partition | seeded, fractions as data |
| `designate_inputs` | declare `input_keys[]` | the `with_inputs` fact, explicit |

Deferred (the `ReplaceField` discipline — absent until a real case forces
ratification): `join`, `group`/aggregate.

Rules:

- **Determinism.** Every verb is pure and seeded. Therefore
  `source_identity ⊕ pipeline ⇒ result_hash` is checkable: the artifact bakes
  all three, load re-runs the pipeline and verifies the hash — declare,
  don't discover. A verifier in another language re-implements the small
  closed verb set (the Starlark move, again).
- **Escape hatch.** An authored-code step is a declared UDF leaf: source +
  deps + `authored_by` provenance, captured-or-refused like every authored
  body. Never a silent lambda. An authored step breaks cross-language
  verification for that pipeline and says so.
- **Optimizability.** The pipeline is data, so its knobs (split seed, sample
  size, filter thresholds) are taggable `choice` fields — but they live on
  the objective side of the fence, so inside one optimization run they are
  **frozen by fixed point 2**; only the outer loop (§3) may move them, as
  recorded diffs.

## 3. The outer loop — metric/devset evolution from production traffic

Fixed point 2 (§e2) forbids an optimizer from rewriting its own objective.
It does not forbid the objective from evolving — it forces the evolution
into a **second loop, one level up**, with its own untouchable objective.
The spec's machinery already carries this; the layering is the design:

- **Inner loop (unchanged):** optimize the program against a *pinned*
  metric + devset identity. Every score warranted against that identity.
- **Outer loop (the proactive part):** a separate process — legitimately a
  dspy program itself — that:
  1. **mines production traffic**: View-2 overlays (cost/quality channels,
     `_trajectory`) plus real inputs, collected in a run-log store;
  2. **ingests feedback**: corrections, ratings, downstream outcomes →
     candidate examples with provenance;
  3. **proposes devset updates**: a new hashed blob per snapshot; the update
     is a labeled diff ("devset: +120 from prod week 34, −8 stale") —
     content-addressing (§c1) makes snapshots cheap;
  4. **proposes metric updates**: the metric is a leaf with authored code
     and possibly a judge LM — *it is a program* — so metric evolution is an
     ordinary optimization run whose own fixed point is **agreement with
     human feedback labels**. Goodhart is not eliminated; it is pushed up
     one level, where a human owns the meta-objective. The regress
     terminates at a human, via the deviation doctrine (owner-recorded,
     attributable).
  5. **re-warrants**: objective moved ⇒ historic scores detach (scores
     attach to configurations); re-run the baked metric over kept
     checkpoints — cheap under §c1 — and continue the trajectory through a
     labeled **objective-boundary node** ("metric v3→v4, devset A→B").
     Score curves are never read across that boundary by accident.

**JTBD of the whole aspect:** *keep the objective honest as the world
drifts, without ever letting the inner optimizer grade its own homework* —
while both loops speak one currency: hashed snapshots, labeled diffs,
warranted scores. "The metric evolved" becomes an auditable trajectory
event, not a silent redefinition of success.

## 4. What this does NOT add to ProgramIR

- **Traffic capture stays outside the artifact.** View 2 remains never-baked
  and never load-bearing; the run-log store is a separate artifact kind with
  its own retention/provenance rules, not a ProgramIR component.
- **No online machinery in the IR.** Continuous cadence is a scheduler
  around batch propose→score→keep; the IR sees only snapshots and diffs.

## 5. Propagation

**`roadmap/IR-program-spec.md`:** component 12 gains the devset sub-slot
(placement + `dataset_identity` + the unpinned⇒unwarranted rule) and the ETL
pipeline block (`etl_version` joins the `versions` block, D-024 pattern);
§e1 View 3 gains the objective-boundary node; §e2 gains one sentence — the
fixed point is *per loop*, and objective evolution is the outer loop's
recorded deviation.

**Decision candidates (`05-decisions.md`):** (a) devset identity pinning +
the unwarranted-score rule; (b) the closed ETL verb vocabulary, versioned,
with authored-code steps captured-or-refused; (c) metric/devset evolution is
an outer-loop optimization with a human-owned meta-objective.

**`dspy/programir/` (when implementation starts):** a `12_metric` /
`12_devset` manifest shape with placement + identity; ETL pipeline
validation (closed verbs, seeded, result-hash verification at load);
credential scan over dataset endpoints; `explain` renders the pipeline.

**`dspy/datasets` (new, later):** the verb implementations (pure functions
over rows), the HF/source resolvers, the result-hash check.
