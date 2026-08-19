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

**Naming rule: dplyr names wherever dplyr semantics apply; explicit dspy
names for what dplyr never covered.** dplyr is the gold-standard verb
vocabulary for tabular transformation (and §e1 already cites it as prior
art); inventing synonyms would break one-word-one-meaning against the
strongest precedent. The one deliberate departure: dplyr's expression slots
are the whole host language (NSE over R); ours are closed grammars — the
very thing that makes the pipeline data (see below).

| verb | meaning | notes |
|---|---|---|
| `rename` | source column → signature field | dplyr name |
| `select` | keep only named fields | dplyr name |
| `filter` | keep rows matching a predicate | dplyr name; closed predicate grammar (below) |
| `mutate` | new/changed field from existing ones | dplyr name (was `derive`); restricted expression grammar (below) + the named coercion-function table (absorbs the earlier `map_field`) |
| `slice_sample` | seeded subset | dplyr's exact name |
| `shuffle` | seeded reorder | dspy name — dplyr spells this `arrange(sample(...))`, which needs expressions we refuse; an explicit seeded verb is more honest |
| `split` | train/dev/test partition | dspy name — rsample territory, outside dplyr's scope; seeded, fractions as data |
| `designate_inputs` | declare `input_keys[]` | dspy name — the `with_inputs` fact, explicit |

Deferred (the `ReplaceField` discipline — absent until a real case forces
ratification): joins, `group_by`/`summarise` (aggregation also breaks
single-row provenance, which the result-hash story leans on).

**The expression slots reuse the node-set expression grammar — no second
grammar.** `filter`'s predicate and `mutate`'s expression are the ratified
node-set **expression subset** (D-034/D-037: `Compare` orderings +
membership, `BoolOp`, `UnaryOp`, `BinOp` under exact-int64, `Format`, the
pinned float repr, the closed builtin/value-method tables) — no statements,
no leaf calls — plus column references and the coercion table. One grammar,
one meaning: a predicate means the same thing in a forward `If` and in an
ETL `filter`, and the existing Go/TS node-set interpreters already implement
it, so cross-language ETL verification is reuse, not new work.

**The expression slots have their own 3-rung ladder** (contract invariant
across rungs, as everywhere):

| rung | expression form | verification |
|---|---|---|
| 0 — bake | the closed grammar above, pure + seeded | `result_hash`, checkable by any grade-2 reader |
| middle — authored | a UDF step: captured source + deps + `effects: pure` + `authored_by`; in-process or rung-walked sidecar (D-022; D-040 requirements gradient) | `result_hash` holds; static cross-language verification lost, stated visibly |
| outer — pushdown | the rung-0 expression compiled to the store's language (SQL on a warehouse, HF datasets filter) — execution moves to where the data lives | `result_hash` remains the oracle: the pushdown must reproduce rung-0 semantics or load refuses |

The outer rung is what makes §1's referenced datasets usable: a warehouse
cannot be pulled to the client to run rung-0 verbs, but a closed-grammar
predicate translates mechanically, and the hash catches semantic drift
(SQL null handling, collation) loudly. Determinism at the outer rung leans
on the store's snapshot discipline — which is exactly why `dataset_identity`
pins a revision (§1): the two rules lock together.

### 2a. The serialized form and the frontend

The pipeline serializes as verbs + expression trees in the node-set JSON
encoding (the `node` discriminator vocabulary, `schema/node-set.schema.json`)
plus **one new node: `Col`** — a column reference, the row-scope sibling of
`Var`. Example step: `{"verb": "filter", "expr": {"node": "Compare", "op":
"lt", "left": {"node": "Call", "builtin": "len", "args": [{"node": "Col",
"name": "context"}]}, "right": {"node": "Const", "value": 4000}}}`. The
grammar is small enough that `explain` renders steps back as readable
expressions (`answer = answers["text"][0]`) — round-trip legibility.

**The authoring surface is the dpyr chain** (`MaximeRivest/dpyr` — dplyr's
verb vocabulary as Python method chains over a `col` expression proxy,
executing on polars or duckdb, semantics differentially verified against
real dplyr):

```python
(dspy.Dataset.hf("rajpurkar/squad", revision="5fe9c5f")
    .mutate(answer=col.answers["text"][0])
    .filter(col.context.len() < 4000)
    .select("question", "context", "answer")
    .slice_sample(n=300, seed=7)
    .split(train=0.8, dev=0.2, seed=7)
    .with_inputs("question", "context"))
```

The `col` proxy is the tree-building primitive (`col.amount > 6` builds a
`Compare` node); a captured-lambda spelling can arrive later as sugar — the
D-011 layering (canonical builder + convenient spellings, one object
underneath). Authored UDF steps use a visibly different spelling
(`.mutate_with(fn)`) because they are a different tier: captured source,
provenance, lost static verification. Pushdown has **no syntax at all** —
it is placement, not authoring: the same tree against a warehouse-bound
dataset compiles to SQL at the binding.

**The dpyr integration seam.** Three roles, strictly separated: dpyr is a
*frontend* (the chain builds the tree) and a *grade-2 engine* (polars at
rung 0, duckdb fused/pushed-down at the outer rung); **the closed IR tree is
the only serialized truth** — never a pickled chain, never backend
expressions. What dpyr must expose for this:

1. **Plan export** — a capture seam yielding the logical plan as plain data
   (the closed tree), not polars/duckdb expression objects; alternatively
   dspy wraps the proxy and builds the tree itself, using dpyr purely for
   execution.
2. **Vocabulary gate** — a chain using verbs outside the ratified ETL
   vocabulary (`group_by`, `summarize`, joins, windows) still *runs* in
   dpyr but **refuses loudly at export**, naming the verb and
   `etl_version`. When a real case forces admission, dpyr's
   dplyr-verified semantics are the candidate semantics for ratification.
3. **Semantics pin reference** — two pins must be reconciled at the edges:
   dpyr/dplyr semantics govern the verb layer (e.g. `filter`'s NA/None row
   handling), node-set semantics govern scalar arithmetic (exact int64,
   float repr). Divergences are ruled in the fixture corpus — dpyr's own
   `SEMANTICS.md` discipline (differential golden files + cross-engine
   fuzzing, divergences written down) is the instrument, extended to the
   IR: the same fixtures that hold dpyr to dplyr hold every ETL engine to
   the contract.

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
