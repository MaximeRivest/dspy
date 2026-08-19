# E-08 — weight regimes: joint, fork, shared-base ⊕ delta, as hash facts

**Proves (gap G6):** the three §b-pools training regimes as bytes —
binding topology + tag pattern — and the regime TRANSITIONS as recorded
trajectory moves. Successor to server example 12 (which proved all
three on hardware as asserted hash facts: joint step moved exactly one
blob; fork free at birth; LoRA step left the frozen base hash
untouched, 2.4 MiB delta vs ~1.2 GiB base).

## 1. One program, two predictors — the three topologies

The program: `draft` and `polish`, both needing a model. The regime is
NOTHING but pool entries + bindings + tags:

### Regime A — joint fine-tune (N bindings → 1 entry, no deltas)

```jsonc
"8_model": {"shared": {"face": "lm15/chat-1", "weights_identity": "PleIAs/Baguettotron",
                       "engine": "transformers",
                       "weights": {"base": "sha256:ab3f…", "frozen": false, "weight_ref": "base"}}},
"1_module_tree": {"children": [
  {"kind": "predict", "name": "draft",  "bindings": {"model": "shared", "adapter": "chat"}},
  {"kind": "predict", "name": "polish", "bindings": {"model": "shared", "adapter": "json"}}]}
```
One `Train` step ⇒ ONE object re-hashes (`ab3f… → 4e21…`); both stages
feel it. The sharing is a stated fact, not repeated content.

### Regime B — forked fine-tunes (N bindings → N entries)

```jsonc
"8_model": {
  "draft_model":  {"…": "…", "weights": {"base": "sha256:ab3f…", "frozen": false}},
  "polish_model": {"…": "…", "weights": {"base": "sha256:ab3f…", "frozen": false}}},
  // fork is FREE AT BIRTH: two entry names, ONE hash, zero new bytes
"1_module_tree": {"children": [
  {"kind": "predict", "name": "draft",  "bindings": {"model": "draft_model"}},
  {"kind": "predict", "name": "polish", "bindings": {"model": "polish_model"}}]}
```
Training `draft_model` diverges its hash (`ab3f… → 9d02…`) while
`polish_model` still points at `ab3f…` — the parent blob untouched,
storage cost = the diverged copy only, and only when training actually
separates them.

### Regime C — shared base ⊕ per-binding LoRA (1 entry, N deltas)

```jsonc
"8_model": {"shared": {"…": "…",
  "weights": {"base": "sha256:ab3f…", "frozen": true, "weight_ref": null}}},   // base FROZEN
"1_module_tree": {"children": [
  {"kind": "predict", "name": "draft",
   "bindings": {"model": "shared", "delta": "sha256:e5a8…"}},                  // 2.4 MiB
  {"kind": "predict", "name": "polish",
   "bindings": {"model": "shared", "delta": "sha256:1b77…"}}]}                 // 2.4 MiB
```
Effective weights per predictor = entry base ⊕ binding delta. A train
step re-hashes one SMALL blob; the frozen base hash never moves — and
`frozen: true` makes that an invariant a grade-1 reader checks, not a
hope.

## 2. The regime table as a conformance check

| regime | topology | tags | a train step re-hashes |
|---|---|---|---|
| A joint | N→1, no deltas | base `weight-ref` | the one base blob |
| B forked | N→N | each base `weight-ref` | that stage's base only |
| C base⊕LoRA | N→1, N deltas | base FROZEN, deltas `weight-ref` | that stage's delta only |

Grade-1 obligations: bindings resolve; a `delta` only rides a binding
whose entry declares a base; `frozen: true` + a training run naming
that blob = malformed (R-38). Grade-2 obligations (ex-12's proofs as
fixtures): reconstruct base ⊕ delta bit-for-bit through the store;
assert exactly the blobs in the table above re-hash per step.

## 3. Transitions are optimizer moves (View-3 diffs, cross-ref E-09)

```jsonc
{"id": "step_11", "diff": {"kind": "structure", "op": "fork_entry",
  "note": "shared → draft_model/polish_model (stage interference in quality channel)",
  "hash_fact": "free at birth: both entries → sha256:ab3f…"}},
{"id": "step_14", "diff": {"kind": "structure", "op": "attach_delta",
  "note": "polish specializes cheaply", "produced": "sha256:1b77…"}},
{"id": "step_17", "diff": {"kind": "structure", "op": "merge_delta",
  "note": "draft delta consolidated into base",
  "hash_fact": "base sha256:ab3f… → sha256:c530…; delta blob unreferenced"}}
```

`fork_entry` / `attach_delta` / `merge_delta`: three named ops, each a
recordable, scoreable, reversible move — no mainstream checkpoint
format expresses this; here it is three lines of diff.

## 4. Refusal hook (cross-ref E-11)

**R-38 · training a frozen blob** `[E-08; §b-pools]`
> `training_run at step_12 names blob sha256:ab3f… which entry 'shared' declares frozen — flip the tag as a recorded diff first, or attach a delta; frozen means the hash may not move`
