# E-09 — the store and the trajectory: checkpoints, diffs, and the objective boundary

**Proves (gaps G7, G8):** the content-addressed serialization (§c1) as
bytes — objects store + manifests-of-hashes; `checkpoint = save` as a
series that costs kilobytes; the View-3 trajectory as data — (snapshot,
labeled diff, metric Δ) triples including rejected branches, a
cross-owner fork, and the D-048 **objective-boundary node**. Successor
to server examples 05/07/12's checkpoint machinery, now byte-fixed.

## 1. The store — one object, second serialization

```
store/
  objects/
    sha256:ab3f…   # weights blob, drafter base       (1.2 GB — written ONCE)
    sha256:e5a8…   # drafter LoRA delta @ step 7      (2.4 MiB)
    sha256:77b1…   # drafter LoRA delta @ step 9      (2.4 MiB)
    sha256:7c02…   # tokenizer bundle
    sha256:9c1d…   # instructions map @ step 3        (few KB)
    sha256:e11a…   # demos map @ step 3
    sha256:41d9…   # devset blob (tickets.dev @ etl result_hash)
    sha256:b201…   # devset blob v2 (after boundary — +120 prod examples)
    sha256:03fa…   # metric source v3
    sha256:0c44…   # metric source v4 (judge rubric tightened)
    sha256:558e…   # manifest skeleton (tree, signatures, adapters, …)
  checkpoints/
    step_00/manifest.json
    step_03/manifest.json
    step_07/manifest.json
    step_07b/manifest.json     # rejected branch — kept, labeled
    step_09/manifest.json      # after the objective boundary
  trajectory.json
```

A checkpoint manifest is the SAME manifest as the self-contained form,
with blob-bearing fields holding hashes instead of inline paths:

```jsonc
// checkpoints/step_03/manifest.json (excerpt)
{"versions": {"…": "…"},
 "components": {
   "3a_instructions": "sha256:9c1d…",
   "3b_demos":        "sha256:e11a…",
   "8_model": {"drafter": {"weights": {"base": "sha256:ab3f…"}}},   // ← unchanged since step_00:
   "12_metric": {"source": "sha256:03fa…",                          //   same hash, zero bytes written
                 "devset": {"blob": "sha256:41d9…"}},
   "skeleton": "sha256:558e…"}}
```

`optimizable ⊆ baked`, made physical: between step_00 and step_03 ONLY
`3a`/`3b` re-hash; the 1.2 GB base is referenced eight times and stored
once. `materialize(step_03) → step_03.ir/` resolves every hash inline —
shipping any checkpoint is always possible; `read`/`explain` accept
both forms (one object, two serializations).

## 2. The trajectory — View 3 as bytes

```jsonc
// trajectory.json
{"trajectory_version": "0.1",                       // ⟂ DRAFT(D-048 family)
 "objective": {"metric": "sha256:03fa…", "devset": "sha256:41d9…"},   // scores below are claims
 "steps": [                                                            //   about THIS pair
  {"id": "step_00", "manifest": "checkpoints/step_00/manifest.json",
   "score": 0.42, "scored_under": {"isolation": "fork_ratchet"}},      // D-043(5): envelope on record

  {"id": "step_03", "parent": "step_00", "score": 0.55,
   "diff": {"kind": "text+demos", "fields": ["3a_instructions/draft", "3b_demos/draft"],
            "by": "optimizer:mipro-v2",
            "note": "reworded task; +2 demos (bootstrap)"},            // optimizer-authored annotation
   "delta": +0.13},

  {"id": "step_07", "parent": "step_03", "score": 0.66,
   "diff": {"kind": "weight-ref", "fields": ["8_model/drafter/delta@draft-binding"],
            "by": "optimizer:train-sgd1",
            "training_run": {"contract": "sgd-1", "loss": "cross_entropy",
                             "trainset": "sha256:41d9…#train", "produced": "sha256:e5a8…"}},
   "delta": +0.11},

  {"id": "step_07b", "parent": "step_07", "score": 0.61,
   "diff": {"kind": "choice", "fields": ["1_module_tree/draft/bindings/adapter"],
            "by": "optimizer:flex", "note": "chat→terse"},
   "delta": -0.05, "status": "rejected"},                              // kept: the dead branch is data

  {"id": "boundary_01", "kind": "objective_boundary",                  // D-048: the loop's output —
   "objective_from": {"metric": "sha256:03fa…", "devset": "sha256:41d9…"},
   "objective_to":   {"metric": "sha256:0c44…", "devset": "sha256:b201…"},
   "by": "outer-loop:prod-w34", "approved_by": "maxime",               // the human terminator
   "diff_note": "+120 prod examples (provenance carried); judge rubric v4",
   "rescored": [{"step": "step_07", "old": 0.66, "new": 0.63}]},       // detach → re-warrant, recorded

  {"id": "step_09", "parent": "step_07", "score": 0.71,                // scores after the boundary
   "objective": {"metric": "sha256:0c44…", "devset": "sha256:b201…"},  //   name the NEW pair —
   "diff": {"kind": "weight-ref", "fields": ["…delta…"],               //   curves never cross a
            "by": "optimizer:train-sgd1", "produced": "sha256:77b1…"}, //   boundary by accident
   "delta": +0.08}]}
```

Every step is a complete, reconstructable ProgramIR (checkpoint = save);
every diff names its kind from the optimizable-kind table, its fields,
its author, and its Δ; `scored_under` carries the D-043 envelope so
scores are comparable; the rejected branch stays in the tree — the
search's shape is part of the record.

## 3. The cross-owner fork (deviation doctrine as trajectory)

A receiver loads `step_09.ir`, rebinds the drafter to their provider,
re-scores with the baked metric — and their trajectory CONTINUES this
one:

```jsonc
{"id": "recv_01", "parent": "step_09", "owner": "acme-mlops",
 "diff": {"kind": "deviation", "fields": ["8_model/drafter"],
          "note": "identity swap: baked Baguettotron → openai/gpt-4o-mini"},
 "score": 0.68,
 "objective": {"metric": "sha256:0c44…", "devset": "sha256:b201…"}}    // same yardstick, new owner
```

Fork-free-at-birth (§c1) + recorded deviations (§e0-binding) = one
program's history legitimately spanning authors, every configuration's
score attributable. (The three weight REGIMES — joint/fork/shared-base —
are E-08's subject; their transitions appear here only as ordinary
`structure` diffs.)

## 4. What a grade-1 reader must do with this (no execution)

Parse both serializations; verify every manifest hash resolves in
`objects/` (R-35); verify `optimizable ⊆ baked` per diff (a diff naming
a frozen field is malformed); render the trajectory (View 3 print);
recompute nothing — hashes are facts, scores are records. A grade-2
engine additionally: materialize, re-score, and verify `rescored`
entries reproduce under the new objective.

## 5. Refusal hooks (cross-ref E-11)

**R-35 · dangling checkpoint hash** `[E-09; §c1]`
> `store: checkpoints/step_07 references sha256:e5a8… which is not present in objects/ — a checkpoint that cannot reconstruct is not a checkpoint`

**R-36 · score curve crossing an objective boundary** `[E-09; D-048]`
> `trajectory: step_09 (objective 0c44…/b201…) compared against step_07's score 0.66 (objective 03fa…/41d9…) — scores attach to configurations; re-read step_07 as 0.63 (rescored) or do not compare`

**R-37 · diff naming a frozen field** `[E-09; §e2 fixed points]`
> `trajectory: diff at step_04 rewrites '2_signature/draft' (external signature) — not in the optimizable set; structure moves live within the §d grammar, the task's type is a fixed point`
