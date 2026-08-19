# Trainable LMs — the weight-training seam

**Status:** design draft. Companion to `roadmap/IR-program-spec.md` (§b-pools
base ⊕ delta, component 8b, §e0 placement) and the greenfield LM layer
(`dspy/lm/lm.py`, `dspy/lm/inprocess.py`). External reference: the Tinker SDK
(Thinking Machines) — its API cut validates this design's contract; its
cookbook is the catalog of things dspy's signatures/adapters make unnecessary.

## Thesis

Weight training joins the optimizer family through one small **structural
training contract** — four verbs an engine may declare — with every host
(in-process torch, a self-hosted training service, a hosted service like
Tinker) being one binding of those verbs on a **training placement ladder**.
The user never sees tokens, loss masks, renderers, or datum objects: the
signature says *what* the model must learn to emit, and the adapter says
*exactly where those bytes sit in the token stream* — the two facts every
hand-rolled fine-tuning pipeline reconstructs manually.

## The contract (four verbs, structural)

An engine that supports training declares, structurally (no base class —
D-036 admission style, like `programir_weight_spec()`):

```python
forward_backward(batch, loss: str) -> LossReport   # accumulate gradients
optim_step(params) -> None                         # apply them
save_state(name) -> WeightsRef                     # checkpoint (weights [+ optimizer])
sample(request) -> Response                        # generate from current weights
```

- `sample` is already `complete()` — the inference face and the on-policy
  sampler are the same seam. `save_state` terminates in the ProgramIR
  weights slot (safetensors base or delta blob).
- `loss` is a closed, versioned vocabulary: `cross_entropy |
  importance_sampling | ppo | ...` — plus an authored-code escape hatch
  (declared leaf, provenance-tracked, never a hidden lambda).
- `batch` is trainer-internal data derived from rendered adapter output;
  it is not a user-facing type.

Hosts on the training ladder (mirrors §e0):

| rung | host | verbs bound to |
|---|---|---|
| 0 | in-process torch (`InProcessEngine`) | a local training loop over the live `nn.Module` |
| middle | self-hosted training service (Prime-Intellect-style) | HTTP, `endpoint_ref` + `credential_ref` |
| outer | hosted service (Tinker) | vendor SDK behind the same four verbs |

Rung 0 bakes weights; outer rungs declare `weights_identity` and pull the
trained blob (base or LoRA delta) back into the artifact at checkpoint time.

## What dspy abstracts away (no user surface)

Derived from the Tinker cookbook — each row is user work there, machinery here:

| Tinker cookbook | dspy owner |
|---|---|
| `renderers` (chat templates, `TrainOnWhat`) | adapters/templates; capability-declared LM |
| `Datum`, token shifting, loss-weight masks | derived: adapter render → tokenize → mask output-field spans |
| `SupervisedDataset` builders | devset of `dspy.Example` + signature; bootstrap traces as SFT data |
| `Env` / `EnvGroupBuilder` / tool-use envs | the dspy program *is* the environment; trajectory = execution |
| reward functions | the metric leaf (component 12) |
| evaluators / benchmarks | `dspy.Evaluate` + metric |
| `save_weights_and_get_sampling_client` | LM pool entry binding — samplers see new weights automatically |
| `checkpoint_utils`, `tinker://` paths | `checkpoint = save`, content-addressed store, base ⊕ delta |
| async futures, pipelining, billing | trainer-internal |

## User-facing surface (the only new API)

1. **The trainer** — a `dspy.optim` optimizer, same call shape as the rest:

   ```python
   tuner = dspy.optim.WeightTune(metric=my_metric, loss="cross_entropy",
                                 lr=1e-4, rank=32)   # rank>0 → delta (LoRA); rank=None → base
   trained = tuner.compile(program, trainset=train)
   ```

   An RL variant carries the shaping knobs (group size, advantage scheme,
   on-policy cadence), defaulted hard.
2. **The training host binding** — declared on the LM entry (placement-style),
   never a trainer argument: in-process needs nothing; a service rung binds
   `endpoint_ref`/`credential_ref`.
3. **Trainable tags** — which entries train, frozen vs `weight-ref`, base vs
   delta: the §b-pools tag surface, spelled per-LM (`frozen=`/`trainable=`)
   and recorded in the artifact.
4. **Custom loss** — authored-code leaf, `authored_by` provenance, loud gates.

## Propagation: `roadmap/IR-program-spec.md`

- **§e0 gains the training ladder.** Placement today covers where inference
  runs. Add: a trainable LM entry may carry a `training` block —
  `{contract: "trainable_lm/1", rung, endpoint_ref, credential_ref,
  loss_vocabulary_version}` — same invariant-contract-across-rungs rule.
  Training at an outer rung is declare-tier; the trained blob returning to
  the artifact is the bake step.
- **Component 8a/8b note.** The `engine` field's training capability becomes
  explicit: `transformers` is training-capable, `vllm-offline` is not; a
  service-trained entry records the service as the training placement while
  weights identity/blob remain the entry's. The loss vocabulary joins the
  `versions` block (D-024 pattern).
- **§b-pools.** No change to base ⊕ delta — LoRA rank lands as delta metadata.
  Add one sentence: a training run's host is placement, not identity — moving
  training from torch to a service does not detach scores (PIR-010 pattern);
  changed *weights* do, as always.
- **§e1 View 3.** Weight steps already fit the trajectory; add the loss curve
  as a cost-channel-adjacent per-step fact.
- **Decision to record** (`05-decisions.md`): the four-verb contract, the
  closed loss vocabulary, and "training host is a binding" as a D-entry.

## Propagation: `dspy/programir/`

- **`weights.py`** — the weight-spec dict grows optional fields:
  `delta_of` (name of the base entry when the blob is a LoRA delta) and
  `rank`; validation extends accordingly. `bake_lm` learns to write a delta
  sidecar beside a referenced base instead of a full base blob.
- **`model.py` + `schema/manifest.schema.json`** — the `8_lm` entry admits an
  optional `training` block (contract id, rung, `endpoint_ref`,
  `credential_ref`) and binding-level `delta` refs; schema stays
  validate-loudly on unknown fields.
- **`versions.py`** — add `loss_vocabulary` to the versions block.
- **`contract_validate.py`** — new checks: a `training` block at an endpoint
  rung must bind through `endpoint_ref` (no baked URLs); a `delta` must
  resolve to a declared base entry (link-error style); DT-004 declared-tier
  wording extends to training blocks.
- **`export.py`/`write.py`** — credential scan covers the training block;
  export refuses a trainable claim on an engine that does not declare the
  contract (capture-or-refuse, DT-003 pattern).
- **`_dspy.py`** — structural detection extends: alongside
  `programir_weight_spec`, detect the four training verbs; record the
  training contract in the emitted entry.

## Propagation: outside `programir/`

- **`dspy/lm/inprocess.py`** — implement the four verbs over the live module
  (rung-0 reference binding).
- **`dspy/optim/weight_tune.py`** — the trainer: adapter-derived datum/mask
  builder, loop over the contract, `checkpoint = save` each kept step.
- **`dspy/adapters`** — one new derived fact: the output-field token-span map
  for a rendered exchange (the mask source). Pure, testable without an LM.
- **Tests** (the new suite, in order): mask derivation from adapter renders
  (pure, no LM); the four-verb contract over a fake engine; rung-0 train step
  moves weights on a tiny model (gated, slow); export of a delta-carrying
  artifact round-trips.

## Non-goals

- No BaseLM-style trainer base class (structural admission only).
- No vendor SDK types in the user surface — Tinker/Prime bindings live
  behind the verbs.
- No guessed training costs; billing facts are the service's, reported or
  absent.
