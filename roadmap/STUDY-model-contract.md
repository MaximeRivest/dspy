# Study: the model contract family under the full model universe

**Status:** design study (greenfield-only). Input to the D-045/D-049/D-050
ratification pass. Method: enumerate every model kind a program might
bind, push each through the four drafted axes — request face · training
contract · weights format · engine/placement — and record where the
design holds, strains, or is honestly silent.

## 1. The census

| model kind | request face | training grain | format(s) | engine(s) | provider market today |
|---|---|---|---|---|---|
| chat LLM (incl. VLM) | `lm15/chat-1` | `sgd-1` (LoRA/full) | safetensors, gguf | transformers, vllm, llama.cpp; hosted | the reference market |
| base/completion LM | chat-1 degenerate (`instruct=False` capability) or a `complete-text` face — **open Q1** | `sgd-1` | same | same | shrinking but real |
| embedder (text/image) | `embed-1` | `sgd-1` (contrastive) | safetensors, onnx | transformers, onnxruntime; hosted | real (OpenAI/Cohere/Voyage) |
| multi-vector embedder (ColBERT) | `embed-1`, output shape = matrix (shape lives in the signature, not the face) | `sgd-1` | safetensors | transformers | niche |
| cross-encoder reranker | `predict-1` structurally — **market alias question, Q2** | `sgd-1` | safetensors, onnx | transformers, onnxruntime; hosted | real (Cohere rerank) |
| promptable segmenter (SAM) | `segment-1` (prompt channel: points/boxes/text) | `sgd-1` | safetensors | transformers; hosted | real (Replicate/fal) |
| plain detector/classifier vision (YOLO, DETR) | `predict-1` (image → boxes; no prompt channel) | `sgd-1` | onnx, safetensors | onnxruntime, transformers | real (Roboflow etc.) |
| diffusion image gen (SDXL, Flux) | `generate-image-1` (prompt + face-scoped config: steps/guidance/seed) | `sgd-1` — **LoRA delta fits base ⊕ delta exactly**; the civitai ecosystem is §b-pools practiced in the wild | safetensors | diffusers; hosted | real (fal/Replicate/OpenAI) |
| STT (Whisper) | `transcribe-1` (audio → text; streaming partials) | `sgd-1` | safetensors, gguf (whisper.cpp) | transformers, whisper.cpp; hosted | real |
| TTS | `speak-1` (text → audio; streaming chunks) | `sgd-1` | safetensors | various; hosted | real (ElevenLabs等) |
| classic ML (sklearn/XGBoost/GAM) | `predict-1` | `fit-1` | skops/joblib, onnx, rds, xgboost-json | sklearn, R, xgboost | real (SageMaker/Databricks/Vertex serving) |
| time-series forecaster (Prophet, TimeGPT) | `predict-1` (history+horizon → series; shapes in signature) | `fit-1` | prophet-json, safetensors | prophet, R, torch; hosted | real (TimeGPT is a *hosted forecaster* — the market extends) |
| retrieval index (BM25, FAISS, vector DB) | `predict-1` (query → neighbors) — **an index IS a fitted model: Q3** | `fit-1` (fit(corpus) → index) | faiss index, lucene dir — new format rows | faiss, tantivy; hosted (every vector DB) | real and huge |
| recommender / online learner (river, VW) | `predict-1` | **`online-1` (partial_fit) — missing grain, Q4** | library-native | river, vw | real (as platforms) |
| RL policy | `predict-1` (obs → action) | **needs env + reward, not a trainset — Q5** | safetensors | torch; hosted (Tinker RL) | emerging |
| differentiable authored code (JAX/torch fn with params) | `predict-1`, authored | `sgd-1` over declared params — authored entry + training contract compose; already legal | safetensors (params) + source | jax, torch | none (in-house) |
| GNN | `predict-1` (graph-shaped I/O — marshaling stress, Q6) | `sgd-1` | safetensors | pyg/dgl | thin |

**Out of scope, correctly:** solvers/OR tools (pure code leaves — not learned);
ensembles/cascades (module composition — blooming territory, §e2); agents
(programs, not models).

## 2. Findings

### F1 — The face-admission rule (the study's main yield)

The census shows two forces: structural (does the exchange have channels
beyond typed I/O?) and market (do providers sell it as an endpoint
class?). Proposed rule, mirroring the role-admission test:

> **A face exists iff the exchange carries at least one channel beyond
> typed input/output** — a prompt channel (chat, SAM), a face-scoped
> config vocabulary (sampling; diffusion steps/guidance), or a streaming
> grammar (tokens, partial transcripts, progress frames). **Everything
> else is `predict-1` with shapes in the signature.** Market existence
> alone does NOT admit a face (Q2: rerank stays `predict-1`; the
> provider string routes it fine) — otherwise the vocabulary forks per
> vendor catalog, the disease the role vocabulary already refused.

Under this rule the drafted family holds: `chat-1`, `embed-1` (borderline
— admitted for its market ubiquity and batch semantics), `segment-1`,
`generate-image-1`, `transcribe-1`, `speak-1`, `predict-1`. Closed,
versioned, small.

### F2 — Streaming is per-face, and lm15's grammar is chat's instance

Whisper streams partial transcripts; diffusion streams progress frames;
chat streams role-typed deltas. The stream grammar must be **declared per
face** (the face version pins it), with lm15's `start|delta|end|error`
shape as the template. Buffered replay stays the degenerate case
everywhere — one parsing contract per face, streamed or not.

### F3 — Config is face-scoped; the no-baked-defaults law generalizes

`temperature` belongs to chat; `guidance_scale/steps/seed` to diffusion;
`beam_size` to STT. Each face owns a closed config vocabulary; unset
kwargs are omitted (the greenfield LM rule, generalized verbatim).
`seed` in generate faces is the determinism hook — rung-0 oracle
relevance.

### F4 — Diffusion LoRA is §b-pools practiced in the wild

The civitai ecosystem (shared bases, thousands of deltas, fork/merge
culture) is empirical validation of base ⊕ delta at scale — worth citing
in D-049's why. No design change needed; the fit is exact.

### F5 — Indexes are fitted models (Q3: recommend YES)

`fit(corpus) → index; predict(query) → neighbors` is `fit-1` +
`predict-1` with no remainder. Consequences if ratified: dspy retrievers
stop being a special component and become model leaves; index formats
(faiss, lucene) join the format vocabulary; corpus identity = the
training-set identity (D-046 pins it); a vector DB is a *hosted fitted
model* (the serving market row). This unification deletes a concept
rather than adding one — the cheapest kind of ruling.

### F6 — The training family needs a third grain, deferred (Q4)

`online-1` (partial_fit: state mutates per observation) is real (river,
VW, recommenders) but breaks the current warranty story: serving-time
state mutation means the artifact's state hash moves outside any
training statement. Honest options: (a) defer — schedule online updates
as outer-loop `fit-1` snapshots (works today, keeps checkpoint = save
clean); (b) admit `online-1` with `effects: stateful` and an explicitly
weakened warranty. Recommendation: **(a) now, (b) only when a real
program forces it** — the ReplaceField discipline.

### F7 — RL policy training is not a trainset contract (Q5, deferred)

`Train(target, trainset=...)` cannot express env-interaction training;
the objective there is `(env, reward)` and the env is a *program*. This
is the D-045 four verbs driven by a rollout loop — the Tinker RL
recipes' shape — and it belongs to a later decision (`rl-1` training
grain taking an env-program ref + the metric as reward). Named, not
designed here.

### F8 — Two honest strains, recorded

- **Base LMs (Q1):** recommend chat-1-degenerate via the existing
  `instruct=False` capability fact — no new face; revisit only if
  completion-only providers matter again.
- **Graph/exotic shapes (Q6):** sealed-Arrow covers tables/tensors;
  graph marshaling (and any shape without a JSON-schema meaning) hits
  the same wall as `Callable` fields — refuse loudly at lowering, leave
  the leaf at an authored/in-language rung. A shape the wire cannot
  carry is a placement constraint, stated, not a silent failure.

## 3. Propagation if ratified

- **D-049:** face-admission rule (F1) + the face list; per-face stream
  grammar + config vocabulary clauses (F2, F3); civitai citation (F4);
  indexes-as-models (F5) with retriever unification flagged as its own
  migration; format vocabulary gains faiss/lucene/prophet-json/
  xgboost-json rows.
- **D-045:** grains stated as `sgd-1 | fit-1` closed-for-now, with
  `online-1` and `rl-1` named as deferred candidates (F6, F7).
- **Contract repo (at ratification):** one fixture family per face
  (request/response/stream vectors); refusal vectors for face-config
  mismatches (`temperature` on `embed-1` must refuse).
