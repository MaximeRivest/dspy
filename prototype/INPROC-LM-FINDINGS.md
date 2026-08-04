# In-Process, Weight-Owning LM for DSPy — Findings

Proof-of-concept: a `dspy.BaseLM` subclass that owns model weights directly,
runs generation locally on CPU, and serializes the actual weights (not a
model-id or a path) so the artifact round-trips to an identical model in a
fresh process. Everything is CPU / in-RAM / in-process — no server, no GPU,
no network at inference.

Files:
- `prototype/inproc_lm.py` — `InProcessLM(BaseLM)`
- `prototype/prove.py` — proof harness (prints EXPECTED vs ACTUAL, exits nonzero on failure)

Result: **ALL CHECKS PASSED** (`python -m prototype.prove`, exit 0).

## Model real specs — PleIAs/Baguettotron

Confirmed by loading `AutoConfig` / `AutoModelForCausalLM` in-process:

- **Params: 320,956,992 (~321.0 M)**, verified by summing `p.numel()`.
- **Architecture: `LlamaForCausalLM`** (`model_type: "llama"`). Deep-and-thin:
  `num_hidden_layers=80`, `hidden_size=576`, `intermediate_size=1536`.
- Attention: GQA, `num_attention_heads=9`, `num_key_value_heads=3`, `head_dim=64`.
- `vocab_size=65536`.
- **Tied embeddings**: `lm_head.weight` is tied to `model.embed_tokens.weight`
  (`_tied_weights_keys = ["lm_head.weight"]`). This matters for serialization (below).
- **Chat template: yes** (ChatML-style). Renders e.g.
  `<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n<think>\n` — note it
  force-opens a `<think>` block on every assistant turn.
- **No `eos_token` / `pad_token` defined** by the tokenizer (`eos_token_id` is
  `None`). The chat template closes turns with `<|im_end|>` (id **65492**), so
  `InProcessLM` uses that id as both the generation stop token and pad id.
- Loaded as **float32** for CPU. `device="cpu"`; a `device` slot ("auto" ->
  cuda/mps/cpu) is threaded through but untested beyond CPU in this PoC.

## dump_state / load_state layout — "the weights slot"

`dump_state()` returns a **flat JSON-serializable dict** (all values are
JSON-native: strings, ints, bools, and nested dicts of strings). Exact keys:

| key | type | meaning |
|-----|------|---------|
| `_dspy_lm_class` | str | `"prototype.inproc_lm.InProcessLM"` — DSPy's `LM_CLASS_STATE_KEY`, used by `BaseLM.load_state` to re-import the class (requires `allow_custom_lm_class=True` when loaded via the generic path). |
| `inproc_lm_version` | int | schema version, `1`. |
| `model` | str | source HF id (`"PleIAs/Baguettotron"`), informational only — **not** used to fetch weights on load. |
| `model_type` | str | `"chat"`. |
| `device` | str | the requested device **slot** (e.g. `"cpu"`), preserved across round-trip. |
| `cache`, `num_retries`, `temperature`, `max_tokens` | misc | ordinary `BaseLM` request defaults. |
| `weights_format` | str | `"safetensors"`. |
| **`weights_b64`** | str | **THE WEIGHTS SLOT.** base64 of the raw bytes produced by `safetensors.torch.save(state_dict)`. This is the entire model `state_dict` minus tied duplicates (see below), as one in-memory safetensors buffer. No path, no URL, no model-id shortcut. |
| `config_json` | str | `hf_config.to_json_string()` — the full HF config, so the architecture is rebuilt with `AutoConfig` + `AutoModelForCausalLM.from_config` (no download). |
| `tokenizer_files_b64` | dict[str,str] | every file from `tokenizer.save_pretrained()` (tokenizer.json, config, special-tokens map, etc.), each base64-encoded, so the tokenizer is rebuilt offline. |

**Tied-weight handling (important detail for an IR spec).** safetensors
refuses tensors that share storage, and Baguettotron ties `lm_head.weight` to
`embed_tokens.weight`. So on **save** we drop every key in
`model._tied_weights_keys` from the state_dict before `safetensors.save`; on
**load** we `load_state_dict(..., strict=False)` (the dropped tied key shows up
as "missing") and then call `model.tie_weights()` to re-establish the sharing.
The weights slot therefore stores each unique parameter exactly once.

`load_state()` is fully offline: writes `config_json` to a temp dir → `AutoConfig`;
writes `tokenizer_files_b64` to a temp dir → `AutoTokenizer`; `from_config` builds
an empty model; `safetensors.load(b64decode(weights_b64))` → `load_state_dict` →
`tie_weights`. No network touched.

## Did same-logits / same-output hold across reload?

**Yes — bit-for-bit identical generated token ids.** The harness captures greedy
(temperature 0, `do_sample=False`) token ids on a fixed prompt, JSON-serializes
the state to bytes, deserializes into a **fresh** `InProcessLM.load_state`, and
regenerates. EXPECTED == ACTUAL:
`[5, 6869, 351, 265, 3854, 284, 2989, 7228, 211, 65507, 10689, 32568, 10711, 17, 30750, 1514]`.

## Did the grad step mutate weights?

**Yes.** One `(prompt="The capital of France is", target=" Paris.")` pair, cross-
entropy masked to the target continuation (`labels[:prompt_len] = -100`), one
SGD step (lr=1.0). Loss was **finite (2.7557)** and the watched tensor
`model.embed_tokens.weight` changed with `max|delta| = 4.737` — clearly mutated.
The model is trainable in-process.

## No api_key / no network

- Serialized state top-level keys contain no `api_key` and no credential-like
  key. Full key set:
  `_dspy_lm_class, cache, config_json, device, inproc_lm_version, max_tokens,
  model, model_type, num_retries, temperature, tokenizer_files_b64,
  weights_b64, weights_format`.
- Inference on the reloaded model succeeds with `socket.socket` monkey-patched
  to raise — proving no network call at generation time.

## Serialized artifact size

- **JSON artifact (as proven, `json.dumps` of the full state): 1638.52 MB.**
  This is the shippable self-contained blob (weights + config + tokenizer).
- Underlying raw float32 weights: ~1224 MB full, ~1080 MB after dropping the
  tied `lm_head`. The JSON figure is larger because base64 inflates bytes ~33%
  and the whole thing is a JSON string.
- Size takeaway for an IR spec: base64-in-JSON is honest and portable but ~1.33x
  bloated and slow (the full round-trip takes >2 min on CPU). A real spec should
  put `weights_b64` bytes in a **binary sidecar** (raw safetensors on disk /
  blob store) rather than inlining base64 in JSON; the logical layout above is
  unchanged.

## DSPy code I had to work around

1. **`safetensors` rejects tied weights.** Not DSPy's fault, but the
   Baguettotron/Llama tie between `lm_head` and `embed_tokens` forced the
   drop-on-save / re-tie-on-load dance described above. An IR "weights slot"
   must decide a canonical policy for tied tensors.
2. **`BaseLM.load_state` gatekeeps custom classes.** The generic
   `BaseLM.load_state` refuses to import a non-builtin `_dspy_lm_class` unless
   `allow_custom_lm_class=True`. The PoC sidesteps this by calling
   `InProcessLM.load_state(state)` directly (the classmethod is invoked on the
   concrete class), which is the clean path for a known weight-owning LM.
3. **transformers 5.x API drift (not DSPy).** `apply_chat_template(...,
   return_tensors="pt")` returns a `BatchEncoding`, not a bare tensor, and
   `dtype=` replaced `torch_dtype=`. Handled defensively in `_build_inputs`.
4. **No signature inspection by DSPy.** The typed contract worked exactly as
   documented: set `forward_contract = "typed_lm"`, implement
   `forward(request: LMRequest) -> LMResponse`, and calls with an explicit
   `dspy.LMRequest` return a typed `LMResponse`. No workaround needed —
   capability flags (`supports_function_calling` etc.) are all reported False
   honestly for this tiny base model.
