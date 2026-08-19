# The artifact, byte by byte — `answerer.ir/` for the everything-program

**Status:** reference exemplar (greenfield-only). The on-disk truth for
`exemplar-program-everything.py` §§1–7, written against the RATIFIED
manifest schema (`dspy/programir/schema/manifest.schema.json`) wherever a
shape exists, with every **drafted** extension marked `⟂ DRAFT(D-0xx)`.
Purpose: (1) nothing in D-045..D-050 is left to prose interpretation;
(2) a ratifiability check — every discussed concept must appear below as
plain data, or the draft is not implementable. Where a shape is invented
here first, the marker says so: this file feeds the contract-repo
fixture corpus at ratification, it does not bypass it.

## 0. The directory

```
answerer.ir/
  manifest.json                     # everything below except raw bytes
  weights/
    drafter/                        # 8_model "drafter" — baked chat entry
      model.safetensors             #   1.2 GB, bit-for-bit story
      rebuild_config.json
      tokenizer/
      tying.json                    #   [{"target":"lm_head.weight","source":"embed_tokens.weight"}]
      device.json                   #   {"device":"cuda"}
    drafter.delta/                  # the LoRA delta blob (binding-level)
      delta.safetensors             #   2.4 MiB
    embedder/
      model.safetensors
      rebuild_config.json
      tokenizer/
    sam/
      model.safetensors
      rebuild_config.json
  models/
    priority_gam.R                  # authored R source (the fitted-GAM version)
    priority_gam.state.rds          # fitted state — DECLARED format, not bit-for-bit
  tools/
    search_policies.py              # authored tool body
  datasets/                         # ⟂ DRAFT(D-046): baked dataset blobs
    tickets.dev.jsonl               #   content-addressed; hash in manifest
  metric/
    ticket_metric.py                # component-12 authored harness
  envs/                             # ⟂ DRAFT(D-025 instance shapes)
    entry.py                        #   PEP 723 block + lock (python env)
    entry.py.lock
    r/renv.lock                     #   the R language env block
```

## 1. `manifest.json` — versions first (D-024: refused if absent)

```jsonc
{
  "versions": {
    "ir_version": "0.4.0",
    "roles": "1.0", "strategies": "1.0", "codecs": "1.0",
    "node_set": "0.5", "adapter_ir": "0.2.0", "lm15": "1.0",
    "interpreter_profile": "1.0",
    "faces": "0.1",              // ⟂ DRAFT(D-049): request-face vocabulary
    "training": "0.1",           // ⟂ DRAFT(D-045/D-049): sgd-1 | fit-1
    "loss_vocabulary": "0.1",    // ⟂ DRAFT(D-045)
    "etl": "0.1",                // ⟂ DRAFT(D-047)
    "formats": "0.1"             // ⟂ DRAFT(D-049): gguf|safetensors|onnx|skops|rds|faiss
  },
```

## 2. Component 1 — module tree: nodes, external signatures, bindings

Ratified shape (`tree_node`/`predict_leaf`, D-029/D-036). Leaf bindings
name pool entries; `delta` rides the binding, never the entry.

```jsonc
  "components": {
    "1_module_tree": {
      "kind": "module", "name": "Answerer",
      "signature": {                       // D-036: external signature per node
        "fields": [
          {"name": "question",      "direction": "input",  "shape": {"type": "string"}, "semantic_role": "plain"},
          {"name": "product",       "direction": "input",  "shape": {"type": "string"}, "semantic_role": "plain"},
          {"name": "customer_tier", "direction": "input",  "shape": {"type": "string"}, "semantic_role": "plain"},
          {"name": "screenshot",    "direction": "input",  "shape": {"$ref": "#/shapes/image"}, "semantic_role": "media"},
          {"name": "answer",        "direction": "output", "shape": {"type": "string"}, "semantic_role": "citations"}
        ]
      },
      "forward_ref": "Answerer.forward",
      "children": [
        {"kind": "predict", "name": "priority",
         "bindings": {"adapter": "identity", "model": "priority_gam"}},   // ⟂ DRAFT(D-049): key is
        {"kind": "predict", "name": "locate",                             //   "model", generalizing "lm"
         "bindings": {"adapter": "sam_points", "model": "sam"}},
        {"kind": "tool_ref", "name": "search", "ref": "search_policies"},
        {"kind": "predict", "name": "embed",
         "bindings": {"adapter": "identity", "model": "embedder"}},
        {"kind": "predict", "name": "draft",
         "bindings": {"adapter": "chat", "model": "drafter",
                      "delta": "sha256:e5a8…",                            // base ⊕ delta (§b-pools)
                      "isolation_floor": null}},
        {"kind": "predict", "name": "polish",
         "bindings": {"adapter": "json", "model": "polisher"}}
      ]
    },
```

## 3. Components 2/3a/3b/3c — per-predictor learned state (ratified shapes)

Keyed by predictor path; instructions live in 3a ONLY; demos carry
`input_keys`; config omits unset values (no baked temperature).

```jsonc
    "2_signature": {
      "draft": {"fields": [
        {"name": "question",  "direction": "input",  "shape": {"type": "string"},  "semantic_role": "plain"},
        {"name": "passages",  "direction": "input",  "shape": {"type": "array", "items": {"type": "string"}}, "semantic_role": "plain"},
        {"name": "priority",  "direction": "input",  "shape": {"type": "number"},  "semantic_role": "plain"},
        {"name": "answer",    "direction": "output", "shape": {"type": "string"},  "semantic_role": "plain"},
        {"name": "reasoning", "direction": "output", "shape": {"type": "string"},  "semantic_role": "reasoning"}
      ]},
      "locate": {"fields": [
        {"name": "screenshot", "direction": "input",  "shape": {"$ref": "#/shapes/image"}, "semantic_role": "media",
         "role_resolved_from": "shape:Image"},        // ⟂ DRAFT(D-050): deduction provenance
        {"name": "ui_query",   "direction": "input",  "shape": {"type": "string"}, "semantic_role": "plain"},
        {"name": "region",     "direction": "output", "shape": {"$ref": "#/shapes/mask"}, "semantic_role": "media"}
      ]}
      // …priority, embed, polish elided: same field-record shape
    },
    "3a_instructions": {"draft": "Answer the customer's question using the passages.", "polish": "…"},
    "3b_demos": {"draft": [
      {"values": {"question": "…", "passages": ["…"], "priority": 0.4, "answer": "…"},
       "input_keys": ["question", "passages", "priority"]}
    ]},
    "3c_predictor_config": {"draft": {"max_tokens": 512}, "polish": {}},
```

## 4. Component 4 — adapter pool: preset entries, codecs, strategies

Ratified adapter-ir entry shape (D-018/D-031/D-032). Three entries used
here; `identity` is the degenerate preset (⟂ DRAFT(D-049) as a *named
builtin*, not a new shape).

```jsonc
    "4_adapter": {
      "chat":  {"preset": "chat", "adapter_ir_version": "0.2.0"},
      "json":  {"preset": "json", "adapter_ir_version": "0.2.0",
                "config": {"response_format_routing": {"resolved": "json_object"}}},
      "identity": {"preset": "identity", "adapter_ir_version": "0.2.0"},   // ⟂ DRAFT(D-049)
      "sam_points": {                                                      // ⟂ DRAFT(D-049): a preset
        "preset": "sam_points",                                            //   for the segment face —
        "adapter_ir_version": "0.2.0",                                     //   same entry record
        "strategies": {"media": "native_spatial_prompts"},                 //   role → strategy binding
        "codecs": {"region": {"output_codec": "mask_rle"}},                //   shape codec, named pool
        "literal_table": {"prompt_point_policy": "center_of_text_match"}   //   text-optimizable literal
      }
    },
```

## 5. Component 5 — forward AST (ratified node-set 0.5 encoding)

One excerpt — the screenshot branch — to fix the encoding; the rest is
mechanical. `Col` exists only in ETL (§9); forwards use `Var`/`Attr`.

```jsonc
    "5_forward": {
      "Answerer.forward": {"body": [
        {"node": "Assign", "target": "pr",
         "value": {"node": "Call", "leaf": "priority",
                   "kwargs": {"question": {"node": "Attr", "value": {"node": "Var", "name": "inputs"}, "attr": "question"},
                              "customer_tier": {"node": "Attr", "value": {"node": "Var", "name": "inputs"}, "attr": "customer_tier"}}}},
        {"node": "Assign", "target": "query",
         "value": {"node": "Attr", "value": {"node": "Var", "name": "inputs"}, "attr": "question"}},
        {"node": "If",
         "test": {"node": "Compare", "op": "ne",
                  "left": {"node": "Attr", "value": {"node": "Var", "name": "inputs"}, "attr": "screenshot"},
                  "right": {"node": "Const", "value": null}},
         "body": [
           {"node": "Assign", "target": "region",
            "value": {"node": "Call", "leaf": "locate", "kwargs": {"…": "…"}}},
           {"node": "Assign", "target": "query",
            "value": {"node": "BinOp", "op": "add",
                      "left": {"node": "BinOp", "op": "add",
                               "left": {"node": "Var", "name": "query"},
                               "right": {"node": "Const", "value": " ui:"}},
                      "right": {"node": "Attr", "value": {"node": "Var", "name": "region"}, "attr": "label"}}}],
         "orelse": []}
        // … search / draft / If(priority>0.8) / polish / Return — same vocabulary
      ]}
    },
```

## 6. Component 6 — tools (ratified `tool_entry`, D-042 floors)

```jsonc
    "6_tools": {
      "search_policies": {
        "name": "search_policies",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                       "required": ["query"]},
        "return_schema": {"type": "array", "items": {"type": "string"}},
        "source": "tools/search_policies.py",
        "deps": ["httpx"],                       // derived from inline "# deps:" (one source of truth)
        "language": "python",
        "authored_by": "human",
        "kind": "call",
        "placement": {"rung": "in_process", "contract": "tool/call-1",
                      "isolation_floor": "none"}
      }
    },
```

## 7. Component 7 — interpreter pool (ratified D-033 structural profile)

This program has none — the honest artifact says so. The shape, for the
corpus, from a program that does:

```jsonc
    "7_interpreter": {},
    // elsewhere-fixture: "main": {
    //   "profile": {"language": "python",
    //     "runtime": {"identity": "cpython", "version": "3.12.3"},
    //     "contract": "execute(code, vars) -> result",
    //     "namespace_policy": "isolated_no_builtins",
    //     "result_convention": "last_expression",
    //     "vars_marshaling": "json",
    //     "packages": [], "resource_limits": {"iteration_cap": 1000},
    //     "isolation_floor": "fork_ratchet"},
    //   "placement": {"rung": "in_process", "contract": "execute(code,vars)->result"}}
```

## 8. Component 8 — the model pool (⟂ DRAFT(D-049): `8_model`, faces, training)

The heart of the drafts. Ratified `lm_entry` fields kept verbatim
(`weights_identity`, `engine`, `weights`, `placement`, `class`);
drafted additions: `face`, `training`, non-chat entries, `format`.

```jsonc
    "8_model": {                                  // ⟂ DRAFT: key generalizes "8_lm" (flagged byte-shape Q)

      "drafter": {                                // baked, trainable chat entry (rung 0)
        "face": "lm15/chat-1",                    // ⟂ DRAFT(D-049); resolved_from: "hf:architectures"
        "forward_contract": "typed_lm",           // ratified key, unchanged
        "weights_identity": "PleIAs/Baguettotron",
        "engine": "transformers",
        "weights": {"root": "weights/drafter", "format": "safetensors",
                    "frozen": false, "weight_ref": "base"},
        "training": {"contract": "sgd-1",         // ⟂ DRAFT(D-045): the four verbs, declared
                     "placement": {"rung": "in_process", "contract": "trainable_lm/1"}},
        "placement": {"rung": "in_process", "contract": "complete(Request)->Response"}
      },

      "polisher": {                               // declared chat entry (endpoint binds)
        "face": "lm15/chat-1",
        "forward_contract": "typed_lm",
        "weights_identity": "openai/gpt-4o-mini",  // provider-scoped id — the closed-weights concession
        "config": {"native_fc": true},
        "placement": {"rung": "http_remote", "contract": "complete(Request)->Response",
                      "endpoint_ref": "LM_ENDPOINT", "credential_ref": "LM_API_KEY"}
      },

      "embedder": {                               // ⟂ DRAFT(D-049): embed face — NO messages,
        "face": "embed-1",                        //   no sampling config, no finish_reason
        "weights_identity": "BAAI/bge-small-en-v1.5",
        "engine": "transformers",
        "weights": {"root": "weights/embedder", "format": "safetensors",
                    "frozen": false, "weight_ref": "base"},
        "training": {"contract": "sgd-1"},
        "placement": {"rung": "in_process", "contract": "embed(input)->vector"}
      },

      "sam": {                                    // ⟂ DRAFT(D-049): segment face (prompt channel)
        "face": "segment-1",
        "weights_identity": "facebook/sam3",
        "engine": "transformers",
        "weights": {"root": "weights/sam", "format": "safetensors", "frozen": true},
        "placement": {"rung": "in_process", "contract": "segment(media, prompts)->masks"}
      },

      "priority_gam": {                           // ⟂ DRAFT(D-049): authored cross-language entry,
        "face": "predict-1",                      //   fit-trained, declared state format
        "class": {"origin": "authored", "language": "r",
                  "source": "models/priority_gam.R", "deps": ["mgcv"],
                  "authored_by": "human", "effects": "pure"},
        "weights_identity": "sha256:9f2c…",       // content hash of the fitted state — registry-grade
        "weights": {"root": "models/priority_gam.state.rds", "format": "rds",
                    "bit_for_bit": false,         // ⟂ DRAFT: the serialization floor, stated as data
                    "frozen": false, "weight_ref": "base"},
        "training": {"contract": "fit-1"},        // ⟂ DRAFT(D-045/D-049): fit(dataset)->state
        "placement": {"rung": "sidecar",          // rung-walked by a non-R engine (D-022)
                      "contract": "predict(inputs)->outputs",
                      "isolation_floor": "fork"}
      }
    },
```

## 9. Components 9/10/11 — env per language, credentials, policy

```jsonc
    "9_environment": {
      "python": {"kind": "pep723", "entry": "envs/entry.py", "lock": "envs/entry.py.lock"},
      "r":      {"kind": "renv",   "lock": "envs/r/renv.lock",          // ⟂ DRAFT(D-025 instance)
                 "system_deps": [{"name": "R", "version": ">=4.3"}]}
    },
    "10_credentials": [
      {"name": "LM_API_KEY",    "scope": "polisher chat calls"},
      {"name": "JUDGE_API_KEY", "scope": "metric judge calls"}
    ],
    "11_ambient_policy": {"max_errors": 10, "async_max_workers": 8},
```

## 10. Component 12 + the dataset pool (⟂ DRAFT(D-046/D-047))

The metric (ratified `evaluation_block` extended), the devset as an
explicit sub-slot with pinned identity, the dataset pool, and one ETL
pipeline with node-set expressions + the `Col` node.

```jsonc
    "12_metric": {
      "identity": {"name": "ticket_metric",
                   "parameters": {"example": "object", "prediction": "object"},
                   "return_schema": {"type": "number"}},
      "source": "metric/ticket_metric.py", "deps": ["rapidfuzz"],
      "language": "python", "authored_by": "human",
      "judge": {"model": "judge"},                 // a pool entry like any other (chat face, declared)
      "devset": {"dataset": "tickets", "split": "dev"},   // ⟂ DRAFT(D-046): pool ref + split
      "placement": {"rung": "in_process", "contract": "metric(example,prediction)->score"}
    },

    "13_datasets": {                               // ⟂ DRAFT(D-046/D-049): the dataset pool
      "tickets": {
        "source": {"kind": "hf", "id": "acme/support-tickets",
                   "revision": "9c1ffe2",          // pinned — resolved at bake if omitted in syntax
                   "revision_resolved_from": "hf:latest@2026-08-19"},   // ⟂ DRAFT(D-050) provenance
        "etl": [                                   // ⟂ DRAFT(D-047): dplyr verbs, node-set exprs
          {"verb": "filter", "expr": {"node": "Compare", "op": "eq",
             "left": {"node": "Col", "name": "language"}, "right": {"node": "Const", "value": "en"}}},
          {"verb": "mutate", "fields": {
             "question": {"node": "BinOp", "op": "add",
               "left": {"node": "BinOp", "op": "add",
                 "left": {"node": "Col", "name": "subject"}, "right": {"node": "Const", "value": "\n"}},
               "right": {"node": "Col", "name": "body"}},
             "answer": {"node": "Col", "name": "resolved_answer"}}},
          {"verb": "select", "fields": ["question", "answer", "product", "customer_tier"]},
          {"verb": "slice_sample", "n": 800, "seed": 7},
          {"verb": "split", "fractions": {"train": 0.7, "dev": 0.3}, "seed": 7},
          {"verb": "designate_inputs", "input_keys": ["question", "product", "customer_tier"]}
        ],
        "result_hash": "sha256:41d9…",             // load re-runs the pipeline and verifies
        "baked": {"dev": "datasets/tickets.dev.jsonl"}   // rung-0 blob for the shipped split
      },
      "embed_pairs": {
        "source": {"kind": "ref", "endpoint_ref": "DW_TICKET_PAIRS",
                   "snapshot": "2026-08-15T00:00Z"},     // unpinned ⇒ unwarranted (stated rule)
        "etl": [{"verb": "filter", "expr": {"node": "Call", "builtin": "is_in",
                  "args": [{"node": "Col", "name": "label"},
                           {"node": "List", "items": [{"node": "Const", "value": "duplicate"},
                                                      {"node": "Const", "value": "related"}]}]}},
                {"verb": "select", "fields": ["text_a", "text_b", "label"]}],
        "pushdown": {"dialect": "sql", "verified_by": "result_hash"}    // ⟂ DRAFT(D-047 outer rung)
      }
    }
  },
```

## 11. Provenance — deviations, training statements, deductions

```jsonc
  "provenance": {
    "deductions": [                                // ⟂ DRAFT(D-050): every inferred fact, sourced
      {"field": "8_model.embedder.face", "value": "embed-1", "resolved_from": "hf:pipeline_tag"},
      {"field": "8_model.priority_gam.class.language", "value": "r", "resolved_from": "path:.R"},
      {"field": "8_model.priority_gam.training.contract", "value": "fit-1", "resolved_from": "structural:train()"},
      {"field": "1_module_tree.draft.bindings.adapter", "value": "chat", "resolved_from": "face-default"}
    ],
    "training_runs": [                             // ⟂ DRAFT(D-045): statements, never entry-internal
      {"target": "draft", "entry": "drafter", "contract": "sgd-1",
       "loss": "cross_entropy", "trainset": {"dataset": "tickets", "split": "train"},
       "produced": {"delta": "sha256:e5a8…"}, "step_of": "trajectory:step_07"},
      {"target": "priority", "entry": "priority_gam", "contract": "fit-1",
       "trainset": {"dataset": "priority_labels"},
       "produced": {"weights": "sha256:9f2c…"}}
    ],
    "deviations": []
  }
}
```

## 12. The ratifiability checklist this file proves

| discussed concept | where it landed as bytes |
|---|---|
| module tree + bindings + delta | §2 |
| roles + shapes + deduction provenance | §3 |
| adapters: presets, strategies, codecs, literal table (incl. spatial) | §4 |
| forward as node-set JSON | §5 |
| tools + floors + effects | §6 |
| interpreter structural profile | §7 (fixture note) |
| model pool: faces, chat baked/declared, embed, segment, authored-R, formats, bit_for_bit flag | §8 |
| training contracts as data (sgd-1 verbs declared, fit-1, state format) | §8 + §11 |
| cross-language: language tags, per-language env blocks, sidecar rung, isolation floor | §6/§8/§9 |
| credentials as names; endpoint refs | §8/§9 |
| dataset pool: pinned identity, ETL verbs, Col node, result_hash, pushdown, baked split | §10 |
| metric + judge + devset sub-slot | §10 |
| training statements + deduction provenance + deviation slot | §11 |
| everything versioned | §1 |

**Open byte-shape questions surfaced by writing this** (each must be
answered at ratification, none silently): (a) `8_model` vs `8_lm` key;
(b) `13_datasets` numbering vs folding under 12; (c) the `sidecar` rung
name vs reusing `local_rpc`; (d) `Col`'s home (node-set minor bump vs an
etl-scoped grammar); (e) whether `deductions` live in provenance (here)
or beside each field (`role_resolved_from` shows the alternative —
pick ONE at ratification).
