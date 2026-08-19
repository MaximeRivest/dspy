# E-05 — lowerings: sugar in the frontend, expansion in the artifact

**Proves (gap G3):** the §d-lowering pipeline as bytes — the artifact
NEVER contains sugar, only the core tree; `lowered_from` provenance on
every emitted node; the sugar view vs core view as two renders of one
program; the four lowering laws as checkable facts. Instances: CoT (the
original lowering) and parse-fallback (the error-policy lowering that
evicted the engine's adapter-layer retry).

## 1. The frontend (sugar — what the author writes)

```python
class QA(dspy.Module):
    def __init__(self):
        self.answer = dspy.ChainOfThought("question -> answer",     # sugar 1: CoT
                                          model=lm,
                                          on_parse_error="json")    # sugar 2: fallback
    def forward(self, inputs):
        return self.answer(question=inputs.question)
```

Neither annotation survives compilation as itself. The artifact holds
the EXPANSION; `explain --sugar` reconstructs the readable view from
provenance.

## 2. CoT lowered (signature rewrite, captured resolved)

```jsonc
"2_signature": {"answer": {"fields": [
  {"name": "question",  "direction": "input",  "shape": {"type": "string"}, "semantic_role": "plain"},
  {"name": "reasoning", "direction": "output", "shape": {"type": "string"},
   "semantic_role": "reasoning",
   "lowered_from": "chain_of_thought"},           // mechanism, not contract: this field
  {"name": "answer",    "direction": "output", "shape": {"type": "string"}, "semantic_role": "plain"}
]}},
```

§d-sacred applied: the module's EXTERNAL signature (component 1 node)
still reads `question -> answer` — `reasoning` exists only on the inner
predictor, and at runtime its content routes to the observability
channel (`_trajectory`), never a prediction field. Law 1 held strictly:
same inputs in, same prediction fields out.

## 3. Parse-fallback lowered (annotation → Try flow)

The `on_parse_error: "json"` annotation expands to ordinary component-5
error flow — the retry becomes a NODE: countable in View 2, attributable
in the trajectory, refusable.

```jsonc
"1_module_tree": {"children": [
  {"kind": "predict", "name": "answer",
   "bindings": {"adapter": "chat", "model": "lm"},
   "lowered_from": null},
  {"kind": "predict", "name": "answer__fallback",              // emitted by the lowering:
   "bindings": {"adapter": "json", "model": "lm"},             //   SAME leaf, different
   "lowered_from": "parse_fallback",                           //   adapter binding
   "optimizable": false}]},                                    //   frozen by default — exists,
                                                               //   visible, not tunable
"5_forward": {"QA.forward": {"body": [
  {"node": "Try",
   "body": [{"node": "Assign", "target": "out",
             "value": {"node": "Call", "leaf": "answer", "kwargs": {"…": "…"}}}],
   "handlers": [{"node": "ExceptHandler", "type": "AdapterParseError", "name": "e",
     "body": [{"node": "Assign", "target": "out",
               "value": {"node": "Call", "leaf": "answer__fallback", "kwargs": {"…": "…"}}}],
     "lowered_from": "parse_fallback"}]},                      // provenance on the emitted node
  {"node": "Return", "value": {"node": "Var", "name": "out"}}]}}
```

## 4. The four laws as grade-1 checks

| law | check on these bytes |
|---|---|
| 1 interface-preserving, strictly | external signature of the lowered node ≡ pre-lowering; no added inputs/outputs (MCC stays an honest program maker, not a lowering) |
| 2 call-honest | every LM exchange in the expansion is a leaf: `answer__fallback` is countable — the single-shot invariant as a closure property |
| 3 provenance-total | every emitted node/field carries `lowered_from`; `explain` renders sugar view ⇄ core view losslessly |
| 4 composition is nesting, declared | `Retry(TwoStep(p))` vs `TwoStep(p, extract=Retry(…))` serialize as DIFFERENT trees; application order is provenance data |

## 5. Lowering application is a `choice` field (cross-ref E-09)

```jsonc
{"id": "step_05", "diff": {"kind": "choice", "op": "apply_lowering",
  "note": "two_step applied to 'answer' (extraction: cheap-lm + json)",
  "fields": ["1_module_tree/answer"], "by": "optimizer:flex"}, "delta": +0.05}
```

Retrofit on a RECEIVED artifact is the same op as a recorded deviation
(§d-lowering author-time ≡ retrofit) — what ambient adapters provided
by accident, provided deliberately.

## 6. Refusal hooks (cross-ref E-11)

**R-39 · lowering that adds an external input** `[E-05; Law 1]`
> `lowering 'hint_injector' on 'answer': expansion adds external input 'hint_' — interface-preserving strictly; added inputs break drop-in substitution (declare an honest program maker instead)`

**R-40 · lowering that smuggles a call** `[E-05; Law 2]`
> `lowering 'clean_json' on 'answer': expansion performs an LM exchange with no emitted leaf (parser-internal call) — the single-shot law; every exchange is a leaf, authored, lowered, or refused`

**R-41 · ambiguous lowering composition** `[E-05; Law 4]`
> `lowerings 'retry' and 'two_step' both annotate 'answer' with no declared order — composition is nesting, declared; write Retry(TwoStep(p)) or TwoStep(p, extract=Retry(…))`
