# E-07 — RLM: the interpreter leaf, generated code as data, isolation as bytes

**Proves (gaps G1, G13):** component 7 in use — an interpreter pool
entry with the full D-033 structural profile; model-generated code
flowing into the `execute` leaf as DATA; `While` + iteration cap;
interpreter failures as catchable typed errors; and the ratified
isolation machinery (D-042/D-043) as bytes — floor, grants row, session
kind, sealed-Arrow vars plane. Successor to server example 15; at rung
0 the oracle is BIT-FOR-BIT with the network blocked (the fully-baked
pole), unlike E-06's structural oracle.

## 1. The program

```python
import dspy

coder = dspy.Model("hf:PleIAs/Baguettotron", device="cuda")   # rung-0 baked chat entry

class Solver(dspy.Module):
    def __init__(self):
        self.write_code = dspy.Predict(
            "question, variables_info, error: str -> code @code",  # the code ROLE:
            model=coder)                                           #   projects onto the
        self.finalize = dspy.Predict(                              #   interpreter leaf
            "question, result -> answer", model=coder)
        self.repl = dspy.Interpreter("main")                       # binds pool entry "main"

    def forward(self, inputs):
        error = ""
        result = None
        while result is None:                       # While — closed by ex-15
            pred = self.write_code(question=inputs.question,
                                   variables_info="x: int, docs: list[str]",
                                   error=error)
            try:
                result = self.repl.execute(pred.code,               # generated code = DATA
                                           vars={"x": 7, "docs": inputs.docs})
            except dspy.InterpreterError as e:                      # NameError inside the
                error = e.message                                   #   leaf surfaces typed,
                result = None                                       #   catchable, loopable
        return self.finalize(question=inputs.question, result=result)
```

The engine-side iteration cap (`LoopCapError`, uncatchable) bounds the
`While` at interpretation — a serialized program cannot spin a loader
forever (ex-15's rule, now a profile field).

## 2. The interpreter pool entry (component 7) — profile + isolation, all data

```jsonc
"7_interpreter": {
  "main": {
    "profile": {                                   // D-033: OPEN STRUCTURAL identity —
      "language": "python",                        //   what the scores are claims about
      "runtime": {"identity": "cpython", "version": "3.12.3"},
      "contract": "execute(code, vars) -> result",
      "namespace_policy": "isolated_no_builtins",
      "result_convention": "last_expression",
      "vars_marshaling": "sealed_arrow",           // D-042(4): memfd + F_SEAL_WRITE, Arrow IPC —
                                                   //   the local data plane; JSON-typed on wire rungs
      "packages": [],
      "resource_limits": {"iteration_cap": 1000, "wall_ms": 5000, "mem_mb": 512},
      "isolation_floor": "fork_ratchet"            // author's MINIMUM demand — bakes (D-042(3))
    },
    "kind": "session",                             // D-043(6): leaf substrate discriminant —
                                                   //   a held stateful context; at fork_ratchet+
                                                   //   lifetime = leaf span (fork, place, ratchet,
                                                   //   run, exit — the zygote pattern)
    "grants": [],                                  // the closed static effect row: NO fds, NO
                                                   //   broker routes — a zero-grant interpreter is
                                                   //   not an integrity sink (spec/flow.md);
                                                   //   generated code physically cannot reach the
                                                   //   network (kernel-backed cheapness, D-043(3))
    "placement": {"rung": "in_process",
                  "contract": "execute(code, vars) -> result",
                  "isolation": "fork_ratchet"}     // receiver's ENVELOPE — may exceed the floor,
  }                                                //   never under-run (R-26); recorded per run
}
```

Versions: `interpreter_profile: "1.0"` (D-033). The fused legacy
`kind: python-inproc-namespace` is readable, never emitted.

## 3. The forward bytes — While + the interpreter leaf call

```jsonc
"5_forward": {"Solver.forward": {"body": [
  {"node": "Assign", "target": "error",  "value": {"node": "Const", "value": ""}},
  {"node": "Assign", "target": "result", "value": {"node": "Const", "value": null}},
  {"node": "While",
   "test": {"node": "Compare", "op": "eq",
            "left": {"node": "Var", "name": "result"}, "right": {"node": "Const", "value": null}},
   "body": [
     {"node": "Assign", "target": "pred",
      "value": {"node": "Call", "leaf": "write_code", "kwargs": {"…": "…"}}},
     {"node": "Try",
      "body": [
        {"node": "Assign", "target": "result",
         "value": {"node": "Call", "leaf": "repl",                 // interpreter leaf: ref into
                   "interpreter_ref": "main",                      //   the component-7 pool (D-029)
                   "kwargs": {
                     "code": {"node": "Attr", "value": {"node": "Var", "name": "pred"}, "attr": "code"},
                     "vars": {"node": "Dict",
                              "keys": [{"node": "Const", "value": "x"}, {"node": "Const", "value": "docs"}],
                              "values": [{"node": "Const", "value": 7},
                                         {"node": "Attr", "value": {"node": "Var", "name": "inputs"}, "attr": "docs"}]}}}}],
      "handlers": [{"node": "ExceptHandler", "type": "InterpreterError", "name": "e",
        "body": [
          {"node": "Assign", "target": "error",
           "value": {"node": "Attr", "value": {"node": "Var", "name": "e"}, "attr": "message"}},
          {"node": "Assign", "target": "result", "value": {"node": "Const", "value": null}}]}]}]},
  {"node": "Return", "value": {"node": "Call", "leaf": "finalize", "kwargs": {"…": "…"}}}
]}}
```

`InterpreterError` is exact-matchable (D-034 handler names);
`LoopCapError`/`MalformedNodeError` remain uncatchable — the cap fires
*outside* the program's own error flow.

## 4. Generated code is data — what that means as bytes

`pred.code` is an ordinary string value in the trace. Ex-15's runs are
the reference behavior: the 321M model produced `result = 17 * 23 + 5`
(executed in the leaf) and `result = France` (NameError → typed error →
the program's own except branch, twice, then loop exit). Nothing about
the generated string is whitelisted — restriction governs the TREE
(component 5); the leaf's insides are governed by the PROFILE
(namespace, no builtins, limits) and the ENVELOPE (isolation level).
Three fences, three owners.

## 5. The isolation story this exemplar fixes (D-042/D-043 as practice)

- **Floor bakes, envelope binds:** the entry demands `fork_ratchet`
  (generated code runs here — the D-040 trust profile would default
  optimizer-authored bodies at least this high); the receiver may run
  `sandbox` or `remote`, never `none` (R-26).
- **Invariance is checkable:** run the devset at `fork_ratchet` and at
  `sandbox`; identical results are REQUIRED (isolation-invariance law,
  D-042(5)) — a divergence indicts the leaf, and for optimizer-authored
  code it is a candidate-refusal class (D-043(5)).
- **Zero grants = kernel-backed claims:** with an empty grants row and
  an empty netns, "the generated code made no LM calls and leaked
  nothing" is a fact, not an honor system.
- **Session lifetime:** `kind: session` + ratchet ⇒ the REPL state
  lives exactly one leaf span; `save`/reuse across spans would be a
  grant, visible in the row.

## 6. The oracle (rung 0): bit-for-bit, network blocked

All entries at rung 0 (baked coder weights + in-process interpreter) ⇒
the roundtrip oracle is exact: per-call token ids + interpreter events
reproduce bit-for-bit with the network socket blocked (ex-15's proof,
now the conformance requirement for this fixture at grade 2). This
exemplar and E-06 together pin BOTH oracle regimes of §c.

## 7. Refusal hooks exercised (cross-ref E-11)

R-27 (profile unsatisfied — run against pyodide), R-26 (floor
under-run — envelope `none`), plus one new vector:

**R-34 · undeclared grant use** `[E-07; D-042/D-043]`
```jsonc
// entry grants: []  — but the envelope passes an open TCP fd anyway
```
> `materialize: envelope supplies grant 'fd:tcp:169.254.0.1:80' not present in leaf 'main' grants row — grants are declared in the artifact, never improvised by the envelope`
