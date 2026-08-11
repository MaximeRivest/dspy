# Staging lessons — literature flags and headroom ranking (2026-08-10)

Two research passes (a literature survey of staged embedded DSLs and a
design pass on what SOTA staging unlocks) produced conclusions that
were driving decisions in conversation but were recorded nowhere. This
doc is that record. Nothing here is ratified; items marked **owner**
name the epic/doc that should absorb them.

## A. The five literature flags (known failure modes of our plan)

1. **Two-executor drift — TorchScript's killer.** We are structurally
   TorchScript-shaped (AST subset + separate interpreter). Three things
   protect us: tiny census-chosen subset, per-node pinned semantics,
   heavy computation in leaves. The mitigation the literature demands:
   **differential testing host-Python vs engine on shared fixtures,
   with a designated oracle** — extend the conformance corpus to
   host-vs-engine, not only runtime-vs-runtime. **Owner: Epic F's
   oracle section** (trace equivalence is already its method; this
   flag adds "make it a standing gate, not a one-time proof").
   Contrast case: Dynamo survived via guards + fallback; fallback
   contradicts portability, so differential testing is our only tool.

2. **The designed-divergence warning.** IR semantics deliberately
   differ from live Python (no NaN/Inf, exact int64 with typed
   overflow errors). Passing export does NOT mean identical behavior:
   live Python produces `inf` where the IR raises. **Decide which
   semantics a reported score belongs to, and say it as loudly as we
   refuse constructs.** Concrete remedy, the strongest single idea
   from the survey: an **IR-strict execution mode in Python before the
   engine exists** — authors run under IR semantics from day one; it
   later doubles as the differential-testing oracle for flag 1.
   **Owner: its own decision (pre-F), cheap; propose alongside the
   exporter epic.**

3. **Branch-merge semantics need explicit pinning.** What an
   `if`/`for` branch may assign and what merges afterward was
   AutoGraph's hardest static-analysis area. Our flow walker already
   embodies answers (guard-tainted assignment, fixpoint loops); the
   node-set spec should state the merge rules normatively, not leave
   them implicit in reference behavior. **Owner:
   programir-contract/spec/node-set.md, next spec round.**

4. **Subset ossification — the LINQ lesson.** C# expression trees
   froze at C# 3 forever; growing a serialized subset is politically
   and semantically hard. Our versioned evidence-gated growth process
   is the right defense, but expect freeze anyway; the pressure valve
   is **leaf ergonomics** — making a tool with a typed contract as
   pleasant as inline code. **Owner: standing design bias; revisit in
   every "should the node set grow" debate — refusal + good leaf UX
   beats admission.**

5. **Leaf bodies are the unchecked purity hole** (JAX's undetectable
   impurity, relocated). Now partially addressed: spec/trust.md's
   pairing rule (trust deficit paid with isolation) and
   declare-and-probe effects. The residue: document the leaf contract
   as loudly as JAX documents purity; runtime detection
   (MetaOCaml-style dynamic checks) is a someday option. **Owner:
   spec/trust.md carries the rule; dspy-side authoring docs owe the
   loud version when tools get their doc page.**

Validations worth keeping (no action, ammunition for RFC/debates):
leaf rule = the literature's answer to cross-stage persistence
(MetaOCaml's 20-year blocker, LINQ's parameter lifting); AST-not-trace
validated by JAX's workaround surface; loud refusal has a decade of
empirical support (Numba's silent object-mode deprecation, EF Core 3
removing silent client evaluation, AutoGraph's transparency failure);
Triton = our shape succeeding (tiny subset + escape hatch); ambient
refusal is stronger than Dynamo guards; SGLang proves structure-as-data
buys systems-level wins; LangGraph is the cautionary middle (portable
topology, opaque nodes).

## B. The headroom ranking (value ÷ additional machinery)

Full analysis lived in conversation; the ranking and the three
cross-cutting findings are the load-bearing part.

**Tier 1 — no engine needed, ship-now adoption bait:**
1. static lint over the manifest (dead branches, unused fields,
   unreachable predictors) — days of work, universal payoff;
2. program diff as the optimizer/review surface (grade-1 `diff` op
   exists; needs text rendering);
3. static cost estimator (bounded loops + closed call graph = cost is
   decidable pre-run; unique to us).

**Tier 2 — the engine's dividends (F/G):** auto-batching of
independent leaves (ex-08: 5.8× hand-arranged), prefix-cache-aware
scheduling (prompt bytes knowable pre-dispatch via pure `preview()`),
EXPLAIN ANALYZE (View 2 annotating View 1), exact replay (LM = the
only nondeterminism, interceptable at one seam → deterministic CI),
budget enforcement, provenance, verifiable score claims, streaming as
an engine property.

**Tier 3 — optimization over the IR:** whole-graph instruction/demo
search (every candidate a shippable artifact), Refine/BestOfN as IR
macros + speculative N-way, structure search behind score gates,
partial evaluation/prompt specialization, subgraph distillation
(signatures = typed cut points). Security as a fourth axis now specced
(spec/trust.md "security as an axis"; IR-program-spec §e3).

**Tier 4 — distribution:** polyglot serving (asterisk: Python tools
need a sidecar), content-addressed registry with verifiable score
gates.

**Three cross-cutting findings:**
- **One shared missing layer**: a typed dataflow/dependence pass over
  the node JSON (def-use chains, loop-carried variables, purity table)
  — five features consume it (auto-parallel, cost, DCE,
  specialization, flow/trust checks already partially embody it).
  Build once, contract-side. Cheapest wide-fan-out investment after
  the engine.
- **Determinism is the quiet enabler** — replay, content-addressing,
  claim verification all rest on it. **Guard it as a hard invariant in
  Epic F: pinned join order in any future fan-out semantics**, decided
  at design time, or half of Tier 2–4 silently dies.
- Tier 1 ships before the engine and makes the export restriction pay
  immediately.

## C. Where the rest already landed

The Odersky/capability thread is fully specced:
`programir-contract/spec/flow.md` (secrecy, TACIT prior art),
`spec/trust.md` (trust record, pairing rule, postures incl.
`workbench`, integrity dual, sandboxed-interpreter dividend, security
axis), `roadmap/flow-capabilities.md` (dspy side),
`IR-program-spec.md` §e3. Adapter-as-data verification (authored-codec
`exec` at load; sha256 = integrity not authorship): recorded in the
pairing rule's link-time consequence and campaign memory.
