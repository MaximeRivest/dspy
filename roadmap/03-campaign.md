# Campaign Plan

The epic sequence from here to the engine. Each epic: doc-first, stacked commits, a named oracle (per `04-process.md`). Dependencies are real — do not reorder without revisiting them.

> **Sequencing rewrite (2026-08-10, Maxime's direction; ratification
> draft D-038):** D complete → **I (exporter) shipped locally** →
> priorities are now: (1) doc truth + ratifications, (2) exporter gaps
> + human-explorable tools (lint/diff/cost — for Maxime, no
> release), (3) **adapter-as-data design stage** (examples +
> handwritten Adapter IR + README, Maxime reviews), (4) **Epic F the
> engine** (started in parallel), then the dspy-wide refactor
> (orphans, H). **Epic E is SHELVED: the current LM stack is assumed
> as-is; lm15 was inspiration, not a dependency — the LM-IR contract
> can be refactored later without blocking anything.**

## The destination

dspy becomes a **frontend and standard library**; a **program engine** (vLLM analogy: a runtime that schedules over program-as-data) executes ProgramIR. The IRs built so far — typed LM (lm15), adapter (plans/formats/strategies/ codecs), module AST, tools/interpreter — are the engine's instruction set.

## Sequence

### Epic D — presets, templates, and the adapter as data (COMPLETE 2026-08-07; ratified D-031/D-032 incl. D-δ; as-built in `epic-D-adapter-serializer.md` v5)
An adapter becomes a **preset** `{template, parser, codec bindings, strategy bindings, config}`; the template (messages + slots + loop blocks + directives, upstreamed from `dspy-community-org/dspy_template_adapter`) is the literal table's full form. Class adapters = constructors over presets; BAML = a codec; the `format_*` method zoo deprecates; serialize → link → load with loud refusal (L5). **Oracle:** server examples 01–04's hand-authored manifests regenerate from the real exporter; `explain` renders them unchanged; roundtrip (same rendered prompts, same parses) machine-guaranteed. Also carries: double-key registry (roles become load-bearing for resolution), public codec/strategy registration, `@role` parser + `dspy.roles` export (cutover PR 1b from the epic-C doc). **Dependencies:** none — everything it needs shipped in A/B/C.

### Epic I — the exporter (SHIPPED LOCALLY v4, 2026-08-07; slot ratification pending — D-038 draft)
`dspy.programir`: canonical ProgramIR, compile/write/read/link, `dspy.export`, weight baking, env locking; dspy grade-1 conformance row green 35/35. Slots between D and E, amending D-016. Full charter: `epic-I-exporter.md`. Open gaps: forward compiler at node-set v0.1 (contract is v0.3); shipped module classes (ReAct/PoT/CodeAct) compile only as minis; shape-lowering for signature types (the two-layer rule).

### Epic E — lm15 adoption (SHELVED 2026-08-10, Maxime's direction)
The current LM stack (`dspy/clients/`, litellm, in-repo `LMRequest`/`LMResponse`) is assumed **as-is** for all downstream epics. lm15 served as design inspiration; adopting it is no longer on the critical path. The LM-IR contract seam remains refactorable later without unwinding other work. The old charter stays in `epic-E-lm15-adoption.md` for whenever it revives.

### Epic F — the engine package (compile/link/execute) — STARTED 2026-08-10 (parallel bootstrap)
The compile step is **seeded by Epic I** (`dspy/programir/compile.py` + `forward.py`) and the contract reference's `node_execute` — F adopts, never rebuilds (epic-I scope conflict #2). Consolidate into one engine package operating on real `dspy.Module`s: pool/binding link, closed-grammar interpret over the current LM stack (E shelved — no lm15 dependency). Includes the lowering substrate — at which point TwoStep expands, `ParseContext.lm` dies, fallback/retry become error-policy lowerings (the kill list's blocked items unblock here). **Oracle:** trace equivalence vs native execution (the ex-13 method: run both, demand identical predict-call traces), both branches of every conditional, both refusal classes firing.

### Epic G — runtime services
Cross-call batching (evaluation/optimization loops present calls to a scheduler — ex-08 proved 5.8× hand-arranged; make it continuous), plan caching, prefix-cache alignment, View-2 overlay as the engine's execution log, budget admission control. **Oracle:** throughput gains + unchanged equivalence traces.

### Epic H — the middle-deletion
With the engine authoritative: retire ambient resolution, callback threading, per-module forward quirks, the legacy adapter bodies. Each deletion gated by the override/trace machinery that kept A–C safe. **Oracle:** public surface unchanged; corpus + trace equivalence throughout.

## Deliberately NOT being built (do not start these)

> Annotation (2026-08-10): authored-component admission (custom
> parsers/strategies/codecs/LMs) is no longer a blanket no — it routes
> through the trust pairing rule + requirements gradient
> (`adapter-north-star.md`, contract `spec/trust.md`, D-026 amendment
> pending). The entries below still stand.

- **Optimizers over the new axes** (strategy/codec/structure search, seed regimes, blooming). The substrate makes them one-field mutations; the search itself is research and gates nothing. Substrate first.
- **Refine/BestOfN redo** (metric leaf + For/If loop). Blocked on Epic F's lowering substrate; doing it early recreates the misfiling disease.
- **Role vocabulary extensions** (refusal, media-out, video). Vocabulary is versioned governance, not a drive-by.
- **Sandboxing the in-process interpreter.** An optional outer layer later; not the default, not now.

## Standing risks

- **Upstream sync friction:** `tests/callback/test_callback.py` conflicts with #10119; run the five-point orthogonality check (see `05-decisions.md` D-014) on every synced PR touching adapters/predict.
- ~~Matrix flake `test_dspy_configure_allowance_async`~~ — **retired** (`d4aa6011e`: test bleed, conftest fix, 500/500 stress). Do not rerun-and-ignore; failures there are real now.
- We will need to redo flex so that it optimizes the program IR and the tools codes etc. its added string only new leaf is a unelegant hack.
