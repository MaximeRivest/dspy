# Campaign Plan

The epic sequence from here to the engine. Each epic: doc-first (written by its implementing engineer, per `04-process.md`), stacked PRs, a named oracle. Dependencies are real — do not reorder without revisiting them.

## The destination

dspy becomes a **frontend and standard library**; a **program engine** (vLLM analogy: a runtime that schedules over program-as-data) executes ProgramIR. The IRs built so far — typed LM (lm15), adapter (plans/formats/strategies/ codecs), module AST, tools/interpreter — are the engine's instruction set.

## Sequence

### Epic D — adapter serializer (NEXT; scoped in `epic-D-adapter-serializer.md`)
Component-4 as data: `format_identity` + `literal_table` export in the fixed key vocabulary, strategies block + codec pool binding surfaces, resolved `adapter.config`, serialize → link → load with loud refusal (L5). **Oracle:** server examples 01–04's hand-authored manifests regenerate from the real exporter; `explain` renders them unchanged; roundtrip (same rendered prompts, same parses) machine-guaranteed. Also carries: double-key registry (roles become load-bearing for resolution), public codec/strategy registration, `@role` parser + `dspy.roles` export (cutover PR 1b from the epic-C doc). **Dependencies:** none — everything it needs shipped in A/B/C.

### Epic E — lm15 adoption (AFTER D; ratified 2026-08-05)
dspy's typed-lm contract becomes a veneer over (or direct adoption of) lm15's `Request`/`Response`/`StreamEvent`; role-typed streaming lands per the spec's streaming section (deltas route by role; undeclared reasoning → `_trajectory` live; buffered replay = the fold); litellm retirement path begins. Sequenced after D because D has no LM-layer dependency and the veneer deserves its own design pass; sequenced before AnthropicLM per the typed-LM family plan (streaming arrives solved). **Oracle:** lm15 conformance corpus (304 checks) + golden parity + the existing streaming tests.

### Epic F — the engine package (compile/link/execute)
Consolidate the server examples' three per-example compilers/interpreters (13/14/15 — 15's node set is complete) into one engine package operating on real `dspy.Module`s: `ast.parse` + whitelist compile, pool/binding link, closed-grammar interpret. Includes the lowering substrate — at which point TwoStep expands, `ParseContext.lm` dies, fallback/retry become error-policy lowerings (the kill list's blocked items unblock here). **Oracle:** trace equivalence vs native execution (the ex-13 method: run both, demand identical predict-call traces), both branches of every conditional, both refusal classes firing.

### Epic G — runtime services
Cross-call batching (evaluation/optimization loops present calls to a scheduler — ex-08 proved 5.8× hand-arranged; make it continuous), plan caching, prefix-cache alignment, View-2 overlay as the engine's execution log, budget admission control. **Oracle:** throughput gains + unchanged equivalence traces.

### Epic H — the middle-deletion
With the engine authoritative: retire ambient resolution, callback threading, per-module forward quirks, the legacy adapter bodies. Each deletion gated by the override/trace machinery that kept A–C safe. **Oracle:** public surface unchanged; corpus + trace equivalence throughout.

## Deliberately NOT being built (do not start these)

- **Optimizers over the new axes** (strategy/codec/structure search, seed regimes, blooming). The substrate makes them one-field mutations; the search itself is research and gates nothing. Substrate first.
- **Refine/BestOfN redo** (metric leaf + For/If loop). Blocked on Epic F's lowering substrate; doing it early recreates the misfiling disease.
- **Role vocabulary extensions** (refusal, media-out, video). Vocabulary is versioned governance, not a drive-by.
- **Sandboxing the in-process interpreter.** An optional outer layer later; not the default, not now.

## Standing risks

- **Upstream sync friction:** `tests/callback/test_callback.py` conflicts with #10119; run the five-point orthogonality check (see `05-decisions.md` D-014) on every synced PR touching adapters/predict.
- **The spec's "absent on this branch" section is stale** (written against qc-03; formats/builder/render/strategies exist now) — refresh when Epic D lands.
- **Matrix flake:** `test_dspy_configure_allowance_async` on py3.14 has flaked twice; if it fails alone, rerun before investigating.
