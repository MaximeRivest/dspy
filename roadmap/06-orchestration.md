# Orchestration Plan — the implementation campaign

How the campaign (`03-campaign.md`) actually gets executed: which agents,
in what order, with what gates, and where Maxime decides. The coordinator
(main session) launches forks, reviews reports, maintains `02-state.md` and
`05-decisions.md`, and never lets two code agents touch the branch at once.
This document is the coordinator's persistent memory — any future session
resumes the campaign from here plus the state map.

## Standing rules (every agent, every wave)

- One code agent on `programir-main` at a time. Parallel work only when
  file-disjoint (docs vs code, server-side vs repo) and explicitly noted.
- Every agent: reads `00`–`05` + its epic doc first; owns and revises its
  epic doc; spec-first on conflicts (fix the contract, then the code —
  ratified-decision changes escalate to Maxime).
- Gates per agent run: golden corpus byte-identical (L8), full `dspy-ci`
  matrix, stacked commits, no push, no PRs. Coordinator verifies the
  report against the gates before the next launch.
- After each agent report: coordinator updates `02-state.md`, appends any
  ratified decisions, relays a compressed summary to Maxime.
- **Maxime's checkpoints** (nothing proceeds past one without his word):
  every push; every public-surface change (flagged per-report); every
  spec change that touches a D-numbered decision; upstream syncs.

## Wave 0 — launch (now)

Push the pre-implementation stack (contract, examples, cookbook, E2E
xfails, Epic D v2). Then D-α launches.

## Epic D — three forks, serial (epic doc: `epic-D-adapter-serializer.md`)

**D-α — template engine + preset parity (PRs D-1, D-2).** The riskiest
stack: build the constrained template language (vocabulary-as-data, four
consumers, eager validation, preview) inside `_engine/` under the import
boundary (mechanical test lands first); define presets `chat`/`json`/`xml`
as templates; class adapters delegate. Oracle: corpus byte-identical at
every commit; parity tests; xfail #4 revision. *After D-α: a dedicated
adversarial review fork* (fresh eyes, not the implementer) walks the
template engine for injection/escaping/capacity-derivation holes before
anything builds on it.

**D-β — codecs, strategies, serialization (PRs D-3, D-4, D-5).**
BAML-as-codec + compat shim; strategy awareness (visible-set iteration,
fragment slots, bake-time triple check, `strategies={}`, double-key
registry — roles become load-bearing); preset dump/load with loud refusal
+ the derived 7-key summary view. Oracle: corpus + xfails #3/#5 flip green.

**D-γ — public surface + proof (PRs D-6, D-7).** `@role` parser,
`dspy.roles` export, the template authoring surface (name/API ratified
with Maxime before merge); registration APIs public with admission gates
(spec §9); docs agent moves cookbook recipes 13–19 from Arriving to Today;
server examples 01–04 regenerate from the real exporter (SSH
192.168.2.24, may run parallel to docs — disjoint); ProgramIR spec
"absent on this branch" refresh. Oracle: xfails #1/#2 flip; reference-repo
README examples run; `explain` renders regenerated manifests unchanged.

**Checkpoint C1 (Maxime):** ratify the public API names, push, decide
whether an upstream sync happens before E.

## Epic E — lm15 adoption, two forks, serial (engineer writes epic doc first)

**E-α — the veneer.** Epic doc first (per process); then: dspy's typed-lm
contract becomes a veneer over lm15 `Request`/`Response` behind the
engine's import boundary; lm15 conformance corpus wired into dspy-ci;
capability declarations mapped. Oracle: 304-check conformance + golden
parity + matrix.

**E-β — role-typed streaming.** Deltas route by role (declared → field
stream; undeclared reasoning → `_trajectory` live); StreamListener
reframed as the textual polyfill; buffered replay = the fold; litellm
retirement path opens (router memory: dspy.LM → pure router). Oracle:
existing streaming tests + new role-routing tests + matrix.

**Checkpoint C2 (Maxime):** push; AnthropicLM sequencing decision (the
typed-LM family plan says it comes after streaming — it can now interleave
with F if wanted).

## Epic F — the engine, three forks, serial (epic doc first)

**F-α — engine package.** Consolidate the server examples' compilers/
interpreters (13/14/15 — 15's node set is complete) into one package
operating on real `dspy.Module`s: compile (ast+whitelist), link
(pools/bindings), interpret. Trace-equivalence harness (the ex-13 method)
becomes a CI gate alongside the corpus.

**F-β — the lowering substrate.** Lowerings as compile passes; TwoStep
expands (ParseContext.lm dies — the kill list's blocked items unblock);
parse-fallback + structured-output retry become error-policy lowerings;
CoT recognized as a signature-rewrite lowering. Oracle: trace equivalence
+ corpus (bytes unchanged — lowerings reproduce today's behavior).

**F-γ — Flex on the IR.** Maxime's campaign note: Flex's string-only leaf
is a pre-unified-leaf hack — redo it to emit/optimize core-tree structure
and tool bodies. Epic doc first; scope negotiated with Maxime (this is
research-adjacent).

**Checkpoint C3 (Maxime):** push; decide Refine/BestOfN redo timing (now
unblocked) and whether G starts or AnthropicLM/provider work interleaves.

## Epic G — runtime services, two forks, serial

**G-α — scheduler:** cross-call batching for evaluate/optimize loops
(ex-08's 5.8× made continuous), plan caching, prefix-cache alignment.
Oracle: throughput measured + trace equivalence unchanged.
**G-β — observability:** View-2 overlay as the engine's execution log
(absorbs callback plumbing — the #10119 kill-by-absorption), budget
admission. Oracle: overlay completeness on the example suite.

## Epic H — the middle-deletion, one fork per kill-list cluster

Ambient resolution, callback threading, legacy adapter bodies + `format_*`
zoo, litellm. Each deletion its own stacked PR gated by the
override/trace machinery; public surface unchanged throughout. Docs agent
does the final truth pass (nothing "Arriving" remains).

## Cross-cutting agents

- **Docs agent** (persistent): end of each epic, moves arriving→today,
  re-verifies recipes, sweeps stale claims.
- **Adversarial reviewer** (fresh fork, per invocation): after D-α
  (template engine) and F-α (interpreter) — the two components where a
  subtle hole is expensive.
- **State keeper**: the coordinator itself — `02-state.md` regenerated at
  each epic end; decision log appended as ratifications happen.

## Failure & escalation protocol

- Agent dies mid-run (auth/API): resume via message with "re-check git
  status, continue" — work on disk survives (proven).
- Agent blocked on a design hole: it reports the hole + a proposal;
  coordinator either resolves within ratified decisions or escalates to
  Maxime; spec updated before the agent resumes.
- Matrix flake: rerun before investigating (known flakes listed in
  `03-campaign.md`).
- Corpus regeneration "needed" by a feature agent: automatic stop — the
  change altered behavior; rethink (L8).

## Budget expectations

Recent agent runs: 240k–440k tokens, 6–16 min each. The campaign ≈ 13
fork runs + 2 review forks + docs invocations → plan for ~5M agent tokens
end to end. Coordinator context is preserved by keeping reports
compressed and this file + state map authoritative (compaction-safe).
