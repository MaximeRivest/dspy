# Exemplar coverage matrix — what is proven where, and what is missing

**Status:** audit (greenfield-only). Inventory = union of (a) the spec's
component/feature surface, (b) the 15 server examples' proof points
(`examples-build/README.md` index, cited in IR-program-spec), (c) drafts
D-045..D-050, (d) the adapter-rfc faces. Compared against:
`exemplar-program-everything.py` (EX-P) and
`exemplar-artifact-everything.md` (EX-A).

**Policy this audit sets:** EX-P/EX-A remain the *integration* pair —
breadth, every axis touched once. Depth belongs to a FAMILY of targeted
exemplars (E-01…), mirroring how server examples 01–15 worked. A feature
is covered when it has (1) a frontend appearance, (2) an artifact byte
shape, (3) — at ratification — a fixture family. This file is the index
the 15-example README was.

## 1. Covered by the everything pair (breadth confirmed)

module tree + bindings + delta · roles/shapes + deduction · adapter
reference AND full entry forms (lens, rules, turns, codecs) · forward
node-set JSON (If/Assign/BinOp/Call) · python tool w/ floor · model pool
faces (chat baked/declared, embed, segment, authored-R) · sgd-1/fit-1 +
training statements · cross-language R (env block, sidecar rung) ·
datasets pool + ETL + pushdown + result_hash · metric + judge + devset ·
credentials/endpoints as names · deviations/rebind/rescore/place ·
minimal-syntax deduction · no-training author · idiomatic formats ·
provider market · versions block · semantic ground audit.

## 2. THE GAPS — present in the spec/15-examples/drafts, absent from both EX-P and EX-A

| # | missing feature | proven before by | belongs in |
|---|---|---|---|
| G1 | **interpreter leaf in use** — generated code as data, While/retry loop, iteration cap, ToolError catch | ex-15 (RLM, rung-0 bit-for-bit) | new E-07; EX-A §7 currently `{}` with a comment — insufficient |
| G2 | **model-dispatched (dynamic) tools** — `tools[pred.name](**pred.args)`, unknown-name ToolError; Try/ExceptHandler/Raise/Break in forward | ex-14 (ReAct) | new E-06 |
| G3 | **lowerings as bytes** — `lowered_from` provenance, parse-fallback expansion (Try + predict′ under a different adapter binding), CoT as expansion | spec §d-lowering; ex-13 era compilers emitted core trees | new E-05 + EX-A addendum (one lowered node) |
| G4 | **authored adapter entry** (authored-source custom adapter, admission-gated) + authored strategy/codec origins | ex-01–04 (custom adapter) | new E-03 |
| G5 | **authored LM class** (`origin: authored` engine source, `constructor: {ref,args}`, D-036 structural admission) | ex-09/15 (authored `_OpenAICompatLM`, inproc LM) | new E-04 |
| G6 | **weight regimes beyond one LoRA** — joint fine-tune (N bindings→1 entry), fork-free-at-birth, merge-delta-into-base as recorded moves | ex-12 (all three regimes as hash facts) | new E-08 |
| G7 | **content-addressed store form** — objects/ + manifest-of-hashes, checkpoint series, materialize-on-demand | spec §c1; ex-05/07 checkpoints | new E-09; EX-A shows ONLY the self-contained form |
| G8 | **View-3 trajectory as bytes** — (snapshot, labeled diff, metric Δ) triples, objective-boundary node (D-048), regime-transition diffs | ex-05/07; D-048 | new E-09 (same exemplar as G7 — they are one story) |
| G9 | **history role + multi-turn** — `history` semantic role, ContinuationState carry-through, `turns` face exercised end-to-end over two calls | spec §Adapter-semantics; adapter-rfc | new E-06 (ReAct exercises turns naturally) |
| G10 | **streaming as data** — role-typed deltas routed (declared reasoning vs exhaust), stream grammar per face (STUDY F2) | spec streaming diagram; lm15 grammar | new E-10 |
| G11 | **refusal exemplars** — the loud-refusal families AS EXEMPLARS: un-whitelisted node, undeclared call, dangling binding, unknown version major, non-invertible template slot, face/config mismatch | ex-13/14 fired both §d classes; D-024; adapter-rfc | E-11 (DONE: exemplar-E11-refusals.md, R-01..R-32) |
| G12 | **MCP-rung tool** — outputSchema-verified identity, sampling-as-declared-binding rule | D-027 | new E-12 (may defer with E/F epics — but the BYTES should exist) |
| G13 | **isolation gradient exercised** — levels beyond a floor field: fork_ratchet session leaf, grants row, broker egress (D-042/D-043 ratified!) | upstream D-042/D-043 | EX-A currently shows floors only; E-07 should carry a real grants block |
| G14 | **flow/trust labels** (§e3) — secret/internal/public field labels, posture, flow_check as scored axis | contract spec/flow.md (PROPOSED) | defer until flow.md ratifies; note the slot |
| G15 | **inputs-bag + record splat** (D-041 upstream) — `predict[x](**inputs, k=v)` byte form | D-041 | EX-A §5 uses Attr-access only; add one splat call |
| G16 | **aforward twin** (D-037: identical graph) | D-037 | E-06 note: one aforward variant, asserting same §5 bytes |
| G17 | **base LM / instruct=False capability** + capability-driven strategy flip shown end-to-end | STUDY Q1; adapter-rfc rules | E-03 |
| G18 | **GPU device slot + serve-engine switch exercised** (device.json, vllm-offline system_deps) | ex-06/08 | EX-A has device.json + engine field; system_deps for vllm missing — one-line fix |
| G19 | **deferred-but-named byte stubs** — online-1/rl-1 grains, refusal-not-silence for them | STUDY F6/F7 | E-11 refusals file |

## 3. The proposed exemplar family (the greenfield 01–15)

| id | proves | source material |
|---|---|---|
| E-01 | the no-training author (EX-P §12 extracted, runnable) | done in EX-P |
| E-02 | minimal-syntax deduction + provenance (§9) | done in EX-P |
| E-03 | adapters deep: authored entry, strategy flip on capabilities, base-model polyfill | G4, G17 |
| E-04 | authored LM class + structural admission | G5 |
| E-05 | lowerings: CoT + parse-fallback, sugar view vs core view | G3 |
| E-06 | ReAct: dynamic tools, Try/Raise/Break, turns, history role, aforward twin | G2, G9, G16 |
| E-07 | RLM: interpreter leaf, While, generated code, grants + isolation levels | G1, G13 |
| E-08 | weight regimes: joint/fork/shared-base+delta as hash facts | G6 |
| E-09 | the store: content-addressed form + View-3 trajectory + objective boundary | G7, G8 |
| E-10 | streaming: role-typed deltas, per-face grammars | G10 |
| E-11 | REFUSALS: every named refusal, one snippet + expected error each | G11, G19 |
| E-12 | MCP tool rung (bytes now, execution with E/F) | G12 |
| EX-P/EX-A | the integration pair — every axis once | stays as is |

## 4. Method note

The 15 server examples were built against running hardware; E-01…E-12
are built against the CONTRACT (bytes + expected behavior), which is
what the greenfield stage needs — they become the seed of the fixture
corpus (FIXTURE-PLAN's golden + refusal vectors are E-XX sliced). Rule
carried over from the server README: each exemplar states *what it
proves* in its header, or it does not merge. Completion definition for
this matrix: every G-row either lands in an E-file + EX-A byte section,
or carries an explicit defer marker naming what it waits on (G12→E/F,
G14→flow.md).
