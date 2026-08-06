# Decision Log

Dated, one entry each, with the why. This is the don't-relitigate file: reopening one of these requires new evidence, not new taste. Append at the bottom; never rewrite history.

- **D-001 (2026-06)** — **The IR is the one-way door.** Representation becomes data inside core; every API tier above it is a two-way door. *Why:* presets/auto-adapter/explain become export decisions, not rewrites.

- **D-002 (2026-08-05)** — **TwoStep compiles away; never an adapter.** Expands to main Predict + frozen extraction Predict. *Why:* component 4 is pure by contract; an LM call in parse is a smuggled binding (invisible to placement census, credentials, cost attribution).

- **D-003 (2026-08-05)** — **Lowerings are a named category** (surface tree → core tree; four laws: interface-preserving, call-honest, provenance-total, composition-as-nesting). *Why:* the "type between modules and adapters" needed a home; CoT was already one.

- **D-004 (2026-08-05)** — **The single-shot law** (one adapter entry = one exchange). *Why:* without it, scores depend on calls the manifest never states.

- **D-005 (2026-08-05)** — **The signature is sacred; exhaust → `_trajectory`.** Prediction fields = declared outputs exactly. *Why:* depending on `result.reasoning` couples programs to the mechanism that answered. Litmus: want it? declare it.

- **D-006 (2026-08-05)** — **Bracket access does not shim exhaust; dot access does (with DeprecationWarning).** *Why:* one strict contract view must exist; the friendly view warns.

- **D-007 (2026-08-05)** — **ReActV2 included in the exhaust migration** (`history`, `termination_reason`). *Why:* same exhaust class; excluding it enshrines v1/v2 inconsistency.

- **D-008 (2026-08-05)** — **MultiChainComparison is an aggregator, not a lowering.** *Why:* it adds an external input (`completions`), which no lowering may do; it honestly declares its contract.

- **D-009 (2026-08-05)** — **Roles/strategies/codecs: adapter types split into shape + semantic role; strategies per role in component 4; codecs per shape.** *Why:* the types conflated data with inference semantics — the reinvents-typing and can't-drop-the-types complaints resolved independently.

- **D-010 (2026-08-05)** — **Role vocabulary grounded in the typed LM contract.** A role exists iff ≥2 materially different strategies, one escaping the token stream; textual strategies are polyfills; the vocabulary version-tracks the contract, never a provider.

- **D-011 (2026-08-05)** — **Role syntax: `Annotated` markers are the primitive; four spellings, one registry object** (`Annotated[str, X]` canonical; `X[str]` sugar; `@X` string shorthand; `role=` kwarg). *Why:* role must vary independently of shape; `Annotated` is type-checker-transparent and works in every surface incl. FunctAI. Conflicting spellings refuse loudly.

- **D-012 (2026-08-05)** — **Unions with mixed role markers derive `plain`** (explicit `role=` overrides; upgrade silence to an info log later). *Why:* either-of-two-semantic-types as data is legal; guessing is worse.

- **D-013 (2026-08-05)** — **Seeds: per-leaf objective declarations** (frozen / semantics-fixed / metric-driven), user-picked at tag time. Semantics-fixed is code→code only; blooming requires the View-2 cost regularizer default-on; optimizer-authored code carries `authored_by`.

- **D-014 (2026-08-05)** — **Upstream #10119 (interpreter callbacks) safe to accept; five-point orthogonality check adopted** for upstream syncs. Callback plumbing is View-2's polyfill: kill-by-absorption, not conflict.

- **D-015 (2026-08-05)** — **lm15 adopted as the engine's LM substrate.** Component 8a's contract becomes a veneer over lm15 `Request`/`Response`/`StreamEvent`; in-repo parallel types and litellm retire over the arc. *Why:* part vocabulary independently confirms the role projection; contract-governed, zero-dep, cross-language, streaming solved (role-typed deltas). Sequenced after Epic D.

- **D-016 (2026-08-05)** — **Campaign order: D (serializer) → E (lm15) → F (engine) → G (runtime) → H (deletion).** *Why:* D has zero LM-layer dependency; the veneer deserves its own design pass; the engine needs D's serialization and E's contract.

- **D-017 (2026-08-05)** — **Strategy registry and codec seam stay engine-private until Epic D.** *Why:* public shapes get designed deliberately with the serializer, not leaked incrementally.

- **D-018 (2026-08-06)** — **The literal table's full form is a template; an adapter is a preset** `{template, parser, codec bindings, strategy bindings, config}`. Proof: `dspy-template-adapter` reproduces exact ChatAdapter message parity declaratively; its slots map 1:1 onto the IR (`{instruction}`=3a, `{demos()}`=3b, directives=plan slots, `{inputs(style)}`=codecs). Constrained template language, not general Jinja. The 7-key vocabulary survives as a derived summary view. Class adapters = thin constructors over presets; the `format_*` method zoo = legacy override surface, deprecate in D, delete in H. Templates must be strategy-aware (natively-served roles render no block).

- **D-019 (2026-08-06)** — **BAML is a codec, not an adapter.** Preset `json` + {indented-pydantic input, schema-prose schema} codec bindings ≡ today's BAMLAdapter bytes; the class becomes a compat shim. *Why:* Epic B already proved BAMLFormat = JSONFormat + input-codec; this completes the reclassification user-visibly.

- **D-020 (2026-08-06)** — **The adapter layer gets its own contract spec** (`roadmap/adapter-ir-spec.md`): signature core, template language, presets, vocabularies, ADP-001..011 invariants, corpus-based conformance — lm15-structured, provisional until Epic D proves it, then graduates to its own contract repo. *Why:* a frozen conformance surface and a moving design doc cannot share a document; the ProgramIR consumes it by reference exactly as it consumes lm15.

- **D-021 (2026-08-06)** — **Standalone library: yes, later; import boundary: now.** The render/parse layer extracts to its own library (any frontend: dspy/FunctAI/dicts; any backend: lm15 reference, litellm/SDK alternates) AFTER D/E stabilize — the Instructor/BAML market proves the demand, FunctAI-without-dspy is the utility test. Until then, Epic D builds inside `_engine/` under a mechanically-enforced import boundary (engine imports only signature core + types + lm15; never settings/modules/clients) so extraction is a move, not a surgery.
