# Epic D — the adapter serializer (component 4 as data)

**Status:** DRAFT — scoped by the coordinator; the implementing engineer owns this doc and should revise it before starting (per `04-process.md`).

**Goal.** The engine's in-RAM half is complete (plans, formats, strategies, codecs, roles); the as-data half does not exist — zero serialization surface in `_engine/` (no `literal_table`, no `format_identity`, no dump/load). Server examples 01–04 hand-authored their component-4 manifests, which is where the literal-table key drift came from. Epic D makes the adapter fully self-describing as data, exported by dspy itself.

**Definition of done (mechanical).**
1. Every format exports `format_identity` + a `literal_table` in the fixed key vocabulary (`IR-program-spec.md` §Adapter-notes: `input_field_render, output_field_render, field_separator, output_structure, completed_marker, output_requirement, parse_pattern`; absent key = no such construct, never a synonym). The vocabulary becomes an enforced contract (a test walks all formats and rejects unknown keys) — key drift dies permanently.
2. Adapter entries serialize to a component-4 JSON block: plan operator refs (transforms, parser identities), format identity + literal table, `strategies` per-role block, codec bindings, resolved `config` (capability-checked decisions, e.g. `response_format_routing`).
3. A loader reconstructs a working adapter from that block alone — zero `dspy.settings` reads; dangling strategy/codec/format references are link errors refused loudly naming the reference (L5).
4. **Binding surfaces land here** (they're the serializer's shapes anyway): per-role `strategies={...}` on adapters; named codec pool with registration + per-field overrides; double-key registry resolution (role first, annotation fallback, `strategy_trace` records which key resolved — roles become load-bearing, byte-identical by construction).
5. Cutover PR 1b from `epic-C-semantic-roles.md`: the `@role` string shorthand (pre-tokenization hazards documented there §2a) + public `dspy.roles` export. This is the epic's one sanctioned public-surface change; flag it in every report.
6. **Oracle:** regenerate server examples 01–04's manifests (`maxime@192.168.2.24:~/docmaker/examples-build/`) from the real exporter; `explain` renders them unchanged; the roundtrip check (same rendered prompts, same parses, reconstructed from the manifest alone) passes. Update the spec's stale "absent on this branch" section when done.

**Suggested PR stack** (revise freely):
- D-1: literal-table/format-identity export + vocabulary-enforcement test (corpus untouched — export only reads).
- D-2: component-4 entry serializer (plan refs + config) + loader with loud refusal; roundtrip tests.
- D-3: double-key registry + per-role `strategies` binding surface (byte-identical: role key resolves to the same strategy the annotation key did; corpus is the gate).
- D-4: codec pool + registration + per-field bindings.
- D-5: `@role` parser + `dspy.roles` export (public-surface PR, eager validation, zero change to signatures not using it).
- D-6: examples 01–04 regeneration on the server + spec refresh (docs/ corpus-style commit).

**Non-goals:** lowering substrate (Epic F); lm15 types (Epic E); TwoStep expansion; deprecation signaling on the legacy hook (needs the public seam to point at — decide in D-3 whether it's ready); any optimizer.

**Gates:** L8 corpus discipline throughout (D-1..D-4 byte-identical; D-5 additive); full `dspy-ci` matrix; stacked commits; no push/PR without Maxime's word.
