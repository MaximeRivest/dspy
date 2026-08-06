# Epic D — presets, templates, and the adapter as data

**Status:** DRAFT v2 (2026-08-06) — rescoped after the template ratification
(D-018/D-019 in `05-decisions.md`); the implementing engineer owns this doc
and revises before starting (per `04-process.md`).

**Ratified design.** An adapter is a **preset**: a named data entry
`{template, parser binding, codec bindings, strategy bindings, config}`. The
template — a message list with interpolation slots, constrained loop blocks,
and directive roles — is the literal table's full form; the old 7-key
vocabulary survives only as a derived summary view. The class adapters become
thin constructors over presets `chat`/`json`/`xml`; **BAML becomes a real
codec** (indented-pydantic input + schema-prose schema) bindable to any
preset, with `BAMLAdapter` as a compat shim; the `format_*` method zoo
(`format_system_message`, `format_field_description`,
`format_field_structure`, …) is a legacy override surface headed for the
kill list (deprecate here — there is finally a replacement to point at;
delete in Epic H).

**Reference implementation to upstream:**
`/home/maxime/Projects/dspy-community-org/dspy_template_adapter/` (~1.6k
lines: slot/loop/directive template rendering, parse modes, **exact
ChatAdapter message parity proven declaratively** (its README carries the
parity template), image content-block splitting, finetune export, optimizer
compat). Upstream its semantics, not necessarily its code verbatim — it
predates strategies/roles and must become strategy-aware (gap #4 below).

**Definition of done (mechanical).**
1. **Template engine in `_engine/`**: slots (`{instruction}`, `{field}`,
   `{inputs(style=…)}`, `{outputs(style=…)}`, `{demos(style=…)}`,
   `{history(style=…)}`), loop blocks (`{% for f in inputs/outputs %}` with
   the `f.*` vocabulary), directive roles (`demos`, `history`), brace
   escaping. Constrained language — no general Jinja; every construct
   analyzable, diffable, serializable.
2. **Presets**: `chat`, `json`, `xml` defined AS templates + parser + codec
   bindings, reproducing today's adapters **byte-identically** (golden
   corpus is the gate). Class adapters become constructors resolving to
   presets; `format_*` methods delegate to the template path and are marked
   legacy.
3. **BAML-as-codec**: schema-prose + indented-pydantic becomes a named codec
   entry; preset `json` + that binding == today's BAMLAdapter bytes;
   `BAMLAdapter` class = compat shim.
4. **Strategy awareness** (what the reference lacks): `{outputs()}` and loop
   blocks render only textually-served fields; natively-served roles
   contribute no block. Per-role `strategies={...}` binding surface +
   double-key registry (role first, annotation fallback, trace records
   which key resolved) land here — roles become load-bearing,
   byte-identical by construction.
5. **Serialization**: a preset dumps to a component-4 JSON entry (template
   as data + bindings + resolved config) and loads back with zero
   `dspy.settings` reads; dangling strategy/codec/preset references are
   link errors refused loudly naming the reference (L5). The 7-key summary
   view derives from the template for explain/cross-language readers.
6. **`@role` + public surface**: the `@role` string shorthand
   (pre-tokenization hazards documented in `epic-C-semantic-roles.md` §2a)
   + public `dspy.roles` export + the upstreamed template authoring surface
   (`TemplateAdapter(messages=[...], parse_mode=...)`-style; exact name/API
   for the engineer to ratify). These are the epic's sanctioned
   public-surface changes; flag each in every report.
7. **Oracle**: golden corpus byte-identical for presets replacing the class
   adapters; the reference repo's README examples run against the
   upstreamed surface; server examples 01–04 manifests
   (`maxime@192.168.2.24:~/docmaker/examples-build/`) regenerate from the
   preset exporter with `explain` rendering unchanged; the E2E xfails flip
   green. Update the spec's stale "absent on this branch" section when done.

**Pre-written acceptance tests.** `tests/adapters/test_end_to_end.py`
carries five strict-xfail tests (`test_epic_d_*`); they flip to hard
failures the moment the surface exists. Post-rescope status:
1. `dspy.roles` public import — unchanged.
2. `dspy.Signature("q -> a: str @citations")` — unchanged.
3. `ChatAdapter(strategies={"reasoning": "textual_field"})` — unchanged.
4. `adapter.literal_table()` — **REVISE in D-1**: preset export exposes the
   template; `literal_table()` becomes the derived summary view (keys ⊆ the
   fixed vocabulary still holds).
5. `dump_entry()`/`load_entry()` roundtrip rendering identical messages —
   unchanged in spirit; the entry now carries the template.
Add in D-2/D-3: parity tests — preset `chat` renders byte-identical messages
to `ChatAdapter` for a representative signature+demos; `BAMLAdapter` ≡
preset `json` + baml codec.

**Suggested PR stack** (revise freely):
- D-1: template engine, pure and unused (corpus untouched) + xfail #4 revision.
- D-2: presets defined as templates; class adapters delegate (corpus gate at
  zero diff); parity tests.
- D-3: BAML-as-codec + compat shim.
- D-4: strategy awareness + `strategies={}` + double-key registry.
- D-5: serialization dump/load + derived summary view + loud-refusal loader.
- D-6: `@role` parser + `dspy.roles` export (public-surface PR).
- D-7: template authoring surface upstream + docs pass + server examples
  regeneration (docs/corpus-style commits).

**Non-goals:** lowering substrate (Epic F); lm15 types (Epic E); TwoStep
expansion; template *optimization* (substrate here, optimizer later);
deleting `format_*` outright (Epic H); `dspy-session` projections from the
reference README (`outer_history`, `node_memory`, …) — session territory,
not adapter territory.

**Gates:** L8 corpus discipline throughout; full `dspy-ci` matrix (stage new
files first); stacked commits; no push/PR without Maxime's word.
