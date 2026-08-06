# Epic D — presets, templates, and the adapter as data

**Status:** v4 (2026-08-06) — D-α (PRs D-1/D-2) and D-β (PRs D-3/D-4/D-5)
SHIPPED; this doc records what was actually built plus the D-γ handoff.
Rescoped v2 after the template ratification (D-018/D-019 in
`05-decisions.md`).

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

**PR stack:**
- D-1: template engine, pure and unused (corpus untouched) + xfail #4
  revision — **SHIPPED** (see "D-α as built").
- D-2: presets defined as templates; class adapters delegate (corpus gate at
  zero diff); parity tests — **SHIPPED** (see "D-α as built").
- D-3: BAML-as-codec + compat shim — **SHIPPED** (see "D-β as built").
- D-4: strategy awareness + `strategies={}` + double-key registry —
  **SHIPPED** (see "D-β as built").
- D-5: serialization dump/load + derived summary view + loud-refusal
  loader — **SHIPPED** (see "D-β as built").
- D-6: `@role` parser + `dspy.roles` export (public-surface PR).
- D-7: template authoring surface upstream + docs pass + server examples
  regeneration (docs/corpus-style commits).

## D-α as built (D-1/D-2, 2026-08-06)

**Module layout.** The template engine is a package,
`dspy/adapters/_engine/template/`: `vocabulary.py` (the ONE vocabulary data
structure + `describe_template_language()`), `parser.py` (eager parser, AST
nodes, `TemplateError` teaching errors that enumerate the valid set from
the vocabulary), `renderer.py` (pure `render_nodes` over a `RenderContext`;
also the single-source demo-classification helper and the historical demo
prefix/missing-message strings, referenced by the Format base),
`capacity.py` (`declared_capacity` → `TemplateCapacity`, the D-4 bake-check
substrate), `preview.py` (the pure template walker +
`preview()`). Presets live in `dspy/adapters/_engine/presets.py`; the codec
name registry (`CODECS`/`resolve_codec`) in `codecs.py`.

**Language deltas the implementation forced (spec section 3 updated
first):**
- loop blocks take an optional bare `strip` flag — the historical
  join-then-`strip()` section shape is unreachable without it (field
  descriptions, structure sections, and assistant field blocks strip;
  user-side input loops must NOT, or interior trailing whitespace diverges);
- `{instruction}` takes `style=` (`raw` | `indented`); `indented` is the
  dedent-then-8-space objective block, placed inline after
  `objective is: ` so the historical trailing space falls out of the
  template text;
- empty `{fragments(...)}` slots swallow their whole line (slot alone on a
  line + empty render → the line and its newline vanish) — the zero-byte
  guarantee made precise;
- value-presence semantics are direction-aware: `inputs` loops iterate
  values-present fields in valued positions; `outputs` loops iterate all
  visible fields (assistant position → missing-field message through the
  codec; user position → schema-side, which is what makes the
  output-requirements enumeration render);
- aggregate `style` names an entry in a per-aggregate closed style
  vocabulary (not directly a codec); `outputs` gained `json_object`
  (typed placeholders in schema position, call values in assistant
  position — exactly `_render_json_object`'s two uses).

**Delegation depth chosen for D-2.** Content-first: ChatFormat gained
`preset_name` + three delegators (`render_system`, `render_user_content`,
`render_assistant_content`) that execute the named preset's template;
JSONFormat/XMLFormat override only `preset_name` (their shadowing legacy
bodies were deleted). Every content string on the engine path now renders
from template data. The message SEQUENCE still walks through
`render.py`'s structural skeleton (demo classification is single-sourced
with the walker via `classify_demos`); the pure walker
(`render_template_messages`) exists, and parity tests prove
walker == forced-legacy `adapter.format()` for chat/json/xml across the
golden payload shapes — full walker-driven assembly on the engine path is
deferred to D-4, where strategy fragments force it anyway. The granular
literal methods (`render_field_description`, `render_field_structure`,
`render_task_description`, `output_requirements`,
`render_fields_with_values`) are KEPT: BAML still composes from them
(`BAMLFormat.render_system` now pins the composed path explicitly), and
`tests/adapters/engine/test_presets.py` diffs them against the template
renders so the two sources cannot drift. They go with the `format_*` zoo in
Epic H.

**Import boundary (D-021).**
`tests/adapters/engine/test_import_boundary.py` AST-scans every engine
module (function-local imports included). Allowed prefixes:
`dspy.adapters._engine`, `dspy.adapters.types`, `dspy.adapters.utils`,
`dspy.signatures`, `dspy.core`, `dspy.utils.exceptions`. Pre-existing
back-edges are pinned as a SHRINKING allowlist (new violation fails; stale
pin fails): `builder/render/postprocess → dspy.adapters.base`,
`migrated.py → the five adapter-class modules`, `formats/baml.py →
baml_adapter`, `formats/twostep.py → chat_adapter` (dies with TwoStep's
lowering, Epic F), `parser_hook.py → dspy.clients.openai_format` (dies with
the lm15 veneer, Epic E), and three `dspy.experimental` re-export imports
of Citations (trivially fixable to `dspy.adapters.types.citation`; left
pinned to keep D-α behavior-neutral).

**Deliberately deferred within D-α (documented limits):** the walker's
history directive expands plain turns only (native tool-call replay stays
with the pipeline machinery); `preview()` leaves custom-type markers
unexpanded (part splitting is the frontend's job); a `SignatureCore`
neutral datatype is NOT introduced — the engine renders from dspy
signatures, and the neutral input type arrives with extraction (spec
section 8 phase 2).

**Handoff to D-γ (post-D-β):**
- version constants: import from `_engine/versions.py`
  (`ADAPTER_IR_VERSION`, `versions_block()`, `check_version_compatible`) —
  the exporter's manifest `versions` block (D-024) mirrors these; never
  restate a number;
- the entry shape is `serde.ENTRY_KEYS` in canonical order; the exporter
  wraps `Adapter.dump_entry()` per bound adapter and `explain` can render
  any entry through `load_entry(...).format(...)` / `literal_table()`;
- registration APIs (spec §9) plug into the refusal sites already in
  `_check_codec_ref` (origin-tagged entries) and the engine-private
  `register_codec`-shaped seams (`CODECS`, `register_field_strategy`/
  `register_role_strategy`, `PRESETS`) — admission gates (round-trip probe
  battery) are D-γ's to build;
- the template authoring surface should construct entries through
  `_make_preset` (eager parse + eager capacity) and serialize through
  `serde.build_entry` — nothing else mints entry shapes;
- `@role` parser (epic-C §2a hazards) and `dspy.roles` export are
  untouched; xfails #1/#2 are the acceptance tests;
- pending Maxime ratifications from D-β: the D-019 letter refinement (BAML
  system arrangement as template data), the `strategies=` kwarg, the
  `{field('name')}` spelling, and the `load_entry`/`dump_entry`/
  `literal_table` surface.

## Review fixes (D-α adversarial review, 2026-08-06)

Nine defects confirmed by the adversarial review, fixed spec-first (the
section 3 edits landed before the code). What changed and why:

1. **Reserved-name collisions.** Bare `{instruction}` silently rendered
   the docstring for signatures declaring a field named `instruction`.
   The bare form now refuses at render naming the collision; the call
   form (`{instruction(style=…)}`) stays unambiguous — the presets spell
   the call form, so legacy signatures with reserved field names render
   byte-identically. Bare aggregate/fragments parse errors name the
   collision; `RESERVED_SLOT_NAMES` is one vocabulary constant.
2. **Schema positions render without call values everywhere.** The
   `{f.value}` refusal keys on `mode == "schema"`, and the walker builds
   its system context with `values=None` — preview bytes are engine
   bytes, pinned by preview-vs-`adapter.format()` parity tests.
3. **`{% section strip %}` + xml zero-outputs parity.** Legacy
   join-then-strip collapses trailing empty structure sections with their
   separators; no per-loop arrangement expresses that, so the language
   gained the section block and the xml system template wraps its
   structure region in one. Zero-output / zero-field / native-FC
   ToolCalls-only renders byte-match forced-legacy.
4. **Bare directives render.** `{"role": "demos"}` / orphan
   `{"role": "history"}` resolve patterns via `directive_pair` (own pair →
   demos pair → language default marker patterns); zero demos/turns
   no-op. Eager validation admits exactly the renderable set.
5. **Import boundary sees relative imports** — resolved per-file against
   the module's package, then through the same pin/ban logic.
6. **Option lexing** is quote/escape-aware (documented `\'`/`\"` escapes
   and `%` now lexable); arity errors state the actual problem.
7. **Capacity API shape change** (unconsumed until D-4):
   `TemplateCapacity` gains `directive_iterates_inputs/outputs` and
   `directive_field_slots` (directive patterns analyzed via the same
   `directive_pair` resolution rendering uses); live-lane bits stay
   content-message-only; **`hosts_role_textually(role, field=…)` — the
   field name is required for `media`/`tools`** (per-field hosting:
   `iterates_inputs or field in field_slots`), refusing the coarse
   per-role question that over-claimed. D-4's bake triple check must ask
   per-field.
8. **Vocabulary is the single source:** loop-option acceptance reads the
   data (`loop_options` entries now carry `takes_value` + description);
   unknown aggregate styles refuse at render naming the valid set, with a
   conformance test walking the vocabulary against renderer coverage.
9. **User-turn assembly declared:** join-then-strip + empty-user-turn
   omission are spec section 3 semantics now, docstringed and pinned.
   `render_user_content` no longer joins an empty body between prefix and
   suffix (`'P\n\nS'`, the legacy chat/json shape); xml's accidental
   `'P\n\n\n\nS'` corner (pipeline-unreachable) unifies on the chat
   shape, pinned by test.

Corpus: zero fixture changes; every parity fix proven against
forced-legacy subclasses, never regenerated fixtures.

## D-β as built (D-3/D-4/D-5, 2026-08-06)

**D-3 — BAML as codec; codec authority flips.** The codec contract grew
its third surface: `render_typed_placeholder(name, field_info)` — the
SCHEMA spelling — with the shared text codec rendering the historical
placeholder-plus-type-note and `{f.typed_placeholder}` routing through the
direction's bound codec. The schema-prose machinery moved from
`baml_adapter.py` into `_engine/codecs.py` (`render_schema_prose`; the
adapter module re-imports under the historical names, retiring the
`formats/baml.py` import-boundary pin), and `BAMLCodec` = indented-pydantic
values + schema-prose placeholders, registered as `baml`. Codec AUTHORITY
flipped **generally**: `Format.input_codec/output_codec` resolve NAMES —
preset bindings, layered class/instance `codec_binding_overrides`, registry
resolution — and the corpus stayed at zero drift. `BAMLFormat` is now the
pairing declaration itself: `preset_name="json"` +
`codec_binding_overrides={input: baml, output: baml}` +
`system_template_message` (the schema-prose arrangement as template data in
`presets.BAML_SYSTEM`, whose self-bracketing empty-separator loops
reproduce the legacy `"\n".join(sections)` collapse byte-for-byte, pinned
by empty-field-set parity tests). Its `render_system`/`render_user_content`
method bodies are GONE; the composed legacy structure body remains only as
the frozen parity reference. **D-019 letter refinement (needs Maxime's
ratification):** "preset json + baml codec bindings ≡ BAMLAdapter bytes" is
implementable only with the system ARRANGEMENT as template data — the
sentences/markers/completed-marker placement is D-018 literal-table
territory and cannot hide inside a shape-generic codec. Spec §4 now states
the pairing as codec spelling + template arrangement; no `baml` preset name
exists, and the pairing serializes as an ordinary entry named `baml`.

**D-4 — roles load-bearing; strategies bound; fragments live; bake
checks.** The double-key registry (`strategy_for(role, annotation)`,
epic-C §6 stage 2): resolution order registered-role → registered-annotation
→ builtin-role → builtin-annotation → legacy hook, with built-ins the SAME
instances under both keys (byte-identical by construction) and
`StrategyTrace.resolved_by` recording which key answered. Admission to the
strategy loop stays annotation-gated (`native_response_types`) — widening
admission by role is a later, deliberate act. The `strategies={role: name}`
constructor kwarg (base + ChatAdapter + JSONAdapter; XML/BAML inherit)
validates eagerly against `strategies/vocabulary.py` — ONE data structure:
names, descriptions, implemented subset, native/textual classification,
`STRATEGIES_VERSION`. Bake semantics: textual binding stands the native
strategy down (field stays visible, traced `bound:`); explicit native
binding the LM cannot serve refuses loudly naming role+field, strategy, LM;
`auto` = today's bytes with per-field resolution recorded in
`plan.metadata["strategy_resolution"]`; a tools binding must agree with
`use_native_function_calling` (disagreement refuses at construction) until
the kwarg deprecates. Fragments: `AdapterPatch.fragments` →
`AdapterPlan.fragments` → a contextvar `plan_scope` entered by
`__call__`/`acall` around the frozen `format()` surface → the preset
delegation's `RenderContext.fragments`. **Walker-cutover depth (recorded
decision):** fragments thread through the DELEGATORS; `render.py`'s
structural skeleton stays (native tool-call history replay lives there and
the walker deliberately does plain turns only); the pure walker remains the
parity/preview surface AND became the loaded-entry render path (D-5), which
is the cutover that mattered. The bake triple check runs on every
engine-rendered plan via eager `Preset.capacity` (+
`effective_capacity` for the BAML pairing): ADP-006 refuses a
textually-served role-bearing field with no textual lane (per-field for
media/tools) naming field+role, LM, and template; ADP-007 refuses explicit
slots on natively-hidden fields in live AND example lanes. Builtin presets
host every role and carry no explicit field slots — existing programs never
trip either check. The `{field('name')}` escape spelling landed
(`field` reserved; every reserved-collision refusal teaches it). Xfail #3
flipped.

**D-5 — serde, rulings, summary view.** Both coordinator rulings, spec §3
first: duplicate loop options REFUSE (call-kwargs parity; last-wins is
gone), and bare-directive defaults key on the preset's parser binding
(json → marker user + `json_object` assistant, xml → tag pairs, full_text
→ bare-value assistant; chat markers stay the no-preset fallback) threaded
through walker/preview/capacity. Versions: per-vocabulary constants beside
their data (`SEMANTIC_ROLES_VERSION` in `dspy/signatures/field.py`,
`STRATEGIES_VERSION`, `CODECS_VERSION`, `TEMPLATE_LANGUAGE_VERSION`)
aggregated in **`_engine/versions.py`** with `ADAPTER_IR_VERSION = 0.1.0`
(semver-0: minors breaking) — D-γ's exporter builds on this module.
`_engine/serde.py` owns the entry (`ENTRY_KEYS` canonical order: name,
adapter_ir_version, versions, template, parser, codecs, strategies,
config): exact serde (unknown keys refuse, absent ≠ null, canonical JSON,
ordering preserved), D-024 version refusals naming both sides, ADP-005
link errors naming refs, D-025 language validation on origin-tagged codec
entries (then refusal until the §9 registration API ships). Zero
`dspy.settings` reads, mechanically tested. `Adapter.dump_entry()` dumps
the effective preset (BAML dumps under entry name `baml`; declared
`strategies` bindings layer over the preset's); `dspy.adapters.load_entry`
links back into `PresetAdapter` (in `dspy/adapters/serde.py`, outside the
engine boundary) rendering through the pure walker + parsing through the
parser binding bound to the entry's codecs (full_text implemented:
exactly-one-output-field, refused otherwise). `literal_table()` derives
the 7-key view from the template (symbolic per-field patterns, structure
classification markers/tags/json_object/values, marker + requirement
literals, parser regex) — never authored. Xfails #4 and #5 flipped; #1/#2
remain for D-6. Round trips proven: identical entries, byte-identical
messages across the golden payloads, identical parses, identical derived
tables.

**Public-surface changes in D-β (each needs flagging at review):** the
`strategies=` constructor kwarg (the sanctioned addition);
`dspy.adapters.load_entry` + `Adapter.dump_entry()`/`literal_table()`
(demanded by acceptance test #5 — the pinned-surface test records them);
the `{field('name')}` template spelling (language addition, spec'd).

**Deliberately deferred within D-β:** per-field codec overrides
(`per_field` in the preset codec shape) — the serde refuses non-{input,
output} keys, so the shape can only grow deliberately; strategy-binding
resolution recorded into preset `config` at export (needs an LM-bound
export step — D-γ); `PresetAdapter` runs the legacy postprocess path (its
format/parse overrides are its contract; engine postprocess needs
registration, an Epic-H-adjacent decision).

**Non-goals:** lowering substrate (Epic F); lm15 types (Epic E); TwoStep
expansion; template *optimization* (substrate here, optimizer later);
deleting `format_*` outright (Epic H); `dspy-session` projections from the
reference README (`outer_history`, `node_memory`, …) — session territory,
not adapter territory.

**Design contract:** `roadmap/adapter-ir-spec.md` is normative for D — the
signature core, pipeline (template renders the PLAN, declared capacity,
bake-time triple check), template grammar, preset shape, vocabularies, and
ADP invariants. Where implementation disproves the spec, fix the spec first,
then the code.

**Import boundary (D-021, enforced from D-1):** `_engine/` imports only the
signature core, the types layer, and (post-E) lm15 — never settings,
modules, clients, or teleprompt. Add a mechanical boundary test in D-1;
extraction to the standalone library later must be a move, not a surgery.

**Gates:** L8 corpus discipline throughout; full `dspy-ci` matrix (stage new
files first); stacked commits; no push/PR without Maxime's word.
