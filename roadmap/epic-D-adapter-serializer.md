# Epic D — presets, templates, and the adapter as data

**Status:** v3 (2026-08-06) — D-α (PRs D-1/D-2) SHIPPED; this doc now
records what was actually built plus the D-β/D-γ handoff. Rescoped v2 after
the template ratification (D-018/D-019 in `05-decisions.md`).

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
- D-3: BAML-as-codec + compat shim.
- D-4: strategy awareness + `strategies={}` + double-key registry.
- D-5: serialization dump/load + derived summary view + loud-refusal loader.
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

**Handoff to D-β/D-γ:**
- the strategy seam is live but empty: `RenderContext.fragments` reaches
  every fragment slot; `declared_capacity()` is ready for the bake triple
  check; preset `strategies` bindings are recorded data consumed by
  nothing;
- codec authority in D-2 is still the Format object (`fmt.input_codec`);
  the preset bindings are validated-equal by test — D-3/D-5 flip authority
  to the preset when BAML becomes `preset json + baml codec bindings`
  (note: BAML already inherits the json preset's assistant delegation;
  only its system section and input codec remain class-owned);
- **preset serde (D-5) must carry `adapter_ir_version` + the
  closed-vocabulary versions block from its FIRST dump/load shape (D-024),
  and origin-tagged code entries carry `language` (D-025)** — no
  versionless shape may ever exist (also noted on the `Preset` docstring);
- the derived 7-key summary view (`literal_table()`) derives from the
  parsed template — `ParsedTemplate` keeps both `raw` and nodes, so
  derivation needs no re-parse; xfail #4 already pins
  `completed_marker == "[[ ## completed ## ]]"`;
- `render.py`'s skeleton vs the walker: when D-4 makes strategies
  contribute fragments, route the engine path through
  `render_template_messages` (or thread `fragments` into the delegators) —
  the walker parity tests are the safety net for that cutover.

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
