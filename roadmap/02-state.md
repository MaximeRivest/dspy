# State Map

What exists right now and its load-bearing status. **Regenerate this page at the end of every epic.** Last updated: 2026-08-07, after **Epic D complete** (checkpoint C1 closed — D-031; D-δ fix wave in flight, then Epic E).

## Shipped on `programir-main`

**Pre-existing (the Quiet Compiler epic, 12 PRs):** the adapter engine — `AdapterPlan`/`RenderField` IR, field transforms, parser hooks, `AdapterPatch`, formats layer (chat/json/xml/baml/twostep), built-in strategies (reasoning/citations/tools), override-gated migration, golden corpus (byte-parity gate). All five core adapters engine-backed.

**The `_trajectory` migration** (`9998a00f1`, `1d43627db`, `1809f8037`): `Prediction._trajectory` channel; ReAct/CodeAct/RLM/CoT/ReActV2 exhaust routed; dot-access deprecation shim; declared-`reasoning` stays contractual; flex bridge merges channel for generated code.

**Epic A — engine closed** (`14cfc3bda`, `1c6b1622e`): field-strategy registry (`field_strategy_for()` single resolution path), `LegacyTypeHookStrategy` auto-wrap, `AdapterPatch.replace_render_signature` compat channel.

**Epic B — shapes + codecs** (`4f99265ff`..`805cf0902`): `_engine/codecs.py` (`ValueCodec`, `TextPythonishCodec`, `PYDANTIC_JSON`); directional codec bindings on `Format`; `shapes--` corpus family; typed `UnserializableTypeError`.

**Epic C — semantic roles** (`0240cc738`..`106a9ad25`): `SEMANTIC_ROLES` vocabulary; validated `role=` kwarg; marker objects with `citations[str]` sugar; derivation table; roles recorded onto `RenderField.metadata`.

**Epic D — presets, templates, the adapter as data** (four waves: D-α `77116604c`..`19ab69e02`; adversarial review + fixes `57ffdf5ad`..`282e5d777`; D-β `c0e9675e6`..`55b9f666d`; D-γ `19f974a1f`..`b13ce00cd`):

- **Template language** in `_engine/template/`: vocabulary-as-data + `describe_template_language()`, eager teaching-error parser, pure renderer, `{% section strip %}` blocks, `{field('name')}` escape spelling, reserved-name collision refusal, capacity in live/example lanes (per-field media/tools), `preview()` (format()-identical bytes, no LM call).
- **Presets** `chat`/`json`/`xml` defined AS templates; formats render through them; **codec authority lives in preset binding data**; a codec is a render/parse/**schema** triple; BAML = the `baml` codec + a pairing declaration (arrangement as template data), `BAMLFormat` render bodies gone.
- **Strategies**: double-key registry (`strategy_for(role, annotation)`) — roles load-bearing, builtin role entries containment-bounded so admitted subclass hooks are never shadowed; `strategies={}` per-role constructor binding with `"auto"` resolved + recorded at bake; strategy fragments flow to template fragment slots; ADP-006/ADP-007 bake checks on every engine plan.
- **Serde** (D-024/D-025 from first shape): canonical component-4 entry `{name, adapter_ir_version, versions, template, parser, codecs, strategies, config}`; exact (unknown keys refuse, absent ≠ null); dangling refs = link errors naming the ref; `dump_entry`/`load_entry`/`literal_table` (derived 7-key view); **loaded bindings govern** (byte-parity source-vs-loaded); version constants single-sourced, aggregated in `_engine/versions.py`.
- **Public surface (C1 ratification pending)**: `dspy.roles`; the `@role` string shorthand; `dspy.TemplateAdapter(messages, parse_mode=...)`; `PresetAdapter.preview()`; `register_codec/strategy/preset` + unregister trio with §9 admission gates (codec round-trip probe battery, strategy capability declarations, eager preset validation; three-origin loading for codecs); `describe_template_language()`; `load_entry` listed in `__all__`; `strategies=` kwarg on core adapters.
- **Proof**: acceptance suite fully green, zero xfails (`tests/adapters/test_end_to_end.py`); corpus zero-drift through all four waves; mechanical import boundary with shrinking pinned allowlist; server examples 01–03 manifests regenerated from the real exporter (`explain` renders unchanged, `load.py` roundtrips bit-for-bit; 04 refused by design — authored legacy-override adapter); DD adapters page: recipes 13–19 are Today.

**D-δ fix wave** (`7b626388b`..`b3e369a0d`, 6 commits, from the ten-persona evaluation): roles + strategies load-bearing end-to-end (role-keyed admission in the builder — split spellings engage strategies; registered strategies bindable by name and serialized; explicit `native` never hijacked; `capability_requirements` consulted at plan time); role conflicts raise at signature declaration naming the field (resolution moved to `dspy/signatures/roles.py`, boundary pin died); bake reachable from the pure surface (`preview(lm=)` = live-call bytes incl. bake refusals; `explain_plan()` read surface; `full_text` arity + unhonorable bindings refuse pre-LM-call); authoring honesty (schema-position bare slots refuse; demos-directive `preamble=` is data — no injected prose in authored patterns; history no longer inherits demo patterns; second `History` field refuses; template-lane parse errors self-identify; new-surface errors de-jargoned); entry `config` carries the behavior flags; **`ADAPTER_IR_VERSION` 0.1.0 → 0.2.0** (breaking: inheritance retirement; old entries refuse naming both versions). Corpus zero-drift; matrix green ×3. **Ratified (D-032):** `Adapter.preview(..., lm=)`, `Adapter.explain_plan()`, `dspy.adapters.PresetAdapter` export, `PresetAdapter.parse_mode`, demos `preamble=` key, `{f.role}`, registered names as `strategies=` values. Server example manifests 01–03 regenerated at 0.2.0 (verified: explain + bit-for-bit load roundtrips; 04 byte-unchanged, exporter refusal on record; note — the examples' load.py uses a pre-engine dspy, so the 0.2.0 entry loader is proven by exporter+explain, not by load). **Campaign position:** Epic E is ON HOLD by Maxime's direction — he is driving the exporter (epic-I draft) himself in a separate session; the coordinator holds until he reports back.

**Flake retired** (`d4aa6011e`): `test_dspy_configure_allowance_async`'s two-day roaming failure was test bleed, not a product race — an async test added in the `_trajectory` migration latched `config_owner_async_task`, and the conftest reset never un-latched ownership; under xdist same-worker scheduling the target then refused configure. Fix: conftest teardown restores settings to import-time state under the lock (product untouched). Deterministic repro of the CI signature; 500/500 stress; three consecutive full matrices green. Product follow-up noted, deferred: ownership latching to dead tasks blocks sequential `asyncio.run` entry points — a deliberate design pass someday.

**Cross-language doctrine** (Maxime, in parallel): D-022..D-029 ratified; `roadmap/cross-language.md` question bank; exemplar programs (`0e9cfb479`) + dialect exemplars and `frontend-contract.md` draft; §e0-lang. D-024/D-025 are implemented in the preset serde; D-029's manifest rulings await the exporter (programir-contract). Untracked draft: `roadmap/epic-I-exporter.md` (proposes the exporter epic between D and E, amending D-016 — awaiting Maxime).

**Docs:** `IR-program-spec.md` (reuse inventory refreshed post-D; "absent" list now a "built" list), epic docs A–D (D at v5 with per-wave as-built sections), this doc set.

## Recorded but not load-bearing (deliberate)

- **`"auto"` strategy resolutions** are recorded per-call (`plan.metadata`) but not into entry `config`; entry config also lacks `use_native_function_calling` — both need the LM-bound export step (Epic F linker territory).
- **`per_field` codec overrides** — absent from the entry shape; exact serde refuses unknown keys, so it can only grow deliberately.
- **Authored parsers** (spec §4 `AuthoredParser`) — callables refuse with a teaching error; the three-origin loader shipped for codecs only.

## Spec'd but not built

- Role-based CoT declared-reasoning check (name-based today; epic-C §9 PR 3).
- Validity enforcement for role/direction/multiplicity/shape (epic-C §3 table; cutover epic).
- Annotation-keyed strategy lookup demotion (warn, then remove — epic-C §6 stage 3; the public seam now exists, so the deprecation arc is unblocked, its own decision).
- Authored parsers; `per_field` codec bindings.

## Kill list (retire deliberately, each its own decision)

- `ParseContext.lm` — dies when TwoStep expands as a lowering (Epic F).
- Chat→JSON fallback + structured-output retry inside `base.Adapter.__call__` — become error-policy lowerings (Epic F).
- `adapt_to_native_lm_feature` legacy hook — the public strategy seam shipped (D-γ), so deprecation signaling is now unblocked; its own decision.
- Legacy adapter method bodies + the `format_*` zoo — docs now point at templates; deletion is Epic H.
- litellm dependency; in-repo `LMRequest`/`LMResponse` parallel contract — the lm15 arc (Epic E).
- Boundary-test pinned back-edges: `parser_hook.py → clients.openai_format` dies with E; `formats/twostep.py → chat_adapter` dies with F; adapter-class back-edges die with H.
- Callback plumbing — absorbed by the engine's run overlay eventually (upstream #10119/#10120 are the polyfill).

## Known deferred items

- Silent-degrade shape fixes (`serialize_for_json`'s `str()` fallback) — byte-changing; dedicated corpus-gated commit.
- `direction` rename (`role` key on RenderField) — resolved in spec, not in code.
- Media-as-output roles; `Video` shape; `refusal` role (vocabulary-ready).
- ReAct-family `_trajectory` key unification (v1 `trajectory` vs v2 `history`/`termination_reason`).
- Guillemet list-in-str quirk (pinned); the codec admission battery excludes bare `(Optional[T], None)` probes for the same pinned quirk.
- Refine/BestOfN redo; optimizers over the new axes — blocked on Epic F substrate; **do not start**.
- Flex redo on the IR (F-γ; scope negotiated with Maxime).

## Checkpoint C1 (closed — D-031)

The Epic D public names are **ratified as shipped** against `roadmap/public-surface-epic-D.md` (D-031; the ten-persona evaluation requested no renames). Upstream sync stays declined until after Epic E (D-030). Pushes are current: `programir-main` is on the fork at the reviewed state. Open work: the **D-δ fix wave** (in flight — the twelve high-impact defects), then **Epic E** (doc-first).
