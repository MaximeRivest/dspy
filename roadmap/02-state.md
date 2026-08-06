# State Map

What exists right now and its load-bearing status. **Regenerate this page at the end of every epic.** Last updated: 2026-08-06, mid-Epic-D (after fork D-α; adversarial review of the template engine in flight).

## Shipped on `programir-main`

**Pre-existing (the Quiet Compiler epic, 12 PRs):** the adapter engine — `AdapterPlan`/`RenderField` IR, field transforms, parser hooks, `AdapterPatch`, formats layer (chat/json/xml/baml/twostep), built-in strategies (reasoning/citations/tools), override-gated migration, golden corpus (byte-parity gate). All five core adapters engine-backed.

**The `_trajectory` migration** (`9998a00f1`, `1d43627db`, `1809f8037`): `Prediction._trajectory` channel; ReAct/CodeAct/RLM/CoT/ReActV2 exhaust routed; dot-access deprecation shim; declared-`reasoning` stays contractual; flex bridge merges channel for generated code.

**Epic A — engine closed** (`14cfc3bda`, `1c6b1622e`): field-strategy registry (`field_strategy_for()` single resolution path), `LegacyTypeHookStrategy` auto-wrap, `AdapterPatch.replace_render_signature` compat channel. Zero `TODO(adapters-plan)` markers remain.

**Epic B — shapes + codecs** (`4f99265ff`, `478fc0db5`, `eb17d119b`, `2d8fa218e`, `805cf0902`): `_engine/codecs.py` (`ValueCodec`, `TextPythonishCodec`, `PYDANTIC_JSON`); directional `input_codec`/ `output_codec` on `Format`; BAMLFormat = JSONFormat + input-codec binding; `shapes--` corpus family (25 fixtures + parse-coercion); typed `UnserializableTypeError` for Callable-in-signature.

**Epic C — semantic roles** (`0240cc738`, `8cc3d37e6`, `c0daccd53`, `9fb957375`, `106a9ad25`): `SEMANTIC_ROLES` vocabulary; validated `role=` kwarg; `dspy/signatures/roles.py` marker objects with `citations[str]` sugar; derivation table (legacy types + `Annotated` unwrapping both nesting orders); roles recorded onto `RenderField.metadata`; design doc `epic-C-semantic-roles.md`.

**Epic D fork D-α — template engine + presets** (`77116604c`, `ef9bf14ac`, `2d1b0837f`, `567ed8a4c`, `19ab69e02`): mechanical engine import-boundary test with pinned back-edges (shrinking allowlist in `tests/adapters/engine/test_import_boundary.py`); the constrained template language in `_engine/template/` (vocabulary-as-data + `describe_template_language()`, eager teaching-error parser, pure renderer, `declared_capacity()`, `preview()`); presets `chat`/`json`/`xml` defined as templates in `_engine/presets.py` with formats rendering every content string through them (`render_template_messages` walker parity-tested against forced-legacy `format()`); codec registry (`CODECS`/`resolve_codec`) in `codecs.py`. Corpus zero-drift; full matrix green; zero public-surface change. Spec §3 grammar refined as-proven (bare `strip` flag, `{instruction}` styles, fragment line-swallow, direction-aware value presence, per-aggregate style vocabulary); epic doc at v3 with as-built + D-β handoff. E2E xfail #4 revised (literal_table = derived summary view, targets D-5).

**Cross-language doctrine** (`d97d3d2ea`, `e5fb3f071`, `d5f0208bf`, Maxime): D-022..D-028 ratified; `roadmap/cross-language.md` question bank; §e0-lang in the program spec. Binding on Epic D: D-024 (versions block in artifact + preset serde) and D-025 (`language` on origin-tagged entries) are byte-shape constraints that MUST precede the exporter — D-5/D-γ serde carries `adapter_ir_version` + vocabulary versions.

**Docs:** `IR-program-spec.md` snapshot (source of truth: docmaker), epic docs A/B/C, this doc set.

## Recorded but not load-bearing (deliberate)

- **Semantic roles** — on every plan; consulted by nothing. Strategies still resolve by annotation type.
- **Codecs** — named registry exists (`CODECS`/`resolve_codec`); codec *authority* in rendering is still the Format object, preset bindings test-asserted equal (authority flips in D-3/D-5). No per-field overrides, no public registration.
- **Strategy registry** — engine-private; no public exposure, no deprecation signaling on the legacy hook.
- **Preset `strategies` bindings** — recorded on every preset; consumed by nothing (D-4).
- **Template capacity + fragments** — `declared_capacity()` computed, `RenderContext.fragments` plumbed to every fragment slot; both unconsumed until D-4's bake-time triple check.
- **Message-sequence walker** — `render_template_messages` exists and is parity-tested; the engine path still walks `render.py`'s skeleton (content-first delegation). Cutover to the pure walker lands with D-4's fragments.

## Spec'd but not built

- `@role` string-signature shorthand (parser hazards documented in epic-C doc §2a; assigned to D-6 with public `dspy.roles` export).
- Per-role `strategies={...}` adapter binding surface; double-key (role-then-annotation) registry resolution; strategy awareness in templates (D-4).
- BAML-as-codec + compat shim (D-3; BAML already inherits the json preset's assistant delegation — only its system section and input codec remain class-owned).
- Preset dump/load + derived 7-key summary view + loud-refusal loader (D-5; serde must carry `adapter_ir_version` + vocabulary versions per D-024, `language` per D-025).
- Role-based CoT declared-reasoning check (name-based today; aliasing hole documented).
- Validity enforcement for role/direction/multiplicity/shape rules.

## Kill list (retire deliberately, each its own decision)

- `ParseContext.lm` — exists only for TwoStep's in-adapter extraction call; dies when TwoStep is expanded as a lowering (needs the lowering substrate).
- Chat→JSON fallback + structured-output retry inside `base.Adapter.__call__` — become error-policy lowerings (same dependency).
- `adapt_to_native_lm_feature` legacy hook — wrapped (Epic A); deprecation after public strategy seam ships.
- Legacy adapter method bodies retained for subclass-override compat — the removal epic's list.
- litellm dependency — retired by lm15 (~/Projects/lm15-dev/lm15-python) adoption (ratified; sequenced after Epic D).
- In-repo `LMRequest`/`LMResponse` parallel contract — same arc.
- Callback plumbing — absorbed by the engine's run overlay, eventually (upstream #10119/#10120 are the polyfill of that overlay; accept freely).

## Known deferred items (consolidated from epic reports)

- Silent-degrade shape fixes (`serialize_for_json`'s `str()` fallback) — alters bytes; needs a deliberate byte-changing commit.
- Schema-prose as a first-class codec (BAML's structure section).
- `direction` rename (`role` key on RenderField means input/output — naming collision with semantic roles, resolved in spec, not in code).
- Media-as-output roles; `Video` shape; `refusal` role (anticipated, vocabulary-ready).
- ReAct-family `_trajectory` key unification (v1 `trajectory` vs v2 `history`/`termination_reason`).
- Guillemet list-in-str-field quirk (pinned as documented behavior).
- Refine/BestOfN redo (metric leaf + retry loop) — blocked on lowering substrate; **do not start**.
- Optimizers over the new axes (strategy/codec/structure search, seed regimes) — substrate first; **do not start**.
