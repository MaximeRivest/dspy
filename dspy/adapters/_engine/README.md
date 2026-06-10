# The adapter engine (private)

**Nothing in this package is public API.** Names, modules, and semantics may
change in any release without deprecation; it is never re-exported from
`dspy` or `dspy.adapters` (CI-tested), and external code must not import it.

## What this is

The engine reifies adapter behavior as data — a small compiler:

```
SignatureCall ──builder.py──▶ AdapterPlan ──render.py + formats/──▶ messages ──▶ LMRequest
                                  │ (collapsed slots, RenderFields,
                                  │  transforms, parsers, trace)
LMResponse ──ResponseView──▶ postprocess.py (plan parsers) ──▶ parsed values
```

- **`ir.py`** — `AdapterPlan` (collapsed slot schema; the rich dotted slot
  taxonomy from the design docs lives in `DebugLink` vocabulary, not fields)
  and `RenderField` (rendered name ≠ semantic `original_name`).
- **`patch.py`** — `AdapterPatch` composes the public `LMRequestPatch` and
  adds response channels (parsers, transforms, warnings, debug links,
  strategy trace). Deterministic, associative merge.
- **`builder.py`** — planning. Native FC runs as a `PlanStep`; built-in
  native types (Reasoning, Citations) as `TypeStrategy` objects whose gates
  are SHARED PREDICATES in the type modules (the legacy hooks import the
  same functions — drift is structurally impossible); third-party types keep
  `Type.adapt_to_native_lm_feature`, silently honored.
- **`formats/`** — every literal string of every wire format. `render.py` is
  pure assembly with zero format literals (source-tested); TwoStep's
  structurally different pipeline assembles in `formats/twostep.py`, never
  in `render.py`.
- **`parse.py` / `postprocess.py`** — parser hooks (`parse(response_view,
  ctx)`, frozen since introduction) and the engine postprocess consuming
  typed LMResponses. Text parsing executes through public `adapter.parse()`
  so callback events fire identically.
- **`overrides.py` / `migrated.py`** — the all-or-nothing routing: user
  subclasses overriding anything on the detection surface run the
  byte-untouched legacy pipeline; registration doubles as the migration
  ledger.

## The dual-path contract

Until the (future, explicitly separate) legacy-retirement epic:

1. **The golden corpus is the law** (`tests/adapters/golden/`): request,
   parse, and callback fixtures recorded from pre-engine code. Fixture
   regeneration only in dedicated corpus-update commits with zero `dspy/`
   changes (rebase protocol in the corpus README).
2. **Both paths stay correct.** Upstream adapter behavior changes absorbed
   during a rebase must land in the legacy bodies AND the Format objects in
   the same rebase; the dual-run harnesses adjudicate.
3. **The kill list** — bodies retained ONLY because legacy orchestration
   dispatches through overridable hooks (so delegation would bypass user
   overrides): `Adapter.format`/`format_demos`/`format_conversation_history`
   /`format_system_message` (orchestration), each adapter's
   `format_field_structure`, `format_user_message_content`,
   `format_assistant_message_content`, and `format_field_with_value` (hook
   dispatchers), XML's `parse` (dispatches `_parse_field_value`), BAML's
   `format_user_message_content`, and TwoStep's
   `_legacy_async_quirks_postprocess` (named quirk branch; divergences
   documented in its docstring). True-leaf bodies already delegate to the
   Format objects. An inventory test pins this list.
4. **Retirement exit criterion**: override-routing telemetry (the env-gated
   debug log) plus a deprecation cycle for `format_*`/`parse` overrides —
   owned by a future epic, not this one.

## Seams for later epics (zero rework by design)

- **Preset string vocabulary** attaches at `formats.resolve_format` /
  `register_format` (the class→Format mapping is the single attachment
  point).
- **`adapter.explain()` / `preview()`** attach at the render-only plan entry
  (`inspect.describe_plan` is the seed).
- **Strategy registry / entry points** expose `strategy.TypeStrategy` /
  `PlanStep` unchanged (name, priority, exclusive_group, applies/contribute
  → AdapterPatch over FieldContext/CallContext, StrategyTrace).
- **Normalized config**: strategies currently mutate `ctx.lm_kwargs` exactly
  as the legacy hooks did (byte parity over purity, documented in
  `strategy.py`); the LMConfig migration deprecates that channel without
  re-signaturing anything.
- **`AdapterSearch` / optimizers**: plans and registrations are data;
  serialization of per-predictor adapter config is deliberately deferred.
