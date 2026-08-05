# Epic A — close the engine (third-party types onto the strategy seam)

**Goal.** Close the adapter engine's last `TODO(adapters-plan)` seam
(`dspy/adapters/base.py`): third-party semantic types currently enter
planning through a special-cased inline block in `_engine/builder.py` that
honors the legacy `Type.adapt_to_native_lm_feature()` hook. After this epic
the builder has **one uniform loop over strategies** — built-in, registered
third-party, and legacy-hook types all flow through the same
`TypeStrategy` contract — and the spec's §Adapter-semantics gains its dspy
substrate: a registration path for authored strategies.

**Non-goals.** No public API change (`dspy/__init__.py` and
`dspy/adapters/__init__.py` exports untouched — the registration path is
engine-private until an exposure epic). No deprecation signaling on the
legacy hook. No role vocabulary, no codecs — those are later epics.

## PR stack

### A-1 `refactor(adapters): legacy type hook as an auto-wrapped engine strategy`

- `AdapterPatch` gains a legacy-compat channel: `replace_render_signature`
  (the hook's contract returns a whole rewritten signature; delete-only
  reconstruction would silently drop any richer rewrite). `merge` treats two
  replacements as a conflict; `_apply_patch` consumes it in place of
  transform-derived deletions.
- New `_engine/strategies/legacy.py`: `LegacyTypeHookStrategy(annotation)` —
  adapts the documented hook contract to `TypeStrategy`. `applies()` is
  always true (the hook was always invoked); `contribute()` runs the hook,
  derives Hide transforms + the `ThirdPartyNativeParserHook`, and
  self-reports its `StrategyTrace` via the patch (`type_hook:<TypeName>`,
  decision by observed effects — byte-identical to the inline block it
  replaces).
- `_engine/strategies/__init__.py`: a lazy registry seeded with the
  built-ins; `field_strategy_for(annotation)` resolves registry entry or
  auto-wraps; `register_field_strategy` / `unregister_field_strategy` are
  the authored-strategy seam. `builtin_field_strategy_for` keeps its exact
  contract (tests pin it).
- `builder.py`: the third-party inline block is deleted; the loop calls
  `field_strategy_for` and appends the standard applies-based trace only
  when the patch did not self-report one (a declared contract, not a
  special case). `base.py`'s TODO comment rewritten to record the closed
  state.

**Definition of done:** golden corpus byte-identical, zero fixture
regeneration; exports unchanged; full `dspy-ci` matrix green.

### A-2 `test(adapters): third-party strategies both ways`

- A custom Type exercised via the legacy hook (unregistered → auto-wrap)
  and via `register_field_strategy` (a proper `TypeStrategy`), asserting
  equivalent plan effects (hidden field, kwargs, parser) and correct trace
  attribution (`type_hook:` prefix vs the strategy's own name).
- Registry hygiene: unregister restores auto-wrap; built-ins resolvable
  through the same path.

**Definition of done:** same gates as A-1.

## Gates (both PRs)

- `tests/adapters/test_golden_parity.py` passes with zero regenerated
  fixtures — any needed regeneration means behavior changed: stop.
- `tests/adapters/engine/*` green, notably the existing
  `test_third_party_type_hook_still_silently_honored` (unchanged — the
  compat contract is pinned by it).
- Full `dspy-ci` matrix (py3.10–3.14 + llm_call) green.
