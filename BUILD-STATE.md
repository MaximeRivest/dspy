# BUILD-STATE — the running log of the greenfield rebuild

Each builder agent appends a dated section. Read `GREENFIELD.md` first.

## 2026-08-10 — A1: core + package layout (agent A1)

**Status: GREEN. `uv run --extra dev pytest tests_greenfield/ -q` → 68 passed.**
`import dspy` works at every commit on this branch.

### What exists now

- `dspy/core/` — the intent layer, one import point:
  - `errors.py` — the CANONICAL typed-error table (programir-contract
    SEM-3): `PirError`, catchable channel (`ToolError`,
    `InterpreterError`, `AdapterParseError`, `LMError`, plus the
    interpreter trio), uncatchable channel (`LoopCapError`,
    `MalformedNodeError`), `RAISEABLE` / `CATCHABLE_NAMES` /
    `HANDLER_NAMES` / `handler_matches`.
    `dspy/programir/engine/errors.py` is now a thin re-export of this
    module — one class identity system-wide.
  - `example.py`, `prediction.py` — moved from `dspy/primitives/`.
    `Prediction` keeps the `_trajectory` exhaust channel (dict; direct
    reads fine; attribute reads of exhaust keys warn `DeprecationWarning`).
  - `types.py` — KEPT UNCHANGED (pydantic-only, standalone): the
    normalized `LMRequest`/`LMResponse`/`LMMessage`/stream types. Not
    re-exported at top level; A2's adapter engine imports
    `dspy.core.types` directly.
  - `__init__.py` re-exports the signature surface from
    `dspy/signatures` (the implementation package) plus
    Example/Prediction/errors.
- `dspy/signatures/` — the authoring surface, kept and simplified:
  - Same metaclass syntax (class signatures, `Signature("a -> b")`,
    `@role` shorthand, `role=` kwarg, `dspy.roles.x[str]` sugar).
  - `IS_TYPE_UNDEFINED` moved into `field.py` (utils/ is gone).
  - Old dsp-era `OldField`/`OldInputField`/`OldOutputField`/
    `new_to_old_field` deleted.
  - **roles.py: the legacy-type derivation table is now an EMPTY
    REGISTRY** + `register_role_derivation(type, role)`. The signature
    layer no longer imports `dspy.adapters.types.*`. Subclass lookup
    still works. Conflict/unknown-role refusals unchanged and eager.
- `dspy/lm/` — the whole model story:
  - `lm.py` — ONE `LM` class: litellm `completion()` chat, synchronous,
    no streaming / callbacks / caching / retries. Any transport or
    provider failure raises the typed `LMError` (with `__cause__`); a
    contentless choice also refuses loudly. `lm.history` records
    model/messages/kwargs/outputs/timestamp. Capability facts
    (`instruct`, `native_reasoning`, `native_fc`, `native_citations`)
    are constructor data, exposed as frozen `LMCapabilities`
    (`.to_dict()` for serialization).
  - `dummy.py` — `DummyLM(outputs)`: list of strings (one per call, in
    order; exhaustion raises `LMError` — nothing repeats silently) or a
    callable `f(messages) -> str`. Records every call in `.calls`.
  - `bindings.py` — `BINDINGS`, a plain module-level dict.
    `dspy.configure(lm=...)` writes it (`None` unbinds); arbitrary
    binding names allowed (A2/A3 will bind adapters etc.).
    `resolve(name, overrides)` — per-predictor overrides win, else the
    table, else `BindingError` (deliberately NOT in the catchable
    table: host misconfiguration, not a program error). NO
    thread-locals, NO `dspy.context`, NO settings object — tests assert
    their absence.
- `dspy/primitives/` — trimmed to interpreter machinery only:
  `code_interpreter.py` (its error base is now the table's
  `InterpreterError`), `python_interpreter.py`, `runner.js`,
  `repl_types.py`, `sandbox_serializable.py`. `Module`/`BaseModule`
  deleted (A3 rebuilds Module on the IR).
- `dspy/__init__.py` — exports core surface + `LM`, `DummyLM`,
  `LMCapabilities`, `configure`, `BindingError`, `roles`, metadata.
  Does NOT import `dspy.adapters` or `dspy.programir` (both dormant
  until A2/A3).
- `tests_greenfield/` — `test_core.py` (44) + `test_lm.py` (24).

### Removed in the carve (commit e9178eb0f; git history keeps them)

`dspy/clients`, `dspy/predict`, `dspy/retrievers`, `dspy/streaming`,
`dspy/teleprompt`, `dspy/evaluate`, `dspy/dsp`, `dspy/propose`,
`dspy/datasets`, `dspy/experimental`, `dspy/utils` (ALL of it, not just
callbacks — everything in it belonged to removed machinery),
`dspy/primitives/{module,base_module}.py`.

### Decisions taken (charter-ambiguity calls)

1. **Signature implementation stays in `dspy/signatures/`**; `dspy/core`
   is a re-export door. Truest to "keep the existing metaclass surface"
   with zero duplication.
2. **One error table**: `dspy/core/errors.py` canonical; the engine
   module re-exports it (identity-shared, verified by test via
   file-path import since the engine package init is dormant).
3. **Role derivations are registrations**, not imports: signatures own
   the vocabulary; the type-defining layer registers `type -> role`.
4. **`BindingError` lives outside the catchable table** — a program's
   `Try` handles runtime errors, not missing host configuration.
5. **LM call surface**: `lm(messages)` or `lm(prompt="...")`
   (exclusive), returns `list[str]` (one per choice). Sync only.
6. **Removed beyond the named list**: `propose/` (teleprompt
   machinery), `datasets/` (depends on dsp utils; not in the new
   layout), `experimental/` (re-exported adapter types only).
7. **uv.lock re-locked** (version marker + pytest-xdist were stale).

### Notes for A2 (adapters v2)

- `dspy/adapters/` is UNTOUCHED but DORMANT: it still imports deleted
  modules and will not import until you rework it. The import-time
  breakage map (what each part reaches for):
  - `dspy.clients.base_lm` (`BaseLM`) — `types/base_type.py`,
    `types/reasoning.py`, `base.py` and most adapter classes.
  - `dspy.utils.callback` (`BaseCallback`, `with_callbacks`) and
    `dspy.dsp.utils.settings` — `base.py`, `types/tool.py`.
  - `dspy.utils.exceptions` (`AdapterParseError`, `LMError`,
    `UnserializableTypeError`) — `_engine/parse.py`, `_engine/formats/*`,
    `_engine/postprocess.py`, `utils.py`. Re-point to
    `dspy.core.errors` (same names, contract semantics; note the old
    rich metadata kwargs are gone — messages only).
  - `dspy.utils.annotation` (`experimental` decorator) —
    `types/citation.py`, `types/document.py`: just delete the decorator.
  - `dspy.experimental` (deleted) — lazy imports inside
    `types/citation.py` / `types/document.py` docstring helpers.
  - `dspy.core.types` (`LMRequestPatch` etc.) — `_engine/*`: still
    fine, `core/types.py` is kept unchanged.
- `dspy/primitives/repl_types.py` imports `dspy.adapters.utils.serialize_for_json`
  at module top — it is dormant until you fix adapters/utils (nothing
  imports it eagerly; `sandbox_serializable` pulls it lazily).
- **Register the legacy-type role derivations** when you rebuild the
  type frontend: call
  `dspy.signatures.roles.register_role_derivation(...)` at import time
  for Reasoning→reasoning, Tool→tools, ToolCalls→tool_calls,
  Citations→citations, History→history, Image/Audio/File/Document→media,
  Code→code (the exact table deleted from roles.py, recoverable at
  commit a911af2eb^).
- LM capability facts for strategy predicates: read
  `lm.capabilities` (frozen dataclass) — never sniff.
- The old `Adapter.__call__` orchestration relied on settings/callbacks;
  the new engine gets its LM via `dspy.lm.resolve("lm", overrides)`.

### Notes for A3+ (execution spine)

- `dspy/programir/` kept whole but its package `__init__` (and
  `engine/materialize.py`, `export.py`, `_dspy.py`, `compile.py`,
  `leaves.py`) still import deleted modules (`dspy.predict.predict`,
  `dspy.clients.base_lm`, `dspy.dsp.utils.settings`, old
  `dspy.primitives.module`). Rewire them to `dspy.core` /
  `dspy.lm` / the A2 adapters; `dspy/__init__.py` then re-grows
  `export`, `load`, etc.
- `engine/errors.py` already re-exports `dspy.core.errors` — keep that
  direction.
- Old `Module.save/load`, callbacks, usage tracking, asyncify,
  streaming all died with the carve; the charter replaces them with
  artifact save/load through the IR. Do not resurrect `dspy/utils`.

### Scope trims (deliberate, per budget)

- No async LM path, no caching, no retries in `dspy/lm` (A2/A3 can add
  where the design demands it — as data, not ambient state).
- `dspy.core.types` stream/delta types survive unused until a stage
  needs them; deleting the streaming subset is A2's call.
- Old `tests/` tree untouched and expected red — not the target.
