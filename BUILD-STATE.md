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

## 2026-08-10 — A2: adapters v2 (agent A2)

**Status: GREEN. `uv run --extra dev pytest tests_greenfield/ -q` → 185
passed (A1's 68 + 117 new: `test_adapters.py`, `test_parser_data.py`,
`test_strategies.py`).** The adapter-ir-stage design is implemented: the
entry IS the adapter.

### What exists now

- `dspy/adapters/adapter.py` — ONE entry-backed `Adapter`:
  `format(signature, inputs, demos, lm=|capabilities=) -> AdapterCall
  (messages, request)`, `parse(signature, completion, channels=, ...)`,
  `preview()`/`parse_preview()` (the same two functions, pure, no LM),
  `dump_entry(for_signature=)`, `lens()`, `explain_strategies()`, and
  the `with_parser`/`with_strategies`/`with_codecs` doors (hybrid:
  callable on the class or an instance).
- `dspy/adapters/presets.py` — `ChatAdapter()`, `JSONAdapter()`,
  `XMLAdapter()` as thin constructors over preset entries (templates
  verbatim from the old preset data), plus `make_adapter(name=,
  template=, parser=, engine_controls=, requires=, ...)`.
  `ChatAdapter().dump_entry()` reproduces example 01's file
  byte-for-byte.
- `dspy/adapters/lens.py` — the centerpiece: `{"kind":"lens","of":
  "template"}` derived mechanically from the template. Derivation
  source order: demos-directive assistant pattern → authored assistant
  message → any outputs loop/`json_object` aggregate → the last user
  message's INPUTS loop (the labeling-convention fallback examples
  05-07's two-message template needs) → full_text degenerate. Modes:
  labeled (boundary regex with a `name` hole; xml suffixes cut; unknown
  labels ignored — the completed marker terminates for free),
  json_object (fenced-block preference, strict-then-repair, exhaust
  unknown keys), full_text (exactly one output field). One output field
  with no labels in the completion degenerates to full_text. A loop
  attribute the lens cannot invert (`{f.desc}` etc.) refuses at
  construction with a teaching error (`LensError`).
- `dspy/adapters/parse.py` — the combinator vocabulary
  (`parse_combinators` 0.1.0): `fenced_block`, `alternatives`,
  `json_object` (repair none|json_repair), `fields_from_object`
  (unknown_keys exhaust|ignore|refuse), `regex` (RE2-subset validated:
  refuses lookaround/backrefs/atomic/conditional, plus non-compiling),
  `fields_from_groups`, `coerce`, and the typed terminals `tool_calls`
  and `citations`. Authoring helpers emit the examples' exact dict
  spellings; `run_pipeline` is pure and tracks consumed spans;
  `remove_spans` is the consume mechanic. ONE vocabulary for entry
  parsers and strategy routings.
- `dspy/adapters/strategy.py` + `strategies.py` — rules as data:
  predicate over DECLARED `LMCapabilities` facts (vocabulary: instruct,
  completion, native_reasoning, native_function_calling,
  native_citations, image_input; `completion` = not instruct; None
  facts = all False), `hides`, `transforms` (rename), `fragments`,
  `engine_controls` (request_patch with the `{"$from": "field:name"}`
  splice — tools lower to litellm function schemas), `routings`
  (channel and text forms). Builtin rules are the example trios
  VERBATIM (05/06/07): reasoning native/prefix_cot/interleaved, tools
  native_fc/cli_text/xml_blocks, citations native/inline.
  `register_strategy(role, name, rule)` is the role-keyed public door;
  `strategies={"reasoning": "prefix_cot"}` binding syntax preserved;
  `"auto"` resolves first-predicate-pass in registration order and
  records `auto->name` (surfaced by `explain_strategies`); an
  explicitly named/inline rule whose predicate fails refuses naming the
  capability.
- `dspy/adapters/codecs.py` — families `text_pythonish`,
  `pydantic_json`, `schema_prose` (the ported BAML codec body), `json`;
  `register_codec` door; shape codecs (`image` at base64/png wire; PIL
  optional-import; `frontend_bindings` annotation); `coerce_shape` —
  the one coercion door (str/int/float/bool/json/ToolCalls/Citations).
- `dspy/adapters/serde.py` — the 0.3.0-draft entry, exact:
  `load_entry`/`build_entry` (as `Adapter.dump_entry`). ENTRY_KEYS + optional
  `requires`; unknown keys refuse; dangling refs refuse naming
  themselves; version tags (`-draft`) parse; 0.x minor-strict, 1.x
  entry-newer-than-engine refuses. Conditional versions block:
  roles/strategies/codecs/template_language always; parse_combinators/
  lm_capabilities/shapes present exactly when used (used-but-missing
  refuses; present-but-unused tolerated). **All 14 data-level example
  entries (01–09, both/all trio members) load and round-trip exactly;
  10/11/12 refuse with named requirements** ("requires python>=3.12
  sidecar for `4_adapter/ledger_recovery/parser` — unbound; refuse or
  bind one").
- `dspy/adapters/types/` — Tool/ToolCalls (settings- and callback-free;
  async `__call__` refuses toward `acall`; from_mcp_tool/from_langchain
  deleted), History, Image, trimmed `Type` base (content-part markers;
  streaming/native hooks gone), and a NEW neutral `Citations`
  (span + 1-based doc index) replacing the anthropic-shaped one.
  Importing the package registers the role derivations
  (Tool→tools, ToolCalls→tool_calls, Citations→citations,
  History→history, Image→media).
- `dspy/__init__.py` now exports the adapter surface (ChatAdapter,
  JSONAdapter, XMLAdapter, make_adapter, Adapter, load_entry, Tool,
  ToolCalls, Citations, History, Image) and `dspy.adapters` carries the
  `parse` / `strategy` authoring modules.

### Decisions taken (deviations from the 12 examples flagged)

1. **String parsers refuse** (open question 1 resolved hard): the four
   builtins ARE lens entries; `"parser": "chat"` gets a teaching error.
2. **Fragment dialect** (open question 5): fragments speak literal text
   plus `{field('name')}` slots ONLY; bare braces are literal. Forced
   by example 06's own bytes (`{"arg": ...}` beside `{field('tools')}`
   in one fragment string).
3. **Rule faces**: canonical order kind, predicate, hides,
   [transforms], fragments, engine_controls, routings; empty faces are
   spelled empty (as in the files); `transforms` present only when
   non-empty (as in 07-native).
4. **Builtin strategy names** replace the legacy vocabulary words:
   native/prefix_cot/interleaved, native_fc/cli_text/xml_blocks,
   native/inline (the examples' authoring intent). Structural bindings
   with no rule: media native_parts|url_reference, history
   directive_turns|inline.
5. **Rules name fields literally** (as drawn in the examples): a rule
   hiding `reasoning` against a signature whose reasoning-role field is
   named `thoughts` refuses loudly. Role-relative field resolution is a
   future nicety, not implemented.
6. **Missing channel → None**, not a parse refusal: a native-FC model
   answering without tool calls is a legitimate completion (A4's ReAct
   needs this). Text routings with zero findall matches yield empty
   values, same logic. Everything else missing refuses.
7. **`requires` is authored-or-derived** (open question 7): loaded
   entries reproduce their stated block verbatim (or its absence);
   python-authored adapters derive `lm_capabilities` from rule
   predicate atoms at dump.
8. **Versions computed at dump**: strategies 1.1.0-draft iff any rule
   object; codecs 1.1.0-draft iff family/leaf per_field entries. This
   reproduces every example file's block exactly.
9. **`with_strategies` keeps the preset entry name** (`chat`); the
   examples' per-entry names (chat_reasoning_native) come from
   `make_adapter`/`load_entry`, which is where names are authored.
10. **engine_controls**: `stop_sequences` maps to request `stop`;
    everything else (e.g. `completion_mode`) passes through the request
    dict verbatim — the executor's concern (A3).
11. **LMCapabilities grew `image_input`** (additive, per the examples'
    vocabulary) — one A1 test updated for the new to_dict key. And
    **Signature got `arbitrary_types_allowed`** so host types
    (`photo: PIL.Image.Image`) are legal annotations (example 09's
    core power); `dump_entry(for_signature=)` lowers media-role
    PIL/Image fields to the shape+wire+binding triple.
12. **Media is structural**: media-role input values lower to image
    parts (the custom-type marker + `split_message_content_for_custom_types`)
    whenever present; `url_reference` unimplemented.
13. Inline-citation spans are captured from the RAW completion (the
    routing runs before the lens), so a first-sentence span can include
    the field marker — the drawn pipeline's own semantics, kept as-is.

### Scope trims (honest)

- No authored (level-3) parser EXECUTION, no leaf codecs, no
  `materialize.interpreter` — all three refuse with the named-
  requirement error, so examples 10/11/12 refuse exactly as receivers
  without the capability should. `python_literal` codec family not
  implemented (10 refuses on the family name first).
- Template capacity checks (`_engine/template/capacity.py`) survive but
  nothing calls them yet; ADP-006-style bake checks (e.g. example 04's
  stop-sequence/field-name collision) are not wired.
- TwoStep/BAML adapter pairings gone; the schema_prose codec family
  carries the BAML value/schema spelling for whoever rebuilds the
  pairing as an entry.

### Notes for A3 (execution spine)

- **The Predict leaf's exchange is four lines:**
  ```python
  lm = resolve("lm", overrides)           # dspy.lm.bindings
  adapter = resolve("adapter", overrides) # default: dspy.ChatAdapter()
  call = adapter.format(signature, inputs=inputs, demos=demos, lm=lm)
  outputs = lm(messages=call.messages, **call.request)
  fields = adapter.parse(signature, outputs[0], lm=lm)
  # -> Prediction(**fields)
  ```
  `call.request` may carry `stop`, `tools`, `tool_choice`, `reasoning`,
  `citations`, `completion_mode` — pass them through to the LM call;
  the bindings table accepts an `"adapter"` binding name already.
- `parse(..., channels={...})` is the native-channel door. A1's LM
  returns bare `list[str]`, so native reasoning/FC/citations strategies
  only fully work once the LM layer surfaces response channels — parse
  fills channel-routed fields with None until then. DummyLM tests pass
  channels explicitly.
- `preview()`/`parse_preview()` are pure and byte-stable — use them for
  explain/lint surfaces; `adapter.lens()` and
  `adapter.explain_strategies(signature, lm=)` return data views.
- The adapter entry (`dump_entry()`) is JSON-able and exact — embed it
  in the ProgramIR artifact as component 4; `load_entry` is the link
  step and refuses dangling/unversioned entries loudly.
- `dspy/programir/` remains dormant and still imports deleted modules
  (`dspy.predict.predict`, old primitives Module) — rewire per A1's
  notes; nothing in the new `dspy/adapters` imports it.

## 2026-08-10 — A3: the execution spine (agent A3)

**Status: GREEN. `uv run --extra dev pytest tests_greenfield/ -q` → 208
passed (A1+A2's 185 + 23 new: `test_spine.py`).** The program IS the IR:
`program(**inputs)` compiles forward to node-set v0.3, materializes
against live bindings, and runs the engine interpreter. The equivalence
test holds byte-for-byte on both branches.

### What exists now

- `dspy/modules/module.py` — `Module`: author exactly as classic dspy
  (children in `__init__`, a `forward`), but `__call__` = compile_ir
  (cached) → materialize(live bindings) → interpret. Also:
  `forward_native()` (the equivalence control arm ONLY), `compile_ir()`,
  `to_manifest()` (which makes any Module a valid target for every
  programir tool via `load_manifest`'s duck check), `save(dir)`,
  `named_predictors()`, `explain()` (str), `lint()` (Finding list),
  `cost()` (bounds dict), `invalidate_ir()`.
- `dspy/modules/predict.py` — `Predict(signature, lm=, adapter=,
  **config)`: the leaf, A2's four-line exchange verbatim (resolve lm +
  adapter → `format` → `lm(messages, **request)` → `parse` →
  `Prediction`). Per-predictor bindings via constructor kwargs or
  `set_lm`/`set_adapter`; a STRING binding is a name looked up in the
  `dspy.configure` table at call time; a live object binds directly.
  LM missing anywhere refuses with `BindingError`; adapter defaults to
  one shared `ChatAdapter()` singleton (stable identity keeps the pool
  and the compile cache honest). `predictor.config` merges under the
  adapter's request (`lm(messages, **{**config, **call.request})`).
- `dspy/programir/_dspy.py` — rewired to the new core, zero settings
  reads. NEW: `compile_with_live(program) -> (ir, live_bindings)` — the
  compiler records pool-name → live object for all four kinds; that
  dict is exactly `materialize`'s `bindings` argument, which is how the
  default execution path runs the freshly compiled IR in-process.
- `dspy/programir/engine/materialize.py` — rebuilt predictors are the
  NEW `Predict` (lm+adapter bound per-predictor); LM bindings must be
  `dspy.lm.LM` instances; adapters rebuild via `load_entry` on the
  carried v2 entry (component 4 = A2's extended shape, byte-exact);
  tools rebuild from source sidecars (verified end to end: authored
  function → sidecar → fresh namespace at load). `ExecutableProgram`
  now aggregates a per-run `_trajectory["predictor_calls"]` exhaust
  (path, inputs, outputs per leaf call).
- `dspy/programir/export.py` — credential harvesting without
  settings/clients: env-var regex + each resolved LM's `kwargs["api_key"]`
  / sensitive headers / `api_key` attributes.
- `dspy/programir/__init__.py` grows `load(path, bindings)` = read +
  link + materialize; `dspy/__init__.py` exports `Module`, `Predict`,
  `load`, `diff` (lazy wrappers over programir/tools).
- Observability wiring (point 5) landed whole: explain/lint/cost as
  Module methods, `dspy.diff` accepting Modules/IRs/paths on both
  sides. NOTHING trimmed this stage.

### The compile-at-call decision

Forward compiles at FIRST `__call__` and at explicit `compile_ir()` —
NOT at `__init_subclass__`: the leaves are instance attributes, unknown
until `__init__` runs, and a teaching error should surface where the
program is used, with the full leaf table in hand. The compiled
executable is cached on the instance, keyed by a live fingerprint per
predictor: (resolved lm id, resolved adapter id, demo ids, instructions,
config repr). Reconfiguring `dspy.configure(lm=...)`, `set_lm`, demo
edits, and instruction edits all recompile automatically (tested);
structural edits (rebinding a child attribute after `__init__`) are NOT
fingerprinted — call `invalidate_ir()` after those.

### Decisions taken

1. **`Predict.__call__` runs the exchange directly**, never through the
   engine: the leaf IS its own machinery on both paths, and the engine's
   rebuilt predictors re-enter exactly this code. A bare Predict still
   compiles to the trivial `5_forward/self` (equivalence tested).
2. **Engine-path predictors are rebuilt from the manifest**, not the
   live objects — signature, instructions, demos, config all flow
   through components 2/3a/3b/3c. What runs is what saves; the
   equivalence test proves the round trip loses nothing (for the
   representable type set — see gaps).
3. **LM capability facts ride in the LM entry's `config` slot**
   (`config.lm_capabilities`) — `lm_entry` is additionalProperties:false
   in the manifest schema, and `config` is its one loose slot. They are
   provenance; at materialize time the BOUND LM's own declared facts
   drive the strategies.
4. **`11_ambient_policy` compiles as `{}`** — the greenfield core has no
   settings object; the component's contents are spec-loose until
   ratified.
5. **Exhaust**: native Predict sets `_trajectory["completion"]` (raw
   text); the engine's root Prediction gets
   `_trajectory["predictor_calls"]`. Leaf-level `_trajectory` does NOT
   survive the engine's record boundary (records are plain dicts —
   PIR-013) — declared outputs only, exhaust is per-layer.
6. **BindingError stays host-side**: compile wraps it naming the
   predictor path; `materialize`/`load` refuse missing LM/interpreter
   bindings with ValueError naming the pool entry (test d covers both).

### IR gaps hit (esp. component 4 / write-read carriage)

- **Signature rebuild is shape-poor**: `materialize._build_predictor`
  maps JSON-Schema `type` → {str,int,float,bool,list,dict} only. A
  signature with a media field (PIL type), pydantic model, or typed
  list refuses at materialize naming the field. So a program using
  A2's media shapes SAVES fine (component 2 carries the schema; the
  adapter entry carries the shape codec) but cannot yet LOAD/engine-run.
  The missing piece is a shape→annotation lifter (and/or driving
  `format` off the field records directly instead of a rebuilt class).
- **`dump_entry(for_signature=)` is never passed by the compiler**:
  component 4 entries are pooled per adapter IDENTITY, but per-field
  shape codecs are per-SIGNATURE. One adapter serving two signatures
  would need signature-lowered variants; the manifest has no
  per-predictor codec slot. A4/A5 hitting media should either pool
  per (adapter, signature) or ratify a 3d-style slot.
- **`prefix` is write-only**: the field API deprecated it, so rebuilt
  signatures carry desc only; identical prompts today because prefixes
  derive from field names.
- **LM entry carries no default kwargs** (temperature/max_tokens of the
  LM object): per-predictor config is 3c; the receiver's bound LM
  brings its own defaults. Fine for DummyLM tests; a real
  cross-process replay will want them ratified into the entry.
- **`write()` shells `uv lock --script`** (~1s warm here; needs uv and
  possibly network for a cold cache). The save/load tests run the real
  path on this machine. If CI ever runs cold/offline, stub
  `_materialize_environment_locks` there.
- Adapter v2 entries round-trip write/read exactly (chat entry loads
  back and reproduces the same messages) — no gap found in component 4
  carriage itself for the data-level entries.

### Notes for A4 (ReAct / ReActV2 / RLM)

- **Leaf declarations you get for free** (compiler + engine already
  handle them; tool leaves smoke-tested end to end including
  load-from-sidecar):
  - tools TABLE: `self.tools = {"search": fn_or_Tool, ...}` — compiles
    each entry into component 6 and admits DYNAMIC dispatch
    `self.tools[name_expr](**kwargs)` in forward (SEM-8);
  - static tool: `self.count = fn` → `self.count(text=...)`;
  - extract predictor: just another `dspy.Predict` child;
  - interpreter leaf: a child with `programir_profile()` (D-033
    structural identity; `PythonInterpreter` itself REFUSES export —
    runtime versions unpinned), called `self.interp(code=...)`, exactly
    one kwarg; at materialize the receiver must bind the runtime.
- **Tool functions must satisfy `extract_tool`**: full type hints +
  return hint, no closures, no global reads, deps via `# deps:` line.
  Tool ERRORS: raise/let-fail inside the leaf → engine wraps as
  `ToolError` → catchable by `except ToolError` in forward.
- **The zero-reach-back trap for loop caps**: forward cannot read
  `self.max_iters`. `For` needs a LITERAL range; so either bake the cap
  as a literal in the class's forward source, take `max_iters` as a
  forward argument (list-form loop or While+break), or use `While` with
  an explicit counter (SEM-6 caps While at 1000). Decide per module and
  note it.
- **Trajectory**: build it as a plain list/dict with the closed
  builtin/method tables (`json_dumps`, Format, append). If users must
  SEE it, declare it an output field (PIR-013: engine records carry
  declared outputs only); `_trajectory` exhaust does not cross the
  engine's record boundary.
- **Native-FC tools**: A2's missing-channel→None ruling means a
  native-FC model answering without tool calls parses clean — branch on
  `x.tool_calls == None`... note `is None` lowers to `== null` in the
  subset. A1's LM returns bare `list[str]` (no channels yet), so
  native-FC strategies need the LM channel surface first or the
  cli_text/xml_blocks strategies.
- **ChainOfThought did not land in A3** (charter lists it; my scope was
  Module+Predict): in this world it is a thin Predict wrapper — extend
  the signature with a reasoning-role field, or bind the `prefix_cot`
  strategy. One of A4's first moves; keep it a leaf, not a composite.
- Engine record access in forward: leaf results are dicts —
  `route.category` (Attr) and `route["category"]` (Index) both work;
  iteration order is insertion order (v0.2 Dict ruling).

## 2026-08-10 — A4: the module library (agent A4)

**Status: GREEN. `uv run --extra dev pytest tests_greenfield/ -q` → 231
passed (A1–A3's 208 + 23 new: `test_modules.py`).** ChainOfThought,
ReAct, ReActV2, and RLM exist, and every one of them compiles clean into
node-set v0.3 BY CONSTRUCTION — the compile-clean assert is a test, the
equivalence test (engine vs `forward_native`, byte-identical LM calls
and identical outputs) holds for ReAct, RLM, ReActV2, and the
ChainOfThought manifest round trip, and ReAct proves
save → load(bindings) → run with the tool leaf rebuilt from its sidecar.

### The literal-loop-cap mechanism (`ir_literals`) — READ THIS ONE

The trap A3 named: `For` needs a literal range, and zero-reach-back
forbids `self.max_iters` reads — yet a loop cap is exactly the kind of
thing users configure per instance. The resolution is a **frontend-only
declared-literal door**; the IR itself is untouched, still pure v0.3:

- A module class declares `ir_literals = ("max_iters",)` — a tuple of
  attribute names whose values are JSON scalars.
- At compile time (`_dspy._declared_literals`), the compiler reads each
  named attribute off the INSTANCE and hands the mapping to
  `compile_forward(..., literals=...)`.
- Inside the compiler, a `self.<name>` read of a declared literal
  lowers to `Const(<value>)` — in any expression position — and
  `range(self.<name>)` in For position lowers to the literal `range`
  count (`forward.py::_range_literal`). Everything else on `self.`
  still refuses with the zero-reach-back teaching error (which now also
  names the `ir_literals` alternative).
- The artifact therefore carries `{"node": "For", "range": 5}` — a
  literal, no reach-back, nothing to resolve at run time. `explain`,
  `lint`, `cost` all see the real bound.
- **Staleness is handled by the fingerprint**: `Module._fingerprint`
  folds every module's `ir_literals` values in (via the new
  `_named_modules()` walk), so `react.max_iters = 1` recompiles on the
  next call — tested, including the recompiled `range` in the manifest.
- Non-scalar or missing declared attributes refuse at compile with the
  class and attribute named.

ReAct uses it for the For cap; ReActV2 (`while turn < self.max_iters`)
and RLM (`if rounds == self.max_iters: break`) use the same door in
While position — the Const substitution works anywhere an expression
does, only For-range needed its own path.

### Generated forwards (the second mechanism)

ReAct/ReActV2/RLM are signature-polymorphic: the forward's parameter
list IS the user signature's input fields, and the v0.3 envelope has no
`**kwargs`. So these modules GENERATE their forward source in
`__init__` (a `str.format` template with the input names spliced),
register it in `linecache` under a unique `<dspy-generated:...>`
filename, `exec` it, and bind it on the instance as a method
(`modules/_generate.py`). The compiler now prefers an instance-bound
`forward` (`module.__dict__.get("forward")`) over the class's — and
`inspect.getsource` returns exactly the generated bytes, so the
compiler and the native control arm read the SAME source. One
mechanism, no second semantics.

### Leaf declarations per module

- **ChainOfThought** — a LEAF, per A3's note: `class ChainOfThought(
  Predict)`. `__init__` prepends a `reasoning` output field carrying
  the reasoning ROLE (`OutputField(role="reasoning")`); conduct comes
  from the adapter's reasoning strategy (default binding `"auto"`:
  native thinking iff the bound LM declares `native_reasoning`, else
  `prefix_cot` for instruct models, else plain). `strategies=`
  passthrough derives the adapter via `with_strategies` at
  construction (refuses on a string adapter binding). Compiles as a
  Predict node — no module node, no forward of its own. NOTE THE API
  CHANGE: `reasoning` is a DECLARED output now (`prediction.reasoning`),
  not `_trajectory` exhaust — engine records carry declared outputs
  only (PIR-013), so the old exhaust routing cannot round-trip.
- **ReAct** — leaves: `self.react` (Predict: inputs + trajectory →
  next_thought, next_tool_name, next_tool_args), `self.extract`
  (ChainOfThought: inputs + trajectory → outputs), and `self.tools`, a
  tools TABLE of dispatch wrappers (below). The loop: For over the
  baked cap; trajectory is a plain dict grown with SetIndex
  (`trajectory["thought_" + str(idx)] = ...`); `finish` is a NAME
  CHECK, not a tool; `except AdapterParseError` ends the loop; `except
  ToolError` turns a failure into an observation; the extract
  predictor terminates. Run journal = the engine's
  `_trajectory["predictor_calls"]` exhaust.
- **ReActV2** — leaves: `self.react` (inputs + history → next_thought,
  tool_calls) + the tools table. The v2 channels became DECLARED
  outputs: the record is `{**outputs, "history": [...],
  "termination_reason": ...}` built in-forward (exhaust cannot cross
  the engine record boundary). `submit` is the termination convention
  (a call whose args carry the outputs); multi-call turns via For over
  the parsed `tool_calls` list; empty batch / parse failure terminate
  with their reason. A toolless ReActV2 is legal (pure submit agent) —
  the generated forward swaps the dispatch arm for an in-forward
  answer, since an empty tools table cannot be declared.
- **RLM** — leaves: `self.gen` (inputs + transcript → code),
  `self.extract` (inputs + transcript + result → outputs), and
  `self.interp`, a DECLARED interpreter leaf (D-033 structural profile
  in component 7). The default runtime is the new
  `dspy.InProcessInterpreter` (programir/interpreters.py): exec, EMPTY
  builtins, `result`-variable convention, `str(result)` back, every
  failure a typed `InterpreterError`. While loop, literal cap, failed
  rounds become transcript notes the next round reads. This is the
  corpus MiniRLM grown usable, deliberately NOT the 817-line legacy.

### The uniform-args dispatch convention (tools)

A dynamic tool call site has exactly ONE compile-time kwargs shape
(kwargs-only convention, no splat), but ReAct dispatches to many tools
with different natural parameter shapes. Resolution: every user tool
becomes a generated **dispatch wrapper** `def <name>(args: dict) ->
<ret>` that embeds the user function's source VERBATIM (docstring,
`# deps:` line and all), calls it with `**args`, and converts any
failure into the typed `ToolError` (`from dspy.core.errors import
ToolError` inside the body keeps it self-contained). The wrapper is
what sits in `self.tools` and what component 6 carries — the artifact's
tool leaf is exactly what runs, and it rebuilds from the sidecar at
load (tested). The LM-facing schemas stay honest because the react
instructions render each ORIGINAL `Tool` (name, args schema, desc).
Native-path parity is deliberate: `ToolTable.__missing__` raises
`ToolError` for unknown names (the engine's refusal), so `except
ToolError` behaves identically on both arms — also tested directly.

### v0.3 subset gaps hit (contract feedback — precise list)

1. **Dynamic dispatch is single-shape** (SEM-8 + the kwargs-only/no-
   splat rule): `self.tools[name](**parsed_args)` is inexpressible, so
   per-tool parameter shapes cannot flow through one dynamic call
   site. The dispatch-wrapper convention is the workaround. If the
   contract ever wants natural tool signatures under dynamic dispatch,
   it needs either a dict-splat admission AT TOOL CALL SITES only, or
   a ratified single-argument dispatch convention.
2. **Exhaust does not cross the engine record boundary** (PIR-013):
   old ReAct's `_trajectory["trajectory"]` and V2's `history`/
   `termination_reason` exhaust are unreproducible. Design change: V2
   channels became declared outputs; ReAct's journal is the engine's
   `predictor_calls`. A ratified per-forward exhaust slot (record +
   exhaust return shape) would let composite modules ship run journals
   without widening their contracts.
3. **Typed-error parity is an engine-side guarantee only**: the engine
   wraps leaf failures into the table, but nothing obliges LIVE leaves
   to do the same, so `except ToolError`/`except InterpreterError`
   would diverge on the native arm. A4 closes it in the frontend
   (ToolTable, wrapper conversion, InProcessInterpreter raising typed
   errors). Worth a contract sentence: leaf implementations speak the
   typed table at the boundary.
4. **Signature rebuild shape-poverty bites EQUIVALENCE, not just
   load** (extends A3's gap): `dict[str, Any]` renders as
   `(dict[str, Any])` in prompts natively but rebuilds as bare `dict`
   → different bytes under the engine. A4's generated signatures use
   BARE `dict`/`list` annotations so the round trip is identity. The
   fix remains a shape→annotation lifter (or carrying the authored
   type name in the field record).
5. **Semantic role was dropped at materialize** — fixed this stage:
   `_build_predictor` now restores `role=` from the field record's
   `semantic_role`. Without it, CoT's prefix_cot fragment vanished on
   the engine path (the strategies predicate on roles). Receiver duty
   worth ratifying.
6. Frontend bug fixed en route: `leaves.py::_check_self_contained`
   missed inner-scope names (nested def/lambda params, `except E as
   e:` bindings) — legitimate self-contained sources refused.

### Scope trims (honest)

- **ReActV2 is the lean v2 shape**, not the experimental file's:
  history is a plain event list (no `dspy.History` typing), tool_calls
  a plain list (no `ToolCalls`/native-FC — that work starts when the
  LM layer surfaces response channels), no forced-submit second pass,
  no tool-call ids. Trimmed per the scope valve, shape preserved.
- No context-window truncation/retry anywhere
  (`ContextWindowExceededError` died in the carve; an LM failure
  propagates as `LMError` — the loop does not catch it).
- Old ReAct's per-CALL `max_iters=` override is gone: the cap is
  instance configuration (edit + automatic recompile), not a call
  argument.
- CoT's `rationale_field`/`rationale_field_type` params dropped (no
  legacy compat by charter).
- RLM: no sub-LM tools (`llm_query`), no batching, no REPL variable
  marshaling, no markdown fence stripping, no output truncation.
- ReAct requires ≥1 tool (a toolless ReAct is CoT wearing a loop —
  refused with that sentence); V2 allows zero (submit-only is a real
  v2 shape).

### Notes for A5 (optimizers)

- **Where the mutable state lives in the IR**: instructions =
  component `3a_instructions` (dotted predictor path → str); demos =
  `3b_demos` (path → list of example records, each `{**values,
  "input_keys": [...]}`); per-predictor config = `3c_predictor_config`.
  On LIVE modules the same state is `predictor.signature.instructions`
  (via `predictor.signature = predictor.signature.with_instructions(
  ...)` — signatures are immutable classes), `predictor.demos` (a list
  of `dspy.Example`, each REQUIRING `.with_inputs(...)` or compile
  refuses), and `predictor.config`.
- **Mutate + rescore, the cheap loop**: mutate the live predictor
  (append demos / swap instructions) and just CALL the program — the
  fingerprint covers demo identities, instructions, config, bindings,
  and `ir_literals` values, so recompile is automatic; no
  `invalidate_ir()` needed for these. Score = run over the devset with
  a scripted or live LM; `named_predictors()` gives you the (path,
  predictor) surface. Keep ONE adapter/LM instance per candidate —
  fingerprints are id-based, fresh instances recompile every call.
- **Propose at the artifact level** if you prefer pure-data mutations:
  `compile_with_live(program)` → edit the manifest components 3a/3b →
  `materialize(ir, live)` and replay. What you keep IS the artifact:
  checkpoint == `program.save(dir)` (12_metric already carries
  metric+devset when passed to `compile_program(metric=, devset=)`).
- The scripted-completion helper for tests lives in both test files
  (`chat_completion(**fields)`); ReAct/RLM scripts in
  `test_modules.py` show the exact multi-step shapes.
- Demos flow into generated-signature predictors exactly as into plain
  ones (component 3b is path-keyed; ReAct's react/extract predictors
  are ordinary paths).
- Generated forwards are per-INSTANCE: deep-copying a module copies
  the bound method's self? No — `copy.deepcopy` on a module with an
  instance-bound MethodType rebinds correctly under default deepcopy
  semantics, but if you clone programs for candidates, prefer
  re-CONSTRUCTING (`ReAct(sig, tools, max_iters)`) over deepcopy and
  carry demos/instructions across; it is the honest path and avoids
  aliasing the tools table.

## 2026-08-10 — A5: optimizers as IR mutations (agent A5)

**Status: GREEN. `uv run --extra dev pytest tests_greenfield/ -q` → 249
passed (A1–A4's 231 + 18 new: `test_optim.py`).** `dspy/optim/` exists:
LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch, and
the loop machinery they share — propose = mutate the program's data,
score = engine replay over a devset, keep = the best state. checkpoint
== save: every accepted candidate is an ordinary loadable artifact.

### What exists now

- `dspy/optim/base.py` — the loop shape:
  - `ProgramState` + `snapshot_state`/`apply_state` — a candidate is
    DATA: `{path: demos}` + `{path: instructions}`, the live mirror of
    manifest components 3b/3a. Apply refuses on structural mismatch.
  - `evaluate(program, devset, metric) -> EvaluationResult(score,
    results, lm_calls)` — sequential; every program runs THROUGH THE
    ENGINE (`run_engine`: a `Module` — leaf Predict included — goes
    `_executable()`; an `ExecutableProgram` already is the engine).
    Catchable failures score 0.0 with `prediction=None`; `lm_calls`
    counts `predictor_calls` exhaust (failed runs contribute none —
    documented).
  - `Checkpointer` — `<dir>/candidate-NNN/` via `program.save` (the one
    save path) + `scores.json` (candidate, score, label; the acceptance
    trajectory in order). Loading a checkpoint is plain `dspy.load`.
  - `check_trainset` — every entry must be a `dspy.Example` with
    `.with_inputs(...)` called; refusal names the index.
  - `Optimizer` — `compile(program, *, trainset, ...) -> program`.
- `dspy/optim/labeled_fewshot.py` — `LabeledFewShot(k=16, seed=0)`: one
  seeded rng samples k demos per predictor sequentially (`sample=False`
  takes the first k). The trivial mutation; proves the plumbing.
- `dspy/optim/bootstrap.py` — `BootstrapFewShot(metric,
  metric_threshold=, max_bootstrapped_demos=4, max_labeled_demos=16,
  seed=0)`: the teacher (default: the student ITSELF; or a provided
  structurally identical program — paths + `signature.equals` checked,
  shared predictor instances refused) gets labeled demos attached, runs
  the trainset through the engine, and each metric-passing run's
  per-predictor call records become `Example(augmented=True, **inputs,
  **outputs)` demos. The current example is dropped from demos while it
  runs (no self-teaching). Teacher/student state snapshot-restored in a
  `finally`; the student then gets augmented demos + a seeded labeled
  fill from the unbootstrapped remainder, both caps honored.
- `dspy/optim/random_search.py` —
  `BootstrapFewShotWithRandomSearch(metric, ..., num_candidate_programs
  =16, stop_at_score=, seed=0)`: the classic schedule verbatim — seed
  -3 zero-shot, -2 labeled-only, -1 unshuffled bootstrap, 0..N-1
  bootstrap over `Random(seed)`-shuffled trainset with a
  `Random(seed).randint(1, max)` demo budget. Each candidate: apply
  baseline → propose → `evaluate` → snapshot → revert; STRICT
  improvement accepts (ties keep the first best); `optimizer.
  trajectory` records every candidate (seed, label, score, lm_calls,
  accepted, checkpoint name). Best state applied on return.
- `dspy/__init__.py` exports `optim` plus the three optimizer classes;
  `evaluate` stays namespaced (`dspy.optim.evaluate`).

### The mutation surface used (exact fields)

- `predictor.demos = [...]` — a list of `dspy.Example` (each with
  `_input_keys`); compiles into component 3b path-keyed.
- `predictor.signature = predictor.signature.with_instructions(...)` —
  component 3a (snapshot/apply carry it; none of the three shipped
  optimizers proposes instruction edits yet — A6+ instruction
  optimizers get the door for free).
- NOTHING else: no config mutation, no structural edits, no
  `invalidate_ir()` needed anywhere — A4's fingerprint (demo ids,
  instructions, bindings, ir_literals) recompiles mutated programs
  automatically, exactly as promised. Candidates on ONE live program
  via snapshot/apply; no deepcopy, no reset_copy (re-construct for a
  fresh original — noted in the Optimizer docstring).

### What the engine records gave vs what was added

- `_trajectory["predictor_calls"]` (path, inputs, outputs per leaf
  call) is EXACTLY the bootstrap trace — no predictor2name id mapping,
  no settings.trace context; path-keyed records match
  `named_predictors()` paths on student and teacher by construction.
  Nothing had to be added to the engine.
- One gap worked around, not fixed: a LEAF program's default call is
  native (`Predict.__call__`), which records no `predictor_calls` — so
  `run_engine` routes Modules through `_executable()` (private access,
  the one impurity; a public `Module.executable()` would be cleaner).
- Failed runs leave no exhaust, so `evaluate.lm_calls` undercounts by
  the calls a failing run made before raising. Honest fix would be an
  engine-side call counter that survives refusal; documented instead.

### Seeds / determinism

- One explicit `seed=` per optimizer (default 0), driving: labeled
  sampling, the teacher's labeled demos, the validation shuffle, and
  the labeled fill. Random-search candidate seeds are the schedule
  itself (-3..N-1), matching the old semantics.
- Everything is sequential; DummyLM scripts in the battery are
  CALLABLES (answer tables keyed off the rendered prompt), so
  determinism holds regardless of call order — copy that pattern.
- Same seed → byte-identical demo sets (tested); different seed →
  different labeled samples (tested).

### Old-semantics deviations (honest trims)

- `max_rounds` dropped: the old retry loop rode `rollout_id`/
  `temperature=1.0` cache-busting that has no greenfield counterpart —
  one engine attempt per trainset example.
- Multiple traces of one predictor in one run: the old 50/50
  hash-rng sampling is now "the LAST call wins" (deterministic; for
  ReAct that is the finish step — arguably the best demo anyway).
- Old `_train`'s raw-demo RESAMPLING quirk (each predictor sampled
  from the previous predictor's sample) dropped: every predictor
  samples its labeled fill from the full validation remainder with the
  one shared rng.
- Errors during a bootstrap attempt: only the TYPED catchable channel
  (`CatchableError`) counts as a failed attempt; anything else
  propagates. No `max_errors` budget, no logging counters.
- Random search: `restrict=`, `num_threads`, `labeled_sample=` gone;
  `stop_at_score` kept. Candidate metadata lives on
  `optimizer.trajectory`, not bolted onto the returned program.
- LabeledFewShot on an empty trainset attaches empty demo lists (old
  returned early keeping stale demos — mutating to the stated state is
  truer).

### Notes for A6 (integration / demo scripts)

- **The showcase arc that writes itself**: one QA task, four beats —
  (1) zero-shot ChainOfThought fails half the devset; (2)
  `LabeledFewShot` lifts it (demos visible in `preview()` and in
  `explain()`); (3) `BootstrapFewShot` produces augmented demos WITH
  the reasoning the teacher actually traced (show one demo's
  `reasoning` field); (4) `BootstrapFewShotWithRandomSearch(...,
  checkpoint_dir=...)` — then `dspy.load(checkpoint_dir /
  "candidate-001", bindings=...)` and run it, proving checkpoint ==
  artifact. `optimizer.trajectory` prints as the search story;
  `dspy.diff(zero_shot_manifest, optimized_manifest)` shows the demos
  as the ONLY delta — the whole thesis in one diff.
- The `copycat` DummyLM pattern in `test_optim.py` (answers correctly
  only when a demo carries the answer) makes optimizer improvement
  REAL under a scripted LM — reuse it for the demo.
- ReAct optimization works end to end (bootstrap + equivalence
  tested); its bootstrapped react-step demos carry `trajectory` dict
  values — they render and round-trip fine.
- `optim.evaluate` is the scoring door for the demo's before/after
  numbers; `result.lm_calls` gives the cost line.
