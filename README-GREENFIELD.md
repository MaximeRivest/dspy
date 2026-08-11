# README-GREENFIELD — the morning tour

**Branch `greenfield-ir`. Start here, then run the six scripts in
`examples_greenfield/`. Everything below is scripted (DummyLM) — zero
network, zero credentials.**

```bash
uv run --extra dev pytest tests_greenfield/ -q   # 256 passed
uv run python examples_greenfield/01_hello.py    # then 02..06
```

## (a) What this is

This branch rebuilds dspy as if ProgramIR, the Adapter IR, and explicit
LM bindings had been the foundation from day one. The syntax is still
dspy — Signature, Module, Predict, ChainOfThought, ReAct, RLM,
optimizers — but the program IS the IR, the engine IS the executor, and
adapters ARE data. Every program compiles to an inspectable, diffable,
shippable artifact, and what runs is exactly what saves.

## (b) The numbers (computed with `wc -l` over `*.py`, honestly)

| | old (`e4086d010`, at the charter) | greenfield (now) |
|---|---|---|
| `dspy/` lines | 50,195 | 21,537 |
| `dspy/` files | 228 | 88 |
| tests | (old `tests/` tree, not the target) | 256 passing, 2,873 lines |
| examples | — | 6 scripts, 384 lines |
| commits | — | 25 on this branch, charter to finish |

Of the 21,537 new-world lines, roughly 11,800 are kept-and-rewired
foundations (`programir/` 7,488, `adapters/_engine/` templates 1,735,
`signatures/` 1,397, `primitives/` interpreter machinery 1,222); the
rebuilt surface on top is ~9,700 lines: `core/` 2,870, `adapters/` v2
~4,600, `modules/` 1,057, `optim/` 671, `lm/` 309.

## (c) What's different

| old dspy | greenfield |
|---|---|
| `dspy.settings` / `dspy.context`, thread-local ambient state | **bindings**: one plain dict written by `dspy.configure(...)`, per-predictor overrides (`lm=`, `set_lm`), loud `BindingError` when unresolved |
| adapter classes with Python `format`/`parse` methods | **entries**: an adapter is data — template + parser + strategies + codecs; `dump_entry()`/`load_entry()` round-trip byte-exact |
| parser enum (`"chat"`, `"json"`) | **lens + pipeline**: the parser is derived mechanically from the template (lens), or declared as combinator steps (pipeline); string parsers refuse with a teaching error |
| `forward` runs your Python | **engine**: `forward` compiles to node-set v0.3 at first call; `program(**inputs)` runs the engine interpreter; `forward_native()` survives only as the equivalence-test control arm |
| `save`/`load` state dicts, cloudpickle | **one artifact path**: `program.save(dir)` writes manifest + tool source sidecars + locked PEP 723 env; `dspy.load(dir, bindings=...)` reads + links + materializes |
| teleprompt mutating live objects via `settings.trace` | **optimizers as IR mutations**: propose = edit demos/instructions as data, score = engine replay, keep = the state; checkpoint == `save`, every candidate a loadable artifact |
| capability sniffing on live clients | **declared LM capability facts** (`instruct`, `native_reasoning`, `native_fc`, ...) as constructor data; adapter strategies predicate on facts only |
| provider/library exceptions, silent fallbacks | **the typed error table** (contract SEM-3): `ToolError`, `AdapterParseError`, `LMError`, `InterpreterError`... catchable in `forward`; refusals are loud and teach |

## (d) The tour — run this, then this

**`01_hello.py`** — a Signature, a Predict, one exchange, then
`program.explain()`: the compiled IR as one readable view (module tree,
signature records, forward as the node set, adapter/LM pools, the works).

```
question : Why is the sky blue?
answer   : Because the atmosphere scatters blue light most.
...
  def forward[self](question):        # restricted-python-ast
      prediction = predict[self](question=question)
```

**`02_chain_of_thought.py`** — ChainOfThought IS Predict with a
reasoning-ROLE field. The `"auto"` strategy resolves per the LM's
declared facts: an instruct model gets the `prefix_cot` fragment, a
native-thinking model gets a request channel. Same program, different
rendered bytes — shown with pure previews.

```
resolved : auto->prefix_cot        |  resolved : auto->native
prefix_cot request: {}  (conduct rides the prompt)
native request:     {'reasoning': {'effort': 'medium'}}
```

**`03_react_tools.py`** — ReAct with two declared tool leaves and a
literal loop cap. One engine run, then the run journal, the static
lint, and the call/token bounds — all views over the same compiled value.

```
predictor calls: ['react', 'react', 'react', 'extract']
total calls  : Bounds(minimum=2.0, expected=3.5, maximum=6.0)
```

**`04_adapters_three_ways.py`** — one program, four wire formats: the
chat/json/xml presets plus a custom entry whose parser is a declared
regex pipeline. Preview bytes for each, parse each convention back,
then run the same predictor under all four.

```
   chat: score = 7
   json: score = 7
    xml: score = 7
 grader: score = 7
```

**`05_save_ship_load.py`** — `save` writes the whole story to one
directory; `load(bindings=...)` rebuilds it (the tool leaf from its
source sidecar) and replays the direct run byte-for-byte.

```
    8672B  manifest.json
     634B  tools/lookup_capital.py
replayed.toDict() == direct.toDict(): True
```

**`06_optimize.py`** — BootstrapFewShot: a teacher's metric-passing
engine traces become demos (reasoning included), scored by engine
replay, and the whole optimizer step is one reviewable diff.

```
score: 0.0 -> 1.0
    ~ demos (3b): 0 -> 2 baked
      + inputs(question="What is the capital of France?") -> labels(answer="Paris", ..., reasoning="Recall France.")
```

## (e) Honest limits

Everything trimmed or unfinished, harvested from BUILD-STATE A1–A5:

- **No async, no streaming, no caching, no retries, no callbacks** in
  the LM layer; synchronous chat completions only (A1).
- **No response channels yet**: `LM` returns bare `list[str]`, so
  native reasoning / function-calling / citations strategies format the
  request correctly but parse their channel-routed fields to `None` on
  live runs. DummyLM tests pass channels explicitly (A1/A2; the reason
  ReAct ships on the text strategies).
- **Media is save-only**: media-role inputs render (base64 image
  parts), and a media program SAVES fine — but the signature rebuild at
  load maps JSON-Schema types to {str, int, float, bool, list, dict}
  only, so media/pydantic/typed-list fields refuse at materialize
  (A2/A3). `url_reference` media strategy unimplemented.
- **Authored code refuses by design, and nothing can bind it yet**: no
  level-3 parser execution, no leaf codecs, no `materialize.interpreter`
  — all refuse with named requirements, exactly as a receiver without
  the capability should (A2). `python_literal` codec family absent.
- Template capacity checks exist but nothing calls them; ADP-006-style
  bake checks (stop-sequence/field-name collisions) not wired (A2).
- **The adapter entry pools per adapter identity**: per-field shape
  codecs are per-SIGNATURE, and the compiler never passes
  `for_signature=` — one adapter serving two media signatures has no
  slot for both lowerings (A3, contract feedback below).
- `prefix` is write-only; the LM entry carries no default kwargs
  (temperature etc.) — the receiver's bound LM brings its own (A3).
- `save()` shells `uv lock --script`: needs `uv`, and a cold cache may
  need network — the one non-hermetic edge of the artifact writer (A3).
- Structural edits after `__init__` (rebinding a child module) are not
  fingerprinted — call `invalidate_ir()` (A3).
- **Exhaust does not cross the engine record boundary** (PIR-013): leaf
  `_trajectory` stays per-layer; composite modules that want a visible
  journal declare outputs (ReActV2's `history`) or use the engine's
  `predictor_calls` (A3/A4).
- ReActV2 is the lean shape: plain history list, plain tool_calls, no
  native-FC, no forced-submit pass, no tool-call ids (A4).
- No context-window truncation/retry anywhere; an LM failure propagates
  as `LMError` (A4).
- ReAct: the loop cap is instance config (no per-call `max_iters=`);
  at least one tool required. CoT: `rationale_field` params gone (A4).
- RLM: `InProcessInterpreter` (exec, empty builtins), not the legacy
  sandbox; no sub-LM tools, no batching, no REPL variable marshaling,
  no fence stripping, no output truncation. `PythonInterpreter` refuses
  export (runtime versions unpinned) (A4).
- Optimizers: sequential evaluation only; one engine attempt per
  example (`max_rounds` retry dropped); multiple traces of a predictor
  in one run = last call wins; no `max_errors` budget; only the
  catchable typed channel counts as a failed attempt;
  `evaluate.lm_calls` undercounts runs that failed mid-way (A5).
- The old `tests/` tree and the docs site are untouched and expected
  red — this branch's target is `tests_greenfield/`.

## (f) Design decisions to review

**`ir_literals` — the declared-literal door (A4).** `For` needs a
literal range and zero-reach-back forbids `self.max_iters` — yet a loop
cap is per-instance configuration. Resolution: a class declares
`ir_literals = ("max_iters",)`; at compile time the named attribute's
JSON-scalar value lowers to `Const` (and to the literal `For` range),
so the artifact carries `{"node": "For", "range": 5}` — pure v0.3,
nothing resolved at run time. The fingerprint folds these values in, so
`react.max_iters = 1` recompiles automatically. The IR is untouched;
the door is frontend-only. Review: is a declared-literal attribute the
right seam, or should the contract grow a parameter concept?

**Generated forwards (A4).** ReAct/ReActV2/RLM are
signature-polymorphic but the v0.3 envelope has no `**kwargs`, so these
modules generate their forward source in `__init__` (input names
spliced into a template), register it in `linecache`, and bind it on
the instance. The compiler prefers the instance-bound forward and
`inspect.getsource` returns exactly the generated bytes — compiler and
native control arm read the SAME source, one semantics. Review: the
generation is honest but novel; is per-instance source the mechanism to
keep?

**Compile at first call, not at subclass (A3).** Leaves are instance
attributes, unknown until `__init__` runs, and the teaching error
should surface where the program is used with the full leaf table in
hand. The compiled executable caches per instance under a live
fingerprint (bindings, demos, instructions, config, ir_literals);
reconfigure/mutate and the next call recompiles. Review: the
fingerprint is id-based — cheap and honest, but structural edits need
the explicit `invalidate_ir()`.

**String parsers refuse (A2).** `"parser": "chat"` — the 0.2.0 enum
spelling — gets a teaching error, not a compatibility shim. The
builtin presets ARE lens entries derived from their own templates;
there is exactly one parser vocabulary (lens + pipeline combinators)
and no privileged names. Review: hard refusal on day one is the
cheapest moment to buy one vocabulary forever.

**The exhaust/record boundary (A3/A4, PIR-013).** Engine records carry
declared outputs only; `_trajectory` exhaust does not cross. This
forced an API change: ChainOfThought's `reasoning` is now a DECLARED
output field (`prediction.reasoning`), and ReActV2's
`history`/`termination_reason` became declared outputs built
in-forward. The run journal composites DO get is the engine's own
`_trajectory["predictor_calls"]`. Review: the boundary keeps records
pure data, but a ratified per-forward exhaust slot would let modules
ship journals without widening their contracts.

## (g) Contract feedback for programir-contract

Harvested from A3/A4 (the precise lists live in BUILD-STATE):

1. **Signature rebuild is shape-poor** (A3, bites equivalence per A4):
   field records carry JSON-Schema `type` only, so rebuilt annotations
   lose `dict[str, Any]` vs `dict` and refuse media/pydantic shapes.
   Wanted: a shape→annotation lifter, or the authored type name in the
   field record.
2. **Component 4 pools per adapter identity**, but per-field shape
   codecs are per-signature — pool per (adapter, signature) or ratify a
   3d-style per-predictor codec slot (A3).
3. **Dynamic dispatch is single-shape** (SEM-8 + kwargs-only): natural
   per-tool parameter shapes cannot flow through one dynamic call site;
   the uniform-args dispatch wrapper is the workaround. Wanted: a
   dict-splat admission at tool call sites only, or a ratified
   single-argument dispatch convention (A4).
4. **A per-forward exhaust slot** (record + exhaust return shape) so
   composite modules can ship run journals without widening their
   declared contracts (A4).
5. **Typed-error parity should be a leaf duty**: the engine wraps leaf
   failures into the table, but nothing obliges LIVE leaf
   implementations to speak it — one contract sentence would close the
   native/engine divergence the frontend currently closes by hand (A4).
6. **Semantic role restoration is a receiver duty** worth ratifying:
   dropping `semantic_role` at materialize silently killed the
   prefix_cot fragment until fixed (A4).
7. Smaller: `prefix` needs a slot or a funeral; the LM entry should
   carry default request kwargs for cross-process replay;
   `11_ambient_policy` compiles as `{}` and stays spec-loose until
   ratified (A3).
