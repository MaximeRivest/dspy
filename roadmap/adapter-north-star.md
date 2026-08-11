# Adapters — the north star (Maxime, 2026-08-10)

**This document is the true intent of adapters. The code in
`dspy/adapters/` is moving toward it and is NOT finished; where this
doc and the code disagree, this doc governs the direction. Companion
mechanics: `adapter-parse-dsl.md` (how each piece becomes data),
`adapter-data-audit.md` (current code truth),
`programir-contract/spec/trust.md` (the ladder for what stays code).**

## What an adapter IS

An adapter is a **signature-independent inference-time language** for
rendering and parsing. The orthogonality that defines the whole
design:

- **Modules act on signatures.** A module (ReAct, CoT) is written
  against particular field structure.
- **Adapters act on LM families.** Chat/JSON/XML adapters apply to
  ANY signature — they don't care what the fields are. What they DO
  care about is the model class: today's adapters all assume
  *instruct* models. A **base model** (no post-training, no chat) needs
  different adapters entirely — few-shot pattern-completion templates,
  stop-sequence discipline — same signatures, same programs.

So the grid is: **programs × signatures** on one axis, **adapter ×
LM-family** on the other. The same program must run unchanged across
adapter choices; the adapter choice is a function of the model, the
token budget, and the inference strategy — never of the program.

## The three customization layers (all research surfaces)

**1. Template — the whole-exchange form.** Token-efficient adapters
for cost; dead-simple adapters for small/weak models; base-model
completion templates; anything in between. A different template
implies a different parser (see the lens principle in
adapter-parse-dsl.md).

**2. Strategies — per-semantic-role inference strategies.** Given a
semantic element in the signature (reasoning, tools, citations,
media), a strategy chooses HOW the exchange conducts it — same
signature, same program, different inference behavior:

- *Reasoning*: native provider thinking; OR a classic prefix-CoT
  section; OR **interleaved thinking — a thinking tag after every
  sentence**, achieved by *instructing* the model to write that way
  (pure prompt + parse data; works today on any instruct model), with
  engine-side token-trigger injection as an optional *enforced*
  variant where the engine offers the hook. Three-plus strategies,
  one `reasoning` role.
- *Tools*: native JSON function calling; OR CLI/bash-style invocation
  parsed from text; OR XML tool blocks. The tool-calling *format* is a
  strategy, not a program property.
- *Citations*: native provider channel; OR inline markers; etc.
- *Media* (future): e.g. image-quality/fidelity levels as a strategy
  over image-typed fields.

Strategies MUST be customizable — this is a rich avenue of research
(strategy search is a named optimizer axis). dspy ships builtins; the
registry admits custom ones; the IR must carry the choice.

**3. Codecs — the type boundary.** The Python-level signature says
`PIL.Image` or `duckdb.Database`; the codec owns the crossing:
a base64 image from an image-capable model *materializes as the PIL
object the signature promised*; a DuckDB database *renders into
whatever LM-request component shows it to the model*. Structured
output belongs here too, and blurs into strategy: "return JSON I
validate" vs **"return Python code an interpreter evals into the
object"** are two codec/strategy hybrids for the same typed field.

**The two-layer rule (Maxime's cross-language test, 2026-08-10):**
"data-only" means *the thing itself crosses languages* — so a codec is
data only when it is expressed against the **neutral shape
vocabulary** (Epic B shapes = the Adapter IR's type system: image,
audio, text, the wire encodings, with pinned decode semantics), and
the host type (`PIL.Image` vs `image.Image` vs `Blob`) is a
**per-frontend binding**, never IR content. `PIL.Image` lowers at
export to `(shape: image, wire: base64/png)` + a Python-binding
annotation; every runtime materializes its native type (the
Arrow/protobuf/i64-vs-BigInt pattern). The criterion doubles as the
cleanest data/code boundary: **expressible against the neutral shape
vocabulary → data; no neutral shape (arbitrary host classes, compute
at the boundary) → code on the trust ladder (authored codec or
declared leaf).**

## The Adapter IR goal

Everything above should be expressible **as data, in a baked,
multi-language representation — an Adapter IR with the same standing
as the ProgramIR** — NOT as "point at the dspy Python and run it".
`ADAPTER_IR_VERSION` (0.2.0) and the entry serde are the seed; the
gap is that strategy *behavior*, parse behavior, and codec behavior
are still name-references to Python. Requirements:

- a shipped program using **customized** adapters travels: the
  customization goes IN the artifact (data levels) or resolves via
  the trust ladder (code levels) — never "install my repo first";
- Go/TS/other runtimes render and parse from the Adapter IR alone;
- the ProgramIR points into it the way it points into the node set.

## What being data requires (the design consequences)

- **Strategy-as-data** is a rule with four faces, all data:
  `{predicate over declared LM-capability facts, render fragments
  (template-language), engine controls (request-side data: stop
  sequences, logit bias, grammar/structured-decode spec, token-trigger
  injection points — the LMRequestPatch generalized), parse routings
  (channel → field, coercion)}`. Most strategies need only fragments
  + routings (+ a mild predicate like "instruct model") —
  interleaved thinking is instruction fragments + a parse routing on
  any instruct model. Engine controls are the *optional* face for
  enforced variants, gated by a capability predicate only when used.
- **LM-capability vocabulary** becomes load-bearing data: instruct vs
  base; native reasoning/citations/FC; engine hooks (trigger
  injection, grammar decode). Artifacts refuse loudly on engines
  lacking a required capability — same versioned-vocabulary growth
  discipline as the node set.
- **Parser-as-data**: template-as-lens + combinators
  (adapter-parse-dsl.md levels 0–1).
- **Codec-as-data**: syntax families + options + the shapes vocabulary
  (Epic B) for media; the eval-Python structured-output strategy is a
  reference to a **sandboxed interpreter leaf** — already representable
  in the ProgramIR/trust vocabulary.
- **The origin collapse** (adapter-parse-dsl.md): data-level
  customization carries no trust question, only vocabulary-version
  compatibility; the trust ladder guards only the code tails.

## Portability is a requirements gradient, not simplicity (Maxime, 2026-08-10)

"Portable" is never a binary and never a virtue of simplicity: it is
the **declared requirement set** of the author's decisions. Simple
choices → tiny requirement set → runs pure-Go anywhere. An authored
Python strategy → the set includes "python≥3.x sidecar for component
X" — a legitimate portability statement, refused by receivers who
don't meet it, *naming the requirement*, never moralized as
"unportable." Consequence for **D-026 (amendment pending)**: authored
adapter code stops being flat-refused and instead **declares its
requirements like every other component** — language env block
(D-025), isolation floor (pairing rule), hence a placement, hence
D-022 rung-walkability. Data-level entries still carry no placement —
data needs none; templates/parse-data are the *zero-requirement floor*
of the gradient, not the definition of portability. Target error
shape: "requires python≥3.12 sidecar for `4_adapter/my_parser` —
unbound; refuse or bind one."

## Status honesty

Current `dspy/adapters/` is a waypoint: the render template is data;
parsers are a closed enum of four; strategies and codecs are
registered Python behind name-refs; `applies()` reads live LM objects
instead of declared capability facts; engine controls exist only as
the native-FC request patch. Directionally right, deliberately
unfinished. The census (adapter-parse-dsl.md) sizes the vocabularies;
the Adapter IR extension is its own design round — likely its own
epic once the exporter arc closes.
