# What can go in a `forward`? How we chose the exportable subset

DSPy 4.0 can save an entire program — signatures, instructions, demos,
adapters, tools, even model weights — as one portable artifact that loads
and runs anywhere, in any language, with no reach into your Python
environment. For that to work, your module's `forward` method has to
travel too. This page explains which Python constructs an exportable
`forward` may use, how we decided, and what to do when something you
wrote is refused.

## Why there is a subset at all

When you export a program, `forward` stops being "whatever Python does"
and becomes **data**: a small, well-defined tree that any conforming
runtime — Python, Go, TypeScript, or one that doesn't exist yet — can
execute identically. That guarantee is only possible if the language of
`forward` is closed: every construct in it has written-down semantics
and a test suite that all runtimes must pass.

The alternative — shipping pickled Python — is how programs stop being
portable, reproducible, or safe to load. We refuse that trade. So the
question became: *which* subset?

## The three-way rule

Every Python construct lands in exactly one bucket:

1. **Admit** — it gets its own node in the exported tree, with semantics
   pinned in about five lines that every language can implement
   identically. `if`, `for`, `while`, `try`, f-strings, lists, dicts,
   indexing, slicing, comparisons, arithmetic.
2. **Desugar** — it's convenience syntax over admitted semantics, so the
   exporter rewrites it at export time and runtimes never see it.
   Comprehensions become loops; `a, b = pair` becomes a destructuring
   node; `x += 1` becomes `x = x + 1`; `is None` becomes `== None`.
   You write natural Python; the artifact stays small and closed.
3. **Refuse, loudly, by name** — the exporter tells you exactly which
   construct, where, and why. Never a silent partial export.

The test for admission is honest and strict: a construct is admitted
only if its semantics are (a) small enough to pin exactly, and (b)
implementable identically everywhere. `mod` (`%`) on negative numbers
means different things in Python, Go, and JavaScript — so it's refused
rather than specified wrongly for someone. Dict ordering *is* pinned
(insertion order, observable), which costs Go an ordered-map type — we
paid that cost because your code genuinely depends on it.

## Chosen by evidence, not taste

We did not sit in a room guessing what people write. We ran two censuses
and admitted what the data said:

- **Every shipped DSPy module** — Predict, ChainOfThought, ReAct (v1
  and v2), ProgramOfThought, CodeAct, RLM, Avatar, MultiChainComparison,
  KNN, Parallel, aggregation — every AST node and call target in their
  `forward` methods and helpers.
- **389 community-written `forward` methods** from 288 files across ~75
  public projects that depend on DSPy — the code people actually write,
  not the code we imagine.

The results shaped the subset directly. 46% of community forwards
needed nothing beyond plain assignments, calls, ifs, and loops. The
rest overwhelmingly needed the same short list: f-strings, dict/list
literals, indexing (`trajectory[key] = v`), slicing (`passages[:k]`),
and light string/list/dict methods (`.strip()`, `.append()`, `.get()`,
`.join()`). All of that is in. What almost nobody used — set literals,
bitwise ops, `zip`, walrus-heavy tricks — is desugared, deferred, or
refused. Every admission in the spec cites its census count.

## What's supported

Control flow and state: `if`/`elif`/`else`, `for` (over ranges and
lists), `while` (with an iteration cap so a loaded program can't spin
forever), `break`/`continue`, `try`/`except`/`raise` with a typed error
table, `return`, assignment, tuple destructuring, subscript writes.

Values: strings (proper Unicode), exact 64-bit ints, floats (no
NaN/Inf — arithmetic that would produce them raises a typed, catchable
error instead of poisoning your data), booleans, `None`, lists and
insertion-ordered dicts with defined aliasing (mutation through one
variable is visible through another, everywhere, identically).

Expressions: f-strings, arithmetic (`+ - * /` with overflow *detected*,
never wrapped), comparisons and orderings, `and`/`or`/`not`,
membership (`in`), conditional expressions, indexing and slicing.

Calls: every call resolves to something *declared* — a Predict, a
sub-module, a tool, an interpreter — or to a small table of pure
builtins (`len`, `str`, `sorted`, `max`, `json_dumps`, …) and value
methods (`.strip()`, `.split()`, `.join()`, `.append()`, `.get()`,
`.items()`, …), each with pinned cross-language semantics.

## What's refused, and what to do instead

Refusals are design positions, not gaps. Each names its reason:

- **`with dspy.context(...)`** — ambient state is the thing exportable
  programs exist to eliminate; a program whose behavior depends on an
  invisible context can't make portable claims about itself. Bind
  models and adapters explicitly instead — per-predictor bindings are
  first-class in 4.0.
- **`import` inside `forward`** — an undeclared dependency is an
  undeclared leaf. Move the code into a tool with a `# deps:` line; the
  artifact then carries and reconstructs it.
- **`lambda`** — closures smuggle hidden state. Almost every census hit
  was a `sorted` key; `sorted` is built in, and anything fancier is an
  honest tool.
- **`async`/`await`** — scheduling belongs to the runtime, not the
  program. Your forward says *what* calls happen; engines batch and
  parallelize underneath without your code changing.
- **`print`** — becomes a `log` that writes to the run's observability
  trace, not to program state. (Outputs you actually want are signature
  fields — declare them.)

**The escape hatch is always the same, and it's a good one:** a leaf's
*body* is full, unrestricted Python. The subset governs orchestration —
the glue between calls. Anything heavier (numerics, regex surgery, an
ODE solver, pandas) belongs in a tool leaf with a typed contract, and
travels with the artifact as introspectable source plus declared
dependencies. Restriction is for the tree; freedom lives at the leaves.

## The promise, and how to change the subset

The subset is versioned. An artifact states the version it was written
against; a runtime states the versions it supports; mismatches refuse
loudly instead of guessing. Additions go through the same pipeline every
time: evidence that real code needs it → proposed semantics small enough
to pin → ratification → conformance fixtures → every runtime implements
and passes. That pipeline is public, and census-style evidence from your
own codebase is exactly what moves it. If the subset refuses something
you write all the time, that's a data point we want — open an issue with
the construct and the use, and it enters the next census round.

We built it this way so that a DSPy program can make a promise no prompt
ever could: *what you measured is what you shipped, and it will do the
same thing on someone else's machine.* The subset is the price of that
promise, and we kept the price as low as the evidence allowed.
