## The idea

Make a DSPy program **fully exportable as one portable artifact**: signatures, instructions, demos, adapter configuration, tools — and the module's `forward` itself. Today everything *except* `forward` can be captured as data; the control flow is the missing piece, so a saved program is state without a program, and loading it means re-importing the author's Python.

The proposal: capture `forward` as a **small, closed graph** — real control-flow nodes (`if`, `for`, `while`, `try`), assignments threading values, and calls that always resolve to a *declared* leaf (a Predict, a sub-module, a tool, a code interpreter). Not a trace (traces lose data-dependent branches), but the actual AST, accepted node-by-node. Your `forward` stays normal Python that runs unmodified; export just checks it against the supported set and serializes it.

## Why

- **What you measured is what you ship.** A program whose score was evaluated on one machine reproduces behaviorally on another — including its branching and loops — with zero reach into the receiver's ambient environment. No pickle, no "install my repo first".
- **Programs become inspectable and optimizable objects.** An optimizer (or a human) can see the whole structure — which predictor is called where, under which branch, in which loop — and rewrite it as data, with every change recorded and scoreable. This is the same move Spark/DuckDB/JAX made: once the program is a structured object, explain/profile/optimize come almost for free.
- **The framework stops being the boundary.** A graph with written-down semantics can be validated, diffed, and executed by lightweight runtimes in other languages (we have working experimental readers in Python, Go, and TypeScript sharing one conformance test corpus). DSPy-Python remains the reference; the artifact becomes the interface.

## The selection rule

Every Python construct lands in exactly one bucket:

1. **Supported** — it becomes a node with semantics you can pin in ~5 lines, implementable identically across languages.
2. **Rewritten at export** — convenience syntax over supported semantics: comprehensions become loops, `a, b = pair` becomes destructuring, `x += 1` becomes `x = x + 1`, `is None` becomes `== None`. You keep writing natural Python; the graph stays small.
3. **Refused, loudly, by name** — the exporter tells you the construct, the line, and the reason, plus the supported way to express the same thing. Never a silent partial export.

The bar for "supported" is deliberately strict: exact, portable semantics or nothing. Two examples of it cutting both ways: `%` (mod) is refused because sign semantics differ across Python/Go/JS — refusing beats specifying it wrongly for someone. Dict insertion ordering *is* guaranteed (observable, like Python) even though it costs non-Python runtimes an ordered-map type — real code depends on it, so we pay.

Other hard guarantees in the same spirit: exact 64-bit integers (overflow is a typed error, never a silent wrap — JS runtimes must use BigInt), no NaN/Inf (arithmetic that would produce them raises a catchable typed error instead of poisoning data), strings as well-formed Unicode, containers as reference values with defined aliasing.

## Chosen by data, not taste

To decide what goes in, we censused what forwards actually contain:

1. **Every shipped DSPy predict module** (Predict, ChainOfThought, ReAct, ProgramOfThought, CodeAct, RLM, Avatar, MultiChainComparison, KNN, Parallel, aggregation) — every AST node and call target in `forward` and its helpers.
2. **389 community-written `Module.forward` methods** from 288 files across ~75 public GitHub repos depending on DSPy.

What the data says:

- **46% of community forwards** (177/389) use nothing beyond plain assignments, calls, ifs, and loops.
- The rest overwhelmingly need one short list: f-strings (737 occurrences), subscripts (610), list/dict literals (339), destructuring (199), comprehensions (102), boolean logic, slicing like `passages[:k]` (42), and light value methods — `.strip()` (100), `.append()` (90), `.get()` (80), `.join()` (60), `.split()` (39).
- Only **~10% of forwards** touch anything we'd refuse — and mostly incidentally (a lambda sort key, an import inside forward), each with a mechanical rewrite.

## What would be supported

**Statements:** assignment, tuple destructuring, subscript writes (`trajectory[key] = v`), `return`, `if`/`elif`/`else`, `for` over ranges and lists, `while` (with an iteration cap, so a loaded artifact can't spin its host forever), `break`/`continue`, `try`/`except`/`raise` over a typed error table.

**Expressions:** f-strings, list/dict literals, indexing and slicing, `and`/`or`/`not`, conditional expressions, comparisons (mathematically exact numeric ordering; code-point string ordering), membership (`in`), `+ - * /` arithmetic.

**Calls — the leaf rule, the heart of the design:** every call resolves to something declared. Four leaf kinds: a **Predict**, a **sub-module** (this is how programs nest), a **tool**, or a **code interpreter** (how PoT/CodeAct/RLM-style modules run model-generated code). Plus a small table of pure builtins (`len`, `str`, `sorted`, `max`, `sum`, `any`/`all`, JSON encode/decode, …) and common value methods (`.strip`, `.split`, `.join`, `.replace`, `.append`, `.extend`, `.get`, `.items`, `.update`, …) — each with pinned cross-language semantics, each backed by census counts.

**Async is supported: `aforward` exports to the same graph as `forward`.** `async def aforward` with `await leaf.acall(...)` is the async mirror of sequential code — same calls, same order, just non-blocking — so it lands in the *rewritten* bucket: sync and async twins lower to the **identical graph**, and the engine decides whether execution is sync, async, batched, or parallel. Your async program exports fine and loses nothing. The census backs this: in the community corpus's async forwards, essentially every `await` wraps a plain `.acall(...)`. (What's *not* covered is deliberate concurrency — see the refused list.)

**And the escape hatch that makes the restriction livable:** a leaf's *body* is full, unrestricted Python. It travels in the artifact as introspectable source with declared dependencies. The subset governs *orchestration* — the glue between calls. Numerics, regex surgery, pandas, an ODE solver: that's a tool with a typed contract, not a reason to grow the graph.

## What would be refused, and why

These are design positions, not coverage gaps — each names its reason and its replacement:

- **`with dspy.context(...)`** — ambient state is precisely what a portable artifact cannot carry: a program whose behavior depends on invisible context can't make portable claims about its own scores. Replacement: explicit per-predictor model/adapter bindings.
- **`import` inside `forward`** — an undeclared dependency is an undeclared leaf. Replacement: a tool with declared deps; the artifact carries and reconstructs it.
- **`lambda` / closures** — hidden state smuggled past the graph. Nearly all observed uses were `sorted` keys; `sorted` is built in, anything fancier is an honest tool.
- **Explicit concurrency** (`asyncio.gather`, tasks, locks — *not* plain `async`/`await`, which is supported; see above) — genuinely new semantics: join order, error propagation, races against the aliasing guarantees. And it's rare where it would live: exactly **one** `gather` in all 389 community forwards, because engines already batch and parallelize underneath. Stays out for now; if evidence demands it, it arrives as an explicit fan-out node with pinned join/error semantics, never as raw event-loop plumbing.
- **`print`** — becomes a log into the run's observability trace, never program state. Output you actually want is a signature field: declare it.
- **Runtime class definition and opaque callables** (the current Refine/BestOfN internals) — not expressible as declared structure. The *concept* ("try N, keep the best by a metric") remains expressible as a metric leaf plus an ordinary loop.
- **Small numeric edges** (`%`, floor-div, set literals, bitwise ops) — cross-language semantics diverge or evidence is ~zero; refused rather than mis-specified.

## Versioning

The supported set is versioned. An artifact states the version it was written against; a runtime states what it supports; mismatches refuse loudly instead of guessing. The set only grows through the same pipeline: evidence that real code needs it → semantics small enough to pin → conformance fixtures → every runtime passes.

## Questions for the community

1. Does the supported/rewritten/refused split — and especially the leaf rule — match how you write forwards? What would your code hit first?
2. Anything in the supported tables that looks wrong, missing, or over-included? Every entry has census counts behind it; happy to share methodology and data.
3. What do you write *constantly* that we should census next? Real repos and snippets are exactly the evidence that moves the set.
4. Reactions to the deliberate refusals (`dspy.context`, lambda, explicit concurrency) and their proposed replacements? In particular: is graph-identical export of `aforward` (with concurrency left to the engine) enough for your async use, or do you write deliberate `gather`-style fan-out inside forwards?
