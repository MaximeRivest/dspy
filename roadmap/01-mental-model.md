# The Mental Model

The taxonomy you must hold to work on this codebase. Read after `00-constitution.md`; deep rationale in `IR-program-spec.md`.

## The thesis — a superset orchestrator (Maxime, 2026-08-10)

DSPy's product is not a Python library. It is a **superset orchestrator
of programming languages**: *I don't care what your language, types,
control-flow syntax, or interpreter are — wherever you are, you will
orchestrate AI components, and that orchestration alone is what I
standardize.* Three IRs carry it, each defined by what it refuses to
own:

- **Program IR** owns orchestration; refuses to own your language
  (idiomatic `forward`; only the closed skeleton is extracted).
- **Adapter IR** owns the inference-time exchange; refuses to own your
  types (host types bind per-frontend; neutral shapes cross the wire —
  the two-layer rule, `adapter-north-star.md`).
- **LM IR** owns the model contract; refuses to own the model: a
  leaf's interface is a signature, the LLM is *today's
  implementation*. Optimizers may replace an LLM call with a trained
  classifier, feature-extraction+ML, or generated code — the program
  never said "LLM"; it said "this signature, this metric" (the
  unified-leaf section below is this thesis in miniature). AI writes
  the tools and programs too; optimizers rewrite implementations; all
  of it over the same three IRs. LLMs are the bootstrap, not the
  commitment.

Downstream consequences are bindings, not architecture: engine
rewrite, new optimizers, many frontends in many languages, mixed
interpreters/sandboxes/permissions, fully distributed or fully
in-memory — all placement and binding choices over unchanged programs.

**The boundary rule** (resolving idiomatic-types vs shippability):
*idiomatic inside, neutral at the edges, and the cost of a non-neutral
edge is local, priced, and stated.* Back-and-forth of host-language
objects between the program and the AI world is a core power and is
never restricted. A language-specific type at an artifact boundary
costs a shim/codec on the receiver's side OR a sidecar call back into
the origin language (the D-022 rung-walk) — per-boundary, never
program-wide. The composition report states residues as facts
("portable except field `db` requires Python at leaf `render_db`");
the receiver decides. Graceful degradation, same shape as trust
profiles.

## Intent vs mechanism

Every piece of the system is one or the other, and the line between them is the design's load-bearing wall:

- **Intent** — what the user asked for. The signature: field names, **shapes** (data types, arbitrary pydantic/JSON-schema), **semantic roles**, instructions. Frozen; defines the task.
- **Mechanism** — how the system delivers it. Everything below the signature. Swappable, optimizable, recorded.

The swap test: if you can change it and the task hasn't changed (same signature, same metric), it's mechanism.

## The two jobs adapters were doing

Historically "adapter" meant two jobs fused: **representation** (deciding what the model sees and how to recover outputs) and **orchestration** (moving a call along — retries, fallbacks, second calls). The whole architecture is the surgical separation of these. Representation became data (plans, formats, strategies, codecs). Orchestration became tree structure (the forward AST, lowerings). Every historical bug traced in this project was a piece of one job serialized in the other's format.

## The four mechanism layers

Each scoped one level down; stratified by the single-shot law (L2 — a strategy may never add a call; a format may never touch structure; a strategy may never own format literals):

| layer | operates on | scope | example choice |
|---|---|---|---|
| **lowering** | the tree | across exchanges | TwoStep, retry, parse-fallback |
| **format** | the whole exchange | one exchange | chat markers vs JSON body vs XML |
| **strategy** | one field's role | within one exchange | reasoning native-channel vs textual; tools native-FC vs textual-JSON |
| **codec** | one value's shape | one value on the wire | BAML schema-prose vs JSON vs Python literal |

## Roles and shapes

A signature field splits into **shape** (what the value *is* — plain data, any JSON-schema-able Python type) and **semantic role** (what the field *means to the inference*). The closed role vocabulary — `plain, reasoning, tools, tool_calls, citations, history, media, code` — is the field-level projection of the typed LM contract (lm15's part types confirm it independently). A role exists iff it admits ≥2 materially different strategies, one escaping the token stream. Textual strategies are **polyfills**; history has channelized three of them (CoT → native reasoning, ReAct → function calling, quote-prompting → native citations).

Authoring: `Annotated[str, citations]` is canonical; `citations[str]`, `"answer: str @citations"`, and `OutputField(role="citations")` are sugars over the same registry object. Legacy types are the fused spelling (`Reasoning` ≡ `Annotated[str, reasoning]`).

## Plans, pools, bindings

- **AdapterPlan** (`dspy/adapters/_engine/ir.py`) — the inspectable artifact between a signature call and the wire request: request slots, the RenderField layer (model-facing names decoupled from semantic names; hiding is a rendering decision, never a semantic deletion), parsers, provenance. Planning is pure — no LM call.
- **Pools** — identity-bearing shareable components (adapters, LMs, tools, strategies, codecs) are named entries declared once.
- **Bindings** — each predictor leaf names its entries. The tree is the use-site table; pools are the symbol table; **load is the link step**. Sharing is a fact the artifact states, not a coincidence readers infer.

## Program makers, lowerings, the core tree

```
signature ──program makers──▶ SURFACE TREE ──lowerings──▶ CORE TREE ──formats──▶ wire
```

Program makers originate structure; lowerings rewrite tree (expand sugar into leaves, bindings, tags — provenance `lowered_from`); the **core tree** (only typed leaves, resolved bindings, explicit tags) is the only serialized form. Forward is restricted-Python AST over a closed node whitelist; every `Call` resolves to a typed leaf: Predict, sub-module, tool/UDF, interpreter.

## The unified leaf

Every leaf's interface is a signature; what varies is the implementation: LM+adapter+bindings, code, interpreter, or sub-module. Predict⇄UDF swaps, distillation (model→code), and blooming (code→module) are all the same mutation: implementation choice.

## Seeds and regimes

User code opened to search declares its objective at tag time: **frozen** (default), **semantics-fixed** (seed is the spec; fidelity+cost objective; code→code only), **metric-driven** (seed is a hint; program metric; the frontier tier, strictest gates).

## Exhaust

Anything generated while answering that isn't a declared output. Goes to `Prediction._trajectory` (instance attr, never in `_store`, never serialized). Dot-access shims it with a DeprecationWarning; bracket access is the strict contract view and does not.

## The misfiling table — historical failure modes

Everything below lived exactly one layer below its true home. When filing something new, check this table first:

| what | was filed as | actually is |
|---|---|---|
| TwoStep | an adapter | a **lowering** (its parse made a second LM call — a smuggled binding) |
| `Reasoning`'s provider hacks | a semantic type | a **strategy** trapped in a type |
| Chat→JSON parse fallback | adapter control flow | an error-policy **lowering** |
| BAMLAdapter | a format | a **codec preference** riding the chat format |
| ambient `settings.adapter`/`lm` | a feature | the absence of **bindings** (one-slot state can't express per-predictor programs) |
| CoT's `reasoning` output | contract | **exhaust** (unless the user declares the role) |
| MultiChainComparison | (candidate lowering) | an honest **aggregator** — it adds an external input, which no lowering may do |
| StreamListener marker-matching | streaming infra | the token-space **polyfill** of role-typed deltas |
