# The frontend contract — authoring surfaces over the ProgramIR (D-030 draft)

**Status:** draft (2026-08-07), distilled from the exemplar iteration
(`roadmap/exemplar-program*.{py,ts,go}` — seven dialects of one program).
Library-neutral: it binds ANY frontend that emits ProgramIR artifacts —
the dspy class dialects, and **FunctAI, the intended first consumer**
(per adapter-ir contract §8, FunctAI is the planned first dspy-free
consumer; going FunctAI-first is that milestone arriving early, and it
requires the exporter/compiler to land extraction-ready, D-021 style).
Each numbered convention is D-030 ratification material.

## The dialects, and the recommendation

| exemplar | dialect | compiler needed |
|---|---|---|
| exemplar-program.py | native class (current dspy) | AST + whitelist |
| exemplar-program-sig.py | class + module signatures, no ambient | AST + whitelist |
| exemplar-program-fn.py | decorated functions | AST + whitelist |
| exemplar-program-plain.py | zero subclassing (dataclass/stub/closure) | AST + whitelist |
| **exemplar-program-graph.py** | **plain + declaration-site bindings + stages (RECOMMENDED)** | AST + whitelist |
| exemplar-program-flow.py | builder/combinator (pyspark-style) | **none** — constructs the tree directly |
| exemplar-program.{ts,go} | foreign surfaces | build-time AST / builder |

All emit the same artifact. The graph dialect is the bet for FunctAI.

## FC-1 · Leaf declaration and the census

A leaf is any name resolvable at compile time to a leaf-typed value —
Predict/predictor stub, module, plain function (tool), interpreter, LM.
Kind is inferred from the value's TYPE, never from a reserved container
name; the binding-site name (attribute or variable) becomes the tree /
pool-entry name. A dict-of-leaves attribute is the dynamic-dispatch
table (its keys are the census). A call to any unresolvable name refuses
loudly at compile. **A leaf reachable from two module bodies refuses
loudly naming both parents** (per-predictor state cannot be shared until
shared-predictor semantics exist).

## FC-2 · Signatures and shapes

- Predictor: typed stub — params = inputs, return annotation = outputs,
  docstring = instructions (component 3a).
- Module: its own annotations are its external signature (manifest
  module-signature proposal, contract spec/manifest.md 2026-08-07);
  every return path constructs the declared record; module docstrings
  describe the contract and are NEVER prompted.
- Shapes: dataclass / class-signature / TypedDict → JSON Schema.
- `Annotated[T, "string"]` = field description; `Annotated[T, marker]` =
  semantic role (D-011); both may coexist on one field. Legacy fused
  types (`dspy.Citations`) imply their role.
- OPEN: single-output naming (a bare `-> dict` return needs a field-name
  rule or a refusal). OPEN: per-element role nesting
  (`list[citations[str]]`) — component 2 has one role slot per field;
  propose refuse-at-lowering with "move the marker outward".

## FC-3 · Deps and effects (leaf self-containment)

Third-party deps are declared by `# deps:` comments INSIDE the leaf's
body, beside the import they justify; absence = stdlib-only. (Go needs
none — imports + go.mod are statically sufficient; TS uses the comment
only for real packages.) Effects: undeclared defaults conservative
(`network`); a leaf declares `pure` to earn caching/oracle rights.

## FC-4 · Bindings and the two ambient dialects

Bindings attach at declaration (`@predict(lm=..., adapter=...)`) or
post-hoc (`set_lm` / `set_adapter` — set_adapter is a required dspy
addition, epic I). Pool entries derive names from binding-site variable
names (OPEN: collision + anonymous-instance rules). Two dialects:

- **strict (FunctAI):** no ambient syntax exists; export refuses any
  unbound predictor naming its path. The export resolution step is a
  lookup + refusal.
- **compat (dspy):** `dspy.configure` ambient defaults resolved at
  export into explicit bindings (epic I's resolution rule).

## FC-5 · Demos, examples, data

Examples are plain dicts. `with_inputs` is retired: `input_keys[]`
(component 3b) derives from the signatures that already declare inputs.
Demos attach at the node (decorator kwarg) or post-hoc; both legal.

## FC-6 · The compile boundary

Forward bodies compile against the ratified node set (v0.1 today; the
v0.2 batch is PROPOSED) with the desugar table in the contract's
spec/node-set.md as the shared lowering law — independent frontends must
lower identical surface constructs to identical trees. Constructs
outside the set refuse with teaching errors naming the proposal that
would admit them. Leaf bodies (tools, LM classes, metrics) are
unconstrained native code. Builder frontends (flow dialect) bypass
compilation entirely — they may emit any ratified node the day it
ratifies.

## FC-7 · Resources

OPEN: lazy construction — importing a program file must not load
weights; proposal: LM/interpreter constructors are lazy by default,
materializing at first call or export.

## Ratification queue (D-030 batch)

FC-1 name/census rules · FC-2 single-output + per-element-role rulings ·
FC-3 deps grammar + effects default · FC-4 entry naming + the two
dialects · FC-5 input_keys derivation · FC-6 desugar-table adherence as
a conformance surface (frontend fixtures: same source → same tree) ·
FC-7 lazy resources · kwargs-splat and parameter-default encodings
(filed contract-side, spec/node-set.md open encoding questions).
