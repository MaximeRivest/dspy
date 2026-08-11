# GREENFIELD — dspy rebuilt as if the IRs existed from day one

**Branch `greenfield-ir`, worktree `/home/maxime/Projects/dspy-greenfield`.
Experimental. NO backward compat, NO legacy. Maxime reviews in the
morning. Every builder agent reads this charter FIRST, then
`BUILD-STATE.md` (the running log each agent appends to).**

## The goal

Prove how nice dspy would be if ProgramIR + Adapter IR + LM bindings
were the foundation. Same dspy *syntax* (Signature, Module, Predict,
ChainOfThought, ReAct, RLM, optimizers) except the specced changes.
The program IS the IR; the engine IS the executor; adapters ARE data.

## Canon (read before building)

- `roadmap/01-mental-model.md` (thesis at top), `roadmap/adapter-north-star.md`,
  `roadmap/adapter-parse-dsl.md`, `roadmap/adapter-ir-stage/README.md`
  (+ its 12 examples — the adapter design to IMPLEMENT),
  `roadmap/IR-program-spec.md` (skim §b, §d, §e0).
- Contract: `/home/maxime/Projects/programir-contract/spec/node-set.md`
  (v0.3 semantics), `reference/interp.py`.
- Reusable code in THIS tree (adapt freely, delete what fights you):
  `dspy/programir/` (v0.3 compiler, engine, tools — keep),
  `dspy/adapters/_engine/template/` (template language — keep core),
  `dspy/signatures/` (authoring surface — simplify).

## The architecture (the specced changes)

1. **New package layout** — rebuild `dspy/` in place on this branch:
   `dspy/core/` (signature, prediction+trajectory, typed errors),
   `dspy/lm/` (ONE `LM` class, litellm-backed, plus `DummyLM`;
   `dspy.configure(lm=...)` sets an explicit DEFAULT BINDING TABLE —
   a plain dict, no thread-local magic, no `dspy.context`),
   `dspy/adapters/` (v2, per the adapter-ir-stage design),
   `dspy/programir/` (compiler+engine+tools, kept and promoted),
   `dspy/modules/` (Predict, ChainOfThought, ReAct, ReActV2, RLM),
   `dspy/optim/` (ported optimizers), `tests_greenfield/`.
   Delete from this branch what the new layout replaces (old clients/,
   predict/, retrieve/, streaming, callbacks, teleprompt/) — git
   history preserves them; this branch is the clean world.
2. **The program is the IR.** `Module.__init_subclass__`/export
   compiles `forward` (v0.3 subset — forwards MUST compile clean;
   no self-attr reads in forward, declared leaves only).
   **AMENDED by D-041 (2026-08-11, ratified): node-set 0.4 adds the
   inputs-bag** — `def forward(self, inputs)` bound to the module's
   own signature + signature-record splat into leaf calls (LM-decided
   kwargs stay refused). Once the 0.4 propagation reaches this tree,
   ReAct/ReActV2/RLM are REWRITTEN as hand-written forwards and the
   generated-forward machinery is deleted. Program makers stay
   CLASSES (makers-as-functions deferred by Maxime's ruling). `program(**inputs)` executes THROUGH the engine
   interpreter by default. `program.save(path)` = the artifact;
   `dspy.load(path, bindings=...)` = read+link+materialize. One path.
3. **Adapters v2 = the adapter-ir-stage design, implemented**: parser
   as `{"kind":"lens"}` derived from the template + `{"kind":
   "pipeline"}` combinators (alternatives, fenced_block, json_object,
   fields_from_object, regex/fields_from_groups, coerce); strategies
   as rule objects (predicate on declared LM capability facts,
   fragments, routings, request patches) — reasoning native/prefix/
   interleaved AND tools native-FC/CLI/XML actually working; codecs as
   families (text_pythonish, pydantic_json, schema_prose) + shape-level
   media (image base64↔PIL at the frontend binding). Entries dump/load
   as the extended shape; `preview()` and `parse_preview()` pure.
4. **LM capability facts are declared data** on the LM binding
   (instruct/base, native_reasoning, native_fc, native_citations) —
   strategies predicate on them, never on live-object sniffing.
5. **Typed errors everywhere** (the contract error table); teaching
   errors in the compiler voice; loud refusal, never silent fallback.
6. **Modules**: ReAct, ReActV2, RLM rewritten with subset-clean
   forwards (the shipped versions' refusal list in the v0.3 commit
   message is the checklist of what to avoid); tools are declared
   leaves; RLM's interpreter is a declared interpreter leaf.
7. **Optimizers as IR mutations** (`dspy/optim/`): LabeledFewShot,
   BootstrapFewShot, and RandomSearch — propose = mutate the IR
   (demos/instructions as data), score = engine replay over a devset,
   keep = the artifact. checkpoint == save. No settings reads.
8. **Observability**: `program.explain()`, `program.lint()`,
   `program.cost()`, `diff(p1, p2)` wired from programir/tools.

## Working rules for builder agents

- Commit locally on `greenfield-ir` as you go (small commits, repo
  style, no AI attribution). NEVER push. NEVER touch files outside
  this worktree.
- Tests: local pytest inside this worktree only (`uv run pytest
  tests_greenfield/ -q`). Keep it green at every commit. The old
  `tests/` tree is not the target and may break — that is fine; do
  not run it.
- Append a dated section to `BUILD-STATE.md` before finishing: what
  you built, what works (test count), decisions taken, what the next
  agent must know.
- When the charter is ambiguous: pick the option truest to the
  north-star docs, note it in BUILD-STATE.md, keep moving.

## Build order (one agent per stage)

A1 core: package layout + core/ + lm/ + bindings + errors (+ carve
   out old dirs). A2 adapters v2. A3 program-is-the-IR execution
   spine. A4 modules (ReAct/ReActV2/RLM). A5 optimizers. A6
   integration: end-to-end demo (`examples_greenfield/demo.py`, all
   DummyLM), README-GREENFIELD.md tour, final green run.
