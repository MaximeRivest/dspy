# Adapter IR — staged review package

**Status: handwritten target artifacts, 2026-08-10. Nothing here
executes.** Each example shows what dspy SHOULD someday produce: an
idealized authoring surface (`authoring.py`), the entry it would dump
(`adapter-ir.json`), and the reasoning (`notes.md`). The purpose is to
co-design the extended entry shape by staring at concrete instances,
before any vocabulary is fixed (the census gate still applies).

Canon this extends: `adapter-north-star.md` (intent),
`adapter-parse-dsl.md` (parse mechanics), `adapter-ir-spec.md` (the
ratified 0.2.0 entry), `01-mental-model.md` (layer law). Every entry
here is a recognizable extension of the ratified ENTRY_KEYS —
`name, adapter_ir_version, versions, template, parser, codecs,
strategies, config` — plus at most one new key (`requires`).

## a. What the Adapter IR is, and the end goal

An adapter is a signature-independent inference-time language for
rendering and parsing. Modules act on signatures; adapters act on LM
families. The grid is programs x signatures on one axis, adapter x
LM-family on the other, and the same program must run unchanged across
adapter choices.

The goal is that ALL THREE customization layers — template (the
whole-exchange form), strategies (per-semantic-role conduct), codecs
(the type boundary) — are expressible as data in one baked,
multi-language representation with the same standing as the ProgramIR.
Today only the template is there. The gap: the parser is a closed enum
of four Python programs; strategies and codecs are name-references
into registered Python; capabilities are read off live LM objects.

The organizing principle throughout is the **origin collapse**:
components pushed into data-only form exit the security problem
entirely — the load question becomes "do you speak vocabulary version
N", never "do you trust the author". The trust ladder (language block,
isolation floor, identity, placement) guards only the honest code
tails, which declare their requirements like every other component
(the requirements gradient — never moralized as "unportable").

## b. The proposed extended entry shape

Unchanged: `name`, `adapter_ir_version` (bumped to `0.3.0-draft`),
`versions`, `template`, `config`. Extended keys below; every new
spelling is also flagged `PROPOSED:` in the example that forced it.

### `parser` — from enum to three forms

Rationale: the parser is the maximum-exposure component; making it
data removes the most-attacked class from the code surface, and the
template already pins the layout the builtin parsers invert.

- `{"kind": "lens", "of": "template"}` — **level 0**: derived
  mechanically from the bound template (labels -> boundaries, slots ->
  captures, values -> the bound codec). No new vocabulary; the four
  builtin parser names become derivations of their own presets.
- `{"kind": "pipeline", "steps": [...]}` — **level 1**: declared
  combinators for the residue (tolerance, formats we did not render).
  Ops drafted here: `fenced_block`, `alternatives`, `json_object`,
  `fields_from_object`, `regex` (RE2 subset only), `fields_from_groups`,
  `tool_calls`, `citations`. Versioned as `versions.parse_combinators`.
- `{"kind": "authored", ...}` — **level 3**: baked source + language
  block + identity + `authored_by` + FORCED `isolation` (the pairing
  rule: parser exposure is total). A string parser name stays valid as
  the builtin spelling during migration.

### `strategies` — from names to rules

Rationale: the north star defines strategy-as-data as a rule with four
faces, all data. A role's value may now be the rule object instead of
a name:

```
{"kind": "rule",
 "predicate":       {"capability": "<fact>"} | {"all"/"any"/"not": ...},
 "hides":           ["field", ...],
 "transforms":      [{"rename": {"from": ..., "to": ...}}],
 "fragments":       [{"target": "system"|"user", "content": <template-language>}],
 "engine_controls": {"request_patch": {...}, "stop_sequences": [...], ...},
 "routings":        [{"channel": ..., "field": ..., "coerce": <shape>}
                     | {"text": <pipeline>, "field": ..., "consume": bool,
                        "materialize": {...}?}]}
```

Predicates test DECLARED LM-capability facts (`instruct`,
`completion`, `native_reasoning`, `native_function_calling`,
`native_citations`, `image_input` — `versions.lm_capabilities`), never
live objects. Most strategies need only fragments + routings;
engine controls are the optional enforced face. Parse pipelines inside
routings reuse the ONE parse-combinator vocabulary.

### `codecs` — three data forms plus the leaf door

Rationale: the two-layer rule. `codecs.input`/`output` stay name refs;
`codecs.per_field` (already in the spec) admits objects:

- `{"kind": "family", "family": ..., "options": ..., "parse_chain": ...}`
  — syntax family + options, the census's data form of codec bodies.
- `{"kind": "shape", "shape": ..., "wire": ..., "frontend_bindings": ...}`
  — the neutral-shape codec: host types (`PIL.Image`) are per-frontend
  bindings, never IR content. Versioned as `versions.shapes`.
- `{"kind": "leaf", "leaf": ..., "language": ..., "effects": ...,
  "placement": ..., "emits": ...}` — no neutral shape means compute at
  the boundary: the codec stays a data REFERENCE; the transformation
  is a declared leaf with effects and placement.

### `requires` — the declared requirement set (NEW key)

Rationale: portability is a requirements gradient. Data-only entries
omit it (the zero-requirement floor). Entries that need capabilities,
languages, or leaves state them:
`{"lm_capabilities": [...], "languages": [{name, requires, packages?,
for, binding, isolation_floor?}], "leaves": [...]}`. Refusals name the
requirement ("requires python>=3.12 sidecar for `.../parser` —
unbound; refuse or bind one"), never the author.

### `versions` — three candidate vocabularies

`parse_combinators`, `lm_capabilities`, `shapes` join the block, drawn
conditionally (present only when used) in these examples — see open
question 6.

## c. Example index

| # | example | what this one proves |
|---|---|---|
| 01 | chat-baseline | today's ChatAdapter with its parser as the derived lens — the bridge from 0.2.0, one changed key |
| 02 | json-tolerant | tolerance as ordered level-1 combinators: visible, diffable, optimizable |
| 03 | token-minimal | a whole custom adapter that is JUST a template edit; lens degenerates to full_text |
| 04 | base-model | the second LM-family column: few-shot completion template, stop-sequence discipline as engine-control data, declared `completion` capability |
| 05 | reasoning-three-ways | the strategy rule language: one role, three conducts (channel / fragments / fragments+regex routing), only `strategies.reasoning` differs |
| 06 | tools-three-ways | tool FORMAT as strategy: native FC request patch (`$from` splice) vs CLI-text vs XML, same program |
| 07 | citations-native-vs-inline | channel routing + field transform vs inline-marker combinator parse; `consume` earning its keep both ways |
| 08 | custom-regex-parser | authored parse-DATA: RE2 groups -> fields, zero trust machinery — the origin collapse in action |
| 09 | media-shapes | the two-layer rule: shape+wire in the IR, `PIL.Image` as a python frontend binding |
| 10 | eval-python-codec | the codec/strategy hybrid: python_literal codec + rule + sandboxed interpreter LEAF with isolation declared |
| 11 | authored-code-parser | the level-3 tail: language block, forced isolation floor, the named-requirement refusal a Go receiver emits |
| 12 | duckdb-boundary | no neutral shape -> codec stays a data reference, compute is a declared leaf with effects and placement |

Bracketing pairs worth reviewing together: 08 vs 11 (data-authored vs
code-authored parsing), 09 vs 12 (neutral shape vs no neutral shape),
02 vs 10 (same typed-output goal, zero-requirement vs
sandbox-requirement conduct).

## d. Open questions (the decisions this package exists to surface)

1. **Parser key polymorphism.** Is `parser: string | {kind: ...}` the
   right migration (strings = builtin names, objects = the three
   levels), or should 0.3.0 drop strings and re-express the four
   builtins as lens entries of their presets from day one? The second
   is cleaner; the first keeps every 0.2.0 entry loadable.

2. **Combinator spelling and size.** Are pipelines flat step lists
   with `alternatives` as the one nesting op (as drawn), or a full
   expression tree? And do typed terminals per role (`tool_calls`,
   `citations` — examples 06/07) scale, or should routings end in a
   generic `{"coerce": "<shape>"}` step so parse_combinators stays
   role-agnostic? This decides the vocabulary's growth curve; the
   census should arbitrate, but the SHAPE must be picked first.

3. **Predicate grammar.** Capability atoms + `all`/`any`/`not` — is
   boolean structure even wanted for v1, or are bare atoms enough
   (every example here needed exactly one atom)? And are capability
   FACTS pure booleans, or typed (e.g. `max_tokens`, context length)
   with comparisons — which would make the predicate a real
   expression language and needs a much stronger justification.

4. **Where engine controls live.** Example 04 puts adapter-level
   controls in `config.engine_controls` (legal today — config is
   open); rules carry their own `engine_controls` face. Should
   engine controls be a first-class entry key with a closed
   vocabulary instead — given they are exactly the "LMRequestPatch
   generalized" and will grow (grammar/structured-decode spec,
   token-trigger injection)? Open dict now vs closed vocabulary now is
   a real fork.

5. **Strategy rule faces.** The north star names four faces
   (predicate, fragments, engine controls, routings); the examples
   needed two more: `hides` (05/06/07) and `transforms` (07). Admit
   both as faces, fold renames into routings, or fold `hides` into
   `transforms`? Also: may fragment content use template-language
   slots (06 embeds `{field('tools')}`), and if so which SUBSET of the
   language do fragments speak?

6. **Versions-block growth.** Three new vocabularies
   (`parse_combinators`, `lm_capabilities`, `shapes`). Present only
   when the entry uses them (as drawn — smaller entries, harder
   validation: "missing or unused?") or always present (the current
   loader's strict symmetric check)? The current serde refuses BOTH
   unknown and missing vocabularies; conditional presence breaks that
   symmetry deliberately.

7. **`requires`: authored or derived?** In every example, `requires`
   is computable from the entry body (capabilities from predicates,
   languages from authored blocks, leaves from materialize/codec
   refs). Is it a real ENTRY_KEY (receivers check one place; risk of
   drift from the body) or a derived view like the literal table
   (never authored, computed at dump)? The literal-table precedent
   says derived.

8. **Codec-family versioning.** Codec families (`python_literal` new
   here) currently sit under one `versions.codecs` number. Do families
   version individually (a family is pinned render+repair+schema
   semantics — like combinators), or does the single codecs number
   cover all builtin families, with growth = minor bump? Same question
   hits `shapes` wire encodings.

9. **Does per-field/shape material belong in the ADAPTER entry at
   all?** Adapters are signature-independent by definition, yet
   `per_field` codecs (09, 10, 12) name signature fields. Options:
   (a) per_field lives in the adapter entry as drawn (simple, but the
   entry is now signature-coupled); (b) shape/leaf codecs live in the
   ProgramIR's per-predictor BINDING, and the adapter entry stays
   pure; (c) per_field keys are role- or shape-patterns, not field
   names. This is probably the deepest structural question in the
   package.

10. **Placement in adapter entries.** Spec section 9: adapter entries
    carry no placement and no credential, so authored adapter code
    never rung-walks. The leaf-codec rule (12) and the interpreter
    materialization (10) both want a placement/rung-walk. Resolution
    options: the leaf is NOT part of the adapter entry (it lives in
    the ProgramIR leaf table; the codec only references it — making 12
    consistent with the no-placement law), or the law relaxes for
    leaf-referencing codecs. The first reading is cleaner and D-022
    already covers it; the entries here draw the fault line on
    purpose.

11. **Is the lens the DEFAULT for authored templates?** Example 03
    omits `parser` at the authoring surface and gets the lens. Silent
    defaulting is convenient and usually right, but a template whose
    labels are ambiguous produces a weak lens silently — should
    `make_adapter` require `parser=` explicitly, or refuse only when
    lens derivation is ambiguous (the teaching-error road)?

12. **Round-trip gating for authored parse-data.** Codec registration
    runs the ADP-003 probe battery. Should authored level-1 pipelines
    paired with authored templates (08) be probe-gated at
    registration too — `parse(render(x)) == x` over schema-generated
    values — or is parse-data exempt because it is inspectable?
