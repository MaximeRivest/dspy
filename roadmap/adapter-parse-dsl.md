# The adapter as a staged embedded DSL — parsing, codecs, strategies as data (design note, 2026-08-10)

**Status: design hypothesis (Maxime + assistant), pre-census. The
question: can rendering, parsing, codecs, AND strategies be expressed
as data for ~95–99% of useful cases, making the whole adapter a staged
embedded DSL with the render template's hygiene?** Working answer: yes
— template-as-lens + a combinator vocabulary for parsing; a
family/options/chain form for codecs; a rule language for strategies —
with an empirical census gate before any vocabulary is fixed, and a
trust ladder only for the surviving procedural tails.

**Organizing principle (the origin collapse):** for a data-only
component, the builtin/packaged/authored origin distinction is
security-irrelevant — authored data is as safe as builtin data,
because analysis replaces trust (total inspectability, zero
authority). Origin and the trust pairing rule govern only what remains
code. For data, the load question is compatibility, not trust:
"does this receiver speak vocabulary version N" — refused loudly as a
version mismatch, never as an authorship judgment. Consequence: every
component class pushed into data-only form exits the security problem
entirely, and embeds into ProgramIR artifacts with no origin/exec
concerns.

## The asymmetry

Epic D made the render side a closed, interpreted, no-eval data DSL
(template vocabulary, capacity, pure `preview()`). The parse side
stayed four hardcoded programs behind a closed enum (`chat`, `json`,
`xml`, `full_text`). "Custom parser" therefore currently means
"authored Python" — the gap D-026 papered over.

## The move: parser = template read backwards

Bidirectional-programming lineage (lenses/Boomerang; invertible syntax
descriptions). The builtin parsers are already secretly inverse
templates: `chat` inverts the chat preset's `[[ ## field ## ]]`
layout; `json`/`xml` mirror their presets; `full_text` is the
degenerate lens. Since the template already pins layout as data:

1. **Derived lens (level 0):** a parser derived mechanically from the
   bound template — labels → section boundaries, slots → captures,
   value parsing → the already-data-bound codec. Most wild custom
   adapters are layout tweaks; under the lens they become template
   edits, not parser classes.
2. **Declared parse-data (level 1):** a small combinator vocabulary
   for the residue (LLM sloppiness, formats we did not render),
   admitted under the node-set selection rule — supported / rewritten
   / refused, census-gated, ~5-line pinned semantics per combinator:
   ordered alternatives (try A else B), tolerance flags
   (case-insensitive labels, whitespace), fenced-code-block extract,
   **regex captures restricted to the RE2 subset** (cross-language
   identical + no backtracking bombs — a DoS guard for free),
   JSON-repair policy enum, truncation policy, strip/split.
3. **Packaged (level 2)** and **authored code (level 3):** the trust
   ladder, but ONLY for the honest code tail (bespoke recovery
   heuristics, stateful or grammar-coupled parsing). Per the origin
   collapse, levels 0–1 need no ladder at all — an authored level-1
   parser is just data, loadable and analyzable anywhere that speaks
   the vocabulary version. Level-3 authored parsers arrive via the
   pairing rule (spec/trust.md): forced isolation rung, since a
   parser's exposure is total by definition.

## The security argument (why data beats sandboxed code here)

A parser is the maximum-exposure component — raw LM output in, always
on the injection path. Parse-data has **total exposure, zero
authority**: no effects to lie about, nothing to exfiltrate with,
grade-1 analyzable, no sandbox needed. Making parsing data removes the
most-attacked component class from the code surface entirely. Receiver
support becomes a *compatibility* clause for levels 0–1 (vocabulary
version) and a *trust-profile* clause only for levels 2–3: a minimal
runtime may speak data-parsers only and loudly refuse code ones —
graceful degradation of customization, not a failure mode.

## Bonuses

- `preview()` gets its dual: `parse_preview(completion)` — pure,
  byte-testable, fixture-corpus-able like the render golden corpus.
- Parse tolerances/format choices become optimizer axes (View-3
  one-field mutations), and diffable.
- Level 0–1 parsers embed directly into ProgramIR artifacts with no
  origin/exec concerns — the "easier to embed" payoff.

## Gate before any vocabulary is fixed

**Census custom parsers, codecs, and strategies in the wild** (the
forward-study method): harvest `Adapter` subclasses overriding
`parse`/`parse_value`/`format*`, plus registered codec/strategy uses,
from public repos depending on dspy; classify each against levels 0–3
per class; size each vocabulary (parse combinators, codec families,
strategy rules) from counts, not taste. The 95–99% claim is a
hypothesis until this runs.

## Extension (same day): codecs and strategies decompose the same way

Verified in code (2026-08-10 night):

- **Codecs** (`_engine/codecs.py`): bodies are thin — render-syntax
  family (json-ish/guillemet/pydantic-indent/baml-schema-prose) +
  fixed parse-repair chain + schema-driven coercion where the schema
  already travels as data. Data form: `{render_family + options,
  parse_chain policy, placeholder/schema style}`. Residue: novel wire
  syntaxes (vocabulary growth) and exotic custom-type serialization.
- **Strategies** (`_engine/strategies/`): the produced `AdapterPatch`
  is already almost data; parser hooks are channel routings
  (citations hook = "channel `citations` → field X, coerce
  `Citations`" in three lines); `applies()` predicates are capability
  checks on declared LM facts. Data form — a rule language: *"when LM
  supports X: hide field F, route channel C → F, add request param
  P."* Residue: the native-FC tools step (real procedural wiring,
  litellm-coupled — Epic E touches it regardless).

**The origin-collapse principle (Maxime's):** for data-only
components, origin is security-irrelevant — authored data is as safe
as builtin data because *analysis replaces trust* (total
inspectability, zero authority). The trust ladder guards only
surviving procedural tails. What replaces trust for data is
**vocabulary-version compatibility** — refusals say "unknown
combinator/family", not "untrusted author". Caveat kept honest:
data cannot act but can persuade (template prompt-steering); the
guarantee is visibility (diff, `preview()` bytes, composition
report), not innocence.

## Relations

- **Governing intent: `adapter-north-star.md`** — this note is the
  mechanics for one slice of that vision; read the north star first.

- Amends the D-026 story the same way the trust arc does: refusal was
  interim; levels 0–1 are the *portable* customization path, levels
  2–3 the trust-paired one.
- Companion: `roadmap/adapter-data-audit.md` (current truth),
  `programir-contract/spec/trust.md` (ladder + profiles),
  `roadmap/flow-capabilities.md` (dspy surface).
