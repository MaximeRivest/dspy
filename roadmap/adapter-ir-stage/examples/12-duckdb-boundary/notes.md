# 12 duckdb-boundary — codec stays data, compute is a leaf

## What this shows

The other half of the two-layer rule's criterion. Example 09's
`PIL.Image` HAS a neutral shape, so its codec is pure data. A DuckDB
connection has none — rendering it IS compute (queries run). The rule
resolves it cleanly: the codec entry stays data (a REFERENCE with
`kind: "leaf"`), and the compute is a declared transformation leaf
with a language block, declared effects (`read:db`), an unbound
placement, and a declared emission (`emits: {shape: "text"}` — what
the template's slot receives, so capacity checking still works
without running anything).

This is the boundary rule from the mental model: idiomatic inside,
neutral at the edges, and the cost of a non-neutral edge is local,
priced, and stated — "portable except field `db` requires
python>=3.11 (+duckdb) at leaf `render_duckdb`". Per-boundary, never
program-wide; the receiver decides.

Contrast with 11: a parser leaf gets a forced isolation floor (raw LM
output in); this render leaf does not (it sees the user's own object,
pre-exchange). Effects and placement, not blanket sandboxing, are the
honest declaration here.

## Data ladder placement

- Codec entry: data (a leaf reference + declared facts about it).
- The leaf body: code on the trust ladder (authored origin; source or
  package per the three-origin rule — elided in this entry for
  review focus; it would carry the same source/identity block as 11).

## What today's dspy does instead

A `duckdb`-typed field refuses at lowering (ADP-010: no JSON-schema
meaning) — correct as far as it goes, but there is no declared door
for "here is the transformation that makes it model-facing"; users
preprocess outside the program and the artifact never knows the field
existed.

## PROPOSED spellings

- Codec `kind: "leaf"` with `leaf`, `direction`, `emits`, `language`,
  `effects`, `placement`, `frontend_bindings`.
- Effect atoms (`read:db`) — which effect vocabulary the adapter
  reuses from the ProgramIR trust spec, open question.
- Whether `placement: "unbound"` may appear in an ADAPTER entry at
  all — adapter-ir-spec.md section 9 currently says adapter entries
  carry no placement; the leaf rule says leaves do. This example
  deliberately sits on that fault line for review.
