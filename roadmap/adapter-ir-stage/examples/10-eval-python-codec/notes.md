# 10 eval-python-codec — the codec/strategy hybrid, with the trust seam placed right

## What this shows

Structured output conducted as "model emits Python; a sandboxed
interpreter materializes it". The decomposition puts each piece where
the layer law says it goes:

- the CODEC (`python_literal` family) owns the value spelling — data;
- the STRATEGY rule owns the conduct: fragments teach the fenced
  block, a routing extracts it — data;
- the INTERPRETER is a declared LEAF reference with an explicit
  isolation declaration (`sandboxed`, no effects, no network). The
  leaf is the ONLY code in the picture, and it is already
  representable in the ProgramIR/trust vocabulary — the adapter entry
  does not smuggle an eval, it NAMES one and declares its floor.

The entry's `requires.leaves` block is the requirements gradient
speaking: this artifact truthfully needs a python>=3.10 sandbox. A
Go receiver refuses naming the leaf — or binds a sidecar (D-022
rung-walk). Compare "return JSON I validate" (example 02): same typed
field, zero-requirement conduct. The pair is the north star's
"blurs into strategy" sentence made into two artifacts you can diff.

## Data ladder placement

- Codec family + rule + pipeline: data.
- The interpreter: **leaf reference** — trust ladder, isolation
  forced (a parser-adjacent component with real authority).

## What today's dspy does instead

Nothing comparable; `PythonInterpreter` exists as a tool, but no
strategy can bind an interpreter to a field materialization, and no
entry can declare the requirement.

## PROPOSED spellings

- `materialize.interpreter` on a routing: `leaf`, `language`,
  `isolation`, `effects`, `network`.
- `requires.leaves` — leaf requirements surfaced in the entry's
  requirement set (duplication with the ProgramIR leaf table is an
  open question: does the adapter entry declare, or only reference?).
- Codec object `kind: "family"` with `family` + `options`;
  `python_literal` as a codec-family name (codecs 1.1.0-draft).
- Using role `code` as the strategy key for code-emission conduct of
  a typed field — versus keeping the field `plain` and keying the
  hybrid another way. Open question.
