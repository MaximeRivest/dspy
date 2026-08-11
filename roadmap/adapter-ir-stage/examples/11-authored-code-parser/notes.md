# 11 authored-code-parser — the requirements gradient, code tail

## What this shows

The honest level-3 case: a stateful, grammar-coupled recovery parser
no combinator vocabulary should try to swallow. Under the pending
D-026 amendment it is NOT flat-refused at export: it declares its
requirements like every other component — language env block (D-025),
`isolation: "sandboxed"` as a FORCED floor (the pairing rule: a
parser's exposure is total by definition, so it never runs
unsandboxed), source baked in, identity-verified, `authored_by`
provenance (ADP-011).

The requirement set is a portability STATEMENT, never a moral defect.
A pure-Go receiver that cannot bind a Python sidecar emits exactly:

    requires python>=3.12 sidecar for `4_adapter/ledger_recovery/parser`
    — unbound; refuse or bind one.

A receiver that CAN bind one runs the parser out-of-process (ADP-002
purity makes that sound) and the artifact never knows the difference.
Contrast with example 08: authored PARSE-DATA needed none of this
machinery. The pair brackets the origin collapse from both sides.

## Data ladder placement

- Template: data.
- Parser: **level 3** (authored code, trust ladder, forced isolation
  rung). This is the level whose existence justifies keeping levels
  0-1 small and honest instead of growing a Turing-complete "data"
  vocabulary.

## What today's dspy does instead

An `Adapter` subclass override — works locally, refused at export
(D-026 as ratified), with no vocabulary to even declare the sidecar.

## PROPOSED spellings

- Parser `kind: "authored"` carrying `origin`, `language`,
  `entrypoint`, `source`, `identity`, `authored_by`, `isolation`.
- `requires.languages[]` with `for` (component path), `binding:
  "sidecar"`, `isolation_floor` — the declared-requirement spelling of
  the north star's gradient. Whether `requires` is derivable from the
  entry body (it is, here) and therefore a DERIVED view rather than an
  authored key — open question, same logic as the literal table.
