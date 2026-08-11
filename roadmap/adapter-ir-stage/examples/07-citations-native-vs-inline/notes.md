# 07 citations — channel routing vs inline-marker parse

## What this shows

The `citations` role conducted two ways. Native: a request patch turns
the provider channel on, a `channel` routing brings it home, and a
`transforms` face renames the text target (`answer` -> `answer_text`)
so the text lens and the native assembler compose — this transform is
already normative in adapter-ir-spec.md section 2; here it gets its
data spelling. Inline: fragments teach `[n]` markers and a combinator
pairs sentence spans with document indices; `consume: false` keeps the
markers visible in the answer (a deliberate contrast with example 05,
where the think tags are consumed — the flag earns its keep).

## Data ladder placement

- Native rule: pure data, no parse vocabulary at all.
- Inline rule: data + a level-1 pipeline; `citations` is the second
  typed terminal combinator (after `tool_calls`), coercing into the
  `Citations` shape by shapes-vocabulary name.

## What today's dspy does instead

The citations hook is three lines of registered Python ("channel
`citations` -> field X, coerce `Citations`" — the audit's own
observation that it is ALMOST data); the inline-markers variant is the
`span_markers` vocabulary word with no shipped implementation.

## PROPOSED spellings

- `transforms` face: `[{"rename": {"from": ..., "to": ...}}]` — the
  fifth rule face, beyond the north star's four. Alternative: fold
  renames into `routings`. Open question.
- `citations` combinator (`span_group`, `doc_group`).
- Whether a typed terminal combinator per role scales, or whether
  routings should end in a generic `{"coerce": "<shape>"}` step —
  open question (this decides how big parse_combinators grows).
