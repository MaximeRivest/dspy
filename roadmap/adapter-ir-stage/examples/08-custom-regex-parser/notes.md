# 08 custom-regex-parser — the origin collapse in action

## What this shows

A fully AUTHORED parser that needs NO trust machinery. The user's own
wire format (SCORE/VERDICT lines) is recovered by an RE2-subset regex
with named groups mapping to same-named fields; values coerce through
the field's declared schema (`score: int`). Contrast with example 11:
same job ("my own parse logic"), opposite side of the data/code line.
This entry carries no `authored_by`, no identity hash, no language
block, no isolation — because parse-data has total exposure and ZERO
authority. Loading asks one question: "do you speak parse_combinators
0.1". A minimal Go receiver runs this parser identically (RE2 is the
cross-language guarantee AND the no-backtracking DoS guard).

Note the template and parser are authored TOGETHER but independently:
the template teaches the format, the pipeline reads it back. ADP-003
(carry your parser) is satisfied by the pair; whether the round-trip
probe battery should gate authored parse-data pairs at registration
is an open question.

## Data ladder placement

- Template: data (authored).
- Parser: **level 1** (authored parse-data — the level where
  "authored" stops mattering).

## What today's dspy does instead

Subclass `Adapter`, override `parse` with Python — instantly level 3,
unshippable without your repo, and D-026's flat refusal at export.

## PROPOSED spellings

- `fields_from_groups` — named regex group -> same-named field, value
  coerced by the field schema (the schema-driven coercion codecs
  already do).
- `mode: "search" | "findall"` on the regex combinator.
