# 02 json-tolerant — level-1 parse-data

## What this shows

The json preset with its LLM-sloppiness tolerance made explicit as an
ordered combinator pipeline: prefer a fenced ```json block if present,
try strict JSON, fall back to a repair policy, then map object keys to
fields (unknown keys become exhaust, ADP-004). Today all of that is
hardcoded inside the `json` parser program; here it is inspectable,
diffable data. A receiver in Go runs the same pipeline from the same
five lines of pinned semantics per combinator.

## Data ladder placement

- Template: data.
- Parser: **level 1** (declared parse-data). This is the residue the
  lens cannot express — recovery from formats we did not render. It
  carries no trust question (origin collapse): loading asks only "do I
  speak `parse_combinators` 0.x", never "who wrote this".
- The repair policy is an enum, not code — `json_repair` names pinned
  repair semantics, exactly like the node-set selection rule.

## What today's dspy does instead

`parser: "json"` plus a fixed, invisible repair chain inside
`_engine/codecs.py` and the json format. The tolerance level is not a
choice, not serialized, and not optimizable.

## PROPOSED spellings

- `"kind": "pipeline"` with `"steps"` — the level-1 parser form.
- Combinator names `fenced_block`, `alternatives`, `json_object`,
  `fields_from_object` and their option keys (`policy`, `repair`,
  `unknown_keys`) — first draft of the census-gated vocabulary.
- `versions.parse_combinators` — a NEW versions-block vocabulary, only
  present when the entry uses a pipeline parser (conditional presence
  vs always-present is an open question).
- `JSONAdapter.with_parser(...)` and the `dspy.adapters.parse`
  authoring module.
