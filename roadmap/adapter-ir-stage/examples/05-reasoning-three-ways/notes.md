# 05 reasoning-three-ways — the strategy rule language

## What this shows

ONE signature with one `reasoning`-role field; THREE adapter entries
that conduct it three ways. This is the north star's strategy-as-data
rule with its four faces, all data:

- **native**: predicate `native_reasoning`; `hides` removes the field
  from the token stream (a rendering decision, never a semantic
  deletion); `engine_controls.request_patch` is the request-side face;
  a `channel` routing brings the provider channel back to the field.
- **prefix-CoT**: fragments only. The field stays visible; the lens
  parses it as a normal section. Predicate is the mild "instruct".
- **interleaved**: fragments INSTRUCT the interleaved style (pure
  prompt — works on any instruct model today), and a `text` routing
  runs a regex combinator over the completion, collecting the tags
  into the field and `consume`-ing them so the lens sees clean text.

The file is a JSON array because the review unit is the contrast; each
element is a complete standalone entry. Note what stays constant: the
template, the parser, the codecs — only the `strategies.reasoning`
value changes. That is the layer law made visible.

## Data ladder placement

- Rules: data (strategy vocabulary, drafted as 1.1.0-draft).
- The interleaved routing's pipeline: level-1 parse-data, reused
  inside a strategy — parse combinators are ONE vocabulary whether
  they appear under `parser` or under a routing.

## What today's dspy does instead

`reasoning: "auto" | "native_channel" | "textual_field" | "prefill"` —
names into registered Python. The interleaved variant does not exist
at all; building it means an authored strategy class. `applies()`
reads live LM objects, not declared capability facts.

## PROPOSED spellings

- The rule object: `kind: "rule"` with faces `predicate`, `hides`,
  `fragments`, `engine_controls`, `routings` (face names lifted
  verbatim from adapter-north-star.md; `hides` is the extra one).
- Predicate atoms: `{"capability": "<name>"}`; combinators `all`/
  `any`/`not` reserved but unused here.
- Routing forms: `{"channel": ..., "field": ..., "coerce": ...}` and
  `{"text": <pipeline>, "field": ..., "consume": bool}`.
- `consume` — removes matched spans from the lens's input stream.
- `strategy.rule(...)` / `strategy.capability(...)` authoring helpers.
