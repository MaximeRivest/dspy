# 06 tools-three-ways — the tool FORMAT is a strategy

## What this shows

The same program calls tools three ways: native FC (a request-side
patch plus a typed channel routing — pure data, no token-stream
bytes), CLI-style `!call name {json}` lines, and XML blocks. The
program never said which; the strategy did. The two textual variants
are polyfills of the native channel (the mental-model's channelization
story), and both are fragments + a parse routing — no code anywhere.

The interesting corner is native FC's request patch: it must reference
THE CALL'S tool declarations, which are a field value, not a literal.
`{"$from": "field:tools"}` is the proposed data spelling for "splice
this field's lowered value here". This is the one place the north star
admits is real procedural wiring today (litellm-coupled); the claim of
this example is that the wiring collapses to one reference form.

Note the fragments embed `{field('tools')}` — a template-language slot
inside a strategy fragment. Fragments are template-language strings by
the north star ("render fragments (template-language)"); this example
leans on that and flags it.

## Data ladder placement

- All three rules: data. Textual variants add level-1 pipelines.
- `tool_calls` combinator: a typed terminal combinator (pairs regex
  groups into ToolCalls). Coercion targets like `ToolCalls` are
  shapes-vocabulary names, not Python classes.

## What today's dspy does instead

`use_native_function_calling: true` as a constructor flag recorded in
`config`; the FC wiring is procedural litellm code; the textual
variants exist only as `textual_json` / `xml_dispatch` vocabulary
words with registered Python behind them.

## PROPOSED spellings

- `{"$from": "field:<name>"}` — request-patch value splice.
- `tool_calls` combinator with `name_group`/`args_group`/`args`.
- Template slots legal inside fragment content (and WHICH subset of
  the language fragments may use — open question).
- Capability name `native_function_calling`.
