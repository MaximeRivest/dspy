# 01 chat-baseline — the bridge example

## What this shows

Today's ChatAdapter, byte-for-byte the entry `dump_entry()` already
emits, with ONE change: `"parser": "chat"` becomes
`{"kind": "lens", "of": "template"}`. This is the level-0 move from
adapter-parse-dsl.md: the chat parser is already secretly the inverse
of the chat template. The template pins `[[ ## name ## ]]` markers as
data; the lens reads them back — labels are section boundaries, slots
are captures, values decode through the bound output codec
(`text_pythonish`). No enum, no second description of the layout, and
`parse(render(x)) = x` (ADP-003) holds by construction.

## Data ladder placement

- Template: data (unchanged, template language 1.1.0).
- Parser: **level 0** (derived lens). Zero new vocabulary — the lens
  is a fixed derivation rule of the IR itself, so the `versions`
  block does not grow.
- Codecs, strategies: name references, as today.

## What today's dspy does instead

`parser: "chat"` names one of four hardcoded parser programs
(`VOCABULARY["parsers"]`). The layout is therefore stated twice: once
in the template (data) and once in `ChatFormat.field_header_pattern`
(Python). A template edit that moves a marker silently desynchronizes
the two. Under the lens, the enum's four builtins become four
derivations of their own presets and the desync class dies.

## PROPOSED spellings (flagged for README open questions)

- `"parser": {"kind": "lens", "of": "template"}` — the object form of
  the parser key, and `"lens"` as the level-0 kind name.
- `adapter_ir_version: "0.3.0-draft"` — the extended shape is a minor
  bump; strings stay valid as builtin names during migration.
- `parse_preview()` — the parse dual of `preview()` (the bonus already
  named in adapter-parse-dsl.md; the method name is the proposal).
