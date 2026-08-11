# 03 token-minimal — the "template edit IS the custom adapter" case

## What this shows

A token-efficient adapter for a small/weak model: no scaffold system
block, `name: value` lines, no completed marker, no type notes. The
whole customization is a template; the parser is the derived lens.
With two output fields the lens would read `answer:`-style labels as
boundaries; with one output field it degenerates to full_text — the
degenerate lens named in adapter-parse-dsl.md. This is the north
star's claim in miniature: most wild custom adapters are layout
tweaks, and under the lens they are template edits, not parser
classes. The entry sits at the zero-requirement floor of the
portability gradient: pure data, loadable by any conforming receiver.

## Data ladder placement

- Template: data.
- Parser: **level 0** (lens; full_text as its degenerate case).
- Everything else: builtin name references.

## What today's dspy does instead

You subclass `Adapter`, override `format` AND `parse`, and keep them
in sync by hand — then your program cannot ship, because the subclass
is your repo's Python. Or you use `full_text`, which is close but
carries the chat preset's heavy scaffold.

## PROPOSED spellings

- `dspy.make_adapter(name=, template=, ...)` — template-first adapter
  construction with the lens as the default parser when `parser` is
  omitted. (Open question: is "lens" the right DEFAULT for authored
  templates, or must it stay explicit?)
- `set_lm(lm, adapter=...)` — per-predictor binding spelling; matches
  the bindings-not-ambient-settings direction, exact surface TBD.
