# 04 base-model — the second column of the LM-family axis

## What this shows

An adapter for a BASE model: one few-shot pattern-completion prompt,
`{demos(style='yaml')}` carrying the pattern inline (no chat-turn
directive — a base model has no turns), stop sequences providing the
discipline that chat markers provide for instruct models. The lens
still inverts it: the trailing `answer:` label the template renders is
the boundary the parser reads from. The signature and the program are
untouched — this is the north-star grid: programs x signatures on one
axis, adapter x LM-family on the other.

Note the stop sequence `"\nquestion:"` names a field literal. That is
deliberate and honest: this template is signature-shaped at its edges,
and bake should refuse a signature whose field names collide with the
stop discipline (an ADP-006-style capacity check).

## Data ladder placement

- Template: data. Demos travel as an aggregate slot, not turn pairs.
- Parser: **level 0** (lens).
- Engine controls: **request-side data** in `config.engine_controls`
  — the north star's "LMRequestPatch generalized" face.
- `requires.lm_capabilities`: declared capability facts; the refusal
  is a compatibility statement, not a failure.

## What today's dspy does instead

Nothing. All four builtin adapters assume instruct models; base-model
use means hand-rolled prompts outside dspy or abusing `full_text`.
`applies()` reads live LM objects instead of declared capability
facts, so there is no vocabulary in which to even SAY "completion
model".

## PROPOSED spellings

- `config.engine_controls` — engine controls nested in the open
  `config` dict (vs a first-class entry key — open question).
- `requires` — a NEW top-level ENTRY_KEY: the declared requirement
  set of the portability gradient. Also plausible inside `config`.
- `versions.lm_capabilities` — the capability vocabulary is versioned
  data, same growth discipline as the node set.
- Capability name `completion` — first entry of that vocabulary.
- `history: "inline"` — spec section 5 names this strategy; unused by
  the builtins today.
