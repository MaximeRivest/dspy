# Epic D — the new public surface, stated simply

Everything Epic D added that a user can see or call, in plain language: the
name, what it does, why it exists. This is the checkpoint-C1 review document —
ratifying these names freezes them. Nothing already in DSPy changed behavior:
every existing program renders byte-identical prompts (the golden corpus is
the proof).

---

## Declaring what a field *means*

**`dspy.roles`** — a module of eight markers: `plain`, `reasoning`, `tools`,
`tool_calls`, `citations`, `history`, `media`, `code`.

*What:* tag a signature field with what it means to the model exchange —
"this answer should carry citations", "this field is the model's thinking".

*Why:* your signature should say **what you want**, not **how it's
delivered**. A citations field might be served by Anthropic's native
citations API on one model and by quote-markers-in-text on another — the
role stays the same, so your program never changes when the delivery does.

```python
answer: dspy.roles.citations[str] = dspy.OutputField()
```

**The `@role` shorthand in string signatures** —

```python
dspy.Signature("question -> answer: str @citations")
```

*What:* the same role tag, in the quick one-line signature form.

*Why:* class signatures already had three ways to declare a role; the
one-liner form had none. Unknown roles fail immediately and list the valid
ones.

---

## Writing your own prompt

**`dspy.TemplateAdapter(messages, parse_mode="json")`**

*What:* your prompt as a literal message list with `{slots}` — nothing is
added that you didn't write. `{question}` fills in an input, `{instruction}`
the signature's docstring, `{demos()}` the few-shot examples, and so on.
`parse_mode` says how to read the model's reply back (`chat`, `json`, `xml`,
or `full_text` = the whole reply is the one output).

*Why:* before, customizing a prompt meant subclassing an adapter and
overriding methods — code that can't be inspected, diffed, or saved. A
template is **data**: it can be printed, versioned, shipped inside a saved
program, and eventually optimized.

```python
adapter = dspy.TemplateAdapter(
    messages=[
        {"role": "system", "content": "You are concise. {instruction}"},
        {"role": "user", "content": "Summarize:\n\n{text}"},
    ],
    parse_mode="full_text",
)
```

**`adapter.preview(signature, demos=(), inputs=...)`**

*What:* returns the exact messages that would be sent — no model call, no
cost.

*Why:* the learn-by-looking loop. If a prompt surprises you, you look at it,
you don't pay for it.

**`dspy.adapters.describe_template_language()`**

*What:* returns the entire template language as data — every slot, loop
option, style, directive.

*Why:* the language is deliberately small and closed (so templates stay
analyzable). A closed language must be discoverable: the validator, the
error messages, and this call all read the same single source, so they can
never disagree.

---

## Saving an adapter as data

**`adapter.dump_entry()`** and **`dspy.adapters.load_entry(entry)`**

*What:* save an adapter to a plain JSON dict and load it back. The loaded
adapter renders **byte-identical** messages and honors the same settings —
including how each role is served.

*Why:* this is the foundation of the whole program-as-a-file goal: a saved
program must carry its exact prompt behavior with it. The entry is
versioned; loading refuses loudly on unknown versions, unknown keys, or
references to things that don't exist — never a silent half-load.

**`adapter.literal_table()`**

*What:* a seven-key summary of the prompt format (field markers, separators,
the "completed" marker, the parse rule…), always **derived** from the
template, never hand-written.

*Why:* a cross-language reader (or a human) can understand a prompt shape at
a glance without parsing the whole template.

---

## Choosing how a role is delivered

**`strategies={...}` on the adapters**

```python
dspy.ChatAdapter(strategies={"reasoning": "textual_field"})
```

*What:* pick, per role, whether it's served by the model's native feature or
by plain text in the prompt. Anything you don't set is `"auto"`: resolved
against what the model actually supports, and the choice is recorded.

*Why:* the same program runs against very different models. This makes the
delivery choice explicit, swappable, and — because it's recorded in the
saved entry — reproducible.

---

## Plugging in your own pieces

**`dspy.adapters.register_codec(name, codec)`** (and `unregister_codec`)

*What:* register your own way of writing values into prompts and parsing
them back (a codec).

*Why the gate:* a broken codec silently corrupts every program that uses it,
so registration runs a round-trip torture battery (nested models, unicode,
empty collections, None) — parse(render(x)) must equal x, or the codec is
refused on the spot.

**`dspy.adapters.register_strategy(strategy, role=... | annotation=...)`**

*What:* register your own way of serving a role.

*Why the gate:* a strategy must declare which model capabilities it needs
and must carry its own parser — every way of writing something into a
prompt has to come with the way of reading it back.

**`dspy.adapters.register_preset(name, messages, parser=..., ...)`**

*What:* register a named prompt shape (template + parser + bindings); then
`PresetAdapter("name")` uses it anywhere.

*Why:* teams can share prompt shapes as named, validated data instead of
copy-pasted subclasses. The template is checked completely at registration —
errors happen at register time, not mid-run.

---

## Two small additions to the template language itself

**`{% section strip %} … {% endsection %}`** — marks a region whose joined
render gets whitespace-collapsed the way the legacy adapters did it. Exists
because the old byte layout genuinely could not be reproduced without it
(empty trailing sections must collapse; interior ones must not).

**`{field('name')}`** — an escape spelling to reference a field whose name
collides with a reserved word (a field literally named `instruction`, say).
Without it those fields were unreferenceable; with it, the collision error
can teach the way out.

---

## What deliberately did NOT change

- Every existing adapter (`ChatAdapter`, `JSONAdapter`, `XMLAdapter`,
  `BAMLAdapter`, `TwoStepAdapter`) works unchanged and renders the same
  bytes.
- Subclassing adapters and overriding `format_*` methods still works — it is
  the legacy customization path, headed for deprecation now that templates
  exist, but nothing breaks today.
- No behavior changes without opting in: the golden corpus (byte-for-byte
  reference outputs across all five adapters) was untouched through the
  entire epic.
