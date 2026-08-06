# Adapter IR — worked examples (by hand)

Hand-traced signature → plan → messages → request → parse maps, written to
make the contract crisp before Epic D implements it. These become fixture
seeds. Companion to `adapter-ir-spec.md`; the two fixes they surfaced
(positional fragment slots; media-renders-as-parts-at-slot) are folded into
the spec.

## Example 1 — minimal: `"question -> answer"`, preset `chat`

**SignatureCore:**
```json
{"instructions": "Answer concisely.",
 "fields": [
  {"name": "question", "direction": "input",  "shape": {"type": "string"}, "role": "plain"},
  {"name": "answer",   "direction": "output", "shape": {"type": "string"}, "role": "plain"}]}
```

**Plan** (bake: all plain, no strategy fires — the plan is the *diff*
between signature and wire, here nearly empty):
```json
{"visible_inputs": ["question"], "visible_outputs": ["answer"],
 "config_patches": {}, "fragments": {},
 "parsers": [{"kind": "chat_markers", "fields": ["answer"]}]}
```

**Messages** (template renders the plan; one demo; fragment slots empty):
```json
[
 {"role": "system",    "content": "Your input fields are:\n1. `question` (str)\n...structure...\nIn adhering to this structure, your objective is:\n  Answer concisely."},
 {"role": "user",      "content": "[[ ## question ## ]]\nCapital of Italy?"},
 {"role": "assistant", "content": "[[ ## answer ## ]]\nRome\n\n[[ ## completed ## ]]"},
 {"role": "user",      "content": "[[ ## question ## ]]\nCapital of France?\n\nRespond with ... `[[ ## answer ## ]]` ... `[[ ## completed ## ]]`."}]
```

**Request:** `Request(messages=(those four), config=Config(...))` — empty
patches meant nothing outside the messages.

**Parse:** `"[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"` →
`{"answer": "Paris"}`. Exactly the declared outputs (ADP-004).

## Example 2 — one signature, two LMs

```python
class RAG(dspy.Signature):
    """Answer from the provided documents."""
    question: str = dspy.InputField()
    docs: list[dspy.Document] = dspy.InputField()      # role: media
    thinking: reasoning[str] = dspy.OutputField()
    answer: citations[str] = dspy.OutputField()
```

### 2-A: LM with native reasoning + native citations

Strategies resolve `reasoning→native_channel`, `citations→native`,
`media→native_parts`.

```json
{"visible_inputs": ["question", "docs"],
 "visible_outputs": ["answer"],
 "config_patches": {"reasoning": {"effort": "medium"}},
 "fragments": {},
 "slot_codecs": {"docs": "document_parts_citable"},
 "field_transforms": [{"kind": "RenameOutputField", "from": "answer", "to": "answer_text", "reason": "native:citations"}],
 "parsers": [
   {"kind": "chat_markers",     "fields": ["answer_text"]},
   {"kind": "native_reasoning", "fills": "thinking", "from": "ThinkingPart"},
   {"kind": "native_citations", "fills": "answer",   "from": "CitationPart[]", "over": "answer_text"}]}
```

Note: `thinking` is in **no** loop — not the field list, not the structure
section, not the "Respond with…" enumeration. `docs` stays **visible**: its
slot renders as citable `DocumentPart`s at its template position (the
media fix). The strategy emitted the rename transform so the text parser
works under `answer_text` while native citations assemble `answer`.

User message text: just the question block + the output requirement naming
only `answer`. Request carries the doc parts inside the user message's part
list and `reasoning` in config. Response: `ThinkingPart(...)` + text +
`CitationPart[]` → `{"thinking": ..., "answer": ...}`.

### 2-B: bare text LM

Strategies resolve `reasoning→textual_field`, `citations→span_markers`,
`media→text_render`.

```json
{"visible_inputs": ["question", "docs"],
 "visible_outputs": ["thinking", "answer"],
 "config_patches": {},
 "fragments": {"system": ["When quoting a document, wrap the quote in <cite doc=\"N\">…</cite>."]},
 "slot_codecs": {"docs": "text_pythonish"},
 "parsers": [
   {"kind": "chat_markers",   "fields": ["thinking", "answer"]},
   {"kind": "span_citations", "fills": "answer", "resolve_against": "docs"}]}
```

Same template, fuller loops: the model is told to produce
`[[ ## thinking ## ]]` then `[[ ## answer ## ]]`; the cite-marker
instruction renders in the system fragment slot; docs render as text at the
same slot position. Empty patches. Same prediction shape back.

## The preset `chat` template (one artifact, all three renders)

This is the artifact as shipped in `dspy/adapters/_engine/presets.py` (D-2,
byte-proven against the golden corpus). Three details the first hand-written
draft got wrong, now normative in the spec: the section loops carry the
`strip` flag (the historical join-then-strip shape), the objective line uses
`{instruction(style='indented')}` on one line after `objective is: ` (the
trailing space and the 8-space indent both come out exactly), and
`{fragments('system')}` sits directly after the completed marker with no
blank line (its empty render removes the whole line).

```python
CHAT = Preset(
    name="chat",
    template=[
        {"role": "system", "content": """\
Your input fields are:
{% for f in inputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
Your output fields are:
{% for f in outputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
All interactions will be structured in the following way, with the appropriate values filled in.

{% for f in inputs separator='\\n\\n' strip %}
[[ ## {f.name} ## ]]
{{{f.name}}}
{% endfor %}

{% for f in outputs separator='\\n\\n' strip %}
[[ ## {f.name} ## ]]
{f.typed_placeholder}
{% endfor %}

[[ ## completed ## ]]
{fragments('system')}
In adhering to this structure, your objective is: {instruction(style='indented')}"""},
        {"role": "demos",
         "user": "{% for f in inputs separator='\\n\\n' %}\n[[ ## {f.name} ## ]]\n{f.value}\n{% endfor %}",
         "assistant": "{% for f in outputs separator='\\n\\n' strip %}\n[[ ## {f.name} ## ]]\n{f.value}\n{% endfor %}\n\n[[ ## completed ## ]]\n"},
        {"role": "history"},
        {"role": "user", "content": """\
{% for f in inputs separator='\\n\\n' %}
[[ ## {f.name} ## ]]
{f.value}
{% endfor %}

{fragments('user')}
Respond with the corresponding output fields, starting with the field \
{% for f in outputs separator=', then ' %}`[[ ## {f.name} ## ]]`{f.chat_type_hint}{% endfor %}, \
and then ending with the marker for `[[ ## completed ## ]]`."""},
    ],
    parser="chat",
    codecs={"input": "text_pythonish", "output": "text_pythonish"},
    strategies={"reasoning": "auto", "citations": "auto", "tools": "auto",
                "media": "auto", "history": "directive_turns"},
)
```

Example 1: fragment slots empty, history no-ops → today's ChatAdapter
bytes. 2-A: loops see fewer fields; `{f.value}` for docs emits parts. 2-B:
loops see more fields; the fragment slot carries the strategy instruction.
Zero conditionals in the template — the plan does all the deciding.

## The crispness theorems

1. **plan = signature − native channels** (strategy claims become patches,
   transforms, and parsers).
2. **request = template(plan) ⊕ config patches** — patches are never
   tokens; the template is never non-tokens; message content is a part
   list and slots decide which part kind they emit.
3. **the prediction is invariant across LMs** — that is what roles buy.
4. **every arrow is data** — core, plan, messages, patches, parsers are all
   printable; no stage lives inside a method.
