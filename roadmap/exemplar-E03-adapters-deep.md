# E-03 — adapters deep: authored entries, the capability flip, the base-model polyfill

**Proves (gaps G4, G17):** an AUTHORED adapter entry (origin, admission,
requirements — and why it never rung-walks); one signature crossing
three capability profiles with only strategy resolution changing; the
`instruct=False` base model served by polyfill. Successor to server
examples 01–04.

## 1. The authored entry (component 4, origin: authored)

A house wire format no preset covers — ANSI-terminal-style sections.
Authored as a template + one authored parser; travels as data + source:

```jsonc
"4_adapter": {"house_ansi": {
  "name": "house_ansi", "adapter_ir_version": "0.2.0",
  "versions": {"strategies": "1.2.0-draft", "codecs": "1.0"},
  "origin": "authored",                              // vs builtin preset refs
  "template": {"messages": [
    {"role": "system", "content": "\u001b[1m{instruction}\u001b[0m\n{% for f in inputs %}\u001b[36m{f.name}\u001b[0m: {f.type}\n{% endfor %}"},
    {"role": "demos"},
    {"role": "user", "content": "{% for f in inputs %}\u001b[36m{f.name}\u001b[0m\n{f.value}\n{% endfor %}\nReply exactly:\n{% for f in outputs %}\u001b[33m{f.name}\u001b[0m\n{f.value}\n{% endfor %}"}]},
  "parser": {"kind": "lens"},                        // authored TEMPLATE, derived parser —
                                                     //   authored ≠ hand-written parser; the
                                                     //   lens law still holds or refuses
  "authored": {"language": "python",                 // only the residue is code: one codec
    "codecs_source": "adapters/ansi_codecs.py",      //   (strips ANSI on parse)
    "deps": [], "authored_by": "human"},
  "codecs": {"__default__": {"output_codec": "ansi_stripped_text"}}
}}
```

The law this fixes as bytes (D-026 + D-040): **authored adapter code
carries no placement and never rung-walks** — a foreign engine refuses
it at the profile level, naming the requirement (`requires python
authored-codec support for '4_adapter/house_ansi'`); the portable
customization path is the template half, which IS data. The requirements
gradient applies: the entry above declares its language + deps, so the
refusal is a named unmet requirement, never a flat "unportable".

## 2. One signature, three capability profiles — the flip as data

Signature: `question -> reasoning @reasoning, answer`. Entry strategy:
`reasoning: "auto"`. Three models, three resolutions — the program and
signature never change:

| bound model declares | resolved rule | prompt contains | reply path |
|---|---|---|---|
| `native_reasoning: true` | `native` | NO reasoning section; request patched `{"reasoning": {"effort": "medium"}}` | API channel → field |
| plain `instruct: true` | `prefix_cot` | reasoning as a normal template section + teaching fragment | the lens parses it |
| `<think>`-style | `interleaved` | fragment teaches think-tags | text routing extracts + consumes; lens parses the rest |

Bytes: `explain_strategies` output is provenance —
`{"reasoning": "auto->native"}` — recorded per call in View 2, so a
score is attributable to the conduct that produced it.

## 3. The base model (`instruct=False`) — the completion polyfill

STUDY Q1's ruling exercised: no new face. A base model declares
`{"instruct": false, "completion": true}`; the chat entry's rules
predicate on it:

```jsonc
"strategies": {"exchange": {"kind": "rule",
  "predicate": {"capability": "completion"},
  "fragments": [{"slot": "system", "content": ""}],       // no chat framing
  "engine_controls": {"request_patch": {"stop": ["\n\n[[ ## "]}},   // stop at next marker
  "render_mode": "single_prompt"}}                        // messages fold to one completion
```

The polyfill loop, third historical instance (CoT→native, textual-FC→
native-FC): chat framing over completions was ITSELF always a polyfill;
here it is a rule, visible, versioned, and droppable per model.

## 4. Refusal hooks (cross-ref E-11)

**R-42 · authored adapter on a foreign profile engine** `[E-03; D-026/D-040]`
> `load (declared-tier engine, ts): entry 'house_ansi' requires python authored-codec evaluation — outside the declared-tier profile; unmet requirement named, template half remains readable (explain works, format refuses)`

**R-43 · strategy resolution with no admissible rule** `[E-03]`
> `strategies: role 'reasoning' bound 'auto' — no registered rule's predicate passes for capabilities {instruct: false, completion: true, native_reasoning: false}; refusing (a base model with no reasoning conduct is a configuration, not a fallback)`
