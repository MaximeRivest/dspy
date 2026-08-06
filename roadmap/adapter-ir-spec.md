# The Adapter IR — a contract for rendering and parsing

**Status:** provisional specification (2026-08-06). Epic D implements against
it; whatever D proves wrong gets fixed *here first*, then in code. When it
stabilizes (post-D/E), this document graduates to its own contract repo,
lm15-style (spec + fixture corpus + language-neutral harness + constitution),
and the implementation extracts to a standalone library (D-021). Until then
the dspy engine (`dspy/adapters/_engine/`) is reference behavior, not
normative authority.

**One sentence.** This contract owns the middle layer of the LLM stack:
**typed signature ⇄ canonical messages** — how a declared I/O contract is
rendered into a prompt and how a model's response is recovered into typed
values — independent of any particular signature frontend (dspy, FunctAI, raw
dicts) and any particular LM backend (lm15, litellm, provider SDKs).

**Layering.** lm15 owns the wire (canonical `Request`/`Response`/
`StreamEvent`). This contract owns signature ⇄ messages, rendering into
lm15 `Message` values (or a structurally identical dict form for non-lm15
backends). The ProgramIR owns program structure and consumes this contract by
reference: its component 4 is a pool of this spec's preset entries plus
bindings — nothing more.

---

## 1. The signature core (the neutral input)

The contract cannot depend on `dspy.Signature`. Its input is a minimal
datatype every frontend lowers to:

```
SignatureCore = {
  instructions: str,                     # ProgramIR component 3a
  fields: [ {
    name: str,
    direction: "input" | "output",
    shape: JSONSchema,                   # arbitrary; no blessed type list
    role: SemanticRole,                  # default "plain"
    desc: str | null,
    prefix: str | null,
  } ],
}
```

plus per-call values: `inputs: {name → value}`, `demos: [ {name → value} ]`
(ProgramIR 3b), and optionally a `history` value. dspy signatures, FunctAI
function signatures, and hand-written dicts all lower to this; the lowering
is the frontend's job and out of scope here. Shapes with no JSON-schema
meaning (callables) are refused at lowering with the field named (ADP-010).

## 2. The pipeline (normative)

**The template renders the plan, never the raw signature.**

```
SignatureCore + LM capabilities + preset
  → strategies resolve per role: native | textual      (bake)
      · hide natively-served fields (visible set)
      · patch the request outside the token stream (tools array, reasoning kwargs)
      · contribute template fragments (textual strategies only)
  → PLAN { visible fields, fragments, request patches, parser set }
  → template renders the plan into messages             (codecs spell values)
  → parse = parse_mode over visible text  ∪  strategy parsers over native channels
```

Consequences, all normative:

- Slot iterators (`{inputs()}`, `{outputs()}`, loop blocks) iterate the
  plan's **visible** fields only. A natively-served field contributes no
  text; its parser fills it from the typed channel.
- A **textual strategy is a template-fragment provider**; preset templates
  carry the slots its fragments fill.
- **Declared capacity:** a template statically declares which roles it can
  host textually — derivable by analyzing its slots (this is why the
  template language is closed). Bake checks the triple (signature roles ×
  LM capabilities × template capacity) and refuses loudly naming all three
  when no lane exists (ADP-006).
- An explicit slot reference to a natively-served field is a bake-time
  conflict, refused loudly — never an empty render (ADP-007).

## 3. The template language (closed, versioned)

A template is a list of messages. Each message is `{role, content}` or a
directive. Content strings may use:

- **Value slots:** `{field_name}` (rendered through the field's bound input
  codec), `{instruction}` (3a).
- **Aggregate slots:** `{inputs(style=…)}`, `{outputs(style=…)}`,
  `{demos(style=…)}`, `{history(style=…)}` — `style` names a codec.
- **Loop blocks:** `{% for f in inputs|outputs [separator='…'] %} … {% endfor %}`
  with the closed `f.*` vocabulary: `i/index, name, type, desc, desc_suffix,
  value, placeholder, typed_placeholder, marker, chat_type_hint`.
- **Strategy slots:** where textual-strategy fragments land (exact naming
  ratified in D-1; per-role, e.g. `{tools.instructions}`).
- **Escapes:** `{{`/`}}` literal braces; `{{{f.name}}}` literal placeholder.

Directive messages expand at render time: `{"role": "demos"}` → user/
assistant pairs per demo (assistant format follows the parser binding);
`{"role": "history"}` → prior turns. Directives are role-named on purpose —
they are the textual strategies of the `history` and demo machinery.

**Deliberately NOT in the language:** general Jinja, arbitrary expressions,
user-defined control flow. Analyzability (capacity derivation, diffing,
optimization) is a spec requirement. Registered helper functions are an
authored-code escape hatch and carry provenance (ADP-011).

## 4. Presets (the canonical entry)

```
Preset = {
  name: str,
  template: [Message | Directive],
  parser: "chat" | "json" | "xml" | "full_text" | AuthoredParser,
  codecs:  { input: CodecRef, output: CodecRef, per_field?: {name → CodecRef} },
  strategies: { role → StrategyRef | "auto" },
  config: { resolved, capability-checked flags },   # e.g. response_format_routing
}
```

Canonical JSON; tuples/ordering preserved; serde is exact (absent ≠ null).
`"auto"` strategies resolve against LM capabilities at bake and the
resolution is recorded into `config` (declare-don't-discover). Loading a
preset resolves every `CodecRef`/`StrategyRef` against the pools; a dangling
reference is a link error naming the reference (ADP-005). Built-in presets:
`chat`, `json`, `xml` — each defined AS a template in this language,
byte-reproducing the historical class adapters (the corpus proves it).
`full_text` requires exactly one plain output field. BAML is **not** a
preset: it is preset `json` + the `baml` codec bindings.

## 5. Closed vocabularies

- **Roles:** `plain, reasoning, tools, tool_calls, citations, history,
  media, code` (+ anticipated: `refusal`). Grounded in the typed LM
  contract; versioned with it.
- **Strategies (per role, initial):** reasoning `native_channel |
  textual_field | prefill`; tools `native_fc | textual_json | xml_dispatch`;
  citations `native | span_markers | json_quotes`; media `native_parts |
  url_reference`; history `directive_turns | inline`.
- **Codecs (initial):** `text_pythonish` (the historical implicit pair),
  `pydantic_json` (indented model dumps), `baml` (schema-prose + indented
  input), `json`, `xml`, `yaml`. Codecs are shape-generic by law.
- **Derived summary view:** the 7-key literal table
  (`input_field_render, output_field_render, field_separator,
  output_structure, completed_marker, output_requirement, parse_pattern`)
  is *derived from* the template for cross-language readers; never authored.

Extending any vocabulary is a versioned act with an admission rule (roles:
must change how the exchange is conducted; codecs: must be shape-generic).

## 6. Invariants (citable)

- **ADP-001 (single-shot):** nothing in render or parse calls an LM. One
  preset application = one exchange.
- **ADP-002 (purity):** rendering is a pure function of (plan, values);
  planning is a pure function of (core, capabilities, preset). No ambient
  reads, no clock, no network.
- **ADP-003 (carry your parser):** every rendering decision ships with the
  parser that inverts it; round-trip `parse(render(x)) = x` on adversarial
  probes gates admissibility.
- **ADP-004 (sacred outputs):** parse returns exactly the declared output
  fields; anything else is exhaust for the caller's observability channel.
- **ADP-005 (loud linking):** dangling refs refuse at load naming the ref.
- **ADP-006 (capacity check):** the (roles × capabilities × capacity)
  triple is verified at bake; no silent degradation.
- **ADP-007 (no silent slot conflicts):** explicit slots referencing hidden
  fields refuse at bake.
- **ADP-008 (strategy awareness):** iterators see visible fields only.
- **ADP-009 (direction of ownership):** user-supplied passthrough and
  provider-returned data are distinct named things; never echoed across.
- **ADP-010 (shape honesty):** non-JSON-schema shapes refuse at lowering,
  naming the field.
- **ADP-011 (provenance):** authored parsers/helpers/codecs carry
  `authored_by` and identity; machine-mutated templates carry their chain.

## 7. Conformance

The fixture corpus is the promoted golden corpus (today: 176+ files —
request, parse, callbacks, shapes families — byte-parity gated, regeneration
only in dedicated commits). Contract form: canonical `SignatureCore` +
values + preset → exact rendered messages; completion (text or typed
channels) → exact parsed values. A conforming implementation passes the
whole corpus with strict typed equality; the harness performs all comparison
itself (lm15's discipline). `dspy_template_adapter` is the intended first
external conformer; the chat-parity template in its README is the historical
proof the corpus's chat family is expressible in this language.

## 8. The extraction path (D-021)

Phase 1 (Epic D, now): implement inside `dspy/adapters/_engine/` under an
**extraction-ready import boundary** — the engine imports only the signature
core, the types layer, and (post-E) lm15; never settings, modules, clients,
or teleprompt. A mechanical test enforces the boundary. Phase 2 (post-D/E):
extract to a standalone library — rendering/parsing over any backend
(lm15 reference; litellm/SDK alternates) and any frontend; FunctAI becomes
the first dspy-free consumer and the utility proof; this document graduates
to the contract repo. dspy keeps a thin compat layer; the engine (Epic F)
depends on the library, not the reverse.
