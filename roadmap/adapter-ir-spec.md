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
  plan's **visible** fields only. **Only non-token-stream channels hide**
  (native reasoning, native citation assembly) — inputs essentially never
  hide, because every input goes *somewhere* in the request and the
  template says where.
- **Message content is a part list; slots decide which part kind they
  emit.** A text slot emits TextParts; a media field's slot emits media
  parts (DocumentPart/ImagePart, citable when the strategy says so) at its
  template position — the `{image}` content-splitting mechanism of the
  reference implementation, promoted to the rule. There is no separate
  part-contribution side channel: `slot_codecs` on the plan decide the
  emission kind per field.
- Strategies may emit **field transforms** into the plan (e.g. native
  citations renames the text target to `answer_text` so the text parser
  and the native assembler compose) — the transform machinery is the
  strategy's to use, recorded like everything else.
- A **textual strategy is a template-fragment provider**; preset templates
  carry the slots its fragments fill.
- **Declared capacity:** a template statically declares which roles it can
  host textually — derivable by analyzing its slots (this is why the
  template language is closed). Capacity separates the live-call lane
  (content messages) from the example lane (demos/history directive
  patterns), and media/tools hosting is a per-field question — an input
  field lands textually only where an inputs iteration or its own named
  slot places it. Bake checks the triple (signature roles ×
  LM capabilities × template capacity) and refuses loudly naming all three
  when no lane exists (ADP-006).
- An explicit slot reference to a natively-served field is a bake-time
  conflict, refused loudly — never an empty render (ADP-007).

## 3. The template language (closed, versioned)

A template is a list of messages. Each message is `{role, content}` or a
directive. Content strings may use:

- **Value slots:** `{field_name}` (rendered through the field's bound input
  codec), `{instruction}` (3a). `{instruction}` takes an optional
  `style=` from a closed set (`raw` default; `indented` is the historical
  dedent-then-eight-space objective block) — presentation transforms on
  the instruction are declared styles, never renderer magic.
- **Aggregate slots:** `{inputs(style=…)}`, `{outputs(style=…)}`,
  `{demos(style=…)}`, `{history(style=…)}` — `style` names an entry in
  that aggregate's closed style vocabulary (codec-aligned names; e.g.
  `outputs` styles include `json_object`, which renders typed placeholders
  in schema position and the call's values in assistant position).
- **Loop blocks:** `{% for f in inputs|outputs [separator='…'] [strip] %} …
  {% endfor %}` with the closed `f.*` vocabulary: `i/index, name, type,
  desc, desc_suffix, value, placeholder, typed_placeholder, marker,
  chat_type_hint`. The bare `strip` flag applies `str.strip()` to the
  joined result — the historical join-then-strip section shape, carried as
  declared data (D-2 proved byte parity is unreachable without it).
  `{f.value}` spells the value through the direction's bound codec;
  `{f.typed_placeholder}` spells the field's *schema* through the same
  binding (the codec's schema spelling — the shared text codec renders the
  historical placeholder-plus-type-note, the `baml` codec renders
  schema prose). A codec is a render/parse/schema triple: how a value is
  shown, how an emission is recovered, and how the expected shape is
  described.
- **Section blocks:** `{% section strip %} … {% endsection %}` — the
  enclosed content renders and the joined result is `str.strip()`'d: the
  historical join-then-strip *region* shape, which is what lets a trailing
  subsection that renders empty collapse together with its literal
  separators (the xml preset's structure region with zero output fields is
  unreachable without it). Sections may contain loops and slots; they do
  not nest, may not appear inside loop bodies, and carry no fragment
  slots.
- **Value-presence semantics (normative, corpus-pinned):** in valued
  positions, `inputs` loops iterate only fields present in the values
  dict (the historical skip of absent inputs), while `outputs` loops
  iterate every visible field — assistant positions render absent values
  through the missing-field message, user positions render schema-side
  (the output-requirements enumeration names every field). Schema
  positions render without call values: they iterate everything and refuse
  `{f.value}` — and every rendering surface, `preview()` included, applies
  the same schema-position context, so preview bytes are engine bytes.
- **User-turn assembly (normative, corpus-pinned):** a rendered user
  message assembles through the historical join-then-strip — prefix, body,
  and suffix join on blank lines, parts that render empty contribute no
  join element, and the joined result is `str.strip()`'d; a user message
  whose assembled content is empty is omitted from the message list
  entirely. System and assistant messages emit their rendered bytes
  verbatim, and always emit. (The asymmetry is the legacy pipeline's,
  carried as declared semantics, not renderer accident.)
- **Reserved names:** `instruction`, `inputs`, `outputs`, `demos`,
  `history`, `fragments`, and `field` are the language's reserved slot
  names; the call forms (`{instruction(...)}`, `{inputs(...)}`, …) always
  denote the reserved construct. A signature may declare a field carrying
  a reserved name — loops and aggregates render it like any other field —
  but such a field has no bare value-slot spelling: the bare
  aggregate/fragments/field spellings refuse at parse naming the possible
  collision, and a bare `{instruction}` rendered against a signature that
  declares a field named `instruction` refuses at render naming the
  collision (`{instruction(style='raw')}` is the unambiguous spelling of
  the signature instructions). Reserved-name shadowing is never silent.
- **The `{field('name')}` escape spelling:** the quoted-name value-slot
  call form denotes the signature field `name` unambiguously, whatever the
  name — it is the ONLY value-slot spelling for a field carrying a
  reserved name (`{field('inputs')}`, `{field('instruction')}`), and every
  reserved-collision refusal names it as the way out. For non-reserved
  names it is equivalent to the bare spelling. `field` itself is reserved:
  a bare `{field}` refuses at parse naming the call form.
- **Fragment slots (positional):** `{fragments('system')}` and
  `{fragments('user')}` — placed once per preset; a textual strategy's
  fragment names the slot it targets. Empty slots render as nothing — a
  slot alone on its line removes the whole line, including its newline —
  so a preset pays zero bytes when no textual strategy fires (this is what
  keeps byte-parity with the historical adapters satisfiable).
- **Escapes:** `{{`/`}}` literal braces; `{{{f.name}}}` literal placeholder.

Directive messages expand at render time: `{"role": "demos"}` → user/
assistant pairs per demo (assistant format follows the parser binding);
`{"role": "history"}` → prior turns. Directives are role-named on purpose —
they are the textual strategies of the `history` and demo machinery.
A directive carrying no `user=`/`assistant=` pair falls back, in order: a
history directive inherits the demos directive's patterns when the
template carries one; otherwise the directive expands through the
language's **default turn patterns** — user
`{% for f in inputs separator='\n\n' %}[[ ## {f.name} ## ]]\n{f.value}{% endfor %}`
and the same shape over `outputs` with `strip` for the assistant turn (the
marker pair, mirroring the reference implementation's format-by-default
behavior). Zero demos and zero history turns expand to nothing.
Consequently every template that parses can render: eager validation
admits exactly the renderable set.

**Deliberately NOT in the language:** general Jinja, arbitrary expressions,
user-defined control flow. Analyzability (capacity derivation, diffing,
optimization) is a spec requirement. Registered helper functions are an
authored-code escape hatch and carry provenance (ADP-011).

**Discoverability (normative).** Because the language is closed, its entire
vocabulary is enumerable — and the contract requires that it be enumerable
*as data, from one source*:

- **The vocabulary is a data structure** (slots, loop variables, directive
  roles, codec/style names, parse modes, fragment slot names), defined once
  in the implementation; the validator, the error messages, the docs table,
  and `describe_template_language()` (a public introspection call returning
  the vocabulary) all read the same structure. A conformance test asserts
  the docs table equals the data — vocabulary drift dies the way
  literal-table key drift did.
- **Errors teach.** Templates parse eagerly at preset construction; every
  unknown construct refuses loudly naming itself AND the valid set in its
  category (`unknown slot {outpts()} — valid slots: instruction, inputs,
  outputs, demos, history, fragments; valid fields here: question, answer`).
  Learning the language by making mistakes must work.
- **`preview()` is part of the contract**: render a preset against a
  SignatureCore + values with no LM call (ADP-002 guarantees this is
  possible), so the learn-by-looking loop is always available.

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
preset: it is preset `json` + the `baml` codec bindings + the pairing's
schema-prose system arrangement, carried as template data. The split
follows the layer law exactly: the codec owns the schema-prose *spelling*
(shape-generic, any annotation), the template owns the *arrangement* (which
sentences appear, where the markers and the completed marker sit — D-018's
literal-table territory, which is why the arrangement cannot hide inside a
codec). No `baml` preset name exists; the pairing serializes as an ordinary
component-4 entry whose template carries the arrangement.

Serialized presets carry a `versions` block — this contract's version plus
the versions of the §5 vocabularies in force, mirroring the ProgramIR
manifest's block (ratified 2026-08-06, D-024). A reader meeting an unknown
major refuses loudly naming both versions, and a preset with no block at all
is refused as malformed (no unversioned grandfathering) — ADP-005's
discipline applied to time; without the block, forward-compat refusal is
unimplementable and vocabulary extension is silent drift.

## 5. Closed vocabularies

- **Roles:** `plain, reasoning, tools, tool_calls, citations, history,
  media, code` (+ anticipated: `refusal`). Grounded in the typed LM
  contract; versioned with it.
- **Strategies (per role, initial):** reasoning `native_channel |
  textual_field | prefill`; tools `native_fc | textual_json | xml_dispatch`;
  citations `native | span_markers | json_quotes`; media `native_parts |
  url_reference`; history `directive_turns | inline`.
- **Codecs (initial):** `text_pythonish` (the historical implicit pair),
  `pydantic_json` (indented model dumps), `baml` (indented-pydantic value
  spelling + the simplified schema-prose schema spelling — ``Output field
  `name` should be of type: …`` over any annotation), `json`, `xml`,
  `yaml`. Codecs are shape-generic by law.
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

## 9. Extension & shipping (customization contract)

Third parties extend the system at the pools; what ships depends on what
the extension *is*:

- **Templates/presets are pure data — always baked, never trusted-code.**
  The constrained language means a custom template serializes into the
  artifact as JSON and loads with no exec and no flags. Programs may carry
  prompt shapes that do not exist in the host implementation.
- **Codecs, strategies, and authored parsers are code — three-origin rule**
  (mirrors ProgramIR §e0-class / tool bodies):
  `builtin` (named ref, resolved internally) · `packaged` (import path +
  dist/version; the program's PEP 723 env manifest provides it; trust flows
  from the baked lockfile; entry-point registration, pytest-plugin style) ·
  `authored` (source baked into the artifact, exec'd in an isolated
  namespace at load, identity-verified, `authored_by` + provenance —
  ADP-011).
- **Registration is the API and carries an admission gate**:
  `register_codec/strategy/preset(...)` (public with Epic D). Codec
  registration runs the schema-generated round-trip probe battery
  (ADP-003) — a codec failing `parse(render(x)) == x` on adversarial
  probes is refused at registration, before it touches any program.
  Strategies self-declare capability requirements and which shapes they
  serve; presets validate their templates eagerly (§3 discoverability).
- **In the ProgramIR**: component-4 pools hold origin-tagged entries;
  bindings reference by name; **load is the link step** — builtin resolves
  internally, packaged must import at the declared version (mismatch
  refuses loudly naming the entry), authored execs baked source and
  verifies identity; dangling refs are link errors (ADP-005). An
  optimizer-discovered template or codec ships identically with
  `authored_by: optimizer` — a search result becomes a distributable
  artifact through the same door.
- **Cross-language receivers (ratified 2026-08-06, D-025/D-026):**
  `packaged`/`authored` entries carry `language`; `builtin` needs none. To a
  receiving engine in another language, `builtin` resolves internally (same
  behavior, its own implementation, corpus-conformant);
  `packaged`/`authored` entries in a foreign language are refusable at the
  profile level (ProgramIR §e0-lang's declared-tier profile).
  Templates/presets-as-data are the portable customization path: a program
  restricted to them plus builtin codecs/strategies loads on any conforming
  engine with no code execution. An engine MAY evaluate foreign authored
  codecs/strategies out-of-process — ADP-002 purity is what makes that
  sound — but that is an engine quality upgrade, invisible to the artifact,
  never a requirement: adapter entries carry no placement and no credential
  (ProgramIR §Adapter-notes — a law to absorb into ADP-002 at graduation),
  so authored adapter code is the one authored-code class that does not
  rung-walk.
