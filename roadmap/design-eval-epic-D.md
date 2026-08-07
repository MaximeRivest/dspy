# Epic D surface — the ten-persona design evaluation

**Method.** Ten independent agents, each a different user persona, exercised the
new public surface with real, executed code: 220 exercises, every judgment
backed by the code or error that produced it. Raw reports (32 bug claims, 79
friction items, full repros): `review_packet/epic-d-stress-personas.json`.
This document is the synthesis: the report card, the convergent findings, the
deduplicated defect list, and what it means for the C1 name ratification.

## Report card

| persona | aspect | grade |
|---|---|---|
| Prompt golfer | exact-control authoring | A- |
| Generic-template author | loops + `{f.*}` vocabulary | A- |
| Portability user | entries as data | A- |
| Newcomer | docs + errors only | A- |
| Few-shot & history user | demos/history mechanisms | B+ (surface A-, semantics B-) |
| Structured-output user | parse modes + typed outputs | B+ (authoring A-, parse errors C+) |
| Roles author | the four spellings | B+ |
| Strategies user | `strategies={}` delivery | B+ |
| Codec author | writing + registering codecs | B+ |
| Strategy author | writing + registering strategies | B- |

Consensus sentence, appearing near-verbatim in seven reports: *the teaching
errors are the best I have used in any library of this kind* — closed
vocabularies quoted back at the moment of the mistake, time-to-first-template
under a minute, recipes run as written, serde round-trip laws hold everywhere
including fresh-process reload from disk, template-syntax injection fully
neutralized.

## The one big theme

**The declaration layer is A-grade; the plan layer beneath it is only
half-surfaced.** Five personas hit the same wall from five directions:

- `format()`/`preview()` bypass planning, so pure inspection *lies* whenever a
  strategy or native channel is in play — the flagship "inspect without
  spending a token" loop shows bytes a live call won't send.
- The `"auto"` strategy resolution is recorded on a plan no public surface
  exposes — no accessor, nothing in `inspect_history`.
- Statically decidable checks fire only after the paid LM call: `full_text`
  with two output fields, conflicting role declarations, an unhonorable
  native binding on an incapable LM (the docs promise all three refuse
  early; none do).

This is one coherent design gap, not scattered bugs. The fix direction is
one move: make bake reachable from the pure surface (an optional `lm=` on
`preview()`/`format()`, construction-time static checks, and a public
plan/explain accessor).

## The other convergent findings

**2. Schema-position silent empties** (3 personas). Bare `{q}` in a system
message renders as empty string with the value present — while the loop
spelling `{f.value}` in the same position refuses loudly and the unknown-slot
error claims the field is "available here". Three surfaces, three behaviors,
one mistake; the single most common hand-tuned pattern ("You are the
assistant for {company}") silently corrupts. Bare value slots in schema
positions should refuse like `{f.value}` does.

**3. Roles are write-only, and the four spellings are not yet one meaning**
(3 personas). No public way to read a field's role back; conflicting
declarations do **not** "raise immediately" as the docs promise (they explode
at serving time, invisible to preview, without naming the field); and — the
biggest semantic gap — **the split spellings never engage strategy
resolution**: only the fused legacy types (`dspy.Reasoning`) do, because the
builder consults strategies only for `Type`-subclass annotations in
`native_response_types`. `"answer @reasoning"` + `strategies={"reasoning":
...}` silently no-ops.

**4. The two extension doors don't compose** (2 personas). A
`register_strategy`'d strategy is not bindable via `strategies=` (closed
builtin-only vocabulary) and is inert unless the annotation is in
`native_response_types`; strategy authoring requires private `_engine`
imports (`AdapterPatch`, `HideOutputField`); codec method signatures are
learnable only by instrumenting the admission battery; register-on-import
(the natural pip-package pattern) is not idempotent; and the two registries
disagree on mutation semantics (codec refuses duplicates, strategy silently
replaces).

**5. Error attribution and vocabulary honesty** (4 personas).
`TemplateAdapter` failures report "Adapter ChatAdapter failed…" — a class the
user never constructed; "valid strategies" lists advertise six names that
then refuse as unimplemented; internal jargon (`L5`, `ADP-003`, "spec
section 3") leaks into user-facing errors.

**6. The json parse lane is the weak parser** (2 personas). It leaks raw
pydantic `ValidationError`s with no field name and no LM response (chat mode
names both); it hands codecs *decoded* JSON values where the contract (and
the admission battery) promise strings — so an admitted custom codec can
crash at runtime; `Literal[1,2,3]` and `IntEnum` cannot parse the values
their own prompts instruct; `Optional` output keys are required rather than
optional.

**7. Legacy semantics leak into the new authoring surface** (3 personas).
User messages join-strip and silently drop when empty, with no opt-out and no
user-facing documentation; incomplete demos inject ChatAdapter's "This is an
example of the task…" preamble into *authored* directive patterns; a bare
history directive inherits the demos directive's custom patterns
(one-directional spooky action); directive defaults inject `[[ ## ]]` chat
markers into `full_text` templates; a second `History` field's turns vanish
silently (an L5 violation by the design's own rule).

**8. Serde is the crown jewel, with one real hole** (all ten used it; the
portability persona confirmed the D-γ report's flag from the outside):
`dump_entry()` does not capture `use_json_adapter_fallback` /
`use_native_function_calling` — behaviorally different adapters serialize
byte-identically, so load can change behavior. Also: `PresetAdapter` is
returned by `load_entry` and named in docs but exported from nowhere;
`format()` crashes on string signatures that `preview()` accepts; the
version-refusal wording doesn't state the semver-0 policy it enforces.

## Deduplicated defect list (32 claims → 24 distinct, all repro-backed)

High-impact (candidates for a fix wave before broad adoption):
1. Split-role spellings never engage strategy resolution (finding 3).
2. Registered strategies not bindable / inert outside `native_response_types` (finding 4).
3. `format()`/`preview()` blind to planning; no public plan accessor (finding 1).
4. Bare value slots in schema positions silently empty (finding 2).
5. `dump_entry` loses constructor behavior flags — identical entries, different behavior (finding 8).
6. Role conflicts don't raise at declaration; kwarg silently wins (finding 3).
7. `full_text` arity + unhonorable-native-binding checks fire post-LM-call (finding 1).
8. json lane: raw ValidationError leaks; codec `parse_value` contract violated with decoded values (finding 6).
9. TemplateAdapter path never invokes strategy-contributed parser hooks — hidden field returns `None` silently (strategy author).
10. A registered role strategy hijacks an explicit `native` binding instead of refusing (strategy author).
11. Second `History` field's turns silently dropped (few-shot persona).
12. Incomplete-demo preamble injected into authored patterns (few-shot persona).

Medium: parse-error misattribution to legacy class names; advertised-but-
unimplemented strategy names in "valid" lists; `Literal[int]`/`IntEnum`
coercion failures; `Optional`-key hard-fail in json mode; unicode field names
unreferenceable even via `{field('name')}`; `PresetAdapter` unexported;
`format()` vs `preview()` string-signature asymmetry; `capability_requirements`
demanded at registration but never consulted at plan time; codec battery
missing the None probe its docs claim; `describe_template_language()` omits
the loop-block grammar; registries' reload/idempotence; authored-codec
exec errors escape unwrapped; `{f.role}` missing from the loop vocabulary;
`str()`-vs-typed inconsistency between `inputs(style='json')` and
`json_object`.

## The C1 names verdict

**No persona wanted a rename.** The nouns — `TemplateAdapter`, `dspy.roles`,
`@role`, `register_codec/strategy/preset`, `dump_entry`/`load_entry`/
`literal_table`, `describe_template_language`, `strategies=` — all graded A
as *names*. Two naming-adjacent actions are needed: **export
`PresetAdapter`** (users already hold instances of it), and note the
`parser` (entry key) vs `parse_mode` (constructor) split as a deliberate
choice or unify it. The names can be ratified as-is; every defect above is
semantics beneath stable names.

## Recommendation

One more fix wave (call it D-δ) before the surface is promoted beyond early
adopters, scoped to the high-impact twelve — most cluster into three real
work items: (a) roles/strategies actually load-bearing end-to-end (items 1,
2, 6, 9, 10), (b) bake reachable from the pure surface + static checks at
construction (3, 7), (c) authoring-surface honesty (4, 5, 11, 12). The
medium list rides along where cheap. The delights — teaching errors, serde
laws, injection safety, recipe fidelity — are the identity of this surface;
every fix should preserve them.
