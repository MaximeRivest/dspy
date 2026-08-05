# Epic C — semantic roles in signatures

**Status:** design ratified; derive-and-record shipped (this branch). Roles are
recorded, not yet load-bearing. The load-bearing cutover is a later epic with
its own PR stack (§7).

Spec grounding: `roadmap/IR-program-spec.md` §Adapter-semantics. dspy's
semantic types conflate a **shape** (what the value is — pydantic territory)
with a **role** (what the field means to the exchange — genuine invention).
This epic separates them at the signature: a field's `semantic_role` is
intent, frozen, component-2 data; the strategy answering a role is mechanism,
component-4 data. The engine already resolves mechanism through one seam
(Epic A's `field_strategy_for`) and encodes values through another (Epic B's
codecs); roles are the intent-side key both will eventually be looked up by.

## 1. The vocabulary, closed and versioned

`plain | reasoning | tools | tool_calls | citations | history | media | code`

Admission rule: **does it change how the exchange is conducted?** A role is
admitted only when at least two genuinely different inference strategies can
answer it (reasoning: native channel vs textual field; tools: native FC vs
textual JSON). If a candidate only changes what the value *is*, it is a shape
— `media` earns its role because media parts route into content blocks, not
because images are bytes. Extending the vocabulary is a versioned act on this
table, never a per-call-site invention.

## 2. User-facing syntax: `role=` kwarg, types as derivation compat

**Recommendation: the explicit kwarg is the canonical syntax; the semantic
types become derivation shorthand and eventually just shapes.**

```python
answer: str = dspy.OutputField(role="citations", desc="cited answer")
thinking: str = dspy.OutputField(role="reasoning")
```

Rationale, in order of force:

1. **The role must be able to vary independently of the shape.** That is the
   entire point (§Adapter-semantics: swapping strategy must not mutate the
   signature). Typed markers can't do this: `Reasoning` pins shape `str`; a
   `role=` kwarg composes with *any* annotation.
2. **It matches the field-metadata precedent.** `desc`, `prefix` already ride
   `json_schema_extra` through `move_kwargs`; `role` is the same kind of
   declaration, and `InputField`/`OutputField` are the natural validation
   point (unknown role → immediate `ValueError`, not a deep engine failure).
3. **Typed markers stay working through derivation** (§4): annotating
   `Reasoning` means `shape str + role reasoning`, forever during the arc.
   Nothing breaks; the kwarg is strictly more expressive.

Kwarg name is `role=` (the spec's own example syntax); stored internally as
`json_schema_extra["semantic_role"]` because `role` already means
input/output *direction* on `RenderField` (the spec renames that key
`direction` in component 2; dspy code will follow suit only in the cutover
epic). Explicit `role=` **wins over derivation**; a conflict (e.g.
`Reasoning`-annotated field with `role="citations"`) is a `ValueError` at
signature build — declaring both and disagreeing is a bug, not a preference.

## 3. Validity rules

| role | direction | multiplicity | shape constraint |
|---|---|---|---|
| `plain` | either | any | any JSON-schematizable shape |
| `reasoning` | output only | ≤ 1 | str-like |
| `tools` | input only | ≤ 1 | `Tool` or `list[Tool]` |
| `tool_calls` | output only | ≤ 1 | `ToolCalls`-shaped (list of call records) |
| `citations` | output only | any | citation-bearing shape |
| `history` | input only | ≤ 1 | conversation-turn list |
| `media` | input only (today) | any | media part (image/audio/file/document) |
| `code` | either | any | str-like source |

Enforcement is **deferred to the cutover epic** (today's derive-and-record
must not reject programs the legacy types accept — e.g. the tools/tool_calls
pairing rule already lives in `NativeFunctionCallingStep` and stays there for
now). The table is ratified here so the cutover has a fixed target. `media`
as output (image generation) is an anticipated table extension, not a day-one
row.

## 4. Derivation table (legacy annotation → role)

Unwrap `Optional[...]`/`X | None` and `list[...]` first; derive on the core
type. All non-None union members must agree on a role, else `plain`.

| annotation | role |
|---|---|
| `Reasoning` | `reasoning` |
| `Tool`, `list[Tool]` | `tools` |
| `ToolCalls` | `tool_calls` |
| `Citations` | `citations` |
| `History` | `history` |
| `Image`, `Audio`, `File`, `Document` (and lists) | `media` |
| `Code` | `code` |
| anything else | `plain` |

Custom `Type` subclasses with `adapt_to_native_lm_feature` derive `plain`
today (their mechanism enters via Epic A's auto-wrapped legacy strategy); the
cutover epic gives third parties a role registration alongside strategy
registration.

## 5. How roles reach the engine

Shipped now (derive-and-record):

- `semantic_role_for(annotation)` — pure derivation, engine-private
  (`_engine/roles.py`); vocabulary constant lives in `dspy/signatures/field.py`
  (roles are signature-level intent; adapters already import signatures, so
  the layering is acyclic).
- `AdapterPlan.from_signature` resolves each field's role (explicit
  `semantic_role` in `json_schema_extra` wins, else derived) into
  `RenderField.metadata["semantic_role"]`. Metadata is consulted by nothing
  yet: zero behavior change, corpus byte-identical.
- `dump_state` persists only `prefix`/`desc`, so recording adds nothing to
  saved programs; explicit `role=` survives in the live signature object
  exactly as `desc` does. The IR's component 2 will bake it; that is the IR
  compiler's concern, not `dump_state`'s.

## 6. The strategy-resolution bridge

Today (post-Epic A) strategies resolve **by annotation type**:
`field_strategy_for(annotation)`. End state (spec): component 4's
`strategies` block binds **per role** — `reasoning: native_channel |
textual_field`, resolved against LM capabilities at bake. The bridge, in
three stages, each shippable alone:

1. **Record** (this epic): every `RenderField` carries its role. Resolution
   unchanged.
2. **Double-key** (cutover epic, PR-stacked): the registry becomes
   role-keyed with the annotation-keyed lookup as compat —
   `strategy_for(role, annotation)` tries the role table first, falls back
   to `field_strategy_for(annotation)`, and `strategy_trace` records which
   key resolved. Built-ins register under both keys; byte-identical by
   construction, proven by the corpus.
3. **Cut over**: adapters gain the per-role `strategies` binding block
   (component 4's shape); annotation-keyed lookup demoted to
   legacy-compat, warned, then removed with the semantic types' mechanism
   half. The types survive as plain shapes (or dissolve into ordinary
   pydantic models) with derivation keeping their role implication.

## 7. `_trajectory` reconciliation

The sacred-signature doctrine and roles compose without tension, and the
role marker *improves* the CoT check:

- **Declared `role="reasoning"` field** → contractual output. Strategies
  fill it (from the native channel or a textual field); it stays in the
  prediction.
- **Undeclared reasoning** (CoT's injected field) → mechanism exhaust →
  `_trajectory`, exactly as migrated. CoT's current check is *name-based*
  (`"reasoning" in signature.output_fields`); after the cutover it becomes
  *role-based* (`any field with role reasoning`), which fixes the aliasing
  hole where a user's unrelated field named `reasoning` is mistaken for a
  declared thinking channel, and conversely lets a user declare their
  channel under any name.

## 8. Deprecation policy

- Semantic-type annotations: **no warning now**; derivation makes them
  first-class citizens of the role system. Warning begins only when the
  double-key stage lands and `role=` has been public for ≥ one minor
  release; removal of their *mechanism* half no earlier than the cutover
  epic completing. Their *shape* half never breaks.
- `adapt_to_native_lm_feature`: unchanged (Epic A's decision stands);
  deprecation belongs to the exposure epic that publishes strategy
  registration.

## 9. PR stack for the cutover epic (future)

1. `role=`-aware validation rules (§3) behind the explicit kwarg only.
2. Double-keyed strategy registry + trace attribution (bridge stage 2).
3. Role-based CoT/_trajectory check (name-based check retired).
4. Per-role `strategies` block on adapters (component 4 shape) + capability
   checking at plan build.
5. Annotation-keyed lookup demoted to compat with warning.

Each PR: corpus byte-identical (or dedicated corpus commit where a *new*
declared-role path adds genuinely new cases), full matrix, zero public
surface change until PR 1 publishes the kwarg documentation.

## Non-goals of this epic

Role-validity enforcement; any resolution-order change; publishing the
strategy/codec registries; the `direction` rename; media-output roles;
per-role capability checks. All sequenced above.
