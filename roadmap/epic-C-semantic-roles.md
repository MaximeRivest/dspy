# Epic C — semantic roles in signatures

**Status:** design ratified; derive-and-record shipped (this branch). Roles are recorded, not yet load-bearing. The load-bearing cutover is a later epic with its own PR stack (§7).

Spec grounding: `roadmap/IR-program-spec.md` §Adapter-semantics. dspy's semantic types conflate a **shape** (what the value is — pydantic territory) with a **role** (what the field means to the exchange — genuine invention). This epic separates them at the signature: a field's `semantic_role` is intent, frozen, component-2 data; the strategy answering a role is mechanism, component-4 data. The engine already resolves mechanism through one seam (Epic A's `field_strategy_for`) and encodes values through another (Epic B's codecs); roles are the intent-side key both will eventually be looked up by.

## 1. The vocabulary, closed and versioned

`plain | reasoning | tools | tool_calls | citations | history | media | code`

Admission rule: **does it change how the exchange is conducted?** A role is admitted only when at least two genuinely different inference strategies can answer it (reasoning: native channel vs textual field; tools: native FC vs textual JSON). If a candidate only changes what the value *is*, it is a shape — `media` earns its role because media parts route into content blocks, not because images are bytes. Extending the vocabulary is a versioned act on this table, never a per-call-site invention.

## 2. User-facing syntax: `Annotated` markers are the primitive; four spellings, one object

**Ratified: role markers as `Annotated` metadata are the cross-surface primitive; every other spelling is sugar over them.** One role vocabulary, one marker object per role (`dspy.signatures.roles`, internal-first), four spellings that all resolve to the same registry entries:

1. **`Annotated[str, citations]`** — canonical, type-checker-transparent: mypy/pyright see `str`; dspy sees the marker. Works in every surface that accepts a Python annotation, including string signatures via the type-resolution namespace (verified: `custom_types={"Annotated": Annotated, "m": marker}` resolves today with the marker landing on `FieldInfo.metadata`).
2. **`citations[str]`** — subscriptable-marker sugar: each marker's `__getitem__(shape)` returns `Annotated[shape, marker]`, nothing else. Nests meaningfully: `list[citations[str]]` (per-item) vs `citations[list[str]]` (whole value) — both declare the field's *participation* in the role (role territory); the per-item/whole distinction is reserved for strategies (mechanism territory) and MUST NOT fork the role vocabulary.
3. **`"answer: str @citations"`** — string-signature shorthand (§2a, spec'd below, implementation deferred to the next PR).
4. **`OutputField(role="citations")`** — the kwarg spelling (shipped).

Why the marker is the primitive and not the kwarg:

1. **The role must vary independently of the shape** (§Adapter-semantics: swapping strategy must never mutate the signature). Markers compose with *any* annotation; the legacy types can't (`Reasoning` pins `str`).
2. **Cross-surface universality.** Class signatures, string signatures, and plain function signatures (FunctAI, §2b) all accept annotations; only dspy's field factories accept kwargs. One primitive, every surface.
3. **Type-checker transparency** — `Annotated` is the standard mechanism for exactly this: metadata that tools may ignore and frameworks may consume.

Conflict rule, uniform across all spellings: any two role declarations on one field that disagree (marker vs kwarg, marker vs marker, either vs a non-`plain` legacy-type derivation) → loud `ValueError` at signature/plan build. Agreement is permitted; redundancy is not an error. **Legacy types are documented as the fused spelling**: `Reasoning` ≡ `Annotated[str, reasoning]` — shape and role in one name, which is precisely the coupling the markers dissolve.

The kwarg stores as `json_schema_extra["semantic_role"]` because `role` already means input/output *direction* on `RenderField` (the IR spec renames that key `direction`; dspy follows suit only in the cutover epic).

### 2a. The `@role` string-signature shorthand (spec'd, deferred)

Grammar: `field_name[: type] @role` — the `@role` token comes after the annotation if one is present, after the name otherwise. `@reasoning` with no type defaults the shape to `str` (matching legacy `Reasoning`); an unknown `@role` errors eagerly, listing the vocabulary. `Annotated` markers in string signatures already work via the namespace (spelling 1); `@` is pure ergonomics.

Deferred from this epic **because it is not cleanly containable in the current parser**: `_parse_field_string` feeds the whole field string to `ast.parse(f"def f({field_string}): pass")`, and `@word` is a syntax error inside an argument list. Implementation therefore needs a pre-tokenization pass with two known hazards, recorded here for the implementing PR: comma splitting must be subscript-depth-aware (`dict[str, int] @plain`), and the untyped form (`answer @reasoning`) must inject the defaulted annotation before `ast.parse` sees the string. Recommended shape: depth-aware field split → per-field `@(\w+)` extraction with eager vocabulary validation → rewrite as `Annotated[<type-or-str>, <marker>]` in the parse namespace. Zero change to signatures not using `@`.

### 2b. Context: FunctAI as a third signature surface

FunctAI (`~/Projects/functai`) exposes plain Python function signatures via `@ai` and consumes the markers natively — `def f(docs: media[list[Document]]) -> citations[str]` — with no dspy field factories in sight, which is exactly why the primitive must live in the annotation, not the kwarg. Doctrine mapping for its intermediate fields, consistent with §7: body-assigned intermediates (`reasoning: str = _ai[...]`) are DECLARED (contractual outputs of the inner program); auto-inserted CoT with no body assignment is mechanism → `_trajectory`. Name-based derivation from variable names is a warned convenience there; `Annotated` is truth.

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

Enforcement is **deferred to the cutover epic** (today's derive-and-record must not reject programs the legacy types accept — e.g. the tools/tool_calls pairing rule already lives in `NativeFunctionCallingStep` and stays there for now). The table is ratified here so the cutover has a fixed target. `media` as output (image generation) is an anticipated table extension, not a day-one row.

## 4. Derivation table (legacy annotation → role)

Unwrap `Optional[...]`/`X | None` and `list[...]` first; derive on the core type. All non-None union members must agree on a role, else `plain`.

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

Custom `Type` subclasses with `adapt_to_native_lm_feature` derive `plain` today (their mechanism enters via Epic A's auto-wrapped legacy strategy); the cutover epic gives third parties a role registration alongside strategy registration.

## 5. How roles reach the engine

Shipped now (derive-and-record):

- `semantic_role_for(annotation)` — pure derivation, engine-private (`_engine/roles.py`); vocabulary constant lives in `dspy/signatures/field.py` (roles are signature-level intent; adapters already import signatures, so the layering is acyclic). Marker objects live in `dspy/signatures/roles.py` (internal-first: importable as `from dspy.signatures.roles import citations`, not exported from `dspy/__init__.py` until the cutover epic publishes the syntax). Resolution scans, in declaration strength: the `role=` kwarg, `FieldInfo.metadata` (where pydantic hoists a top-level `Annotated`'s metadata), and markers nested anywhere in the annotation tree — all must agree.
- `AdapterPlan.from_signature` resolves each field's role (explicit `semantic_role` in `json_schema_extra` wins, else derived) into `RenderField.metadata["semantic_role"]`. Metadata is consulted by nothing yet: zero behavior change, corpus byte-identical.
- `dump_state` persists only `prefix`/`desc`, so recording adds nothing to saved programs; explicit `role=` survives in the live signature object exactly as `desc` does. The IR's component 2 will bake it; that is the IR compiler's concern, not `dump_state`'s.

## 6. The strategy-resolution bridge

Today (post-Epic A) strategies resolve **by annotation type**: `field_strategy_for(annotation)`. End state (spec): component 4's `strategies` block binds **per role** — `reasoning: native_channel | textual_field`, resolved against LM capabilities at bake. The bridge, in three stages, each shippable alone:

1. **Record** (this epic): every `RenderField` carries its role. Resolution unchanged.
2. **Double-key** (cutover epic, PR-stacked): the registry becomes role-keyed with the annotation-keyed lookup as compat — `strategy_for(role, annotation)` tries the role table first, falls back to `field_strategy_for(annotation)`, and `strategy_trace` records which key resolved. Built-ins register under both keys; byte-identical by construction, proven by the corpus.
3. **Cut over**: adapters gain the per-role `strategies` binding block (component 4's shape); annotation-keyed lookup demoted to legacy-compat, warned, then removed with the semantic types' mechanism half. The types survive as plain shapes (or dissolve into ordinary pydantic models) with derivation keeping their role implication.

## 7. `_trajectory` reconciliation

The sacred-signature doctrine and roles compose without tension, and the role marker *improves* the CoT check:

- **Declared `role="reasoning"` field** → contractual output. Strategies fill it (from the native channel or a textual field); it stays in the prediction.
- **Undeclared reasoning** (CoT's injected field) → mechanism exhaust → `_trajectory`, exactly as migrated. CoT's current check is *name-based* (`"reasoning" in signature.output_fields`); after the cutover it becomes *role-based* (`any field with role reasoning`), which fixes the aliasing hole where a user's unrelated field named `reasoning` is mistaken for a declared thinking channel, and conversely lets a user declare their channel under any name.

## 8. Deprecation policy

- Semantic-type annotations: **no warning now**; derivation makes them first-class citizens of the role system. Warning begins only when the double-key stage lands and `role=` has been public for ≥ one minor release; removal of their *mechanism* half no earlier than the cutover epic completing. Their *shape* half never breaks.
- `adapt_to_native_lm_feature`: unchanged (Epic A's decision stands); deprecation belongs to the exposure epic that publishes strategy registration.

## 9. PR stack for the cutover epic (future)

1. `role=`-aware validation rules (§3) behind the explicit kwarg only. 1b. `@role` string-signature shorthand per §2a (depth-aware pre-tokenizer, eager vocabulary validation, `Annotated` rewrite) + marker export decision (`dspy.roles` public path).
2. Double-keyed strategy registry + trace attribution (bridge stage 2).
3. Role-based CoT/_trajectory check (name-based check retired).
4. Per-role `strategies` block on adapters (component 4 shape) + capability checking at plan build.
5. Annotation-keyed lookup demoted to compat with warning.

Each PR: corpus byte-identical (or dedicated corpus commit where a *new* declared-role path adds genuinely new cases), full matrix, zero public surface change until PR 1 publishes the kwarg documentation.

## Non-goals of this epic

Role-validity enforcement; any resolution-order change; publishing the strategy/codec registries; the `direction` rename; media-output roles; per-role capability checks. All sequenced above.
