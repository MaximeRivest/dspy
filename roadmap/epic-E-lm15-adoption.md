# Epic E — lm15 adoption

> **SHELVED (2026-08-10, Maxime's direction — D-038 draft).** The
> current LM stack is assumed as-is by all downstream epics; lm15 was
> inspiration, not a dependency. This charter is kept for whenever the
> LM-contract seam revives. Nothing below is current planning.

**Status:** v1 (2026-08-07) — FRESH DOC, PLANNING STAGE. Written doc-first
per `06-orchestration.md`'s rule for Epic E ("engineer writes epic doc
first"); **no code changes accompany this doc.** Dependencies: Epic D is
complete (`roadmap/epic-D-adapter-serializer.md` status line: "EPIC
COMPLETE + D-δ fix wave", checkpoint C1 closed, D-031 ratified). Epic I
shipped locally (`roadmap/epic-I-exporter.md` status line: "v4 SHIPPED
LOCALLY (2026-08-07)"). Per `02-state.md`'s D-δ entry, Epic E was on hold
pending Maxime's exporter work; that condition is now satisfied — **the
hold is lifted by this doc's dependency check, not by fiat**; Maxime should
confirm before implementation forks are cut (see Open Questions).

## Charter

Epic D made the adapter layer's render/parse contract into data. Epic E
does the same for the *LM* side of component 8a: dspy's typed-LM contract
— today `dspy/core/types.py`'s in-repo `LMRequestPatch`/`LMConfig`/
`LMMessage`/part hierarchy plus `dspy/clients/`'s `forward_contract =
"typed_lm"` seam (`base_lm.py`) — becomes a veneer over (or direct
adoption of) **lm15**'s `Request`/`Response`/`StreamEvent` types
(`~/Projects/lm15-dev/lm15-python/lm15/`, governed by
`~/Projects/lm15-dev/lm15-contract/`).

This is ratified design, not a proposal from scratch: D-015 ("lm15 adopted
as the engine's LM substrate") and the `IR-program-spec.md` §Adapter-
semantics grounding note (component 8a, lines ~640-729) already commit to
this outcome empirically — lm15's part vocabulary
(`Text, Thinking, ToolCall, ToolResult, Citation, Image, Audio, Video,
Document, Binary, Refusal`) converged independently on the same role list
dspy derived top-down from the adapter's semantic roles (component 2), and
lm15 patches two known holes: `refusal` as an anticipated role, and
`ContinuationState` as the `history` role's missing replay mechanism for
opaque provider state (e.g. thinking-block replay signatures). This epic's
job is the *engineering* of that adoption — the veneer, the streaming
routing, the retirement path — not re-litigating whether it happens.

Three concrete outcomes fall out:

1. The engine's LM-facing contract stops being a parallel in-repo type
   system and starts being a thin projection over lm15.
2. Streaming becomes role-typed: deltas route by declared/undeclared role
   through the `AdapterPlan`, per the streaming mermaid diagram in
   `IR-program-spec.md` (lines 717-729) — buffered replay becomes the fold
   case of the same routing, not a separate code path.
3. **litellm is removed in full, inside this epic** (Maxime's ruling,
   2026-08-07 — overrides the ambiguity between `02-state.md`'s kill list
   and `06-orchestration.md`'s Epic H framing noted in prior drafts):
   the dependency is dropped from `pyproject.toml`, every `import litellm`
   in `dspy/` is deleted, and `dspy.LM` finishes its move to the pure
   router shape (per the `lm-router-end-state` design note) with
   `OpenAICompatLM`-family conventions as the load-bearing data-plane
   contract. Epic H's kill-list entry for litellm is satisfied by E, not
   deferred to H.

## Scope split (from `06-orchestration.md`, verbatim structure)

**E-α — the veneer.** Epic doc first (this doc); then: dspy's typed-lm
contract becomes a veneer over lm15 `Request`/`Response` behind the
engine's import boundary; lm15 conformance corpus wired into dspy-ci;
capability declarations mapped.
**Oracle:** 304-check conformance + golden parity + matrix.

**E-β — role-typed streaming.** Deltas route by role (declared → field
stream; undeclared reasoning → `_trajectory` live); `StreamListener`
reframed as the textual polyfill; buffered replay = the fold; litellm
retirement path opens (router memory: `dspy.LM` → pure router).
**Oracle:** existing streaming tests + new role-routing tests + matrix.

These are the two forks 06-orchestration.md names, run serially (E-α
before E-β — streaming routing needs the veneer's typed `Response`/
`StreamEvent` surface to route against). `03-campaign.md`'s one-paragraph
summary folds both under one oracle line ("lm15 conformance corpus (304
checks) + golden parity + the existing streaming tests") — this doc treats
that as the whole-epic oracle, with E-α and E-β each owning the subset
relevant to their own DoD below.

**Checkpoint C2 (Maxime)**, per `06-orchestration.md`, lands after Epic E:
push, then an AnthropicLM sequencing decision — the typed-LM family plan
says AnthropicLM comes after streaming lands, but 06-orchestration.md
notes it "can now interleave with F if wanted." That decision is
Maxime's, not this doc's.

## Definition of done (mechanical)

**E-α:**

1. **Back-edge retired:** `parser_hook.py → dspy.clients.openai_format`
   — the sole `KNOWN_BACK_EDGES` pin in
   `tests/adapters/engine/test_import_boundary.py` annotated "retired by
   the lm15 veneer (Epic E)". `ResponseView.from_lm_response` currently
   calls `dspy.clients.openai_format.legacy_outputs_from_lm_response` to
   derive legacy `str`/`dict` output shapes from a typed `LMResponse`;
   E-α's veneer must let this facade construct itself directly from
   lm15's typed `Response`/parts without reaching into `dspy.clients`.
   Deleting the pin (not just leaving it stale) is part of the DoD per
   the boundary test's own shrink-only discipline.
2. **`ALLOWED_PREFIXES` gains `lm15`:** the boundary test's own assertion
   text already anticipates this ("the engine imports only the signature
   core, the types layer, and (post-E) lm15"); E-α adds the prefix and
   proves no other new back-edges appear.
3. **`dspy/core/types.py` becomes a veneer, not a parallel contract:**
   `LMMessage`, the `LMBasePart` hierarchy (`LMTextPart`, `LMThinkingPart`,
   `LMToolCallPart`, `LMToolResultPart`, `LMCitationPart`, `LMRefusalPart`,
   `LMImagePart`/`LMAudioPart`/`LMVideoPart`/`LMDocumentPart`/
   `LMBinaryPart`), `LMConfig`, `LMRequestPatch` either project onto or
   are replaced by lm15's `Request`/`Response`/`StreamEvent` and part
   types. The `extensions` (request-bound, user-supplied) vs
   `provider_data` (response-bound, provider-returned) direction-of-
   ownership asymmetry — already adopted as adapter-config law per
   `IR-program-spec.md` — must hold in the veneer, not just in prose.
4. **`base_lm.py`'s `forward_contract` seam absorbs lm15 types:** the
   `"typed_lm"` contract (`forward(request) -> response`) currently
   validates against the in-repo `LMResponse`; it validates against
   lm15's `Response` (or the veneer wrapping it) instead. `"legacy"`
   stays as-is — this epic does not touch the legacy shim path.
5. **lm15 conformance corpus wired into dspy-ci:** the 304-check harness
   (request 110 / response 102 / stream 8 / error 16 / serde 68, per
   `cross-language.md`'s asset description) runs as a CI gate, not just
   a manual check.
6. **Capability declarations mapped:** lm15's per-provider capability
   surface (whatever shape it ships today in `lm15-python`) is mapped to
   the adapter engine's `capability_requirements` consultation point
   (D-δ shipped this consultation at plan time; E-α wires the LM side of
   it).
7. **Golden corpus zero-drift** through the veneer swap (same discipline
   D used throughout: corpus is the gate).

**E-β:**

1. **Delta routing implemented per the streaming mermaid diagram**
   (`IR-program-spec.md` lines 717-729): `ThinkingDelta` for a declared
   `reasoning[str]` field streams as that field's partial value;
   `ThinkingDelta` for an undeclared reasoning role streams into
   `_trajectory` live (exhaust is exhaust even mid-flight, per
   §d-sacred); `ToolCallDelta`/`CitationDelta` accumulate under the
   role's bound strategy, whose parser owns the fold; `TextDelta` for
   textual strategies is split by the format's section-parser.
2. **`StreamListener` reframed, not rewritten out from under itself:**
   today's marker-matching `StreamListener` becomes documented and
   implemented as the textual-strategy polyfill of the typed routing —
   existing behavior for textual strategies must not regress.
3. **Buffered replay = the fold:** streamed and unstreamed execution
   share one parsing contract (fold the events, then parse as
   unstreamed — lm15's `Result` assembler is named as the reference
   fold in the spec).
4. **litellm removed in full:** the dependency is dropped entirely inside
   this epic (Maxime's ruling — see Charter item 3), not staged into
   Epic H. `dspy.LM` reaches the pure-router shape the `lm-router-end-state`
   note describes, with `OpenAICompatLM`-family conventions as the
   load-bearing data-plane contract. DoD includes: `litellm` absent from
   `pyproject.toml`'s dependencies, zero `import litellm` remaining under
   `dspy/`, and the full matrix green with it gone.
5. **Streaming test suites green:** existing `tests/streaming/*` plus new
   role-routing tests, run through the matrix (48-core server, Pythons
   3.10–3.14 per the remote-test-server memory — not local pytest).
6. Known gap from the typed-lm-family-plan memory (dated 2026-07-20,
   verify current before treating as fact): DSPy stream listeners
   reportedly emit zero chunks for reasoning-first models (thinking-
   before-text streams never match field-boundary detection). E-β's
   role-typed routing is a plausible fix path since it routes by
   declared delta type rather than textual marker-matching — the DoD
   should include a regression check against this specific failure
   mode, gated on re-verifying the gap still exists.

## Known risks / open questions

- **lm15-go / lm15-ts immaturity** (`cross-language.md`, "lm15-go and
  lm15-ts exist" asset entry): both pass the full 304-check harness but
  are unpublished (no npm/module release engineering), have no
  `CONTRACT_PIN` discipline (sibling-path checkout only), and are frozen
  at a pre-review API. Corpus holes downstream: no streamed-tool-call/
  thinking/mid-stream-error fixtures, no `openai_chat` error fixtures,
  compat presets unpinned, serde/mapping rules living outside the
  contract repo. **Epic E as scoped (E-α/E-β) is a Python-only veneer
  over `lm15-python`**, which does have a `CONTRACT_PIN` file and is the
  most mature of the three — so the go/ts holes likely don't block this
  epic directly. But they do matter for the *next* consumer: if Epic E's
  veneer becomes the reference implementation other-language dspy ports
  build against (per D-028's grade-1/grade-2 framing), the go/ts holes
  become that port's blocker, not this epic's. Flagging this distinction
  explicitly rather than assuming it away.
- **litellm retirement blast radius — resolved.** Maxime's ruling
  (2026-08-07): litellm is removed in full inside Epic E, not staged
  into Epic H. See Charter item 3 and the E-β DoD above.
- **`ContinuationState` — resolved, not open.** Maxime's ruling
  (2026-08-07): this is an lm15 (wire-layer) concern, not a
  ProgramIR/adapter-IR concern, and lm15 already ships the answer —
  `ContinuationState` is a shipped, frozen dataclass in
  `lm15-python/lm15/types.py` (`{provider, kind, data}`), already
  produced by lm15's Anthropic provider on thinking-block responses, and
  designed to travel attached to the `Message`/`Part` it describes rather
  than as a detached side-channel. No new design work is needed at the
  adapter-IR layer: the veneer's job is simply to not strip unknown/
  opaque fields when it saves and replays demos or history turns —
  `ContinuationState` round-trips for free if the veneer treats it as
  ordinary part data. This replaces the "unresolved design decision"
  framing in prior drafts; the only DoD item is a round-trip test proving
  a `ContinuationState`-bearing part survives a save/replay cycle
  unchanged.
- **Role-typed streaming vs the reasoning-model gap.** The typed-lm-
  family-plan memory (17-18 days old, unverified against current code)
  flags reasoning-first streams (e.g. qwen3.6 on vLLM) as producing zero
  chunks under today's marker-matching `StreamListener`. If E-β's routing
  genuinely fixes this it's a welcome side effect, but the epic's DoD
  should not silently assume the fix without an explicit regression test
  naming the failure mode, since the underlying claim is stale and
  unverified.
- **"The veneer deserves its own design pass"** — `03-campaign.md`'s own
  Epic E paragraph says this explicitly, as the reason E is sequenced
  after D rather than concurrently. This doc is the epic doc, not that
  design pass; a closer design note on the exact `Request`/`Response`/
  `StreamEvent` ↔ `dspy/core/types.py` field mapping (which fields
  project 1:1, which need adapter logic, which in-repo types are deleted
  outright vs. kept as thin aliases) should precede E-α's implementation
  fork.
- **`dspy.core` is a new top-level package** (not `dspy/clients/`) already
  carrying `LMMessage`/`LMConfig`/the part hierarchy — separate from
  `dspy/clients/base_lm.py`'s `forward_contract` seam and
  `dspy/clients/openai_format.py`'s legacy-shape derivation. The veneer
  touches at least these three files/modules plus
  `dspy/adapters/_engine/parser_hook.py`; a fuller module inventory
  belongs in the pre-implementation design note above, not guessed here.

## Architectural invariant: render/parse stays wire-library-agnostic

Maxime's ruling (2026-08-07), stated explicitly because Epic E is exactly
the epic that could tempt a violation: **the adapter-IR rendering/parsing
layer must remain independent of whichever wire library sits underneath
it.** This is not a new requirement — `adapter-ir-spec.md` line 17 already
states the layering ("lm15 owns the wire... This contract owns signature
⇄ messages, rendering into lm15 `Message` values (**or a structurally
identical dict form for non-lm15 backends**)") — but Epic E is where it
gets tested for real, because it is the first epic to actually wire a
concrete backend (lm15) underneath the engine.

**What this means concretely for E-α:** `dspy/adapters/_engine/` renders
to and parses from a message/part *shape* (fields, roles, parts), never
to lm15's client, transport, or provider-request machinery directly. If
the veneer is built correctly, the same rendering/parsing code should
work unmodified against litellm's message shape, the raw OpenAI SDK's
message shape, or a hand-rolled dict — swapping the wire library should
be a translation at the boundary (lm15's `Request`/`Response` ↔ the
engine's `RenderField`/`ResponseView`), never a change inside the
template/codec/strategy machinery itself. The mechanical import-boundary
test already enforces the *dependency* direction (engine never imports
`dspy.clients`); it does not yet enforce this *shape-independence*
property, and it should grow a check for it as part of E-α: something
that proves the engine's render/parse path can be pointed at a
structurally-equivalent non-lm15 message dict and produce byte-identical
output, so this stays a provable property, not an aspiration.

## Sequencing / dependencies

- **Epic D: complete.** `roadmap/epic-D-adapter-serializer.md` status
  line reads "v6 (2026-08-07) — EPIC COMPLETE + D-δ fix wave"; checkpoint
  C1 closed (D-031, D-032 ratified in `05-decisions.md`). D-016 ratified
  the campaign order D → E → F → G → H, with the explicit rationale "D
  has zero LM-layer dependency; the veneer deserves its own design pass."
- **Epic I: shipped locally.** `roadmap/epic-I-exporter.md` status line
  reads "v4 SHIPPED LOCALLY (2026-08-07)"; `02-state.md`'s D-δ entry
  recorded Epic E as on hold specifically because "he [Maxime] is driving
  the exporter (epic-I draft) himself in a separate session; the
  coordinator holds until he reports back." Epic I's shipped status
  satisfies that reported-back condition as far as this doc can verify
  from the state map — **Maxime should confirm the hold is intentionally
  lifted**, since the hold was procedural (waiting on his session) rather
  than a dependency this doc can unilaterally clear.
- **D-030** ("no upstream sync before Epic E") stays in force through
  this epic — E does not change that ruling, it operates under it.
- **D-015 / D-016** are the ratifying decisions this epic's charter
  executes; this doc does not re-ratify them, only cites them.
- No new D-numbers are proposed or ratified by this doc. Where a design
  question surfaces above that needs a ruling (litellm blast radius,
  ContinuationState design, the hold-lift confirmation), it is named as
  an open question for Maxime, not decided here.

## Non-goals / explicitly deferred

- **`ParseContext.lm` retirement** — dies when TwoStep expands as a
  lowering, which is Epic F ("F-β — the lowering substrate... TwoStep
  expands (`ParseContext.lm` dies)"), not E.
- **Chat→JSON fallback + structured-output retry becoming error-policy
  lowerings** — also Epic F-β, not E.
- **`formats/twostep.py → dspy.adapters.chat_adapter` back-edge** —
  dies with F per the boundary test's own comment, not E.
- **AnthropicLM / OpenAIResponsesLM / GoogleLM** — the typed-LM family
  sequence continues after streaming lands (per the typed-lm-family-plan
  memory), but which epic they land in is Checkpoint C2's decision
  ("can now interleave with F if wanted"), not scoped by this doc.
- **Role vocabulary extensions** (refusal made concrete beyond
  "anticipated", media-out, video) — `03-campaign.md`'s "Deliberately
  NOT being built" list: "Vocabulary is versioned governance, not a
  drive-by."
- **Optimizers over any new axes this epic touches** — same list:
  "Substrate first."
- **Standalone extraction of the render/parse layer as its own library**
  — D-021 defers this until "AFTER D/E stabilize"; E does not start it,
  only continues building inside the mechanically-enforced import
  boundary that makes the eventual extraction "a move, not a surgery."
