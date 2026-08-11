# Classified flow — the dspy side of the capability story

> 2026-08-10 night: the trust arc gained its last two asks (spec/
> trust.md 6–7): **trust profiles** — named requirement sets as data,
> the artifact-level "loads cleanly under `hardened`" claim (D-023
> pattern generalized; advisory export stamp, receiver re-checks) —
> and the **`harden` op** — posture application as a deterministic
> grade-1 transform (minimal re-placement/re-binding + full deviation
> record; forward/signatures/learned-state/metric untouched; detached
> scores re-warranted by the baked metric). dspy surface when
> ratified: `program.export(trust_profile="hardened")` preflight and
> `dspy.load(..., posture=...)` / `dspy.harden(program, profile)`.

> 2026-08-10 evening: the story grew a second half — `spec/trust.md`
> in the contract repo (PROPOSED): trust record {origin, authority,
> exposure}, the pairing rule (trust deficit paid with isolation;
> authored code never executes at link), receiver postures
> (`workbench` = full-trust local data-science mode, zero ceremony,
> non-transitive; `reviewed`; `hardened`), and the **integrity dual**
> (prompt injection as the same taint walk with sources/sinks
> reversed; sanitizers = dual of declassifiers). PoT/CodeAct/RLM are
> representable as safe: a sandboxed (`isolation: "sandbox"`)
> zero-`grants` interpreter is not an integrity sink; its output stays
> tainted; granting a powerful tool re-arms it. Prototype: 15/15
> selfcheck vectors green. Custom parsers/strategies/LMs become
> supportable under the pairing rule instead of refused.

**Status: design note (2026-08-10), companion to the PROPOSED contract
section `programir-contract/spec/flow.md` (working prototype:
`reference/flow.py` + `selfcheck_flow.py`, 7 vectors green).**

Odersky's agent-harness argument (capabilities + effects; the
classified-contracts summary story) maps onto the ProgramIR because the
artifact is already a closed object: every call is a declared leaf,
every exfiltration capability is a declared placement/effect/credential
slot, and the forward grammar is closed. "No path from this classified
input to anything that can reach the outside world" is therefore a
static grade-1 walk — the contract prototype demonstrates it end to
end, including the two-LLM topology (hosted orchestrator + rung-0
in-process summarizer) and the environment-held declassifier grant.

The dspy-side surface, when the contract section ratifies:

- **Authoring spelling:** `Annotated[T, classified]` — a field *marker*
  in the D-011 marker system, NOT a `semantic_role` (it does not change
  how the exchange is conducted; it labels the data). Sugar spellings
  (`classified[Document]`, `@classified`) follow the four-spellings
  rule. Component-2 storage is an additive schema change riding the
  exporter epic, like D-036's additions.
- **Export-time preflight:** the exporter runs `flow_check` with the
  author's markers and warns/refuses per a strictness flag — the same
  preflight pattern as the declared-tier profile check.
- **Receiver-side check:** a receiver can run `flow_check` with *their
  own* policy over any received artifact before binding endpoints —
  the capability (declassifier grant, endpoint binding) is the
  receiver's act, never the artifact's. This is the part permissions
  can't do and capabilities can (Odersky's distinction), expressed as
  data.
- **Honest limit:** explicit-flow taint + the enumerable branch-shaped
  implicit channels (branch-assign laundering, the exception channel,
  the break channel — all closed in the prototype, 9 selfcheck vectors
  green as of 2026-08-10); sink-selection signaling stays a coarse
  `control` flag; timing/covert channels are out of scope. A lint with
  teeth over the whole orchestration layer, not Scala's
  capture-checked proofs; the leaf boundary stays declare-and-probe
  (`effects: pure`).

**Ratification asks (positions drafted 2026-08-10 in
`programir-contract/spec/flow.md`, awaiting Maxime):**

1. **Lattice, not a single marker:** `secret < internal < public`,
   policy maps labels to refused sink classes; graded declassifier
   grants. Single-bit is unfixable later without a breaking schema
   change; the lattice costs one enum now. dspy spelling follows the
   four-spellings rule: `Annotated[T, secret]` etc.
2. **Export-time default: refuse.** Labeled field + violating flow =
   export refusal naming source field → leaf → entry. Explicit
   opt-down `export(..., flow="warn")` is stamped into provenance so
   receivers can see it. Receiver-side `flow_check` always runs the
   receiver's own policy regardless.
