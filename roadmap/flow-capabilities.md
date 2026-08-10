# Classified flow — the dspy side of the capability story

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
- **Honest limit:** explicit-flow taint + coarse control-flow flags —
  a lint with teeth over the whole orchestration layer, not Scala's
  capture-checked proofs; the leaf boundary stays declare-and-probe
  (`effects: pure`).

Decision needed from Maxime before any of this ratifies: whether
`classified` is one marker or a small lattice (e.g. `secret < internal
< public`), and whether export-time enforcement defaults to warn or
refuse.
