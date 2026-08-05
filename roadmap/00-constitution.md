# The Constitution

The invariants of the ProgramIR work. Every PR review question reduces to "which law does this break?" — cite them by name. Full rationale lives in `IR-program-spec.md`; this page is the enforceable digest.

## L1 — The sacred signature
The signature is the user's declared intent, honored exactly: **prediction fields = declared output fields, no more, no less.** Everything else a strategy generates (reasoning, trajectories, REPL histories, extraction intermediates) is mechanism exhaust → the `_trajectory` observability channel. The channel is readable by instrumentation, never load-bearing for the contract, never serialized as program state. Want an output? Declare it.

## L2 — The single-shot law
One adapter entry (component 4) denotes **one LM exchange** — one `Request → Response` through the typed contract (`n>1` samples in one request are one exchange; a retry or fallback is a second). Every multi-call behavior is component-5 control flow: **authored, lowered, or refused**. There is no third place for an LM call to live. Nothing in the adapter layer — format, strategy, codec, parser — may call an LM.

## L3 — Optimizable ⊆ baked
Every field an optimizer may mutate is a field the artifact bakes. Anything optimizable-but-unbaked is a checkpoint that cannot reconstruct.

## L4 — Checkpoint = save
An optimizer checkpoint is a full `ProgramIR` snapshot; the save path and the checkpoint path are one code path.

## L5 — Loud refusal, never silent partial
Compile, link, and load either accept or refuse **naming the exact node, binding, or capability** — never a silent partial result, never a fallback to ambient state. A dangling binding is a link error. `settings.X or Default()` is the disease this law exists to kill.

## L6 — Declare, don't discover
Anything behavior depends on is declared data, verified at load: weight identities, tensor ties, interpreter profiles, capabilities, deps. Verification is the load step, not an optional check.

## L7 — Direction of ownership
User-supplied request passthrough and provider-returned response data are distinct named things (lm15's `extensions` vs `provider_data`). Echoing provider-returned data back as user-supplied is a bug.

## L8 — The corpus rule
Golden fixtures regenerate **only** in dedicated corpus commits containing zero `dspy/` source changes. Fixture regeneration inside a feature or refactor commit is an automatic failure — it means the change altered behavior. Fixtures for a planned refactor land *before* the refactor.

## L9 — Every representation decision carries its parser
Render and parse are one unit (the `AdapterPatch` discipline). A mutation to one side without the other is invalid; admissibility is gated by round-trip (`parse(render(x)) = x`) on adversarial probes before any LM spend.

## L10 — Interface-preserving lowerings
A lowering keeps the node's external signature exactly — no added inputs, no added outputs. Additions are mechanism (→ L1) or a different module that honestly declares its contract (the MCC rule).

## L11 — Provenance is non-negotiable
Machine-written or machine-mutated code carries `authored_by` and its mutation chain into the artifact. Strategy traces record selected/skipped with reasons. Deviations are recorded, never silent; scores attach to configurations, not artifacts.

## L12 — Objectives are ends, never searched
The external signature (task's type), the metric + devset (task's value), and these invariants (the physics) never carry an optimizable tag. The user picks a seed's regime at tag time; the optimizer never chooses its own objective.
