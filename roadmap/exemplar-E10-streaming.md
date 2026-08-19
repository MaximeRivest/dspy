# E-10 — streaming: role-typed deltas, per-face grammars, one parsing contract

**Proves (gap G10):** the stream as data — role-tagged deltas routed by
the baked plan; declared vs undeclared reasoning mid-flight; per-face
stream grammars (STUDY F2); buffered replay as the degenerate case.

## 1. The law, restated as routing bytes

The engine knows from the AdapterPlan which roles are in flight, hence
which delta types to expect and where each lands. For the E-03
signature (`question -> reasoning @reasoning, answer`) on a
native-reasoning model:

```jsonc
// the derived stream-routing table (computed at bake from plan + strategies;
// serialized in explain output, not authored)
"stream_plan": {
  "face": "lm15/chat-1", "grammar": "lm15/stream-1",     // start | delta* | end | error
  "routes": [
    {"delta": "ThinkingDelta", "role": "reasoning", "declared": true,
     "to": "field:reasoning"},                            // partial value, streams to consumer
    {"delta": "TextDelta", "to": "section_parser:chat"},  // the format's marker-splitter
    {"delta": "ToolCallDelta", "to": "strategy:tools"},   // the strategy's parser owns the fold
    {"delta": "ContinuationDelta", "to": "history:carry"}]}
```

Same program, reasoning UNDECLARED in the signature: the ThinkingDelta
route flips to `"to": "trajectory"` — exhaust is exhaust even
mid-flight (§d-sacred, live). One field in the signature is the entire
difference between API surface and observability.

## 2. Per-face grammars (STUDY F2 as vocabulary)

| face | grammar id | delta kinds |
|---|---|---|
| `lm15/chat-1` | `lm15/stream-1` | Text, Thinking, ToolCall, Citation, Continuation |
| `transcribe-1` | `transcribe/stream-1` | PartialTranscript (revisable), FinalSegment |
| `generate-image-1` | `genimg/stream-1` | ProgressFrame, FinalImage |
| `embed-1` · `predict-1` | none | non-streaming faces: request/response only |

A face version PINS its grammar; an engine streaming deltas outside the
face's kind set is nonconformant. `transcribe`'s `PartialTranscript` is
REVISABLE (later deltas may rewrite earlier text) — a semantics chat
does not have, which is exactly why grammars are per-face, not shared.

## 3. Buffered replay is the degenerate case (one parsing contract)

`fold(events) == response` — the lm15 Result assembler — then parse as
unstreamed. Grade-2 conformance: for every streaming fixture, the
folded stream and the buffered call parse to IDENTICAL predictions, and
history records once at completion. Engines that cannot stream natively
replay the finished response as events (the greenfield LM already does
this); one vocabulary, no backend branching.

## 4. Refusal hooks (cross-ref E-11)

**R-46 · delta outside the face grammar** `[E-10]`
> `stream (face embed-1): received delta kind 'TextDelta' — embed-1 declares no stream grammar; a streaming embed backend is nonconformant, buffer it`

**R-47 · declared-field delta after end** `[E-10; lm15 grammar]`
> `stream: ThinkingDelta received after 'end' event — grammar lm15/stream-1 is start delta* end|error; late deltas are a protocol violation, not extra data`

## 5. Per-field streaming is DERIVED, with two pinned rules

The lens gate does double duty: a template the lens accepts is a
template an online segmenter can split — anchors and field order are
static bake-time facts. Per output field, the plan carries a derived
`field_stream: native | textual | none` (native channel; anchored
textual span, codec decodes at field close; aggregate/non-anchored —
one more reason JSON lives INSIDE fields). Two rules pinned so engines
cannot drift:

- **Holdback:** an engine MUST NOT emit segment text that is a prefix
  of any anchor; holdback buffer = longest-anchor length. When a delta
  arrives is conformance surface, not vibes.
- **Reorder:** segmentation follows anchors ACTUALLY SEEN, never the
  expected order; the fold reconciles (folded ≡ buffered stays the
  oracle).

## 6. The trajectory stream — program-level progress as a typed grammar

`_trajectory` was always live (§d-sacred: "exhaust is exhaust even
mid-flight"); this gives it the grammar it never had. Tree-addressed
events, versioned `trajectory_stream: 0.1` (⟂ DRAFT):

```jsonc
run_start        {run_id, manifest_hash}
node_enter       {path}                          // "draft", "drafter.classify"
leaf_request     {path, leaf, rendered_ref?}     // View-2 cost channel, live
delta            {path, channel, payload}        // exhaust deltas: undeclared
                                                  //   reasoning, interleaved think
tool_observation {path, tool, obs_ref}
interpreter_event{path, kind}                    // exec start/ok/error (E-07)
branch           {path, node_id, taken}          // If/While decisions as they happen
except           {path, error_type}              // the Try paths, live
leaf_response    {path, usage, cost}
node_exit        {path}
run_end          {prediction_ref, usage_total}
```

Because the tree is baked, a UI renders the progress skeleton FROM THE
MANIFEST before the run starts (nodes, loop caps, branch points), then
lights it up — progress display is a pure function of artifact + event
stream. Two subscriptions, split by the intent/mechanism razor:
**value channel** (declared fields, per §5 — streams once the returning
leaf is known) and **progress channel** (this grammar, immediately).
That split DISSOLVES the which-leaf-streams question: no speculation,
no retractions; intermediate leaves are progress, never contract.

One vocabulary, three consumers: live UI, View-2 debugging overlays,
and the D-048 run-log store (the outer loop ingests these same events —
no second logging system exists).

Laws carried over, restated for the live case: the trajectory stream is
NEVER load-bearing (a program that reads its own progress channel has
mechanism re-entering the loop — refuse at compile: the channel has no
leaf); never baked; and it is a flow surface — field labels (§e3) apply
to exhaust payloads, and a posture may redact `delta.payload` while
keeping the skeleton events (progress without content — the hardened-
deployment UI).

## 7. Refusal hooks (cross-ref E-11)

**R-50 · program consuming its own trajectory** `[E-10 §6]`
> `forward: call to 'dspy.trajectory' at modules.py:18 — the observability channel is instrumentation-only, never load-bearing; no leaf exposes it to program logic`

**R-51 · trajectory event outside the grammar** `[E-10 §6]`
> `trajectory stream: event kind 'user_rating' — not in trajectory_stream 0.1; feedback enters through the run-log store's ingestion (D-048), not by extending the live grammar ad hoc`
