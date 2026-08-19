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
