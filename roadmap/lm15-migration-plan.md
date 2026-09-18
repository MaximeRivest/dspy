# Plan: DSPy 3.4.0 on the lm15 contract

Owner: Max. Reviewer: Isaac. Release: 3.4.0.

Ratified premises:

- The 3.3 typed LM contract (`dspy/core/types.py` and everything downstream
  of it) was experimental. It can be removed with **no backward-compat
  obligations**.
- DSPy is fully committed to the lm15 contract. Use it in the simplest,
  most maintainable way: lm15 types **are** the DSPy LM types. No facade,
  no wrappers, no dual naming.
- The *legacy* LM surface (`lm(...) -> list[str | dict]`, OpenAI-shaped
  message dicts, `n=`, `inspect_history`) predates 3.3 and stays stable.
  Only the typed contract is being replaced.

This plan also carries the three linked Max rows: LiteLLM opt-in /
OpenAI-compatible default, model detail hydration, and enterprise auth
preparation — they all land on lm15 machinery.

---

## 1. Facts: upstream `main` today (3f06959eb)

- `dspy/core/types.py`: 2,063-line pydantic contract (`LMRequest`,
  `LMResponse`, parts, config, streams, history, message constructors).
  Shipped in 3.3.0 behind the experimental gate: typed responses only for
  explicit `LMRequest` calls, `experimental=True`, or
  `forward_contract = "typed_lm"`. Default calls still return the legacy
  list.
- `dspy/clients/openai_format.py`: a second in-repo contract
  implementation — the wire mapping (LM types ↔ OpenAI Chat / Responses /
  text).
- Consumers: `dspy/adapters/base.py`, `dspy/clients/base_lm.py`,
  `dspy/clients/lm.py`, `dspy/adapters/_legacy_type_markers.py`,
  ten re-exports in `dspy/__init__.py`, LM error family in
  `dspy/utils/exceptions.py`.
- `litellm` is a required dependency, with an extra already defined and a
  forced pin on `win32` / Python 3.14.
- Tests touching the contract: `tests/core/test_types.py`,
  `tests/clients/test_lm.py`, `tests/clients/test_disk_serialization.py`,
  `tests/clients/test_lm_direct_live.py`, `tests/adapters/test_tool.py`,
  `tests/predict/test_rlm.py`.

## 2. Facts: lm15

- `lm15-python` 1.0.0a1, stdlib-only, zero dependencies, frozen dataclasses,
  strict validation, canonical serde, typed errors, providers
  (OpenAI Chat + Responses, Anthropic, Gemini, compatible servers, async
  variants), router, auth/authkit, model registry, conformance harness in
  `lm15-contract`. Not yet on PyPI.
- Generation core is fixture-proven. Live/batch incomplete — not needed for
  DSPy 3.4.
- `Response` holds exactly one assistant `Message`. `Config` has no `n`.
  This is deliberate contract design, not a gap.

## 3. Architecture: lm15 types are the types

```
dspy user code (legacy call surface, unchanged)
      │
      ▼
dspy/clients/lm.py        one boundary module converts legacy inputs
  (prompt/messages/kwargs ──► lm15.Request)
      │
      ▼
lm15 providers            wire serde, transport, streaming, errors
      │
      ▼
lm15.Response ──► adapters parse ──► Prediction
```

- **Delete** `dspy/core/` entirely. **Delete** `openai_format.py` entirely.
- Internal code imports lm15 directly: `from lm15 import Request, Message`.
- `BaseLM` typed contract becomes
  `forward(request: lm15.Request) -> lm15.Response`. The
  `forward_contract = "legacy" | "typed_lm"` switch survives — it protects
  the *stable* legacy surface, which is not ours to break.
- Users who want typed calls write lm15 idioms:
  `Message.user(...)`, `Request(system=..., messages=...)`. DSPy documents
  this; it does not rename it.

### Re-export policy (expert check applied)

The cheap move is `dspy.LMRequest = lm15.Request`. An API designer rejects
it: two names for one concept, forever, in two projects — and lm15's first
principle is one name per concept. A 3.3-experimental user reads one
migration note; every future user reads the aliases. So:

- `dspy/__init__.py` drops `LMRequest`, `LMResponse`, `LMConfig`,
  `LMMessage`, `System`, `Developer`, `User`, `Assistant`, `ToolCall`,
  `ToolResult`.
- No `dspy.lm15` namespace mirror. Users `import lm15`. One import line is
  not friction worth a permanent alias surface.
- Trade-off stated: 3.3 typed-path users get a hard break with an exact
  migration table (old name → lm15 idiom) in the release notes. We accept
  this because the premise says we may, and aliases would be a permanent tax
  to avoid a one-time, allowed break.

### Where 3.3-contract behavior goes

| 3.3 surface | Destination |
|---|---|
| `LMRequest`, `LMResponse`, parts, `LMConfig` | lm15 types, used directly |
| `System()` message constructor | `lm15.Request.system` field (system is request-level in lm15; the constructor encouraged a message-role model lm15 rejects) |
| `User/Assistant/Developer/ToolResult` constructors | `lm15.Message.user/assistant/developer/tool` |
| OpenAI-dict leniency (`from_call`, `normalize_parts`) | One dspy boundary module for the **legacy** call surface only. Strictness lives in lm15; leniency exists solely where stability demands it. |
| `openai_format.py` wire mapping | lm15 providers + serde (fixture-proven) |
| dspy `LMError` family | Deleted. lm15 errors propagate directly; `dspy.utils.exceptions` keeps only dspy-level errors (`AdapterParseError`, ...). Optimizer/retry logic switches to lm15's taxonomy (`ContextLengthError`, retryable codes). Trade-off: 3.3 `except dspy.LMRateLimitError` breaks — allowed, documented. |
| `LMHistoryEntry`, `inspect_history` | Stays in dspy (it is UX, not contract), rebuilt on `lm15.Request`/`Response` + lm15 canonical serde |
| `LMStream` / stream events | lm15 stream events on the lm15 path; see D3 |

## 4. Decisions

- **D1 — `n` (multiple candidates). RATIFIED 2026-08-31: `n` lives
  entirely in DSPy, outside the lm15 contract.** Decided by the contract
  *and by live evidence* (see appendix A): native `n` is already broken on most
  paths DSPy ships today. `lm15.Response` cannot represent multiple
  candidates, and the provider landscape has moved away from native `n` —
  OpenAI's Responses API rejects the parameter outright. DSPy implements
  `n` as client-side fan-out (async, shared prefix) in the boundary module.
  One code path, uniform across providers. This is a repair, not a
  regression: today `n > 1` hard-errors on Anthropic, 400s on Groq and on
  the Responses path, and only works on OpenAI Chat, Gemini, and
  vLLM-style servers. Trade-offs stated: where native `n` did work, it
  billed input tokens once — fan-out bills them per sample, mitigated by
  provider prompt caching (OpenAI, Anthropic, Gemini) and vLLM prefix
  caching; independent samples lose nothing statistically for DSPy's use
  (temperature sampling). `n` stays a stable public kwarg; `predict`,
  COPRO, and GEPA keep working unchanged — on every provider, for the
  first time.
- **D2 — lm15 release engineering.** lm15 publishes to PyPI
  (`>=1.0.0b1,<1.1` pin) before dspy 3.4.0b1. dspy CI vendors the small
  lm15-contract JSON fixtures it checks against, so dspy tests stay offline.
  No code vendoring: lm15 is the contract's reference implementation and has
  zero transitive deps. Supply-chain note for reviewers: stdlib-only, MIT,
  pinned range — this *reduces* the dependency tree versus litellm.
- **D3 — streaming.** lm15 stream events drive `streamify` on the
  lm15-provider path; the litellm opt-in path keeps its current chunk
  handling. If event mapping is not clean by 3.4.0b1, the lm15 path ships
  non-streaming-first and the beta says so. No half-typed stream API.
- **D4 — default LM path.** `dspy.LM` routes through lm15 providers:
  `OpenAIChatLM` for OpenAI-compatible endpoints (the default),
  `OpenAILM` for Responses, Anthropic/Gemini native. litellm becomes the
  existing `dspy[litellm]` extra, lazy-imported, selected explicitly.
  Remove the `win32`/3.14 forced litellm pin after the CI matrix passes on
  lm15 (stdlib-only, so it should).
- **D5 — pydantic at the LM boundary.** DSPy keeps pydantic everywhere it
  already uses it (signatures, adapters). The LM boundary itself carries
  frozen dataclasses. Adapters receive `lm15.Response` and parse
  `response.text` / parts as plain values. No pydantic wrapper around lm15
  objects — wrapping a validated immutable value in a second validator is
  pure overhead.

## 5. Phases

### Phase 0 — Unblock
- Confirm D1–D5 with Isaac and the lm15 project. D1 needs lm15 to confirm
  `n`-fan-out is the intended pattern (it follows from contract design).
- Publish lm15 to PyPI (beta), tags + changelog.

### Phase 1 — The cut
- Add `lm15` dependency. Delete `dspy/core/`, delete
  `dspy/clients/openai_format.py`, drop the ten `dspy/__init__.py`
  re-exports and the LM error family.
- Write the one boundary module (`dspy/clients/lm_boundary.py`, name TBD):
  legacy inputs → `lm15.Request`; `lm15.Response` → legacy outputs list;
  `n` fan-out; lm15 errors pass through.
- Rewire `adapters/base.py`, `base_lm.py`, `lm.py`,
  `_legacy_type_markers.py` to lm15 types.
- Port tests: `tests/core/test_types.py` shrinks to boundary-module tests
  (lm15 already tests its own types — do not duplicate its suite).

### Phase 2 — Default path and litellm opt-in
- Default `dspy.LM` on lm15 providers per D4; litellm behind the extra with
  a clear error message when absent.
- Cache keys move to lm15 canonical request JSON. One-time disk-cache
  invalidation, stated in release notes.
- Usage/cost from `lm15.Usage` + model registry pricing.

### Phase 3 — Hydration and auth rows
- Model detail hydration via lm15 model registry: capabilities, prices,
  context windows; refresh command; off-switch (`hydrate=False` / env var)
  with on-disk cached defaults for older-model users.
- Enterprise preparation: runnable Azure / AWS Bedrock / GCP Vertex examples
  on lm15 auth + compat paths; where a path genuinely needs litellm, the
  docs say so instead of pretending.

### Phase 4 — Proof
- Conformance cross-check in dspy CI: boundary-built requests serialized via
  lm15 must match vendored lm15-contract fixtures byte-for-byte.
- Full battery via `dspy-ci` (never local): Pythons 3.10–3.14 + real-LLM
  stage on local vLLM. Add fan-out `n` tests there.
- Fresh-venv benchmark, litellm absent: install time, import time, first
  call. Publish numbers as release evidence.

### Phase 5 — Docs and release
- Migration table: every removed 3.3 name → its lm15 idiom; error-class
  mapping; `n` semantics; cache reset; litellm opt-in.
- Land the guide in the versioned 3.4 docs (coordinate with the Zensical
  migration row).

## 6. Risks

- lm15 pre-release churn → pinned range + vendored fixtures + CI gate.
- Hidden dependents on the removed error classes inside dspy itself
  (retry logic, optimizers) → grep-driven sweep in Phase 1, tests per error
  code path.
- Fan-out `n` cost surprise on OpenAI Chat → docs state the economics;
  release notes note that `n` previously errored on Anthropic, Groq, and
  the Responses path, so most users never had native `n` anyway.
- Streaming mapping slips → D3 fallback is explicit, decided at 3.4.0b1.
- Two-repo release coupling → lm15 publishes first; dspy beta pins it.

## 7. Order

```
Phase 0 ──► lm15 on PyPI ──► Phase 1 cut ──► Phase 2 default path
                                                  │
                                   Phase 3 (hydration, auth) ── parallel
                                                  │
                                   Phase 4 proof ──► Phase 5 docs/release
```

## Appendix A — Live evidence: does native `n` still work? (2026-08-31)

Probes: direct HTTP calls with tiny budgets (`/tmp/n-probe/probe.sh`),
plus offline checks through DSPy's installed litellm (1.68.0) and DSPy's
own typed mapper on upstream `main`.

| Path | `n=3` result | Evidence |
|---|---|---|
| OpenAI Chat Completions (gpt-4.1-nano) | Supported per docs; live blocked by billing on our key | Reference docs: "How many chat completion choices... Keep `n` as `1` to minimize costs"; litellm forwards `n=3` |
| OpenAI Chat Completions (gpt-5-mini, reasoning) | **Unverified** (billing); litellm forwards `n=3` blindly | Reasoning models historically reject `n>1`; must re-test before relying on it |
| OpenAI Responses API | **Rejected**: `Unknown parameter: 'n'.` | Live probe. DSPy's own mapper emits `"n": 3` in the Responses payload → guaranteed 400 on upstream main today |
| Anthropic Messages | **Rejected** twice over | Live: `n: Extra inputs are not permitted`. Through DSPy: `LMUnsupportedFeatureError` raised by litellm before any network call |
| Gemini generateContent | **Works**: `candidateCount=3` → 3 candidates | Live probe; litellm maps `n` → `candidate_count` |
| Groq (OpenAI-compatible) | **Rejected**: `'n' : number must be at most 1` | Live probe; litellm wrongly claims `n` supported and forwards it → live 400 through DSPy |
| vLLM (local, OpenAI-compatible) | **Works**: 3 choices | Live probe against 192.168.2.24:8000 |
| OpenRouter | Untested — key rejected (`User not found`) | Probe blocked; re-test if OpenRouter matters for 3.4 |

Conclusion: native `n` survives only on OpenAI Chat Completions, Gemini,
and vLLM-style local servers. OpenAI's strategic API (Responses) dropped
it. Uniform client-side fan-out makes `n` work everywhere and removes two
live-400 paths and one hard error that exist in DSPy today.
