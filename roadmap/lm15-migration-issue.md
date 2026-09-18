# LM Changes from 3.3.0 to 3.4.0

If your code looks like this, **nothing changes for you**:

```python
import dspy

lm = dspy.LM("openai/gpt-4.1-mini")
dspy.configure(lm=lm)

qa = dspy.Predict("question -> answer")
print(qa(question="Why is the sky blue?").answer)
```

## What changes

### 1. The experimental typed API from 3.3 is replaced

In 3.3 we shipped an *experimental* typed API: `dspy.LMRequest`, `dspy.LMResponse`, `dspy.System(...)`, `dspy.User(...)`, and friends. It was clearly gated behind `experimental=True`. These names aare removed in 3.4. lm15's types take their place, used directly:

| 3.3 experimental (removed) | 3.4 (lm15) |
|---|---|
| `dspy.LMRequest(...)` | `lm15.Request(...)` |
| `dspy.LMResponse` | `lm15.Response` |
| `dspy.User("hi")` | `lm15.Message.user("hi")` |
| `dspy.Assistant("...")` | `lm15.Message.assistant("...")` |
| `dspy.System("...")` | `lm15.Request(system="...", ...)` |
| `dspy.LMConfig(...)` | `lm15.Config(...)` |
| `dspy.LMRateLimitError` etc. | lm15's error types |

Why not keep the old names as aliases? Because then the same concept has
two names forever, in two projects.

### 2. litellm becomes optional

Today `pip install dspy` pulls in litellm and its whole dependency tree,
even if you only ever call OpenAI. In 3.4:

- The default path uses lm15 (zero extra dependencies). OpenAI,
  Anthropic, Gemini, and any OpenAI-compatible server (Ollama, vLLM,
  LM Studio, Groq, ...) work out of the box.
- litellm remains fully supported as an opt-in extra:
  `pip install dspy[litellm]`, for the long tail of providers lm15 does
  not cover. If you select a litellm-backed model without the extra
  installed, you get a clear error telling you exactly what to install.

Your install gets smaller and `import dspy` gets faster. We will publish
before/after numbers (install time, import time, time to first call) with
the release.

### 3. `n` (asking for several answers at once) finally works everywhere

Here is a concrete example of why we want a fixture-tested foundation.

DSPy lets you ask for several sampled answers in one call — `n=5` — and
optimizers like COPRO and GEPA use this. We tested what *actually*
happens today when you send `n=3` to real providers (2026-08-31):

| Provider | Native `n=3` today |
|---|---|
| OpenAI Responses API | ❌ Rejected: `Unknown parameter: 'n'` |
| Anthropic | ❌ Rejected — DSPy raises an error before even calling |
| Groq | ❌ Rejected: `'n' : number must be at most 1` |
| Gemini | ✅ Works |
| Local vLLM | ✅ Works |

So `n` is already broken on three major paths in released DSPy — and one
of those (the Responses API) is OpenAI's own newest API, which removed
the parameter entirely. The industry is walking away from native `n`.

In 3.4, DSPy implements `n` itself: it sends the request `n` times in
parallel and collects the answers. One simple mechanism, identical on
every provider.

**The honest trade-off:** on the providers where native `n` still worked,
the prompt was billed once for all answers; with fan-out it is billed
once per answer. Provider prompt caching softens this a lot. In exchange,
`n` stops crashing on Anthropic, Groq, and the Responses API — meaning
for most users, `n` goes from broken to working.

### 4. Smaller changes to know about

- **Cache reset (one-time).** DSPy caches identical requests on disk to
  save you money. The cache key format changes, so old cached entries
  will not be found once. Nothing breaks; previously-cached calls are
  paid for again the first time.
- **Error types.** Errors raised by providers now come from lm15's typed
  error family (rate limit, context too long, auth, etc.). If you wrote
  `except dspy.LMRateLimitError` (a 3.3-experimental name), you update
  the import.
- **Model details from the cloud.** Model capabilities, prices, and
  context windows can be refreshed from lm15's registry (with an
  off-switch and on-disk defaults if you use older models or run
  offline).
- **Enterprise auth examples.** Runnable examples for Azure, AWS Bedrock,
  and GCP Vertex, built on lm15's auth helpers. Where a path genuinely
  still needs litellm, we will fallback to litellm and warn to install it if not installed.

## What we get in exchange

- **Fewer dependencies, faster startup.** lm15 needs nothing but Python.
- **A contract that is *proven*, not assumed.** lm15's request building
  and response parsing are checked byte-for-byte against recorded real
  provider traffic. When a provider changes something, the fixtures catch it.
- **One mental model.** A request is data. A response is data. You can
  print them, save them, test them — the same way on every provider.

## What could go wrong (and our answers)

- *"lm15 is new. What if it has bugs?"* It does — all software does. But
  it is tested against recorded real provider behavior, and DSPy pins an exact version range,
  so an lm15 release can never silently change DSPy under you.
- *"Is this a new supply-chain risk?"* It is one new dependency with
  zero transitive dependencies, replacing a much larger tree on the
  default path. Net risk goes down.
- *"I need a provider lm15 doesn't support."* `pip install dspy[litellm]`
  and everything works as before.
- *"Streaming?"* Streaming keeps working. If the lm15-backed streaming
  path is not fully clean by 3.4.0b1, the beta will say so explicitly and
  streaming users can stay on the litellm path for that release.

## Questions for the community

1. Do you depend on `n > 1` today? On which provider? Does the fan-out
   billing trade-off concern you?
2. Any objection to the one-time cache reset?

## Rollout

1. lm15 publishes a pinned beta to PyPI (it is stdlib-only; DSPy pins a
   strict version range and vendors its test fixtures).
2. DSPy 3.4.0b1: the swap lands behind the stable public surface; full CI
   across Python 3.10–3.14 plus real-model tests; benchmark numbers
   published.
3. 3.4.0: migration guide in the versioned docs.

