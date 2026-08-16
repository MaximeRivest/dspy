# The typed LM API: the landed contract

This page used to be the *migration plan* for DSPy's typed language-model
boundary. The migration has landed, and it landed simpler than the plan.
This page is now the contract: every behavior below is a promise, each
promise is enforced by `tests/lm/test_api_promises.py` (offline, always
run) and exercised against real providers by `tests/live/` (the live
provider matrix). The [table at the end](#what-changed-from-the-plan)
maps every name the old plan promised to what actually shipped.

One idea to hold on to: every call builds one canonical **request**, and
the model sends back one canonical **response**. The canonical types are
[lm15](https://pypi.org/project/lm15/)'s `Request` and `Response`; DSPy
does not wrap them in a second vocabulary.

## The call surface

`dspy.LM` has two call faces, split by how the input arrives.

**Typed (positional) — the default, no flag.** Strings, typed turns, and
previous responses in; the canonical `lm15.Response` out:

```python
lm = dspy.LM("openai:gpt-5-nano")
response = lm("hello")
response.text          # 'Hello! How can I help you today?'
response.usage         # Usage(input_tokens=..., output_tokens=..., ...)
response.tool_calls    # data when the model used tools, () otherwise
response.finish_reason # 'stop'
```

**Legacy (keyword) — the adapter convenience.** `prompt=` or
`messages=[{"role": ..., "content": ...}]` in; a list of strings out.
The result compares equal to a plain `list[str]`, and the canonical
response rides on `.response`:

```python
outputs = lm(prompt="hello")   # ['Hello! How can I help you today?']
outputs.response.usage         # the same canonical Response underneath
```

The rule of thumb: positional inputs → rich `Response`; keyword
`prompt=`/`messages=` → plain strings.

The old plan gated typed returns behind `dspy.context(experimental=True)`.
That gate is gone: the typed face is the default, and there is no
`experimental` flag.

## Conversations are typed turns

Turn constructors are named after the speaker. A previous `Response`
drops in as its own turn:

```python
follow = lm(
    dspy.System("Answer in five words or less."),
    dspy.User("What is DSPy?"),
    first,                       # a previous lm15.Response, folded in
    dspy.User("Say it shorter."),
)
```

Tool exchanges replay with two more constructors:

```python
answer = lm(
    dspy.User("What is the weather in Paris?"),
    dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"city": "Paris"})),
    dspy.ToolResult('{"temperature": "22 C"}', call_id="call_1", name="get_weather"),
    dspy.User("Summarize the result."),
)
```

Multimodal input rides the same face — `dspy.Image` values become image
parts:

```python
response = lm(dspy.User("Describe this image.", dspy.Image("https://example.com/dog.png")))
```

## The full-surface escape hatch: `lm.complete`

When the convenience faces are too small — tool declarations, custom
parts, provider extensions — build the canonical request yourself:

```python
from lm15 import FunctionTool, Message, Request, TextPart

response = lm.complete(Request(
    model=lm.model,
    messages=(Message(role="user", parts=(TextPart(text="Weather in Paris?"),)),),
    tools=(FunctionTool(name="get_weather", description="...", parameters={...}),),
))
response.tool_calls[0].name    # 'get_weather'
```

`complete()` routes through the same engine as the typed face: an
`api_base`/`api_key` endpoint pin on the LM is honored, and a request
naming the LM's model is rewritten to the pinned wire model, so the
provider prefix never leaks onto the wire.

## Streaming: a separate method, one event vocabulary

`lm.stream(...)` accepts every input `lm(...)` accepts and returns an
`lm15.ResponseStream`. It is a method, never a `stream=True` flag that
forks the return type:

```python
stream = lm.stream("Write a haiku about rivers.")
for text in stream:            # text pieces as the model writes
    print(text, end="")
stream.response.usage          # the finished canonical Response
```

Under the text is the typed event stream, one `.events()` away. Every
stream reads **start → deltas → end**, on every provider — dialects
without a native start frame get one synthesized (lm15 mapping rule
MAP-4), so consumers never branch on the backend:

```python
for event in lm.stream("hi").events():
    ...   # StreamStartEvent, StreamDeltaEvent(TextDelta...), StreamEndEvent
```

Streamed and buffered calls are observationally identical afterward:
history and usage are recorded once, when the stream completes, in the
same shape as a buffered call. A stream that fails records nothing.

Async streaming (`astream`) is not part of the landed surface yet; see
the table at the end.

## Model strings and routing

The model string is lm15-native: `provider:model`, or a bare family
name resolved by built-in rules:

```python
dspy.LM("openai:gpt-5-nano")           # explicit provider prefix
dspy.LM("claude-haiku-4-5")            # bare family, rule-resolved
dspy.LM("openai-codex:gpt-5.6-luna")   # OAuth subscription provider
dspy.LM("hf:PleIAs/Baguettotron")      # in-process weights, lazy-loaded
dspy.LM("/path/to/model-dir")          # local directory with config.json
```

Routing (model string → provider → credential) is lm15's `LMRouter`.
One shared router serves the process; `router=` overrides it per-LM and
always wins. Any object with `complete(Request) -> Response` — and
optionally `stream(Request)` — can stand in as the engine. That
structural seam **is** the custom-LM contract; there is no base class
to subclass and no `forward()` to implement.

**Endpoint pinning.** `api_base` moves the endpoint without changing
the dialect; `api_key` rides along. This is the self-hosted convention
(vLLM, SGLang, ollama, gateways):

```python
lm = dspy.LM(
    "openai-chat:gemma/gemma-4-26b-a4b-it",
    api_base="http://192.168.2.24:8000/v1",
    api_key="inktype-local",
)
```

An unresolvable bare model plus `api_base` defaults to the
OpenAI-compatible chat dialect.

## Credentials

- API-key providers pick up their usual environment variables
  (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, ...).
- OAuth subscription providers (`openai-codex:`, `claude-code:`) read
  the local CLI credential files; no API key exists for them.
- A per-LM `api_key` (endpoint pinning, above) always beats ambient
  environment keys.
- Program export (`dspy.programir`) scans live credential values —
  environment keys and per-LM `api_key` kwargs — and the writer refuses
  to serialize an artifact that would leak one.

## Usage and cost: always on, always honest

Every `Response` carries `usage`. Every call — typed, legacy, or
streamed — appends one history entry:

```python
lm.history[-1]   # {"model", "messages", "kwargs", "outputs", "usage", "cost", "response", "timestamp"}
```

`cost` is estimated from the model catalog's per-token prices.
`None` is an honest answer, not a gap: subscription billing
(`openai-codex:`, `claude-code:`) and local serving have no per-token
price. Predictions aggregate through `prediction.get_lm_usage()` and
`prediction.get_lm_cost()` with no flag to set.

## Errors: one typed error, canonical codes

Every routing, transport, or provider failure surfaces as
`dspy.core.errors.LMError` — never a provider-specific exception. The
message names the model and lm15's canonical error code (`auth`,
`billing`, `rate_limit`, `invalid_request`, `context_length`,
`timeout`, `server`, `unsupported_model`, `unsupported_feature`,
`not_configured`, `transport`, `provider`), and the lm15 exception
rides as `__cause__` for callers that need structured fields:

```text
LMError: LM request to 'openai:gpt-5-nano' failed (billing): You have no credits remaining. ...
```

The old plan's named hierarchy (`LMAuthError`, `LMBillingError`, ...)
collapsed into this single type plus the code table.

## Testing your program: the same seam, scripted

`dspy.DummyLM` is an ordinary `dspy.LM` bound to a scripted engine at
the same `complete(Request) -> Response` seam every real backend uses —
the two call faces, history, streaming, and error wrapping are the
production code paths:

```python
lm = dspy.DummyLM(["Paris", "Berlin"])
lm(prompt="Capital of France?")     # ['Paris']
lm.calls[0]["messages"]             # recorded for assertions
```

For wire-shape assertions, script full responses with
`dspy.LM("fake", router=lm15.testing.FakeLM([...]))`.

## What changed from the plan

| The old plan promised | What shipped |
| --- | --- |
| `dspy.LMRequest` / `dspy.LMResponse` wrapper types | lm15's `Request` / `Response` used directly — no second vocabulary |
| Typed returns behind `dspy.context(experimental=True)` | Typed face is the default; the flag does not exist |
| `dspy.BaseLM` subclassing with `forward(request)` | The structural engine seam: any `complete(Request) -> Response` object via `router=` |
| `forward_contract = "legacy" \| "typed_lm"` declarations | Gone — there is no `forward()` to declare a contract for |
| `LMResponse.from_text` for minimal custom LMs | `dspy.DummyLM` / `lm15.testing.FakeLM` |
| `dspy.LMStream` / `.result()` | `lm15.ResponseStream` / `.response` |
| `lm.astream(...)` | Not landed yet — open item |
| `LMAuthError`, `LMBillingError`, ... hierarchy | One `LMError` carrying lm15's canonical code; cause chained |
| `_OpenAICompatLM` reference engine | lm15 provider engines (`OpenAIChatLM` and family), outside DSPy |
| Credential ladder inside each typed LM | lm15 router resolution + per-LM `api_key`; export-time leak refusal |
| `dspy.streamify` / `StreamListener` bridge | Not landed in the greenfield tree — open item |
| LiteLLM as routing fallback | LiteLLM is gone; lm15 routes everything |

The old `dspy.core.types` `LM*` vocabulary (`LMRequest`, `LMResponse`,
`LMTextPart`, ...) still imports for compatibility, but it is not the
LM path's vocabulary and new code should not build on it.
