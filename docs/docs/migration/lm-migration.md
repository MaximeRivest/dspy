# Migrating custom LMs to `dspy.LanguageModel`

DSPy has a new normalized language model contract: custom LMs receive one
`dspy.LMRequest` and return one `dspy.LMResponse`.

There are two changes to know about:

1. **Direct LM calls are richer.** They accept strings, message constructors,
   OpenAI-style messages, media, tools, tool results, previous `LMResponse`
   objects, and explicit `LMRequest` objects.
2. **Custom LM authors get a typed contract.** New custom LMs should subclass
   `dspy.LanguageModel`, not `dspy.BaseLM`.

The normalized path is opt-in today. Existing programs keep using the legacy
LiteLLM-backed `dspy.LM` unless you enable it before constructing the LM.

```python
import dspy

dspy.configure(experimental_lm=True)

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

response = lm("hello")
print(response.text)
```

`experimental_lm` is read when `dspy.LM(...)` is constructed. Set it first,
then create the LM. You can also use `dspy.context(experimental_lm=True)` when
you want to construct a normalized LM for one block without changing the
process-wide default.

## What changes?

The legacy custom LM contract is provider-shaped:

```text
forward(prompt=None, messages=None, **kwargs) -> OpenAI/LiteLLM-shaped response
```

The normalized contract is DSPy-shaped:

```text
forward(request: LMRequest) -> LMResponse
```

A minimal custom LM looks like this:

```python
import dspy


class EchoLM(dspy.LanguageModel):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("hello", model=request.model)


dspy.configure(experimental_lm=True)
dspy.configure(lm=EchoLM(model="test/echo"))

predict = dspy.Predict("question -> answer")
result = predict(question="Say hello")
```

This text-only LM supports text requests because `forward()` is implemented. It
does **not** support images, audio, files, tools, tool results, structured
outputs, reasoning controls, prompt caching, logprobs, or multiple outputs until
you implement the relevant mapping hooks described below.

## Try the normalized `dspy.LM` path

With `experimental_lm=True`, `dspy.LM(...)` returns an `LMRouter`. The router is
itself a `LanguageModel`, but delegates provider behavior to a backend such as
`LiteLLMChatLM`, `LiteLLMTextLM`, or `LiteLLMResponsesLM`.

```python
import dspy

dspy.configure(experimental_lm=True)

chat_lm = dspy.LM("openai/gpt-4o-mini")
text_lm = dspy.LM("openai/davinci", model_type="text")
responses_lm = dspy.LM("openai/gpt-4o-mini", model_type="responses")
```

You can also construct normalized backends directly:

```python
lm = dspy.LiteLLMChatLM("openai/gpt-4o-mini", cache=False)
```

To compare with the legacy path, opt out before construction:

```python
dspy.configure(experimental_lm=False)
lm = dspy.LM("openai/gpt-4o-mini")
```

## Direct calls

The normalized path accepts several public input shapes and immediately turns
them into one `LMRequest`.

```python
lm("hello")
lm(prompt="hello")
lm(messages=[{"role": "user", "content": "hello"}])

lm(
    dspy.System("Be terse."),
    dspy.User("What is DSPy?"),
    dspy.Assistant("DSPy is a framework for programming LM pipelines."),
    dspy.User("Say that in five words."),
)

lm("describe this", dspy.Image("https://example.com/dog.png"))
lm(dspy.Reasoning("Prior reasoning supplied by the caller."), dspy.User("Continue."))
lm(dspy.ToolResult(content='{"temperature": 22}', call_id="call_1", name="weather"))
lm(dspy.LMRequest(model="test/model", messages=[dspy.User("hello")]))
```

You can inspect the normalized request without calling the provider:

```python
request = lm.normalize_request(
    dspy.System("Be concise."),
    dspy.User("Explain DSPy in one sentence."),
    temperature=0.2,
)

request.messages
request.config.temperature
```

Do not mix an explicit `LMRequest` with direct-call inputs, and do not mix
`messages=` with positional prompt or media inputs. DSPy raises `ValueError` for
those ambiguous calls.

## `LMRequest`

A `LanguageModel` receives one `LMRequest`:

```python
request.model       # str
request.messages    # list[LMMessage]
request.tools       # list[LMToolSpec]
request.config      # LMConfig
request.metadata    # dict[str, Any]
```

Messages are role-attributed parts:

```python
for message in request.messages:
    print(message.role)
    print(message.parts)
```

Common request part types are:

- `LMTextPart`
- `LMImagePart`
- `LMAudioPart`
- `LMFilePart`
- `LMThinkingPart`
- `LMToolCallPart`
- `LMToolResultPart`
- `LMCitationPart`
- `LMRefusalPart`

Common config fields are:

```python
request.config.temperature
request.config.max_tokens
request.config.top_p
request.config.stop
request.config.n
request.config.logprobs
request.config.response_format
request.config.reasoning
request.config.tool_choice
request.config.cache
request.config.prompt_cache
request.config.extensions
```

Unknown generation kwargs are preserved in `request.config.extensions`, so a
provider backend can still use provider-specific options. Common secret and
endpoint fields such as `api_key`, `api_base`, and `base_url` are sanitized from
saved LM state and normalized cache keys.

### DSPy cache vs provider prompt cache

DSPy's memoization cache and provider prompt caching are separate.

```python
request = lm.normalize_request(
    "hello",
    cache=False,                 # DSPy memoization cache
    prompt_cache=True,            # provider-side prompt/token cache
    prompt_cache_key="prefix-1",
)

request.config.cache.enabled          # False
request.config.prompt_cache.enabled   # True
request.config.prompt_cache.key       # "prefix-1"
```

DSPy's cache can skip the provider call entirely. Provider prompt caching still
sends a provider request, but may reuse prompt prefixes or KV state.

## `LMResponse`

A `LanguageModel.forward()` returns one `LMResponse`:

```python
return dspy.LMResponse(
    model=request.model,
    outputs=[...],
    usage=dspy.LMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
    cost=0.00012,
    cache_hit=False,
)
```

For text-only responses, use `from_text()`:

```python
return dspy.LMResponse.from_text("Hello!", model=request.model)
```

`LMResponse` is list-like for compatibility:

```python
response[0]
list(response)
response.to_outputs()
response.to_legacy_outputs()
```

It also exposes normalized views:

```python
response.output
response.parts
response.text
response.reasoning_content
response.tool_calls
response.citations
response.images
response.audio
response.files
response.usage
response.cost
response.cache_hit
```

A response can contain richer candidate parts:

```python
return dspy.LMResponse(
    model=request.model,
    outputs=[
        dspy.LMOutput(
            parts=[
                dspy.LMThinkingPart(text="I should call the weather tool."),
                dspy.LMTextPart(text="I will check Paris now."),
                dspy.LMToolCallPart(
                    id="call_1",
                    name="get_current_weather",
                    args={"location": "Paris"},
                ),
            ],
            finish_reason="tool_calls",
        )
    ],
)
```

Usage and cost are first-class:

```python
usage = dspy.LMUsage(input_tokens=12, output_tokens=8, total_tokens=20)
return dspy.LMResponse.from_text("Hello!", model=request.model, usage=usage, cost=0.00012)
```

On a DSPy cache hit, the base class marks the returned response as a cache hit
and removes new accounting. In practice, cached responses report
`cache_hit=True`, no new cost, and empty usage accounting.

## Request support and mapping hooks

`LanguageModel` validates request shape before calling `forward()`. If the
request uses a feature your implementation cannot map, DSPy raises
`LMUnsupportedFeatureError` before the provider call.

For example, a text-only LM rejects image input before `forward()` runs.

Text generation support is inferred from `forward()`. For every richer request
shape, implement the corresponding hook:

| Request feature | Hook |
| --- | --- |
| text | `map_request_text()` |
| image input | `map_request_input_image()` |
| audio input | `map_request_input_audio()` |
| file input | `map_request_input_file()` |
| tools | `map_request_tools()` |
| tool choice | `map_request_tool_choice()` |
| assistant tool calls | `map_request_assistant_tool_calls()` |
| tool results | `map_request_tool_results()` |
| response schema | `map_request_response_schema()` |
| reasoning config | `map_request_reasoning_config()` |
| provider prompt cache | `map_request_prompt_cache()` |
| logprobs | `map_request_logprobs()` |
| multiple outputs | `map_request_multiple_outputs()` |
| provider extensions | `map_request_provider_extensions()` |

To support a response feature, implement the corresponding response hook:

| Response feature | Hook |
| --- | --- |
| text | `map_response_text()` |
| reasoning | `map_response_reasoning()` |
| tool calls | `map_response_tool_calls()` |
| citations | `map_response_citations()` |
| generated images | `map_response_output_image()` |
| generated audio | `map_response_output_audio()` |
| generated files | `map_response_output_file()` |
| refusal | `map_response_refusal()` |

Start with the smallest set your provider truly supports. DSPy will explain the
rest through feature reports and unsupported-feature errors.

## Feature reports and capabilities

The normalized path has two related concepts:

- `lm.capabilities`: model- or deployment-level hints, such as whether a model
  can use tools, images, reasoning, or streaming.
- `lm.features`: implementation support and observed runtime behavior.

Request validation uses implementation support. A model may be capable of
vision, but a custom LM still rejects images unless it implements
`map_request_input_image()`.

```python
print(lm.features.report())

lm.features.request.input_image
lm.features.response.tool_calls
lm.features.supports("streaming")
lm.features.explain("usage")
lm.features.report(format="json")
lm.features.to_json()
```

Feature states are:

- `inferred`: DSPy can infer support from implemented hooks or overridden methods.
- `observed`: DSPy saw the feature work at runtime.
- `unsupported`: DSPy knows the implementation cannot support the feature.
- `unknown`: DSPy has not observed enough information yet.

## Streaming

Normalized streaming uses DSPy event objects, not provider chunks.

A streaming LM implements `forward_stream()`:

```python
class StreamingEchoLM(dspy.LanguageModel):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("hello", model=request.model)

    def forward_stream(self, request: dspy.LMRequest):
        yield dspy.LMStreamStartEvent(model=request.model)
        yield dspy.LMStreamDeltaEvent(
            output_index=0,
            part_index=0,
            delta=dspy.LMTextDelta(text="hello"),
        )
        yield dspy.LMStreamOutputEndEvent(output_index=0, finish_reason="stop")
        yield dspy.LMStreamEndEvent()
```

Callers iterate the stream, then read the final response:

```python
stream = lm.stream("hello")
for event in stream:
    print(event.type)

response = stream.result()
print(response.text)
```

Async streaming uses `aforward_stream()` and `astream()`:

```python
stream = lm.astream("hello")
async for event in stream:
    print(event.type)

response = stream.result()
```

`astream()` returns an async iterator directly. Do not `await lm.astream(...)`
before iterating it.

The core event types are:

- `LMStreamStartEvent`
- `LMStreamDeltaEvent`
- `LMStreamOutputEndEvent`
- `LMStreamEndEvent`
- `LMStreamErrorEvent`

`LMOutputBuilder` assembles deltas into a final `LMResponse`. Use stable
`output_index` and `part_index` values so reasoning, text, and tool-call parts
remain distinct during interleaved streams.

## Callbacks

Normalized LMs use DSPy's callback system for sync calls, async calls, cache
hits, normalized exceptions, and streams.

Callbacks receive a normalized request under `inputs["request"]`. Raw call
inputs are included under `inputs["raw"]`. API keys and common auth fields are
redacted before callbacks see them.

For regular sync and async calls, callbacks fire around the call:

```text
on_lm_start(...)
on_lm_end(...)
```

For cache hits, callbacks still fire. For streaming calls, callbacks start when
the stream is consumed and end after the stream finishes or errors.

If `normalize_error()` maps a provider exception to a DSPy exception, callbacks
receive the normalized exception.

## Error normalization

Custom LMs should map provider errors that DSPy understands. The most important
one is context-window overflow.

```python
class MyLM(dspy.LanguageModel):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        try:
            ...
        except ProviderContextError as error:
            raise dspy.ContextWindowExceededError(model=request.model, provider="my-provider") from error
```

You can also centralize this in `normalize_error()`:

```python
class MyLM(dspy.LanguageModel):
    def normalize_error(self, error: Exception, request: dspy.LMRequest) -> Exception:
        if isinstance(error, ProviderContextError):
            return dspy.ContextWindowExceededError(model=request.model, provider="my-provider")
        return error
```

When DSPy observes a normalized context-window error, it records that support in
`lm.features.context_window_errors`.

## History

`LanguageModel.__call__()` records normalized request and response objects in
history unless history is disabled.

```python
entry = lm.history[-1]
entry.request
entry.response
```

History entries also support legacy dictionary-style keys:

```python
entry["outputs"]
entry["usage"]
entry["cost"]
entry["prompt"]
entry["messages"]
entry["kwargs"]
entry["model"]
entry["response_model"]
entry["timestamp"]
entry["uuid"]
```

`entry["prompt"]` is present for a single plain user text prompt.
`entry["messages"]` is derived from the normalized messages in an OpenAI-style
shape for compatibility.

Use `lm.inspect_history()` or `dspy.inspect_history()` to print recent
interactions.

## Copy, save, and load

Optimizers rely on `lm.copy(...)` to create variants:

```python
hot_lm = lm.copy(temperature=1.0, rollout_id=7)
```

The expected behavior is:

- inference defaults are copied and can be overridden
- history is reset
- feature observations are reset on the copy
- provider resources remain valid for the concrete LM

The default `copy()` uses `deepcopy`. Override it if your LM holds SDK clients,
HTTP sessions, sockets, local model weights, subprocesses, or other non-copyable
state.

The default `dump_state()` returns a sanitized constructor state:

```python
{
    "model": self.model,
    "cache": self.cache,
    **filtered_kwargs,
}
```

API keys and `api_*` values are not saved by the default implementation. If your
LM needs custom save/load behavior, override both `dump_state()` and
`load_state()`.

## Using provider SDKs directly

A custom LM is a small adapter from `LMRequest` to your SDK call, and from your
SDK response back to `LMResponse`.

The examples below are intentionally text-first. They support normal DSPy
modules and direct text calls. Add mapping hooks when you want images, files,
tools, structured outputs, provider prompt caching, logprobs, or multiple
outputs.

!!! info "A few direct SDK-backed `LanguageModel` subclasses"

    === "OpenAI SDK"
        OpenAI's SDK returns OpenAI Chat Completions-shaped objects, so you can
        subclass `OpenAIChatLM` and implement only the transport method.

        ```python linenums="1"
        import os

        from openai import OpenAI
        import dspy


        class OpenAISDKLM(dspy.OpenAIChatLM):
            def __init__(self, model: str, api_key: str | None = None, **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

            def completion(self, request: dict):
                return self.client.chat.completions.create(**request)

            async def acompletion(self, request: dict):
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=self.client.api_key)
                return await client.chat.completions.create(**request)


        dspy.configure(experimental_lm=True)
        dspy.configure(lm=OpenAISDKLM("gpt-4o-mini"))
        ```

        If you prefer the Responses API, subclass `OpenAIResponsesLM` and
        implement `responses()` instead.

    === "Anthropic SDK"
        Anthropic's native Messages API is not OpenAI-shaped, so subclass
        `LanguageModel` directly and map messages yourself.

        ```python linenums="1"
        import os

        import anthropic
        import dspy


        class AnthropicSDKLM(dspy.LanguageModel):
            def __init__(self, model: str, api_key: str | None = None, **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                system = []
                messages = []
                for message in request.messages:
                    text = message.text or ""
                    if message.role == "system":
                        system.append(text)
                    elif message.role in {"user", "assistant"}:
                        messages.append({"role": message.role, "content": text})

                response = self.client.messages.create(
                    model=request.model,
                    system="\n\n".join(system) or None,
                    messages=messages,
                    max_tokens=request.config.max_tokens or 1024,
                    temperature=request.config.temperature,
                )

                return dspy.LMResponse.from_text(
                    response.content[0].text,
                    model=response.model,
                    usage=dspy.LMUsage(
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    ),
                    provider_response=response,
                )

            def normalize_error(self, error: Exception, request: dspy.LMRequest) -> Exception:
                if error.__class__.__name__ == "BadRequestError" and "context" in str(error).lower():
                    return dspy.ContextWindowExceededError(model=request.model, provider="anthropic")
                return error


        dspy.configure(experimental_lm=True)
        dspy.configure(lm=AnthropicSDKLM("claude-sonnet-4-5-20250929"))
        ```

    === "Gemini / Google GenAI"
        The GenAI SDK accepts text prompts directly. This wrapper flattens the
        normalized messages into one prompt.

        ```python linenums="1"
        import os

        from google import genai
        import dspy


        class GenAILM(dspy.LanguageModel):
            def __init__(self, model: str, api_key: str | None = None, **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = genai.Client(api_key=api_key or os.environ.get("GEMINI_API_KEY"))

            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                prompt = "\n\n".join(
                    f"{message.role}: {message.text or ''}" for message in request.messages
                )
                response = self.client.models.generate_content(
                    model=request.model,
                    contents=prompt,
                    config={
                        "temperature": request.config.temperature,
                        "max_output_tokens": request.config.max_tokens,
                    },
                )
                return dspy.LMResponse.from_text(
                    response.text,
                    model=request.model,
                    provider_response=response,
                )


        dspy.configure(experimental_lm=True)
        dspy.configure(lm=GenAILM("gemini-2.5-pro"))
        ```

    === "Groq SDK"
        Groq's SDK follows the OpenAI Chat Completions shape, so reuse
        `OpenAIChatLM`'s mapping and supply Groq's transport.

        ```python linenums="1"
        import os

        from groq import Groq
        import dspy


        class GroqSDKLM(dspy.OpenAIChatLM):
            def __init__(self, model: str, api_key: str | None = None, **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

            def completion(self, request: dict):
                return self.client.chat.completions.create(**request)


        dspy.configure(experimental_lm=True)
        dspy.configure(lm=GroqSDKLM("llama-3.3-70b-versatile"))
        ```

    === "Mistral SDK"
        Mistral's SDK has its own response objects, but the text mapping is
        small.

        ```python linenums="1"
        import os

        from mistralai import Mistral
        import dspy


        class MistralSDKLM(dspy.LanguageModel):
            def __init__(self, model: str, api_key: str | None = None, **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = Mistral(api_key=api_key or os.environ.get("MISTRAL_API_KEY"))

            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                messages = [
                    {"role": message.role, "content": message.text or ""}
                    for message in request.messages
                    if message.role in {"system", "user", "assistant"}
                ]
                response = self.client.chat.complete(
                    model=request.model,
                    messages=messages,
                    temperature=request.config.temperature,
                    max_tokens=request.config.max_tokens,
                )
                usage = getattr(response, "usage", None)
                return dspy.LMResponse.from_text(
                    response.choices[0].message.content,
                    model=request.model,
                    usage=(
                        dspy.LMUsage(
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens,
                            total_tokens=usage.total_tokens,
                        )
                        if usage is not None
                        else None
                    ),
                    provider_response=response,
                )


        dspy.configure(experimental_lm=True)
        dspy.configure(lm=MistralSDKLM("mistral-small-latest"))
        ```

    === "Ollama Python SDK"
        For local models, the Ollama SDK can be wrapped directly without going
        through LiteLLM.

        ```python linenums="1"
        import ollama
        import dspy


        class OllamaSDKLM(dspy.LanguageModel):
            def __init__(self, model: str, host: str = "http://localhost:11434", **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = ollama.Client(host=host)

            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                messages = [
                    {"role": message.role, "content": message.text or ""}
                    for message in request.messages
                    if message.role in {"system", "user", "assistant"}
                ]
                response = self.client.chat(
                    model=request.model,
                    messages=messages,
                    options={
                        "temperature": request.config.temperature,
                        "num_predict": request.config.max_tokens,
                    },
                )
                return dspy.LMResponse.from_text(
                    response["message"]["content"],
                    model=request.model,
                    usage=dspy.LMUsage(
                        input_tokens=response.get("prompt_eval_count"),
                        output_tokens=response.get("eval_count"),
                    ),
                    provider_response=response,
                )


        dspy.configure(experimental_lm=True)
        dspy.configure(lm=OllamaSDKLM("llama3.2"))
        ```

    === "Internal SDK or gateway"
        Use the same pattern for an in-house SDK. Keep authentication and client
        objects on the LM instance, and keep provider secrets out of `kwargs` so
        they do not enter normalized request config.

        ```python linenums="1"
        import dspy


        class AcmeLM(dspy.LanguageModel):
            def __init__(self, model: str, client, **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = client

            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                response = self.client.generate(
                    model=request.model,
                    prompt="\n\n".join(message.text or "" for message in request.messages),
                    temperature=request.config.temperature,
                    max_tokens=request.config.max_tokens,
                )
                return dspy.LMResponse.from_text(
                    response.text,
                    model=request.model,
                    usage=dspy.LMUsage(
                        input_tokens=response.input_tokens,
                        output_tokens=response.output_tokens,
                    ),
                    cost=response.cost,
                    provider_response=response,
                )
        ```


## Anthropic modality mapping examples

Anthropic's native SDK is a good example of when to subclass
`dspy.LanguageModel` directly. The SDK has its own message, content-block, tool,
and usage shapes, so your LM should map each normalized DSPy part explicitly.

Read these examples left to right. Each tab defines one class that subclasses the
class from the previous tab. Start with text, then add images, tools, built-in
tools, reasoning, citations, and streaming.

!!! info "A progressive Anthropic `LanguageModel`"

    === "1. Text"
        Start with text. This class supports system/user/assistant text and
        returns text plus usage.

        ```python linenums="1"
        import os

        import anthropic
        import dspy


        class AnthropicTextLM(dspy.LanguageModel):
            def __init__(self, model: str, api_key: str | None = None, **kwargs):
                super().__init__(model=model, **kwargs)
                self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

            def map_request_text(self, value):
                return {"type": "text", "text": str(value)}

            def map_response_text(self, value):
                return dspy.LMTextPart(text=str(value))

            def _message_content_blocks(self, message: dspy.LMMessage) -> list[dict]:
                return [
                    self.map_request_text(part.text)
                    for part in message.parts
                    if isinstance(part, dspy.LMTextPart)
                ]

            def _message_kwargs(self, request: dspy.LMRequest) -> dict:
                system = []
                messages = []
                for message in request.messages:
                    if message.role in {"system", "developer"}:
                        system.extend(block["text"] for block in self._message_content_blocks(message))
                    elif message.role in {"user", "assistant"}:
                        messages.append({"role": message.role, "content": self._message_content_blocks(message)})
                    elif message.role == "tool":
                        # Anthropic carries tool results as user content blocks.
                        messages.append({"role": "user", "content": self._message_content_blocks(message)})
                data = {"system": "\n\n".join(system) or None, "messages": messages}
                return {key: value for key, value in data.items() if value is not None}

            def _request_kwargs(self, request: dspy.LMRequest) -> dict:
                data = {
                    "model": request.model,
                    "max_tokens": request.config.max_tokens or 1024,
                    "temperature": request.config.temperature,
                    **self._message_kwargs(request),
                }
                return {key: value for key, value in data.items() if value is not None}

            def _parts_from_response(self, response) -> list[dspy.LMPart]:
                parts = []
                for block in self._get(response, "content", []):
                    if self._get(block, "type") == "text":
                        parts.append(self.map_response_text(self._get(block, "text", "")))
                return parts

            def _usage_from_response(self, response):
                usage = self._get(response, "usage")
                return dspy.LMUsage(
                    input_tokens=self._get(usage, "input_tokens"),
                    output_tokens=self._get(usage, "output_tokens"),
                )

            def _get(self, value, name, default=None):
                if isinstance(value, dict):
                    return value.get(name, default)
                return getattr(value, name, default)

            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                response = self.client.messages.create(**self._request_kwargs(request))
                return dspy.LMResponse(
                    model=self._get(response, "model", request.model),
                    outputs=[
                        dspy.LMOutput(
                            parts=self._parts_from_response(response),
                            finish_reason=self._get(response, "stop_reason"),
                        )
                    ],
                    usage=self._usage_from_response(response),
                    provider_response=response,
                )


        lm = AnthropicTextLM("claude-sonnet-4-5-20250929")
        print(lm("Say hello").text)
        ```

    === "2. Add images"
        Add `map_request_input_image()`. Because this hook exists, DSPy will
        allow image parts through request validation before `forward()` runs.

        ```python linenums="1"
        import base64

        import dspy


        class AnthropicVisionLM(AnthropicTextLM):
            def map_request_input_image(self, value: dspy.LMImagePart):
                if value.data is None:
                    raise ValueError("This example accepts base64 image data only.")
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": value.media_type,
                        "data": value.data,
                    },
                }

            def _message_content_blocks(self, message: dspy.LMMessage) -> list[dict]:
                blocks = []
                for part in message.parts:
                    if isinstance(part, dspy.LMTextPart):
                        blocks.append(self.map_request_text(part.text))
                    elif isinstance(part, dspy.LMImagePart):
                        blocks.append(self.map_request_input_image(part))
                return blocks


        image = base64.b64encode(open("dog.png", "rb").read()).decode()
        lm = AnthropicVisionLM("claude-sonnet-4-5-20250929")
        response = lm("Describe this image.", dspy.Image(f"data:image/png;base64,{image}"))
        ```

        This example rejects URL images on purpose. If your provider path can
        fetch URLs, map `value.url` too.

    === "3. Add native tool calls"
        Add tool request hooks, tool-continuation hooks, and tool-call response
        mapping. This supports Python tools normalized to `LMToolSpec`, plus
        multi-turn `Assistant(ToolCall(...))` and `ToolResult(...)` messages.

        ```python linenums="1"
        import dspy


        class AnthropicToolLM(AnthropicVisionLM):
            def map_request_tools(self, value: dspy.LMToolSpec):
                return {
                    "name": value.name,
                    "description": value.description or "",
                    "input_schema": value.parameters,
                }

            def map_request_tool_choice(self, value: dspy.LMToolChoice):
                if value.mode == "required":
                    return {"type": "any"}
                if value.mode == "none":
                    return {"type": "none"}
                return {"type": "auto"}

            def map_request_assistant_tool_calls(self, value: dspy.LMToolCallPart):
                if value.id is None:
                    raise ValueError("Anthropic tool-use continuations require a tool call id.")
                return {
                    "type": "tool_use",
                    "id": value.id,
                    "name": value.name,
                    "input": value.args,
                }

            def map_request_tool_results(self, value: dspy.LMToolResultPart):
                if value.call_id is None:
                    raise ValueError("Anthropic tool results require the matching tool call id.")
                return {
                    "type": "tool_result",
                    "tool_use_id": value.call_id,
                    "content": self._text_from_parts(value.content),
                    "is_error": value.is_error,
                }

            def map_response_tool_calls(self, value):
                return dspy.LMToolCallPart(
                    id=self._get(value, "id"),
                    name=self._get(value, "name", ""),
                    args=dict(self._get(value, "input", {}) or {}),
                    provider_data={"raw_type": self._get(value, "type")},
                )

            def _text_from_parts(self, parts: list[dspy.LMPart]) -> str:
                texts = []
                for part in parts:
                    if not isinstance(part, dspy.LMTextPart):
                        raise ValueError("This example maps text-only tool results.")
                    texts.append(part.text)
                return "".join(texts)

            def _message_content_blocks(self, message: dspy.LMMessage) -> list[dict]:
                blocks = []
                for part in message.parts:
                    if isinstance(part, dspy.LMTextPart):
                        blocks.append(self.map_request_text(part.text))
                    elif isinstance(part, dspy.LMImagePart):
                        blocks.append(self.map_request_input_image(part))
                    elif isinstance(part, dspy.LMToolCallPart):
                        blocks.append(self.map_request_assistant_tool_calls(part))
                    elif isinstance(part, dspy.LMToolResultPart):
                        blocks.append(self.map_request_tool_results(part))
                return blocks

            def _request_kwargs(self, request: dspy.LMRequest) -> dict:
                kwargs = super()._request_kwargs(request)
                if request.tools:
                    kwargs["tools"] = [self.map_request_tools(tool) for tool in request.tools]
                if request.config.tool_choice is not None:
                    kwargs["tool_choice"] = self.map_request_tool_choice(request.config.tool_choice)
                return kwargs

            def _parts_from_response(self, response) -> list[dspy.LMPart]:
                parts = []
                for block in self._get(response, "content", []):
                    block_type = self._get(block, "type")
                    if block_type == "text":
                        parts.append(self.map_response_text(self._get(block, "text", "")))
                    elif block_type == "tool_use":
                        parts.append(self.map_response_tool_calls(block))
                return parts


        def get_weather(city: str) -> str:
            """Get the weather for a city."""
            return "sunny"


        lm = AnthropicToolLM("claude-sonnet-4-5-20250929")
        response = lm("What is the weather in Paris?", tools=[dspy.Tool(get_weather)])
        response.tool_calls
        ```

    === "4. Add built-in tools"
        Provider built-in tools are not Python functions. Keep their raw
        provider shape in `LMToolSpec.provider_data`, then let
        `map_request_tools()` pass that shape through.

        ```python linenums="1"
        import dspy


        web_search = dspy.LMToolSpec(
            name="web_search",
            description="Search the web with Anthropic's hosted tool.",
            parameters={},
            provider_data={
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 3,
            },
        )


        class AnthropicBuiltinToolLM(AnthropicToolLM):
            def map_request_tools(self, value: dspy.LMToolSpec):
                if value.provider_data:
                    return dict(value.provider_data)
                return super().map_request_tools(value)


        lm = AnthropicBuiltinToolLM("claude-sonnet-4-5-20250929")
        response = lm("Find the latest DSPy release notes.", tools=[web_search])
        ```

    === "5. Add native reasoning"
        Add `map_request_reasoning_config()` for DSPy's `reasoning=` and
        `reasoning_effort=` kwargs, then map provider thinking blocks to
        `LMThinkingPart`.

        ```python linenums="1"
        import dspy


        class AnthropicReasoningLM(AnthropicBuiltinToolLM):
            def map_request_reasoning_config(self, value: dspy.LMReasoningConfig):
                thinking = {"type": "enabled"}
                if value.max_tokens is not None:
                    thinking["budget_tokens"] = value.max_tokens
                return thinking

            def map_response_reasoning(self, value):
                return dspy.LMThinkingPart(text=str(value))

            def _request_kwargs(self, request: dspy.LMRequest) -> dict:
                kwargs = super()._request_kwargs(request)
                if request.config.reasoning is not None:
                    kwargs["thinking"] = self.map_request_reasoning_config(request.config.reasoning)
                return kwargs

            def _parts_from_response(self, response) -> list[dspy.LMPart]:
                parts = []
                for block in self._get(response, "content", []):
                    block_type = self._get(block, "type")
                    if block_type == "thinking":
                        parts.append(self.map_response_reasoning(self._get(block, "thinking", "")))
                    elif block_type == "text":
                        parts.append(self.map_response_text(self._get(block, "text", "")))
                    elif block_type == "tool_use":
                        parts.append(self.map_response_tool_calls(block))
                return parts


        lm = AnthropicReasoningLM("claude-sonnet-4-5-20250929")
        response = lm("Think briefly, then answer.", reasoning=dspy.LMReasoningConfig(max_tokens=1024))
        response.reasoning_content
        ```

    === "6. Add documents + citations"
        Add file input for documents and map provider citations back to
        `LMCitationPart`.

        ```python linenums="1"
        import dspy


        class AnthropicCitationLM(AnthropicReasoningLM):
            def map_request_input_file(self, value: dspy.LMFilePart):
                if value.data is None:
                    raise ValueError("This example accepts base64 document data only.")
                return {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": value.media_type,
                        "data": value.data,
                    },
                    "title": value.filename,
                    "citations": {"enabled": True},
                }

            def map_response_citations(self, value):
                return dspy.LMCitationPart(
                    text=self._get(value, "cited_text") or self._get(value, "text"),
                    title=self._get(value, "document_title") or self._get(value, "title"),
                    url=self._get(value, "url"),
                    metadata={
                        key: item
                        for key in ("document_index", "start_char_index", "end_char_index")
                        if (item := self._get(value, key)) is not None
                    },
                )

            def _message_content_blocks(self, message: dspy.LMMessage) -> list[dict]:
                blocks = []
                for part in message.parts:
                    if isinstance(part, dspy.LMFilePart):
                        blocks.append(self.map_request_input_file(part))
                    else:
                        blocks.extend(super()._message_content_blocks(dspy.LMMessage(role=message.role, parts=[part])))
                return blocks

            def _parts_from_response(self, response) -> list[dspy.LMPart]:
                parts = super()._parts_from_response(response)
                for block in self._get(response, "content", []):
                    for citation in self._get(block, "citations", []) or []:
                        parts.append(self.map_response_citations(citation))
                return parts
        ```

        The exact citation shape depends on the SDK response. Keep anything not
        represented by `text`, `title`, or `url` in `metadata`.

    === "7. Add streaming"
        Finally, add `forward_stream()`. It reuses the same request mapping and
        converts provider stream events into normalized `LMStreamEvent` objects.

        ```python linenums="1"
        import dspy


        class AnthropicStreamingLM(AnthropicCitationLM):
            def forward_stream(self, request: dspy.LMRequest):
                yield dspy.LMStreamStartEvent(model=request.model)

                with self.client.messages.stream(**self._request_kwargs(request)) as stream:
                    for event in stream:
                        if self._get(event, "type") != "content_block_delta":
                            continue
                        delta = self._get(event, "delta")
                        delta_type = self._get(delta, "type")
                        if delta_type == "thinking_delta":
                            yield dspy.LMStreamDeltaEvent(
                                output_index=0,
                                part_index=0,
                                delta=dspy.LMThinkingDelta(text=self._get(delta, "thinking", "")),
                            )
                        elif delta_type == "text_delta":
                            yield dspy.LMStreamDeltaEvent(
                                output_index=0,
                                part_index=1,
                                delta=dspy.LMTextDelta(text=self._get(delta, "text", "")),
                            )

                yield dspy.LMStreamOutputEndEvent(output_index=0)
                yield dspy.LMStreamEndEvent()


        lm = AnthropicStreamingLM("claude-sonnet-4-5-20250929")
        stream = lm.stream("Say hello")
        for event in stream:
            print(event.type)
        print(stream.result().text)
        ```

        Use stable `part_index` values. Here reasoning is part `0` and text is
        part `1`; add separate indexes for tool-call deltas if your stream emits
        them.


## OpenAI-compatible endpoints

If your provider exposes an OpenAI-compatible Chat Completions endpoint, reuse
DSPy's OpenAI-format mapping and provide only the transport.

```python
from openai import OpenAI
import dspy


dspy.configure(experimental_lm=True)

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

lm = dspy.CompletionLM(
    "meta-llama/Llama-3.1-8B-Instruct",
    completion=client.chat.completions.create,
)

dspy.configure(lm=lm)
```

For the Responses API, use `ResponsesLM`:

```python
lm = dspy.ResponsesLM(
    "openai/gpt-4o-mini",
    responses=client.responses.create,
)
```

For text completions, use `TextCompletionLM`:

```python
lm = dspy.TextCompletionLM(
    "my-text-model",
    completion=client.completions.create,
)
```

If you need more control, subclass `dspy.OpenAIChatLM`,
`dspy.OpenAIResponsesLM`, or `dspy.OpenAITextLM` and implement only the
transport method: `completion()`, `responses()`, or `text_completion()`.

## Registering a backend for `dspy.LM`

Register a backend factory if you want `dspy.LM("acme/...")` to route to your
custom normalized LM.

```python
import dspy


@dspy.register_lm_backend
def route_acme(model: str, *args, **kwargs):
    if model.startswith("acme/"):
        return AcmeLM(model, *args, **kwargs)
    return None


dspy.configure(experimental_lm=True)
dspy.configure(lm=dspy.LM("acme/small"))
```

Factories are tried in reverse registration order. Return `None` when your
factory does not own the model.

## Migration checklist

- [ ] Set `dspy.configure(experimental_lm=True)` before constructing normalized
      `dspy.LM(...)` objects.
- [ ] Change custom LM subclasses from `dspy.BaseLM` to `dspy.LanguageModel`.
- [ ] Change `forward(prompt=None, messages=None, **kwargs)` to
      `forward(request: dspy.LMRequest)`.
- [ ] Convert `request.messages`, `request.tools`, and `request.config` to your
      provider request.
- [ ] Return `dspy.LMResponse`, not a provider-shaped response.
- [ ] Put generated text, reasoning, tool calls, citations, files, and media in
      `LMOutput.parts`.
- [ ] Put usage on `response.usage` and cost on `response.cost` when available.
- [ ] Implement request mapping hooks for every non-text request shape you
      support.
- [ ] Implement response mapping hooks for every non-text response part you
      support.
- [ ] Normalize recognizable provider errors, especially context-window errors.
- [ ] Implement streaming with normalized `LMStreamEvent` objects if your
      provider streams.
- [ ] Override `copy()` if your LM holds non-copyable runtime resources.
- [ ] Override `dump_state()` and `load_state()` if your LM should round-trip
      through program save/load.

## Legacy `BaseLM`

`dspy.BaseLM` remains the legacy prompt/messages contract. Existing custom LMs
can continue to use it during the migration window, but new custom LMs should
use `dspy.LanguageModel`.

Subclassing `dspy.BaseLM` outside DSPy may emit a `DeprecationWarning`. You can
silence it while migrating:

```python
import dspy

dspy.configure(warn_legacy_lm=False)
```

This only silences the warning. It does not migrate the LM.

## See also

- [`dspy.LM`](../api/models/LM.md)
- [`dspy.configure`](../api/utils/configure.md)
- [Language Models](../learn/programming/language_models.md)
