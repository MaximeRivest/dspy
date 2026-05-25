# Migrating `dspy.BaseLM` subclasses to the typed LM contract

DSPy 3.3 introduces an experimental typed language-model boundary. The goal is to let every LM backend receive the same `dspy.LMRequest` object and return the same `dspy.LMResponse` object, while keeping existing custom LMs working during the migration window. In 3.3, this path is opt-in with `dspy.configure(experimental=True)` or `dspy.context(experimental=True)`.

!!! warning "Experimental in DSPy 3.3"
    The typed LM contract is public for experimentation, but its API is not yet stable. You must opt in with `experimental=True` in DSPy 3.3. We may rename fields, adjust helper methods, or change edge-case behavior before it becomes the default. The planned migration path is: introduce the typed contract experimentally in 3.3, warn more actively for legacy custom LMs in 3.4, make the typed contract the default in 3.5, and remove the legacy `forward(prompt=None, messages=None, **kwargs)` contract in a later release, likely 3.6 or 4.0.

## What changed

A new-style `BaseLM` subclass implements one method:

```python
import dspy

class MyLM(dspy.BaseLM):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("hello", model=request.model)
```

Instead of receiving OpenAI-shaped `prompt`, `messages`, and loose keyword arguments, the backend receives:

- `request.model`: the model or deployment identifier.
- `request.messages`: normalized `LMMessage` objects with typed content parts.
- `request.tools`: normalized tool schemas.
- `request.config`: generation options such as `temperature`, `max_tokens`, `cache`, `rollout_id`, `reasoning`, `tool_choice`, and provider-specific `extensions`.
- `request.metadata`: optional request metadata.

The backend returns an `LMResponse`, which contains one or more `LMOutput` candidates. Each output contains typed parts such as text, reasoning, tool calls, citations, images, audio, and files.

## What still works

Existing custom LMs can keep using the legacy signature during the 3.3 migration window:

```python
class LegacyLM(dspy.BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        ...  # return an OpenAI-shaped provider response
```

DSPy 3.3 does not warn for this signature yet. Without `experimental=True`, direct calls to legacy subclasses keep the stable behavior and return legacy `list[str | dict]` outputs. With `experimental=True`, normalized calls with `request=` translate the `LMRequest` into legacy `prompt`/`messages`/`kwargs` and wrap the provider response into an `LMResponse`.

## Responsibilities of the base class

With `experimental=True`, new-style subclasses get the common DSPy machinery around `forward(request)`:

- direct-call input normalization, e.g. `lm("hello", temperature=0.7)`;
- `request=` overrides, e.g. `lm(request=req, temperature=0.2)`;
- retries for structured retryable LM errors;
- DSPy request caching for non-streaming calls;
- callbacks;
- usage tracking;
- history updates;
- redaction of secrets in history and callbacks;
- compatibility properties such as `supports_function_calling`.

The subclass is responsible for provider I/O and for returning an `LMResponse`.

## Porting a v1 subclass

A typical legacy implementation looks like this:

```python
class FakeLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="fake/test")

    def forward(self, prompt=None, messages=None, **kwargs):
        return provider.chat(
            model=self.model,
            messages=messages or [{"role": "user", "content": prompt}],
            **kwargs,
        )
```

The new implementation accepts `LMRequest` and returns `LMResponse`:

```python
class FakeLM(dspy.BaseLM):
    def __init__(self):
        super().__init__(model="fake/test")

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        provider_response = provider.chat(
            model=request.model,
            messages=[message.to_dict() for message in request.messages],
            temperature=request.config.temperature,
            max_tokens=request.config.max_tokens,
        )
        return dspy.LMResponse.from_text(provider_response.text, model=request.model)
```

For OpenAI-compatible providers, prefer the conversion helpers in `dspy.clients.openai_format` instead of manually serializing messages.

## Direct-call UX for typed LMs

Once an LM uses the v2 `BaseLM` contract, direct calls can use typed messages and parts instead of provider-specific dictionaries. This is useful for custom LM authors, tests, local backends, and advanced users who want to exercise the LM boundary directly. In DSPy 3.3, these calls require `experimental=True`.

```python
import dspy

class EchoLM(dspy.BaseLM):
    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text(
            f"received {len(request.messages)} messages",
            model=request.model,
        )

lm = EchoLM(model="test/echo")
with dspy.context(experimental=True):
    response = lm(
        dspy.System("You are concise."),
        dspy.User("What is DSPy?"),
    )

assert response.text == "received 2 messages"
```

### Single-turn multimodal calls

For a single user turn, pass content parts directly to `lm(...)`. DSPy turns them into one `dspy.User(...)` message.

```python
response = lm(
    "Describe these inputs.",
    dspy.Image("https://example.com/dog.png"),
    dspy.Audio(data="...base64...", audio_format="wav"),
    dspy.File(file_data="data:application/pdf;base64,...", filename="paper.pdf"),
)
```

The request received by `forward()` contains typed parts such as `LMTextPart`, `LMImagePart`, `LMAudioPart`, and `LMBinaryPart`. Backend code can inspect those parts directly or convert the request to a provider shape.

### Multi-turn calls with role helpers

Use `dspy.System`, `dspy.Developer`, `dspy.User`, and `dspy.Assistant` to build multi-turn requests without writing raw message dictionaries.

```python
response = lm(
    dspy.System("Answer in one sentence."),
    dspy.User("What is DSPy?"),
    dspy.Assistant("DSPy is a framework for programming LM pipelines."),
    dspy.User("Say that in five words."),
)
```

You can also feed a previous `LMResponse` back into the next request. DSPy converts the response's first output into an assistant message.

```python
first = lm("What is DSPy?")
second = lm(
    dspy.User("Use this prior answer."),
    first,
    dspy.User("Make it shorter."),
)
```

### Tools and tool results

Tool inputs can be passed positionally with a single-turn request. DSPy keeps the tool out of message content and adds it to `request.tools`.

```python
def crop_image(x1: int, y1: int, x2: int, y2: int) -> str:
    """Crop an image to a bounding box."""
    return "cropped-image-id"

request = lm.normalize_request(
    "Crop the dog.",
    dspy.Image("https://example.com/dog.png"),
    dspy.Tool(crop_image),
)

assert request.tools[0].name == "crop_image"
```

For explicit tool-call conversations, use `dspy.ToolCall` inside an assistant message and `dspy.ToolResult` for the tool output.

```python
response = lm(
    dspy.User("What is the weather in Paris?"),
    dspy.Assistant(
        dspy.ToolCall(id="call_1", name="get_weather", args={"location": "Paris"}),
    ),
    dspy.ToolResult(
        '{"temperature": "22", "unit": "celsius"}',
        call_id="call_1",
        name="get_weather",
    ),
    dspy.User("Summarize the result."),
)
```

### Config overrides

Direct-call kwargs become `request.config` fields. Provider-specific kwargs are preserved in `request.config.extensions`.

```python
response = lm(
    "Think briefly, then answer.",
    temperature=0.2,
    max_tokens=200,
    reasoning_effort="low",
    rollout_id="experiment-1",
    provider_specific_flag=True,
)
```

If you already have an `LMRequest`, pass it with `request=` and provide only the config overrides you want to change. Existing grouped config such as `cache`, `tool_choice`, `prompt_cache`, and `reasoning` is preserved.

```python
request = dspy.LMRequest.from_call(
    model="test/echo",
    prompt="hello",
    temperature=0.0,
    rollout_id="old",
)

response = lm(request=request, temperature=0.7, rollout_id="new")
```

### Response conveniences

Typed LMs return `LMResponse`, which is friendly for direct use and still exposes legacy-compatible views.

```python
response = lm("Say hello.")

response.text              # first output's text, if any
response.parts             # first output's typed parts
response.tool_calls        # tool-call parts from the first output
response.reasoning_content # reasoning text from the first output
response.images            # generated image parts from the first output
response.usage_as_dict()   # normalized usage metadata
list(response)             # output values for all candidates
response.to_outputs()      # legacy list[str | dict] shape
```

## Returning richer outputs

Use `LMResponse.from_text()` for simple text-only responses:

```python
return dspy.LMResponse.from_text("Paris", model=request.model)
```

For richer responses, construct an `LMResponse` with `LMOutput` and typed parts:

```python
from dspy.core import types as lm

return dspy.LMResponse(
    model=request.model,
    outputs=[
        lm.LMOutput(
            parts=[
                lm.LMThinkingPart(text="I should check the weather tool."),
                lm.LMTextPart(text="I will look that up."),
                lm.LMToolCallPart(id="call_1", name="get_weather", args={"city": "Paris"}),
            ],
            finish_reason="tool_calls",
        )
    ],
)
```

`LMResponse` also exposes compatibility views:

```python
response.text
response.reasoning_content
response.tool_calls
response.to_outputs()         # legacy list[str | dict] shape
response.to_legacy_outputs()  # alias for compatibility
```

## Capabilities

New subclasses can declare model/deployment capabilities with `get_capabilities()`:

```python
class MyLM(dspy.BaseLM):
    def get_capabilities(self) -> dspy.LMCapabilities:
        return dspy.LMCapabilities(
            function_calling=True,
            reasoning=False,
            response_schema=True,
        )
```

`LMCapabilities` currently records the capabilities used by DSPy adapters:

- `function_calling`
- `reasoning`
- `response_schema`

The legacy boolean properties (`supports_function_calling`, `supports_reasoning`, and `supports_response_schema`) read from `get_capabilities()`.

## Async and streaming

Async and streaming are opt-in:

```python
class MyLM(dspy.BaseLM):
    async def aforward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        ...

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

`supports_streaming` is true only when the subclass overrides `forward_stream()`. Stream caching is intentionally not part of this first typed-LM release; non-streaming calls still use DSPy's request cache.

## Error handling

Custom LMs should normalize provider failures into DSPy's structured LM errors when possible:

```python
class MyLM(dspy.BaseLM):
    def normalize_error(self, error: Exception, request: dspy.LMRequest) -> Exception:
        if provider_is_rate_limit(error):
            return dspy.LMRateLimitError("rate limited", model=request.model, status=429)
        if provider_is_context_window(error):
            return dspy.ContextWindowExceededError(model=request.model)
        return error
```

Already-normalized `dspy.LMError` instances pass through unchanged. Retryable LM errors (`LMRateLimitError`, `LMTimeoutError`, `LMServerError`, and `LMTransportError`) are retried according to `num_retries`.

## Saving and loading custom LMs

`dump_state()` records the module-qualified LM class path and the constructor state captured by `BaseLM.__init__`. If your subclass has extra persistent state, override both `dump_state()` and `load_state()`.

```python
class MyLM(dspy.BaseLM):
    def __init__(self, model: str, *, deployment: str, **kwargs):
        super().__init__(model=model, **kwargs)
        self.deployment = deployment

    def dump_state(self):
        return {**super().dump_state(), "deployment": self.deployment}

    @classmethod
    def load_state(cls, state):
        state = dict(state)
        state.pop("_dspy_lm_class", None)
        return cls(**state)
```

Custom LM classes must be importable when loading saved programs. Loading custom LM classes requires trusted opt-in with `allow_unsafe_lm_state=True`.

## Migration timeline

The current expected timeline is:

- **3.3**: introduce the typed LM contract as experimental while preserving legacy custom LMs.
- **3.4**: warn more actively when custom LMs use the legacy `forward(prompt=None, messages=None, **kwargs)` signature.
- **3.5**: make the typed contract the default custom-LM path.
- **3.6 or 4.0**: remove the legacy signature, contract-version detection, and v1 translation shim. The exact removal release is still to be decided.

After removal, custom LMs should implement `forward(request: LMRequest) -> LMResponse` only.
