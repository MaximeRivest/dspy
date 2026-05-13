# Migrating custom LMs to the normalized LM API

DSPy’s normalized language-model path has one core idea:

```text
DSPy task/request  ->  provider SDK call  ->  DSPy response
```

In code, that means custom integrations should usually be built with
`dspy.LM.from_sdk(...)`, not by subclassing `dspy.LanguageModel`.

The stable contracts are:

- `dspy.LMRequest`: DSPy’s canonical input to an LM.
- `dspy.ProviderRequest`: the SDK/provider call you prepared.
- `dspy.LMResponse`: DSPy’s canonical output from an LM.
- `dspy.LMSupport`: what your wrapper explicitly supports.

The preferred integration shape is:

```python
lm = dspy.LM.from_sdk(
    model="my-sdk/model",
    support=dspy.LMSupport(...),
    map_request=map_request,
    call=call_sdk,
    map_response=map_response,
)
```

Then add features progressively:

```python
lm = (
    lm
    .with_images(...)
    .with_tools(...)
    .with_usage(...)
    .with_streaming(...)
    .with_errors(...)
)
```

Subclassing `dspy.LanguageModel` is still possible for DSPy internals and rare
escape hatches, but it is no longer the idiomatic custom-SDK path.

## Enable the normalized path

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

`experimental_lm` is read when `dspy.LM(...)` is constructed. Set it first, then
create the LM.

## The new mental model

Legacy custom LMs often implemented provider-shaped methods like:

```text
forward(prompt=None, messages=None, **kwargs) -> provider response
```

The normalized plugin path is more explicit:

```text
LMRequest -> ProviderRequest -> provider response -> LMResponse
```

A provider request is intentionally tiny:

```python
dspy.ProviderRequest(
    args=(...),          # positional SDK arguments
    kwargs={...},        # keyword SDK arguments
    data=...,            # optional native request object
    preview=...,         # optional display/debug representation
    metadata={...},
)
```

This works for JSON APIs:

```python
ProviderRequest(kwargs={"model": "...", "messages": [...]})
```

and for SDKs that use positional or top-level media arguments:

```python
ProviderRequest(args=("Describe this image",), kwargs={"images": [...]})
```

## A minimal text SDK wrapper

Suppose your SDK looks like this:

```python
response = sdk.generate("hello")
print(response.text)
```

Wrap it as a DSPy LM:

```python
import dspy


def map_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
    prompt = "\n\n".join(message.text or "" for message in request.messages)
    return dspy.ProviderRequest(args=(prompt,), preview=prompt)


def call_sdk(provider_request: dspy.ProviderRequest):
    return sdk.generate(*provider_request.args, **provider_request.kwargs)


def map_response(response, request: dspy.LMRequest) -> dspy.LMResponse:
    return dspy.LMResponse.from_text(response.text, model=request.model, provider_response=response)


lm = dspy.LM.from_sdk(
    model="my-sdk/model",
    support=dspy.LMSupport(text=True, messages=True),
    map_request=map_request,
    call=call_sdk,
    map_response=map_response,
)
```

Now DSPy modules can use it:

```python
dspy.configure(experimental_lm=True)
dspy.configure(lm=lm)

predict = dspy.Predict("question -> answer")
result = predict(question="Say hello")
print(result.answer)
```

## Inspect before calling

Diagnostics are first-class. You can inspect both DSPy’s normalized request and
the provider-shaped request without calling the SDK.

```python
request = lm.explain_request("hello", temperature=0.2)
provider_request = lm.explain_provider_request("hello", temperature=0.2)

print(request.messages)
print(provider_request)
```

Use `preview()` when a wrapper provides a nicer display representation:

```python
lm.preview("hello")
```

## `LMRequest`

A custom SDK mapper receives one `LMRequest`:

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

Common part types:

- `LMTextPart`
- `LMImagePart`
- `LMAudioPart`
- `LMFilePart`
- `LMThinkingPart`
- `LMToolCallPart`
- `LMToolResultPart`
- `LMCitationPart`
- `LMRefusalPart`

Common config fields:

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

Unknown generation kwargs are preserved in `request.config.extensions` so your
wrapper can pass provider-specific options through.

## `LMResponse`

Return one `LMResponse`:

```python
return dspy.LMResponse(
    model=request.model,
    outputs=[...],
    usage=dspy.LMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
    cost=0.00012,
    cache_hit=False,
)
```

For text-only responses:

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

## Explicit support

DSPy validates a request before calling your SDK. Validation uses `lm.support`.
Do not make DSPy guess.

A text-only wrapper can declare:

```python
dspy.LMSupport(text=True, messages=True)
```

A richer wrapper can declare:

```python
dspy.LMSupport(
    text=True,
    messages=True,
    images=dspy.ImageSupport(
        urls=True,
        base64=True,
        media_types=("image/png", "image/jpeg", "image/webp"),
        placement="any_message",
    ),
    tools=dspy.ToolSupport(schemas=True, calls=True, results=True),
    response_schema=True,
    reasoning=True,
    prompt_cache=True,
    streaming=True,
    citations=True,
)
```

If a request needs something unsupported, DSPy raises
`LMUnsupportedFeatureError` before your SDK is called.

```python
print(lm.support.report())
print(lm.features.report())
```

`support` is declared capability of the wrapper. `features` includes runtime
observations such as usage/cost appearing in responses.

## DSPy cache vs provider prompt cache

DSPy memoization and provider prompt caching are separate.

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

DSPy’s cache can skip the provider call entirely. Provider prompt caching still
sends a request, but may reuse prompt prefixes or KV state.

## Direct calls

The normalized path accepts several input shapes and immediately turns them into
one `LMRequest`.

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
lm(dspy.ToolResult(content='{"temperature": 22}', call_id="call_1", name="weather"))
lm(dspy.LMRequest(model="test/model", messages=[dspy.User("hello")]))
```

Do not mix an explicit `LMRequest` with direct-call inputs, and do not mix
`messages=` with positional prompt or media inputs. DSPy raises `ValueError` for
ambiguous calls.

## Add images

Images have two concerns:

1. mapping one `LMImagePart` into the SDK’s image object/block, and
2. placing those mapped images where the SDK expects them.

For OpenAI-style APIs, images live inside message content. For Agno-style APIs,
images might be a top-level `images=` argument.

```python
import base64
import dspy


def map_image(image: dspy.LMImagePart):
    if image.url:
        return SDKImage(url=image.url)
    return SDKImage(
        content=base64.b64decode(image.data),
        format=image.media_type.removeprefix("image/"),
    )


def place_images(request: dspy.LMRequest, images: list, provider_request: dspy.ProviderRequest):
    return dspy.ProviderRequest(
        args=(request.messages[-1].text or "",),
        kwargs={"images": images},
        preview={"message": request.messages[-1].text, "images": images},
    )


lm = lm.with_images(
    map_image=map_image,
    place_images=place_images,
    support=dspy.ImageSupport(urls=True, base64=True, placement="latest_user_message_only"),
)
```

If a multi-turn request contains images outside the latest user message, DSPy
will reject it for this wrapper before the SDK call.

## Add usage

```python
def extract_usage(response):
    return dspy.LMUsage(
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        total_tokens=response.usage.total_tokens,
    )


lm = lm.with_usage(extract_usage=extract_usage)
```

## Add streaming

Streaming uses DSPy event objects, not provider chunks.

```python
def stream(request: dspy.LMRequest):
    yield dspy.LMStreamStartEvent(model=request.model)
    for chunk in sdk.stream(request.messages[-1].text):
        yield dspy.LMStreamDeltaEvent(
            output_index=0,
            part_index=0,
            delta=dspy.LMTextDelta(text=chunk.text),
        )
    yield dspy.LMStreamOutputEndEvent(output_index=0, finish_reason="stop")
    yield dspy.LMStreamEndEvent()


lm = lm.with_streaming(stream=stream)
```

Callers iterate the stream, then read the final response:

```python
stream = lm.stream("hello")
for event in stream:
    print(event.type)

response = stream.result()
print(response.text)
```

Use stable `output_index` and `part_index` values so reasoning, text, and tool
call parts remain distinct during interleaved streams.

## Error normalization

Normalize provider errors that DSPy understands.

```python
def normalize_error(error: Exception, request: dspy.LMRequest) -> Exception:
    if error.__class__.__name__ == "ContextLengthError":
        return dspy.ContextWindowExceededError(model=request.model, provider="my-sdk")
    return error


lm = lm.with_errors(normalize=normalize_error)
```

When DSPy observes a normalized context-window error, it records that in
`lm.features.context_window_errors`.

## History, copy, save, and load

`LanguageModel.__call__()` records normalized request and response objects in
history unless history is disabled.

```python
entry = lm.history[-1]
entry.request
entry.response
entry["outputs"]
entry["usage"]
entry["prompt"]
entry["messages"]
```

Optimizers rely on `lm.copy(...)` to create variants:

```python
hot_lm = lm.copy(temperature=1.0, rollout_id=7)
```

`SDKLanguageModel` handles the common case. If you write a custom
`LanguageModel` subclass with non-copyable state, override `copy()`.

## Using provider SDKs directly

!!! info "Direct SDK wrappers with `LM.from_sdk`"

    === "OpenAI Responses"
        ```python linenums="1"
        import os
        from openai import OpenAI
        import dspy

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        lm = dspy.openai_responses_lm(
            "gpt-4o-mini",
            responses=client.responses.create,
            temperature=0.2,
        )
        ```

    === "OpenAI-compatible Chat"
        ```python linenums="1"
        from openai import OpenAI
        import dspy
        from dspy.clients.language_models.openai_format import (
            completion_to_lm_response,
            to_openai_chat_provider_request,
        )

        client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

        lm = dspy.LM.from_sdk(
            model="meta-llama/Llama-3.1-8B-Instruct",
            support=dspy.LMSupport(
                images=dspy.ImageSupport(urls=True, base64=True, placement="any_message"),
                tools=dspy.ToolSupport(schemas=True, calls=True, results=True),
                response_schema=True,
            ),
            map_request=to_openai_chat_provider_request,
            call=lambda provider_request: client.chat.completions.create(**provider_request.kwargs),
            map_response=completion_to_lm_response,
        )
        ```

    === "Anthropic text"
        ```python linenums="1"
        import os
        import anthropic
        import dspy

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        def map_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
            system = []
            messages = []
            for message in request.messages:
                text = message.text or ""
                if message.role in {"system", "developer"}:
                    system.append(text)
                elif message.role in {"user", "assistant"}:
                    messages.append({"role": message.role, "content": [{"type": "text", "text": text}]})
            kwargs = {
                "model": request.model,
                "system": "\n\n".join(system) or None,
                "messages": messages,
                "max_tokens": request.config.max_tokens or 1024,
                "temperature": request.config.temperature,
            }
            return dspy.ProviderRequest(kwargs={k: v for k, v in kwargs.items() if v is not None})

        def map_response(response, request: dspy.LMRequest) -> dspy.LMResponse:
            usage = response.usage
            return dspy.LMResponse.from_text(
                response.content[0].text,
                model=response.model,
                usage=dspy.LMUsage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
                provider_response=response,
            )

        lm = dspy.LM.from_sdk(
            model="claude-sonnet-4-5-20250929",
            support=dspy.LMSupport(text=True, messages=True),
            map_request=map_request,
            call=lambda provider_request: client.messages.create(**provider_request.kwargs),
            map_response=map_response,
        )
        ```

    === "Gemini text"
        ```python linenums="1"
        import os
        from google import genai
        import dspy

        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        def map_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
            prompt = "\n\n".join(f"{m.role}: {m.text or ''}" for m in request.messages)
            return dspy.ProviderRequest(
                kwargs={
                    "model": request.model,
                    "contents": prompt,
                    "config": {
                        "temperature": request.config.temperature,
                        "max_output_tokens": request.config.max_tokens,
                    },
                }
            )

        lm = dspy.LM.from_sdk(
            model="gemini-2.5-pro",
            support=dspy.LMSupport(text=True, messages=True),
            map_request=map_request,
            call=lambda provider_request: client.models.generate_content(**provider_request.kwargs),
            map_response=lambda response, request: dspy.LMResponse.from_text(
                response.text,
                model=request.model,
                provider_response=response,
            ),
        )
        ```

    === "Internal SDK"
        ```python linenums="1"
        import dspy

        def map_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
            prompt = "\n\n".join(message.text or "" for message in request.messages)
            return dspy.ProviderRequest(
                kwargs={
                    "model": request.model,
                    "prompt": prompt,
                    "temperature": request.config.temperature,
                    "max_tokens": request.config.max_tokens,
                }
            )

        lm = dspy.LM.from_sdk(
            model="acme/small",
            support=dspy.LMSupport(text=True, messages=True),
            map_request=map_request,
            call=lambda provider_request: acme_client.generate(**provider_request.kwargs),
            map_response=lambda response, request: dspy.LMResponse.from_text(
                response.text,
                model=request.model,
                usage=dspy.LMUsage(
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                ),
                cost=response.cost,
                provider_response=response,
            ),
        )
        ```

## Anthropic modality mapping examples

These examples show the function-first style for a provider with native content
blocks. They build one wrapper progressively by adding plain functions and
support declarations.

!!! info "A progressive Anthropic plugin"

    === "1. Text"
        ```python linenums="1"
        import os
        import anthropic
        import dspy

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        def get(value, name, default=None):
            return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)

        def anthropic_text(part: dspy.LMTextPart):
            return {"type": "text", "text": part.text}

        def anthropic_blocks(parts):
            return [anthropic_text(part) for part in parts if isinstance(part, dspy.LMTextPart)]

        def anthropic_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
            system = []
            messages = []
            for message in request.messages:
                if message.role in {"system", "developer"}:
                    system.extend(block["text"] for block in anthropic_blocks(message.parts))
                elif message.role in {"user", "assistant"}:
                    messages.append({"role": message.role, "content": anthropic_blocks(message.parts)})
            kwargs = {
                "model": request.model,
                "system": "\n\n".join(system) or None,
                "messages": messages,
                "max_tokens": request.config.max_tokens or 1024,
                "temperature": request.config.temperature,
            }
            return dspy.ProviderRequest(kwargs={key: value for key, value in kwargs.items() if value is not None})

        def anthropic_response(response, request: dspy.LMRequest) -> dspy.LMResponse:
            parts = [dspy.LMTextPart(text=get(block, "text", "")) for block in get(response, "content", []) if get(block, "type") == "text"]
            usage = get(response, "usage")
            return dspy.LMResponse(
                model=get(response, "model", request.model),
                outputs=[dspy.LMOutput(parts=parts, finish_reason=get(response, "stop_reason"))],
                usage=dspy.LMUsage(input_tokens=get(usage, "input_tokens"), output_tokens=get(usage, "output_tokens")),
                provider_response=response,
            )

        AnthropicTextLM = dspy.LM.from_sdk(
            model="claude-sonnet-4-5-20250929",
            support=dspy.LMSupport(text=True, messages=True),
            map_request=anthropic_request,
            call=lambda provider_request: client.messages.create(**provider_request.kwargs),
            map_response=anthropic_response,
        )

        lm = AnthropicTextLM
        print(lm("Say hello").text)
        ```

    === "2. Add images"
        ```python linenums="1"
        import base64
        import dspy

        def anthropic_image(part: dspy.LMImagePart):
            if part.data is None:
                raise ValueError("This example accepts base64 image data only.")
            return {
                "type": "image",
                "source": {"type": "base64", "media_type": part.media_type, "data": part.data},
            }

        def anthropic_blocks(parts):
            blocks = []
            for part in parts:
                if isinstance(part, dspy.LMTextPart):
                    blocks.append(anthropic_text(part))
                elif isinstance(part, dspy.LMImagePart):
                    blocks.append(anthropic_image(part))
            return blocks

        AnthropicVisionLM = AnthropicTextLM.with_images(
            map_image=anthropic_image,
            support=dspy.ImageSupport(base64=True, placement="any_message"),
        ).with_request_mapper(anthropic_request, support=dspy.LMSupport(
            text=True,
            messages=True,
            images=dspy.ImageSupport(base64=True, placement="any_message"),
        ))

        image = base64.b64encode(open("dog.png", "rb").read()).decode()
        lm = AnthropicVisionLM
        response = lm("Describe this image.", dspy.Image(f"data:image/png;base64,{image}"))
        ```

    === "3. Add native tool calls"
        ```python linenums="1"
        import dspy

        def anthropic_tool(tool: dspy.LMToolSpec):
            return {"name": tool.name, "description": tool.description or "", "input_schema": tool.parameters}

        def anthropic_tool_choice(choice: dspy.LMToolChoice):
            if choice.mode == "required":
                return {"type": "any"}
            if choice.mode == "none":
                return {"type": "none"}
            return {"type": "auto"}

        def anthropic_tool_call(part: dspy.LMToolCallPart):
            return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.args}

        def text_from_parts(parts):
            return "".join(part.text for part in parts if isinstance(part, dspy.LMTextPart))

        def anthropic_tool_result(part: dspy.LMToolResultPart):
            return {"type": "tool_result", "tool_use_id": part.call_id, "content": text_from_parts(part.content), "is_error": part.is_error}

        def anthropic_blocks(parts):
            blocks = []
            for part in parts:
                if isinstance(part, dspy.LMTextPart):
                    blocks.append(anthropic_text(part))
                elif isinstance(part, dspy.LMImagePart):
                    blocks.append(anthropic_image(part))
                elif isinstance(part, dspy.LMToolCallPart):
                    blocks.append(anthropic_tool_call(part))
                elif isinstance(part, dspy.LMToolResultPart):
                    blocks.append(anthropic_tool_result(part))
            return blocks

        def anthropic_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
            provider_request = AnthropicVisionLM.explain_provider_request(request=request)
            kwargs = dict(provider_request.kwargs)
            if request.tools:
                kwargs["tools"] = [anthropic_tool(tool) for tool in request.tools]
            if request.config.tool_choice is not None:
                kwargs["tool_choice"] = anthropic_tool_choice(request.config.tool_choice)
            return dspy.ProviderRequest(kwargs=kwargs)

        def anthropic_response(response, request: dspy.LMRequest) -> dspy.LMResponse:
            parts = []
            for block in get(response, "content", []):
                if get(block, "type") == "text":
                    parts.append(dspy.LMTextPart(text=get(block, "text", "")))
                elif get(block, "type") == "tool_use":
                    parts.append(dspy.LMToolCallPart(id=get(block, "id"), name=get(block, "name", ""), args=dict(get(block, "input", {}) or {})))
            usage = get(response, "usage")
            return dspy.LMResponse(
                model=get(response, "model", request.model),
                outputs=[dspy.LMOutput(parts=parts, finish_reason=get(response, "stop_reason"))],
                usage=dspy.LMUsage(input_tokens=get(usage, "input_tokens"), output_tokens=get(usage, "output_tokens")),
                provider_response=response,
            )

        AnthropicToolLM = dspy.LM.from_sdk(
            model="claude-sonnet-4-5-20250929",
            support=dspy.LMSupport(
                text=True,
                messages=True,
                images=dspy.ImageSupport(base64=True, placement="any_message"),
                tools=dspy.ToolSupport(schemas=True, calls=True, results=True),
            ),
            map_request=anthropic_request,
            call=lambda provider_request: client.messages.create(**provider_request.kwargs),
            map_response=anthropic_response,
        )
        ```

    === "4. Add built-in tools"
        ```python linenums="1"
        import dspy

        web_search = dspy.LMToolSpec(
            name="web_search",
            description="Search the web with Anthropic's hosted tool.",
            parameters={},
            provider_data={"type": "web_search_20250305", "name": "web_search", "max_uses": 3},
        )

        def anthropic_tool(tool: dspy.LMToolSpec):
            if tool.provider_data:
                return dict(tool.provider_data)
            return {"name": tool.name, "description": tool.description or "", "input_schema": tool.parameters}

        AnthropicBuiltinToolLM = AnthropicToolLM.with_request_mapper(
            anthropic_request,
            support=AnthropicToolLM.support,
        )
        ```

    === "5. Add native reasoning"
        ```python linenums="1"
        import dspy

        def anthropic_reasoning(reasoning: dspy.LMReasoningConfig):
            thinking = {"type": "enabled"}
            if reasoning.max_tokens is not None:
                thinking["budget_tokens"] = reasoning.max_tokens
            return thinking

        def anthropic_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
            provider_request = AnthropicBuiltinToolLM.explain_provider_request(request=request)
            kwargs = dict(provider_request.kwargs)
            if request.config.reasoning is not None:
                kwargs["thinking"] = anthropic_reasoning(request.config.reasoning)
            return dspy.ProviderRequest(kwargs=kwargs)

        def anthropic_response(response, request: dspy.LMRequest) -> dspy.LMResponse:
            parts = []
            for block in get(response, "content", []):
                if get(block, "type") == "thinking":
                    parts.append(dspy.LMThinkingPart(text=get(block, "thinking", "")))
                elif get(block, "type") == "text":
                    parts.append(dspy.LMTextPart(text=get(block, "text", "")))
                elif get(block, "type") == "tool_use":
                    parts.append(dspy.LMToolCallPart(id=get(block, "id"), name=get(block, "name", ""), args=dict(get(block, "input", {}) or {})))
            usage = get(response, "usage")
            return dspy.LMResponse(
                model=get(response, "model", request.model),
                outputs=[dspy.LMOutput(parts=parts, finish_reason=get(response, "stop_reason"))],
                usage=dspy.LMUsage(input_tokens=get(usage, "input_tokens"), output_tokens=get(usage, "output_tokens")),
                provider_response=response,
            )

        AnthropicReasoningLM = dspy.LM.from_sdk(
            model="claude-sonnet-4-5-20250929",
            support=AnthropicToolLM.support.with_updates(reasoning=True),
            map_request=anthropic_request,
            call=lambda provider_request: client.messages.create(**provider_request.kwargs),
            map_response=anthropic_response,
        )
        ```

    === "6. Add documents + citations"
        ```python linenums="1"
        import dspy

        def anthropic_file(part: dspy.LMFilePart):
            if part.data is None:
                raise ValueError("This example accepts base64 document data only.")
            return {
                "type": "document",
                "source": {"type": "base64", "media_type": part.media_type, "data": part.data},
                "title": part.filename,
                "citations": {"enabled": True},
            }

        def anthropic_blocks(parts):
            blocks = []
            for part in parts:
                if isinstance(part, dspy.LMFilePart):
                    blocks.append(anthropic_file(part))
                elif isinstance(part, dspy.LMTextPart):
                    blocks.append(anthropic_text(part))
                elif isinstance(part, dspy.LMImagePart):
                    blocks.append(anthropic_image(part))
                elif isinstance(part, dspy.LMToolCallPart):
                    blocks.append(anthropic_tool_call(part))
                elif isinstance(part, dspy.LMToolResultPart):
                    blocks.append(anthropic_tool_result(part))
            return blocks

        def citation_part(citation):
            return dspy.LMCitationPart(
                text=get(citation, "cited_text") or get(citation, "text"),
                title=get(citation, "document_title") or get(citation, "title"),
                url=get(citation, "url"),
                metadata={key: item for key in ("document_index", "start_char_index", "end_char_index") if (item := get(citation, key)) is not None},
            )

        def anthropic_request(request: dspy.LMRequest) -> dspy.ProviderRequest:
            system = []
            messages = []
            for message in request.messages:
                if message.role in {"system", "developer"}:
                    system.extend(block["text"] for block in anthropic_blocks(message.parts) if block["type"] == "text")
                elif message.role in {"user", "assistant"}:
                    messages.append({"role": message.role, "content": anthropic_blocks(message.parts)})
                elif message.role == "tool":
                    messages.append({"role": "user", "content": anthropic_blocks(message.parts)})
            kwargs = {
                "model": request.model,
                "system": "\n\n".join(system) or None,
                "messages": messages,
                "max_tokens": request.config.max_tokens or 1024,
                "temperature": request.config.temperature,
            }
            if request.tools:
                kwargs["tools"] = [anthropic_tool(tool) for tool in request.tools]
            if request.config.tool_choice is not None:
                kwargs["tool_choice"] = anthropic_tool_choice(request.config.tool_choice)
            if request.config.reasoning is not None:
                kwargs["thinking"] = anthropic_reasoning(request.config.reasoning)
            return dspy.ProviderRequest(kwargs={key: value for key, value in kwargs.items() if value is not None})

        def anthropic_response(response, request: dspy.LMRequest) -> dspy.LMResponse:
            base = AnthropicReasoningLM._map_response_fn(response, request)
            citations = []
            for block in get(response, "content", []):
                citations.extend(citation_part(citation) for citation in get(block, "citations", []) or [])
            base.outputs[0].parts.extend(citations)
            return base

        AnthropicCitationLM = dspy.LM.from_sdk(
            model="claude-sonnet-4-5-20250929",
            support=AnthropicReasoningLM.support.with_updates(files=True, citations=True),
            map_request=anthropic_request,
            call=lambda provider_request: client.messages.create(**provider_request.kwargs),
            map_response=anthropic_response,
        )
        ```

    === "7. Add streaming"
        ```python linenums="1"
        import dspy

        def anthropic_stream(request: dspy.LMRequest):
            yield dspy.LMStreamStartEvent(model=request.model)
            with client.messages.stream(**AnthropicCitationLM.explain_provider_request(request=request).kwargs) as stream:
                for event in stream:
                    if get(event, "type") != "content_block_delta":
                        continue
                    delta = get(event, "delta")
                    if get(delta, "type") == "thinking_delta":
                        yield dspy.LMStreamDeltaEvent(output_index=0, part_index=0, delta=dspy.LMThinkingDelta(text=get(delta, "thinking", "")))
                    elif get(delta, "type") == "text_delta":
                        yield dspy.LMStreamDeltaEvent(output_index=0, part_index=1, delta=dspy.LMTextDelta(text=get(delta, "text", "")))
            yield dspy.LMStreamOutputEndEvent(output_index=0)
            yield dspy.LMStreamEndEvent()

        AnthropicStreamingLM = AnthropicCitationLM.with_streaming(stream=anthropic_stream)

        lm = AnthropicStreamingLM
        stream = lm.stream("Say hello")
        for event in stream:
            print(event.type)
        print(stream.result().text)
        ```

## Registering a backend for `dspy.LM`

Register a backend factory if you want `dspy.LM("acme/...")` to route to your
custom normalized LM.

```python
import dspy


@dspy.register_lm_backend
def route_acme(model: str, *args, **kwargs):
    if model.startswith("acme/"):
        return make_acme_lm(model, *args, **kwargs)
    return None


dspy.configure(experimental_lm=True)
dspy.configure(lm=dspy.LM("acme/small"))
```

Factories are tried in reverse registration order. Return `None` when your
factory does not own the model.

## Migration checklist

- [ ] Enable `dspy.configure(experimental_lm=True)` before constructing normalized `dspy.LM(...)` objects.
- [ ] Prefer `dspy.LM.from_sdk(...)` over subclassing.
- [ ] Write `map_request(request) -> ProviderRequest`.
- [ ] Write `call(provider_request)`.
- [ ] Write `map_response(response, request) -> LMResponse`.
- [ ] Declare `support` explicitly.
- [ ] Use `explain_request(...)` and `explain_provider_request(...)` while debugging.
- [ ] Put generated text, reasoning, tool calls, citations, files, and media in `LMOutput.parts`.
- [ ] Put usage on `response.usage` and cost on `response.cost` when available.
- [ ] Normalize recognizable provider errors, especially context-window errors.
- [ ] Add streaming with normalized `LMStreamEvent` objects if your provider streams.

## Legacy `BaseLM`

`dspy.BaseLM` remains the legacy prompt/messages contract. Existing custom LMs
can continue to use it during the migration window, but new custom LMs should
use the normalized `LM.from_sdk(...)` path.

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
