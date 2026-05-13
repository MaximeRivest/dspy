"""Adapt lm15 requests, responses, and streams to DSPy's LM types.

The main object exported by this module is `lm15_lm()`. It returns an
`SDKLanguageModel`, not a custom subclass. The flow is deliberately ordinary:

```text
LMRequest -> lm15.Request -> client.complete(...) -> lm15.Response -> LMResponse
```

Streaming follows the same shape, but the provider returns events:

```text
LMRequest -> lm15.Request -> client.stream(...) -> lm15 events -> LMStreamEvent
```

Most functions below are small translators for one edge of that flow. Read the
file in this order when learning it:

1. `lm15_lm()` wires the plugin together.
2. `to_lm15_request()` maps DSPy requests into lm15 requests.
3. `from_lm15_response()` maps lm15 responses back into DSPy responses.
4. `stream_event_to_dspy()` maps lm15 stream events into DSPy stream events.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from functools import partial
from typing import Any, Literal

import lm15.providers as providers
import lm15.types as lm15

from dspy.clients.language_models.provider import ProviderRequest
from dspy.clients.language_models.sdk import SDKLanguageModel
from dspy.clients.language_models.support import ImageSupport, LMSupport, ToolSupport
from dspy.clients.language_models.types import (
    LMAudioDelta,
    LMAudioPart,
    LMCitationDelta,
    LMCitationPart,
    LMConfig,
    LMFilePart,
    LMImageDelta,
    LMImagePart,
    LMOutput,
    LMRequest,
    LMResponse,
    LMStreamDeltaEvent,
    LMStreamEndEvent,
    LMStreamErrorEvent,
    LMStreamEvent,
    LMStreamOutputEndEvent,
    LMStreamStartEvent,
    LMTextDelta,
    LMTextPart,
    LMThinkingDelta,
    LMThinkingPart,
    LMToolCallDelta,
    LMToolCallPart,
    LMToolChoice,
    LMToolResultPart,
    LMToolSpec,
    LMUsage,
)

ProviderName = Literal["openai", "anthropic", "gemini"]
DOCUMENT_MEDIA_TYPES = frozenset({"application/pdf", "text/plain", "text/markdown", "text/html"})

__all__ = [
    "lm15_lm",
    "lm15_support",
    "make_client",
    "to_provider_request",
    "to_lm15_request",
    "to_lm15_message",
    "to_lm15_part",
    "to_lm15_config",
    "from_lm15_response",
    "from_lm15_part",
    "stream_event_to_dspy",
    "stream_delta_to_dspy",
    "provider_from_model",
    "strip_provider_prefix",
]


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def lm15_lm(
    model: str,
    *,
    provider: ProviderName | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    cache: bool = True,
    callbacks: list[Any] | None = None,
    **kwargs: Any,
) -> SDKLanguageModel:
    """Create a DSPy LM backed by lm15.

    Args:
        model: DSPy-facing model name. Provider prefixes such as
            `"openai/"`, `"anthropic/"`, and `"gemini/"` are stripped before
            the request is sent to lm15.
        provider: Provider to use. If omitted, inferred from `model`.
        client: Optional prebuilt lm15 provider client. Supplying this is useful
            for tests, dependency injection, or custom authentication.
        api_key: Provider API key. If omitted, read from the provider's standard
            environment variable.
        base_url: Optional provider base URL.
        cache: Whether DSPy request memoization is enabled by default.
        callbacks: Optional DSPy callbacks.
        **kwargs: Default LM config values, such as `temperature` or
            `max_tokens`.

    Returns:
        An `SDKLanguageModel` composed from plain functions.
    """
    provider = provider or provider_from_model(model)
    get_client = lazy_client(provider=provider, client=client, api_key=api_key, base_url=base_url)

    # SDKLanguageModel expects one-argument callables. `partial()` keeps the
    # public translators named and testable while binding this LM's provider and
    # lazy client once, here at construction time.
    map_request = partial(to_provider_request, provider=provider)
    call = partial(complete_provider_request, get_client=get_client)
    stream = partial(stream_lm15_request, provider=provider, get_client=get_client)

    lm = SDKLanguageModel(
        model=model,
        support=lm15_support(),
        map_request=map_request,
        call=call,
        map_response=from_lm15_response,
        metadata={"provider": "lm15", "protocol": "lm15", "lm15_provider": provider},
        cache=cache,
        callbacks=callbacks,
        **kwargs,
    )

    return lm.with_streaming(stream=stream)


def lm15_support() -> LMSupport:
    """Declare the DSPy request and response shapes handled by lm15.

    The declaration is intentionally optimistic: lm15 has provider-specific
    clients underneath, so this says what the adapter can map, not that every
    model deployment supports every feature.
    """
    return LMSupport(
        images=ImageSupport(urls=True, base64=True, file_ids=True, paths=True, placement="any_message"),
        audio=True,
        files=True,
        tools=ToolSupport(schemas=True, calls=True, results=True),
        response_schema=True,
        reasoning=True,
        prompt_cache=True,
        provider_extensions=True,
        streaming=True,
        citations=True,
        output_images=True,
        output_audio=True,
        output_files=True,
        refusal=True,
    )


# ---------------------------------------------------------------------------
# SDK boundary: client construction and the one real provider call
# ---------------------------------------------------------------------------


def lazy_client(
    *,
    provider: ProviderName,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable[[], Any]:
    """Return a function that creates and then reuses one lm15 client.

    A prebuilt `client` is returned as-is. Otherwise the client is created on
    first use, so constructing `dspy.LM(...)` does not require credentials until
    the first provider call.
    """
    holder = {"client": client}

    def get_client() -> Any:
        if holder["client"] is None:
            holder["client"] = make_client(provider=provider, api_key=api_key, base_url=base_url)
        return holder["client"]

    return get_client


def make_client(*, provider: ProviderName, api_key: str | None = None, base_url: str | None = None) -> Any:
    """Create the concrete lm15 client for one provider."""
    client_kwargs: dict[str, Any] = {"api_key": api_key or os.environ.get(api_key_env_var(provider), "")}
    if base_url is not None:
        client_kwargs["base_url"] = base_url

    if provider == "anthropic":
        return providers.AnthropicLM(**client_kwargs)
    if provider == "gemini":
        return providers.GeminiLM(**client_kwargs)
    return providers.OpenAILM(**client_kwargs)


def complete_provider_request(provider_request: ProviderRequest, *, get_client: Callable[[], Any]) -> Any:
    """Send one mapped lm15 request to `client.complete()`.

    `to_provider_request()` stores the native `lm15.Request` in
    `provider_request.data`; this function is the only non-streaming place that
    calls the external SDK.
    """
    return get_client().complete(provider_request.data)


# ---------------------------------------------------------------------------
# DSPy request -> ProviderRequest -> lm15 request
# ---------------------------------------------------------------------------


def to_provider_request(request: LMRequest, *, provider: ProviderName | str) -> ProviderRequest:
    """Wrap an lm15 request in DSPy's provider-call container.

    `data` stores the native `lm15.Request` used by `client.complete()` and
    `client.stream()`. `preview` stores the same object so `lm.preview(...)`
    can show the exact mapped request without calling the provider.
    """
    native_request = to_lm15_request(request, provider=provider)
    return ProviderRequest(
        args=(native_request,),
        data=native_request,
        preview=native_request,
        metadata={"provider": provider, "protocol": "lm15"},
    )


# ---------------------------------------------------------------------------
# DSPy request -> lm15 request
# ---------------------------------------------------------------------------


def to_lm15_request(request: LMRequest, *, provider: ProviderName | str) -> Any:
    """Convert a normalized DSPy request into an `lm15.Request`.

    System messages become lm15's separate `system` field. All other messages
    stay in order in the lm15 message list. Provider prefixes such as
    `openai/` are removed because lm15 already knows which provider client is
    being used.
    """
    system_parts = []
    messages = []

    for message in request.messages:
        if message.role == "system":
            system_parts.extend(to_lm15_part(part) for part in message.parts)
        else:
            messages.append(to_lm15_message(message))

    return lm15.Request(
        model=strip_provider_prefix(request.model, provider),
        system=to_lm15_system(system_parts),
        messages=tuple(messages),
        tools=tuple(to_lm15_tool(tool) for tool in request.tools),
        config=to_lm15_config(request.config),
    )


def to_lm15_message(message: Any) -> Any:
    """Convert one DSPy message into the matching lm15 message constructor."""
    if message.role == "tool":
        return lm15.Message.tool(tuple(to_lm15_tool_result(part) for part in tool_result_parts(message)))

    parts = tuple(to_lm15_part(part) for part in message.parts)
    if message.role == "developer":
        return lm15.Message.developer(parts)
    if message.role == "assistant":
        return lm15.Message.assistant(parts)
    return lm15.Message.user(parts)


def tool_result_parts(message: Any) -> Iterator[LMToolResultPart]:
    """Yield tool-result parts from a DSPy tool message."""
    for part in message.parts:
        if isinstance(part, LMToolResultPart):
            yield part


def to_lm15_part(part: Any) -> Any:
    """Convert one DSPy content part into the matching lm15 part."""
    if isinstance(part, LMTextPart):
        return lm15.text(part.text)
    if isinstance(part, LMImagePart):
        return to_lm15_image(part)
    if isinstance(part, LMAudioPart):
        return to_lm15_audio(part)
    if isinstance(part, LMFilePart):
        return to_lm15_file(part)
    if isinstance(part, LMThinkingPart):
        return lm15.ThinkingPart(text=part.text, redacted=part.redacted)
    if isinstance(part, LMCitationPart):
        return lm15.CitationPart(text=part.text, title=part.title, url=part.url)
    if isinstance(part, LMToolCallPart):
        return lm15.ToolCallPart(id=part.id or "", name=part.name, input=part.args)
    if isinstance(part, LMToolResultPart):
        return to_lm15_tool_result(part)
    raise TypeError(f"Cannot convert {type(part).__name__} to an lm15 part.")


def to_lm15_image(image: LMImagePart) -> Any:
    return lm15.image(
        data=image.data,
        url=image.url,
        path=image.path,
        file_id=image.file_id,
        media_type=image.media_type,
        detail=image.detail,
    )


def to_lm15_audio(audio: LMAudioPart) -> Any:
    return lm15.audio(
        data=audio.data,
        url=audio.url,
        path=audio.path,
        file_id=audio.file_id,
        media_type=audio.media_type,
    )


def to_lm15_file(file: LMFilePart) -> Any:
    factory = lm15.document if is_document_media_type(file.media_type) else lm15.binary
    return factory(
        data=file.data,
        url=file.url,
        path=file.path,
        file_id=file.file_id,
        media_type=file.media_type,
    )


def to_lm15_system(system_parts: list[Any]) -> Any | None:
    """Represent system content as a string when possible, otherwise as parts."""
    if not system_parts:
        return None
    if all(is_lm15_text_part(part) for part in system_parts):
        return "".join(part.text for part in system_parts)
    return tuple(system_parts)


def to_lm15_tool(tool: LMToolSpec) -> Any:
    return lm15.FunctionTool(name=tool.name, description=tool.description, parameters=tool.parameters)


def to_lm15_tool_result(result: LMToolResultPart) -> Any:
    return lm15.tool_result(
        result.call_id or "",
        tuple(to_lm15_part(item) for item in result.content),
        name=result.name,
        is_error=result.is_error,
    )


def to_lm15_tool_choice(choice: LMToolChoice) -> Any:
    return lm15.ToolChoice(mode=choice.mode, allowed=tuple(choice.allowed), parallel=choice.parallel)


def to_lm15_reasoning(reasoning: Any) -> Any:
    data = {}
    if reasoning.effort is not None:
        data["effort"] = reasoning.effort
    if reasoning.max_tokens is not None:
        data["thinking_budget"] = reasoning.max_tokens
    if reasoning.summary is not None:
        data["summary"] = reasoning.summary
    return lm15.Reasoning(**data)


def to_lm15_prompt_cache(cache: Any) -> Any:
    mode = "auto" if cache.enabled is not False else "off"
    return lm15.CacheConfig(mode=mode, key=cache.key)


def to_lm15_config(config: LMConfig) -> Any:
    """Convert common DSPy generation settings into `lm15.Config`."""
    return lm15.Config(
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        stop=tuple(config.stop),
        response_format=config.response_format,
        tool_choice=to_lm15_tool_choice(config.tool_choice) if config.tool_choice is not None else None,
        reasoning=to_lm15_reasoning(config.reasoning) if config.reasoning is not None else None,
        cache=to_lm15_prompt_cache(config.prompt_cache) if config.prompt_cache is not None else None,
        extensions=dict(config.extensions) if config.extensions else None,
    )


# ---------------------------------------------------------------------------
# lm15 response -> DSPy response
# ---------------------------------------------------------------------------


def from_lm15_response(response: Any, request: LMRequest | None = None) -> LMResponse:
    """Map an lm15 response to DSPy's `LMResponse`."""
    output = LMOutput(
        parts=[from_lm15_part(part) for part in response.message.parts],
        finish_reason=response.finish_reason,
        truncated=response.finish_reason == "length",
        provider_output=response,
    )
    return LMResponse(
        model=response.model,
        outputs=[output],
        usage=usage_to_dspy(response.usage),
        response_id=response.id,
        provider_data=dict(response.provider_data or {}),
        provider_response=response,
    )


def from_lm15_part(part: Any) -> Any:
    """Map one lm15 response part to a DSPy part."""
    part_type = type(part).__name__
    if part_type == "TextPart":
        return LMTextPart(text=part.text)
    if part_type == "ThinkingPart":
        return LMThinkingPart(text=part.text, redacted=getattr(part, "redacted", False))
    if part_type == "ImagePart":
        return image_from_lm15(part)
    if part_type == "AudioPart":
        return audio_from_lm15(part)
    if part_type in {"DocumentPart", "BinaryPart"}:
        return file_from_lm15(part)
    if part_type == "ToolCallPart":
        return LMToolCallPart(id=part.id, name=part.name, args=dict(part.input))
    if part_type == "ToolResultPart":
        return LMToolResultPart(
            call_id=part.id,
            name=part.name,
            content=[from_lm15_part(item) for item in part.content],
            is_error=part.is_error,
        )
    if part_type == "CitationPart":
        return LMCitationPart(text=part.text, title=part.title, url=part.url)
    if part_type == "RefusalPart":
        from dspy.clients.language_models.types import LMRefusalPart

        return LMRefusalPart(text=part.text)
    return LMTextPart(text=str(part))


def image_from_lm15(image: Any) -> LMImagePart:
    return LMImagePart(
        data=image.data,
        url=image.url,
        file_id=image.file_id,
        path=str(image.path) if image.path is not None else None,
        media_type=image.media_type,
        detail=getattr(image, "detail", None),
    )


def audio_from_lm15(audio: Any) -> LMAudioPart:
    return LMAudioPart(
        data=audio.data,
        url=audio.url,
        file_id=audio.file_id,
        path=str(audio.path) if audio.path is not None else None,
        media_type=audio.media_type,
    )


def file_from_lm15(file: Any) -> LMFilePart:
    return LMFilePart(
        data=file.data,
        url=file.url,
        file_id=file.file_id,
        path=str(file.path) if file.path is not None else None,
        media_type=file.media_type,
    )


# ---------------------------------------------------------------------------
# lm15 stream -> DSPy stream
# ---------------------------------------------------------------------------


def stream_lm15_request(
    request: LMRequest,
    *,
    provider: ProviderName | str,
    get_client: Callable[[], Any],
) -> Iterator[LMStreamEvent]:
    """Stream one DSPy request through lm15 and yield DSPy stream events.

    This is the streaming sibling of the normal `map_request -> call ->
    map_response` path. It maps the request once, calls `client.stream()`, then
    delegates each provider event to `stream_event_to_dspy()`.
    """
    native_request = to_lm15_request(request, provider=provider)
    for event in get_client().stream(native_request):
        yield from stream_event_to_dspy(event)


def stream_event_to_dspy(event: Any) -> Iterator[LMStreamEvent]:
    """Map one lm15 stream event to zero or more DSPy stream events."""
    event_type = getattr(event, "type", None)
    if event_type == "start":
        yield LMStreamStartEvent(model=getattr(event, "model", None))
        return
    if event_type == "delta":
        delta = stream_delta_to_dspy(event.delta)
        if delta is not None:
            yield LMStreamDeltaEvent(
                output_index=0,
                part_index=getattr(event.delta, "part_index", 0),
                delta=delta,
            )
        return
    if event_type == "end":
        yield LMStreamOutputEndEvent(output_index=0, finish_reason=event.finish_reason)
        yield LMStreamEndEvent(usage=usage_to_dspy(event.usage) if event.usage is not None else None)
        return
    if event_type == "error":
        yield LMStreamErrorEvent(error=RuntimeError(str(event.error)))


def stream_delta_to_dspy(delta: Any) -> Any | None:
    """Map one lm15 stream delta to a DSPy delta."""
    delta_type = type(delta).__name__
    if delta_type == "TextDelta":
        return LMTextDelta(text=delta.text)
    if delta_type == "ThinkingDelta":
        return LMThinkingDelta(text=delta.text)
    if delta_type == "ToolCallDelta":
        return LMToolCallDelta(id=delta.id, name=delta.name, args_delta=delta.input)
    if delta_type == "ImageDelta":
        return LMImageDelta(
            image=LMImagePart(
                data=delta.data,
                url=delta.url,
                file_id=delta.file_id,
                media_type=delta.media_type or "image/png",
            )
        )
    if delta_type == "AudioDelta":
        return LMAudioDelta(
            audio=LMAudioPart(
                data=delta.data,
                url=delta.url,
                file_id=delta.file_id,
                media_type=delta.media_type or "audio/wav",
            )
        )
    if delta_type == "CitationDelta":
        return LMCitationDelta(citation=LMCitationPart(text=delta.text, title=delta.title, url=delta.url))
    return None


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def provider_from_model(model: str) -> ProviderName:
    """Infer the lm15 provider from a DSPy model string."""
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    if provider in {"anthropic", "gemini", "openai"}:
        return provider  # type: ignore[return-value]
    return "openai"


def strip_provider_prefix(model: str, provider: str) -> str:
    """Remove this provider's prefix from a DSPy-facing model name."""
    return model.removeprefix(f"{provider}/")


def api_key_env_var(provider: str) -> str:
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "OPENAI_API_KEY")


def usage_to_dspy(usage: Any) -> LMUsage | None:
    """Convert lm15 token usage into DSPy's `LMUsage` model."""
    if usage is None:
        return None
    return LMUsage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        total_tokens=usage.total_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        input_audio_tokens=usage.input_audio_tokens,
        output_audio_tokens=usage.output_audio_tokens,
    )


def is_document_media_type(media_type: str) -> bool:
    return media_type in DOCUMENT_MEDIA_TYPES


def is_lm15_text_part(part: Any) -> bool:
    return type(part).__name__ == "TextPart"
