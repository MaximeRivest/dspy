"""lm15-backed implementation of the normalized DSPy LM contract."""

from __future__ import annotations

import os
from typing import Any, Iterator, Literal

import lm15.providers as providers
import lm15.types as lm15

from dspy.clients.language_models.base import LanguageModel, LMCapabilities
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


class LM15LM(LanguageModel):
    """Call an lm15 provider with normalized DSPy requests.

    `LM15LM` maps DSPy's `LMRequest` and `LMResponse` types to lm15's
    provider-independent `Request` and `Response` types. lm15 then maps those
    requests to OpenAI, Anthropic, Gemini, or a configured provider profile.

    Args:
        model: Model name. DSPy-style names like `"openai/gpt-4.1-mini"` are
            accepted; the provider prefix is used for routing and stripped
            before calling lm15.
        provider: Provider to use. If omitted, DSPy infers it from the model
            prefix and falls back to `"openai"`.
        api_key: Provider API key. If omitted, the matching environment variable
            is used.
        base_url: Optional provider base URL.
        cache: Whether this LM should use DSPy's cache by default. lm15 itself
            receives the normalized cache preference in request config.
        **kwargs: Default request configuration, such as `temperature` or
            `max_tokens`.

    Examples:
        ```python
        import dspy

        lm = dspy.LM15LM("openai/gpt-4.1-mini", temperature=0.2)
        response = lm("Write one sentence about DSPy.")
        print(response.text)
        ```

    See Also:
        [`dspy.LanguageModel`][dspy.LanguageModel]
        [`dspy.LMRequest`][dspy.LMRequest]
        [`dspy.LMResponse`][dspy.LMResponse]
    """

    def __init__(
        self,
        model: str,
        *,
        provider: Literal["openai", "anthropic", "gemini"] | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self.provider = provider or _provider_from_model(model)
        self.lm15_model = _strip_provider_prefix(model, self.provider)
        self.api_key = api_key
        self.base_url = base_url
        self._client = None

    def get_capabilities(self) -> LMCapabilities:
        client = self._get_client()
        capabilities = getattr(client, "capabilities", None)
        features = set(getattr(capabilities, "features", ()) or ())
        input_modalities = set(getattr(capabilities, "input_modalities", ()) or ())
        output_modalities = set(getattr(capabilities, "output_modalities", ()) or ())
        return LMCapabilities(
            function_calling="tools" in features,
            reasoning="reasoning" in features,
            response_schema="json_output" in features,
            streaming="streaming" in features,
            input_image="image" in input_modalities,
            input_audio="audio" in input_modalities,
            input_file=bool({"document", "binary"} & input_modalities),
            output_image="image" in output_modalities,
            output_audio="audio" in output_modalities,
            tool_results="tools" in features,
            extensions={"provider": self.provider},
        )

    def forward(self, request: LMRequest) -> LMResponse:
        lm15_request = self._map_request(request)
        lm15_response = self._get_client().complete(lm15_request)
        return self._map_response(lm15_response)

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        lm15_request = self._map_request(request)
        for event in self._get_client().stream(lm15_request):
            yield from self._map_stream_event(event)

    def map_request_text(self, value: Any) -> Any:
        """Map request text to an lm15 text part."""
        return lm15.text(str(value))

    def map_request_input_image(self, value: LMImagePart) -> Any:
        return lm15.image(
            data=value.data,
            url=value.url,
            path=value.path,
            file_id=value.file_id,
            media_type=value.media_type,
            detail=value.detail,
        )

    def map_request_input_audio(self, value: LMAudioPart) -> Any:
        return lm15.audio(
            data=value.data,
            url=value.url,
            path=value.path,
            file_id=value.file_id,
            media_type=value.media_type,
        )

    def map_request_input_file(self, value: LMFilePart) -> Any:
        factory = lm15.document if _is_document_media_type(value.media_type) else lm15.binary
        return factory(
            data=value.data,
            url=value.url,
            path=value.path,
            file_id=value.file_id,
            media_type=value.media_type,
        )

    def map_request_tools(self, value: LMToolSpec) -> Any:
        return lm15.FunctionTool(name=value.name, description=value.description, parameters=value.parameters)

    def map_request_assistant_tool_calls(self, value: LMToolCallPart) -> Any:
        return lm15.ToolCallPart(id=value.id or "", name=value.name, input=value.args)

    def map_request_tool_results(self, value: LMToolResultPart) -> Any:
        content = tuple(self._map_request_part(item) for item in value.content)
        return lm15.tool_result(
            value.call_id or "",
            content,
            name=value.name,
            is_error=value.is_error,
        )

    def map_request_response_schema(self, value: Any) -> Any:
        return value

    def map_request_reasoning_config(self, value: Any) -> Any:
        kwargs = {}
        if value.effort is not None:
            kwargs["effort"] = value.effort
        if value.max_tokens is not None:
            kwargs["thinking_budget"] = value.max_tokens
        if value.summary is not None:
            kwargs["summary"] = value.summary
        return lm15.Reasoning(**kwargs)

    def map_request_tool_choice(self, value: LMToolChoice) -> Any:
        return lm15.ToolChoice(mode=value.mode, allowed=tuple(value.allowed), parallel=value.parallel)

    def map_request_prompt_cache(self, value: Any) -> Any:
        mode = "auto" if value.enabled is not False else "off"
        return lm15.CacheConfig(mode=mode, key=value.key)

    def map_request_provider_extensions(self, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    def map_response_text(self, value: Any) -> LMTextPart:
        return LMTextPart(text=value.text)

    def map_response_reasoning(self, value: Any) -> LMThinkingPart:
        return LMThinkingPart(text=value.text, redacted=getattr(value, "redacted", False))

    def map_response_tool_calls(self, value: Any) -> LMToolCallPart:
        return LMToolCallPart(id=value.id, name=value.name, args=dict(value.input))

    def map_response_citations(self, value: Any) -> LMCitationPart:
        return LMCitationPart(text=value.text, title=value.title, url=value.url)

    def map_response_output_image(self, value: Any) -> LMImagePart:
        return LMImagePart(
            data=value.data,
            url=value.url,
            file_id=value.file_id,
            path=str(value.path) if value.path is not None else None,
            media_type=value.media_type,
            detail=getattr(value, "detail", None),
        )

    def map_response_output_audio(self, value: Any) -> LMAudioPart:
        return LMAudioPart(
            data=value.data,
            url=value.url,
            file_id=value.file_id,
            path=str(value.path) if value.path is not None else None,
            media_type=value.media_type,
        )

    def map_response_output_file(self, value: Any) -> LMFilePart:
        return LMFilePart(
            data=value.data,
            url=value.url,
            file_id=value.file_id,
            path=str(value.path) if value.path is not None else None,
            media_type=value.media_type,
        )

    def map_response_refusal(self, value: Any) -> Any:
        from dspy.clients.language_models.types import LMRefusalPart

        return LMRefusalPart(text=value.text)

    def dump_state(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "provider": self.provider,
            "base_url": self.base_url,
            "cache": self.cache,
            **{k: v for k, v in self.kwargs.items() if not k.startswith("api_") and k != "api_key"},
        }

    def _map_request(self, request: LMRequest) -> Any:
        system_parts = []
        messages = []

        for message in request.messages:
            if message.role == "system":
                system_parts.extend(self._map_request_part(part) for part in message.parts)
            else:
                messages.append(self._map_request_message(message))

        system = self._map_system(system_parts)
        return lm15.Request(
            model=_strip_provider_prefix(request.model, self.provider),
            system=system,
            messages=tuple(messages),
            tools=tuple(self.map_request_tools(tool) for tool in request.tools),
            config=self._map_config(request.config),
        )

    def _map_request_message(self, message: Any) -> Any:
        if message.role == "tool":
            tool_results = [part for part in message.parts if isinstance(part, LMToolResultPart)]
            return lm15.Message.tool(tuple(self.map_request_tool_results(part) for part in tool_results))

        parts = tuple(self._map_request_part(part) for part in message.parts)
        if message.role == "developer":
            return lm15.Message.developer(parts)
        if message.role == "assistant":
            return lm15.Message.assistant(parts)
        return lm15.Message.user(parts)

    def _map_request_part(self, part: Any) -> Any:
        if isinstance(part, LMTextPart):
            return self.map_request_text(part.text)
        if isinstance(part, LMImagePart):
            return self.map_request_input_image(part)
        if isinstance(part, LMAudioPart):
            return self.map_request_input_audio(part)
        if isinstance(part, LMFilePart):
            return self.map_request_input_file(part)
        if isinstance(part, LMThinkingPart):
            return lm15.ThinkingPart(text=part.text, redacted=part.redacted)
        if isinstance(part, LMCitationPart):
            return lm15.CitationPart(text=part.text, title=part.title, url=part.url)
        if isinstance(part, LMToolCallPart):
            return self.map_request_assistant_tool_calls(part)
        if isinstance(part, LMToolResultPart):
            return self.map_request_tool_results(part)
        raise TypeError(f"Cannot convert {type(part)!r} to an lm15 part.")

    def _map_system(self, system_parts: list[Any]) -> Any | None:
        if not system_parts:
            return None
        if all(_is_lm15_text_part(part) for part in system_parts):
            return "".join(part.text for part in system_parts)
        return tuple(system_parts)

    def _map_config(self, config: LMConfig) -> Any:
        return lm15.Config(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            stop=tuple(config.stop),
            response_format=(
                self.map_request_response_schema(config.response_format) if config.response_format is not None else None
            ),
            tool_choice=self.map_request_tool_choice(config.tool_choice) if config.tool_choice is not None else None,
            reasoning=(self.map_request_reasoning_config(config.reasoning) if config.reasoning is not None else None),
            cache=self.map_request_prompt_cache(config.prompt_cache) if config.prompt_cache is not None else None,
            extensions=(self.map_request_provider_extensions(config.extensions) if dict(config.extensions) else None),
        )

    def _map_response(self, response: Any) -> LMResponse:
        output = LMOutput(
            parts=[self._map_response_part(part) for part in response.message.parts],
            finish_reason=response.finish_reason,
            truncated=response.finish_reason == "length",
            provider_output=response,
        )
        return LMResponse(
            model=response.model,
            outputs=[output],
            usage=_usage_to_dspy(response.usage),
            response_id=response.id,
            provider_data=dict(response.provider_data or {}),
            provider_response=response,
        )

    def _map_response_part(self, part: Any) -> Any:
        name = type(part).__name__
        if name == "TextPart":
            return self.map_response_text(part)
        if name == "ThinkingPart":
            return self.map_response_reasoning(part)
        if name == "ImagePart":
            return self.map_response_output_image(part)
        if name == "AudioPart":
            return self.map_response_output_audio(part)
        if name in {"DocumentPart", "BinaryPart"}:
            return self.map_response_output_file(part)
        if name == "ToolCallPart":
            return self.map_response_tool_calls(part)
        if name == "ToolResultPart":
            return LMToolResultPart(
                call_id=part.id,
                name=part.name,
                content=[self._map_response_part(item) for item in part.content],
                is_error=part.is_error,
            )
        if name == "CitationPart":
            return self.map_response_citations(part)
        if name == "RefusalPart":
            return self.map_response_refusal(part)
        return LMTextPart(text=str(part))

    def _map_stream_event(self, event: Any) -> Iterator[LMStreamEvent]:
        event_type = getattr(event, "type", None)
        if event_type == "start":
            yield LMStreamStartEvent(model=getattr(event, "model", None))
        elif event_type == "delta":
            delta = self._map_stream_delta(event.delta)
            if delta is not None:
                yield LMStreamDeltaEvent(
                    output_index=0,
                    part_index=getattr(event.delta, "part_index", 0),
                    delta=delta,
                )
        elif event_type == "end":
            yield LMStreamOutputEndEvent(output_index=0, finish_reason=event.finish_reason)
            yield LMStreamEndEvent(usage=_usage_to_dspy(event.usage) if event.usage is not None else None)
        elif event_type == "error":
            yield LMStreamErrorEvent(error=RuntimeError(str(event.error)))

    def _map_stream_delta(self, delta: Any) -> Any | None:
        name = type(delta).__name__
        if name == "TextDelta":
            return LMTextDelta(text=delta.text)
        if name == "ThinkingDelta":
            return LMThinkingDelta(text=delta.text)
        if name == "ToolCallDelta":
            return LMToolCallDelta(id=delta.id, name=delta.name, args_delta=delta.input)
        if name == "ImageDelta":
            return LMImageDelta(
                image=LMImagePart(
                    data=delta.data,
                    url=delta.url,
                    file_id=delta.file_id,
                    media_type=delta.media_type or "image/png",
                )
            )
        if name == "AudioDelta":
            return LMAudioDelta(
                audio=LMAudioPart(
                    data=delta.data,
                    url=delta.url,
                    file_id=delta.file_id,
                    media_type=delta.media_type or "audio/wav",
                )
            )
        if name == "CitationDelta":
            return LMCitationDelta(citation=LMCitationPart(text=delta.text, title=delta.title, url=delta.url))
        return None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        api_key = self.api_key or os.environ.get(_api_key_env_var(self.provider), "")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url is not None:
            kwargs["base_url"] = self.base_url

        if self.provider == "anthropic":
            self._client = providers.AnthropicLM(**kwargs)
        elif self.provider == "gemini":
            self._client = providers.GeminiLM(**kwargs)
        else:
            self._client = providers.OpenAILM(**kwargs)
        return self._client


def _provider_from_model(model: str) -> str:
    provider = model.split("/", 1)[0] if "/" in model else "openai"
    if provider in {"anthropic", "gemini", "openai"}:
        return provider
    return "openai"


def _strip_provider_prefix(model: str, provider: str) -> str:
    prefix = f"{provider}/"
    return model.removeprefix(prefix)


def _api_key_env_var(provider: str) -> str:
    return {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }.get(provider, "OPENAI_API_KEY")


def _usage_to_dspy(usage: Any) -> LMUsage | None:
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


def _is_document_media_type(media_type: str) -> bool:
    return media_type in {"application/pdf", "text/plain", "text/markdown", "text/html"}


def _is_lm15_text_part(part: Any) -> bool:
    return type(part).__name__ == "TextPart"
