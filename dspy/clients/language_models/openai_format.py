"""OpenAI-format implementations of the normalized DSPy LM contract."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any, Callable

import pydantic

from dspy.clients.language_models.base import LanguageModel, LMCapabilities
from dspy.clients.language_models.features import FeatureStatus, feature_status
from dspy.clients.language_models.types import (
    LMAudioPart,
    LMCitationPart,
    LMConfig,
    LMFilePart,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMRefusalPart,
    LMRequest,
    LMResponse,
    LMStreamEvent,
    LMTextPart,
    LMThinkingPart,
    LMToolCallPart,
    LMToolChoice,
    LMToolResultPart,
    LMToolSpec,
    LMUsage,
)


class OpenAIFormatMixin:
    """Shared mapping hooks for OpenAI-shaped chat and responses protocols."""

    # This mixin owns protocol shape only: normalized DSPy parts in,
    # OpenAI-compatible dictionaries out. It must not know how a request is
    # sent, retried, authenticated, billed, or cached by a provider.

    def map_request_text(self, value: Any) -> str:
        return str(value)

    def map_request_input_image(self, value: LMImagePart) -> list[dict[str, Any]]:
        image_url: dict[str, Any] = {"url": _media_source(value)}
        if value.detail is not None:
            image_url["detail"] = value.detail
        return [{"type": "image_url", "image_url": image_url}]

    def map_request_input_audio(self, value: LMAudioPart) -> list[dict[str, Any]]:
        if value.data is None:
            raise ValueError("OpenAI-format audio input requires base64 `data`; url, file_id, and path are not supported.")
        return [{"type": "input_audio", "input_audio": {"data": value.data, "format": _media_format(value.media_type)}}]

    def map_request_input_file(self, value: LMFilePart) -> list[dict[str, Any]]:
        file_data: dict[str, Any] = {}
        if value.data is not None:
            file_data["file_data"] = _data_uri(value.media_type, value.data)
        elif value.url is not None or value.path is not None:
            file_data["file_data"] = _media_source(value)
        if value.file_id is not None:
            file_data["file_id"] = value.file_id
        if value.filename is not None:
            file_data["filename"] = value.filename
        return [{"type": "file", "file": file_data}]

    def map_request_tools(self, value: LMToolSpec) -> dict[str, Any]:
        data = {"type": "function", "function": {"name": value.name, "parameters": value.parameters}}
        if value.description is not None:
            data["function"]["description"] = value.description
        data.update(value.provider_data)
        return data

    def map_request_tool_choice(self, value: LMToolChoice) -> dict[str, Any]:
        if value.allowed:
            if len(value.allowed) != 1 or value.mode not in {"required", "auto"}:
                raise ValueError("OpenAI-format tool_choice only supports constraining to a single allowed tool.")
            data: dict[str, Any] = {"tool_choice": {"type": "function", "function": {"name": value.allowed[0]}}}
        else:
            data = {"tool_choice": value.mode}
        if value.parallel is not None:
            data["parallel_tool_calls"] = value.parallel
        return data

    def map_request_assistant_tool_calls(self, value: LMToolCallPart) -> dict[str, Any]:
        data = {"type": "function", "function": {"name": value.name, "arguments": json.dumps(value.args)}}
        if value.id is not None:
            data["id"] = value.id
        data.update(value.provider_data)
        return data

    def map_request_tool_results(self, value: LMToolResultPart) -> dict[str, Any]:
        return {"content": self._map_parts_to_openai_content(value.content)}

    def map_request_response_schema(self, value: Any) -> Any:
        return value

    def map_request_reasoning_config(self, value: Any) -> dict[str, Any]:
        data = {}
        if value.effort is not None:
            data["reasoning_effort"] = value.effort
        if value.max_tokens is not None:
            data["thinking_budget"] = value.max_tokens
        if value.summary is not None:
            data["reasoning_summary"] = value.summary
        return data

    def map_request_prompt_cache(self, value: Any) -> dict[str, Any]:
        data = {}
        if value.key is not None:
            data["prompt_cache_key"] = value.key
        if value.enabled is False:
            data["prompt_cache"] = False
        return data

    def map_request_logprobs(self, value: Any) -> Any:
        return value

    def map_request_multiple_outputs(self, value: Any) -> Any:
        return value

    def map_request_provider_extensions(self, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    def map_response_text(self, value: Any) -> LMTextPart:
        return LMTextPart(text=str(value))

    def map_response_reasoning(self, value: Any) -> LMThinkingPart:
        return LMThinkingPart(text=str(value))

    def map_response_tool_calls(self, value: Any) -> LMToolCallPart:
        function = _get_value(value, "function") or {}
        name = _get_value(function, "name") or _get_value(value, "name")
        arguments = _get_value(function, "arguments", _get_value(value, "arguments", "{}"))
        provider_data = _model_dump(value)
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        except Exception as error:
            args = {}
            provider_data["raw_arguments"] = arguments
            provider_data["arguments_parse_error"] = str(error)
        call_id = _get_value(value, "call_id") or _get_value(value, "id")
        return LMToolCallPart(id=call_id, name=name or "", args=args, provider_data=provider_data)

    def map_response_citations(self, value: Any) -> LMCitationPart:
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        if not isinstance(value, dict):
            value = {"text": str(value)}
        citation_fields = {"cited_text", "text", "supported_text", "document_title", "title", "url"}
        return LMCitationPart(
            text=value.get("cited_text") or value.get("text") or value.get("supported_text"),
            title=value.get("document_title") or value.get("title"),
            url=value.get("url"),
            metadata={k: v for k, v in value.items() if k not in citation_fields},
        )

    def map_response_output_image(self, value: Any) -> LMImagePart:
        data = _model_dump(value)
        image_url = data.get("image_url")
        if isinstance(image_url, dict):
            image_url = image_url.get("url")
        source = image_url or data.get("url")
        b64_data = data.get("b64_json") or data.get("data")
        file_id = data.get("file_id")
        media_type = data.get("media_type") or data.get("mime_type") or "image/png"
        detail = data.get("detail")
        if b64_data is not None:
            if isinstance(b64_data, str) and b64_data.startswith("data:"):
                media_type, b64_data = _split_data_uri(b64_data)
            return LMImagePart(data=b64_data, media_type=media_type, detail=detail)
        if source is not None:
            return LMImagePart(url=source, media_type=media_type, detail=detail)
        if file_id is not None:
            return LMImagePart(file_id=file_id, media_type=media_type, detail=detail)
        raise ValueError("Provider image output did not include data, url, or file_id.")

    def map_response_output_audio(self, value: Any) -> LMAudioPart:
        data = _model_dump(value)
        audio = data.get("audio") if isinstance(data.get("audio"), dict) else data
        source = audio.get("url")
        b64_data = audio.get("data") or audio.get("b64_json")
        file_id = audio.get("file_id")
        media_type = audio.get("media_type") or audio.get("mime_type") or "audio/wav"
        if b64_data is not None:
            if isinstance(b64_data, str) and b64_data.startswith("data:"):
                media_type, b64_data = _split_data_uri(b64_data)
            return LMAudioPart(data=b64_data, media_type=media_type)
        if source is not None:
            return LMAudioPart(url=source, media_type=media_type)
        if file_id is not None:
            return LMAudioPart(file_id=file_id, media_type=media_type)
        raise ValueError("Provider audio output did not include data, url, or file_id.")

    def map_response_output_file(self, value: Any) -> LMFilePart:
        data = _model_dump(value)
        file = data.get("file") if isinstance(data.get("file"), dict) else data
        source = file.get("url")
        b64_data = file.get("file_data") or file.get("data")
        file_id = file.get("file_id") or file.get("id")
        filename = file.get("filename")
        media_type = file.get("media_type") or file.get("mime_type") or "application/octet-stream"
        if b64_data is not None:
            if isinstance(b64_data, str) and b64_data.startswith("data:"):
                media_type, b64_data = _split_data_uri(b64_data)
            return LMFilePart(data=b64_data, media_type=media_type, filename=filename)
        if source is not None:
            return LMFilePart(url=source, media_type=media_type, filename=filename)
        if file_id is not None:
            return LMFilePart(file_id=file_id, media_type=media_type, filename=filename)
        raise ValueError("Provider file output did not include data, url, or file_id.")

    def map_response_refusal(self, value: Any) -> LMRefusalPart:
        text = _get_value(value, "refusal") or _get_value(value, "text") or _get_value(value, "content") or str(value)
        return LMRefusalPart(text=str(text))

    def _map_message_to_openai(self, message: LMMessage) -> dict[str, Any]:
        output: dict[str, Any] = {"role": message.role}
        if message.name is not None:
            output["name"] = message.name
        if message.role == "assistant":
            tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
            content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
            output["content"] = self._map_parts_to_openai_content(content_parts)
            if tool_calls:
                output["tool_calls"] = [self.map_request_assistant_tool_calls(part) for part in tool_calls]
            return output
        if message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
            result = message.parts[0]
            output.update(self.map_request_tool_results(result))
            if result.call_id is not None:
                output["tool_call_id"] = result.call_id
            if result.name is not None:
                output["name"] = result.name
            return output
        output["content"] = self._map_parts_to_openai_content(message.parts)
        return output

    def _map_parts_to_openai_content(self, parts: list[Any]) -> str | list[dict[str, Any]]:
        if len(parts) == 1 and isinstance(parts[0], LMTextPart):
            return self.map_request_text(parts[0].text)
        blocks: list[dict[str, Any]] = []
        for part in parts:
            blocks.extend(self._map_request_part_to_openai_blocks(part))
        return blocks

    def _map_request_part_to_openai_blocks(self, part: Any) -> list[dict[str, Any]]:
        if isinstance(part, LMTextPart):
            return [{"type": "text", "text": self.map_request_text(part.text)}]
        if isinstance(part, LMImagePart):
            return self.map_request_input_image(part)
        if isinstance(part, LMAudioPart):
            return self.map_request_input_audio(part)
        if isinstance(part, LMFilePart):
            return self.map_request_input_file(part)
        if isinstance(part, LMThinkingPart):
            return [{"type": "text", "text": part.text}]
        if isinstance(part, LMCitationPart):
            citation = " ".join(value for value in (part.title, part.text, part.url) if value)
            return [{"type": "text", "text": citation}]
        if isinstance(part, LMToolResultPart):
            return self._map_request_part_to_openai_blocks(LMTextPart(text="".join(_part_text(v) for v in part.content)))
        return [{"type": "text", "text": str(part)}]

    def _common_config_kwargs(self, config: LMConfig) -> dict[str, Any]:
        data = self.map_request_provider_extensions(config.extensions)
        for key in ("temperature", "max_tokens", "top_p"):
            value = getattr(config, key)
            if value is not None:
                data[key] = value
        if config.stop:
            data["stop"] = config.stop
        if config.logprobs is not None:
            data["logprobs"] = self.map_request_logprobs(config.logprobs)
        if config.n is not None:
            data["n"] = self.map_request_multiple_outputs(config.n)
        if config.response_format is not None:
            data["response_format"] = self.map_request_response_schema(config.response_format)
        if config.reasoning is not None:
            data.update(self.map_request_reasoning_config(config.reasoning))
        if config.prompt_cache is not None:
            data.update(self.map_request_prompt_cache(config.prompt_cache))
        if config.cache is not None and config.cache.rollout_id is not None:
            data["rollout_id"] = config.cache.rollout_id
        return data


class OpenAIChatLM(OpenAIFormatMixin, LanguageModel):
    """Map normalized requests to OpenAI Chat Completions-shaped calls."""

    # OpenAIChatLM owns chat-completions request and response semantics.
    # Subclasses own transport: OpenAI SDK, LiteLLM, HTTP, local server, etc.

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(function_calling=True, response_schema=True, input_image=True, input_audio=True, input_file=True, tool_results=True)

    def forward(self, request: LMRequest) -> LMResponse:
        provider_request = self._to_completion_request(request)
        return _completion_to_lm_response(self.completion(provider_request), request, lm=self)

    async def aforward(self, request: LMRequest) -> LMResponse:
        provider_request = self._to_completion_request(request)
        return _completion_to_lm_response(await self.acompletion(provider_request), request, lm=self)

    def completion(self, request: dict[str, Any]) -> Any:
        raise NotImplementedError("Subclasses must implement completion(request).")

    async def acompletion(self, request: dict[str, Any]) -> Any:
        raise NotImplementedError("Subclasses must implement acompletion(request).")

    def _to_completion_request(self, request: LMRequest) -> dict[str, Any]:
        data = {"model": request.model, "messages": [self._map_message_to_openai(message) for message in request.messages]}
        data.update(self._common_config_kwargs(request.config))
        if request.config.tool_choice is not None:
            data.update(self.map_request_tool_choice(request.config.tool_choice))
        if request.tools:
            data["tools"] = [self.map_request_tools(tool) for tool in request.tools]
        return data


class CompletionLM(OpenAIChatLM):
    """Call a provider that accepts OpenAI Chat Completions-shaped requests."""

    def __init__(
        self,
        model: str,
        *,
        completion: Callable[..., Any] | None = None,
        acompletion: Callable[..., Any] | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self._completion_callable = completion
        self._acompletion_callable = acompletion

    def completion(self, request: dict[str, Any]) -> Any:
        if self._completion_callable is None:
            raise NotImplementedError("Pass completion=... or override completion(request).")
        return self._completion_callable(**request)

    async def acompletion(self, request: dict[str, Any]) -> Any:
        if self._acompletion_callable is None:
            raise NotImplementedError("Pass acompletion=... or override acompletion(request).")
        return await self._acompletion_callable(**request)


class OpenAIResponsesLM(OpenAIFormatMixin, LanguageModel):
    """Map normalized requests to OpenAI Responses API-shaped calls."""

    # OpenAIResponsesLM owns Responses API item/content semantics.
    # Subclasses own transport: OpenAI SDK, LiteLLM, HTTP, local server, etc.

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities(function_calling=True, reasoning=True, response_schema=True, input_image=True, input_file=True, tool_results=True)

    def map_request_reasoning_config(self, value: Any) -> dict[str, Any]:
        reasoning = {}
        if value.effort is not None:
            reasoning["effort"] = value.effort
        if value.max_tokens is not None:
            reasoning["max_tokens"] = value.max_tokens
        if value.summary is not None:
            reasoning["summary"] = value.summary
        elif reasoning:
            reasoning["summary"] = "auto"
        return {"reasoning": reasoning} if reasoning else {}

    def forward(self, request: LMRequest) -> LMResponse:
        provider_request = self._to_responses_request(request)
        return _responses_api_to_lm_response(self.responses(provider_request), request, lm=self)

    async def aforward(self, request: LMRequest) -> LMResponse:
        provider_request = self._to_responses_request(request)
        return _responses_api_to_lm_response(await self.aresponses(provider_request), request, lm=self)

    def responses(self, request: dict[str, Any]) -> Any:
        raise NotImplementedError("Subclasses must implement responses(request).")

    async def aresponses(self, request: dict[str, Any]) -> Any:
        raise NotImplementedError("Subclasses must implement aresponses(request).")

    def _to_responses_request(self, request: LMRequest) -> dict[str, Any]:
        config = request.config
        data: dict[str, Any] = {
            "model": request.model,
            "input": [item for message in request.messages for item in self._map_message_to_responses_input_items(message)],
        }
        data.update(self._map_config_to_responses_kwargs(config))
        if config.tool_choice is not None:
            data.update(self.map_request_tool_choice(config.tool_choice))
        if request.tools:
            data["tools"] = [self.map_request_tools(tool) for tool in request.tools]
        return data

    def _map_config_to_responses_kwargs(self, config: LMConfig) -> dict[str, Any]:
        data = self.map_request_provider_extensions(config.extensions) if config.extensions else {}
        for key in ("temperature", "max_tokens", "top_p"):
            value = getattr(config, key)
            if value is not None:
                data[key] = value
        if config.n is not None:
            data["n"] = self.map_request_multiple_outputs(config.n)
        if config.logprobs is not None:
            data["logprobs"] = self.map_request_logprobs(config.logprobs)
        if config.stop:
            data["stop"] = config.stop
        if config.reasoning is not None:
            data.update(self.map_request_reasoning_config(config.reasoning))
        if config.prompt_cache is not None:
            data.update(self.map_request_prompt_cache(config.prompt_cache))
        if config.response_format is not None:
            text = data.pop("text", {})
            data["text"] = {**text, "format": self._map_response_format_to_responses(config.response_format)}
        if config.cache is not None and config.cache.rollout_id is not None:
            data["rollout_id"] = config.cache.rollout_id
        return data

    def _map_response_format_to_responses(self, value: Any) -> Any:
        value = self.map_request_response_schema(value)
        if isinstance(value, type) and issubclass(value, pydantic.BaseModel):
            return {"name": value.__name__, "type": "json_schema", "schema": value.model_json_schema()}
        return value

    def _map_message_to_responses_input_items(self, message: LMMessage) -> list[dict[str, Any]]:
        if message.role == "tool" and len(message.parts) == 1 and isinstance(message.parts[0], LMToolResultPart):
            result = message.parts[0]
            mapped = self.map_request_tool_results(result)
            item = {"type": "function_call_output", "output": self._responses_tool_output_text(mapped["content"])}
            if result.call_id is not None:
                item["call_id"] = result.call_id
            return [item]
        tool_calls = [part for part in message.parts if isinstance(part, LMToolCallPart)]
        content_parts = [part for part in message.parts if not isinstance(part, LMToolCallPart)]
        content = self._map_parts_to_responses_content(content_parts)
        items: list[dict[str, Any]] = []
        if content or message.role != "assistant" or not tool_calls:
            item: dict[str, Any] = {"role": message.role, "content": content}
            if message.name is not None:
                item["name"] = message.name
            items.append(item)
        if message.role == "assistant":
            for tool_call in tool_calls:
                items.append(self._map_tool_call_to_responses_input(tool_call))
        return items

    def _map_parts_to_responses_content(self, parts: list[Any]) -> list[dict[str, Any]]:
        blocks = self._map_parts_to_openai_content(parts)
        if isinstance(blocks, str):
            return [{"type": "input_text", "text": blocks}]
        return [self._map_content_block_to_responses(block) for block in blocks]

    def _map_tool_call_to_responses_input(self, value: LMToolCallPart) -> dict[str, Any]:
        tool_call = self.map_request_assistant_tool_calls(value)
        function = tool_call.get("function", {})
        item = {"type": "function_call", "name": function.get("name", ""), "arguments": function.get("arguments", "{}")}
        call_id = tool_call.get("id") or tool_call.get("call_id")
        if call_id is not None:
            item["call_id"] = call_id
        return item

    def _map_content_block_to_responses(self, block: dict[str, Any]) -> dict[str, Any]:
        if block.get("type") == "text":
            return {"type": "input_text", "text": block.get("text", "")}
        if block.get("type") == "image_url":
            return {"type": "input_image", "image_url": block.get("image_url", {}).get("url", "")}
        if block.get("type") == "input_audio":
            return block
        if block.get("type") == "file":
            file = block.get("file", {})
            return {"type": "input_file", "file_data": file.get("file_data"), "filename": file.get("filename"), "file_id": file.get("file_id")}
        return block

    def _responses_tool_output_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(block.get("text", "") if isinstance(block, dict) and block.get("type") in {"text", "input_text"} else str(block) for block in content)
        return str(content)


class ResponsesLM(OpenAIResponsesLM):
    """Call a provider that accepts OpenAI Responses API-shaped requests."""

    def __init__(
        self,
        model: str,
        *,
        responses: Callable[..., Any] | None = None,
        aresponses: Callable[..., Any] | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self._responses_callable = responses
        self._aresponses_callable = aresponses

    def responses(self, request: dict[str, Any]) -> Any:
        if self._responses_callable is None:
            raise NotImplementedError("Pass responses=... or override responses(request).")
        return self._responses_callable(**request)

    async def aresponses(self, request: dict[str, Any]) -> Any:
        if self._aresponses_callable is None:
            raise NotImplementedError("Pass aresponses=... or override aresponses(request).")
        return await self._aresponses_callable(**request)


class OpenAITextLM(LanguageModel):
    """Map normalized text-only requests to OpenAI text-completion calls."""

    # OpenAITextLM is intentionally narrow: text prompt in, text choices out.
    # Subclasses own transport and should not silently serialize rich content.

    def get_capabilities(self) -> LMCapabilities:
        return LMCapabilities()

    def get_request_feature_statuses(self) -> dict[str, FeatureStatus]:
        statuses = super().get_request_feature_statuses()
        for name in ("input_image", "input_audio", "input_file", "tools", "tool_choice", "assistant_tool_calls", "tool_results", "response_schema", "reasoning_config", "prompt_cache"):
            statuses[name] = feature_status(f"request.{name}", "unsupported", f"{type(self).__name__} maps normalized messages to a plain text prompt and does not support request.{name}.")
        return statuses

    def get_response_feature_statuses(self) -> dict[str, FeatureStatus]:
        statuses = super().get_response_feature_statuses()
        for name in ("reasoning", "tool_calls", "citations", "output_image", "output_audio", "output_file", "refusal"):
            statuses[name] = feature_status(f"response.{name}", "unsupported", f"{type(self).__name__} returns text-completion choices and does not support response.{name}.")
        return statuses

    def map_request_text(self, value: Any) -> str:
        return str(value)

    def map_request_logprobs(self, value: Any) -> Any:
        return value

    def map_request_multiple_outputs(self, value: Any) -> Any:
        return value

    def map_request_provider_extensions(self, value: dict[str, Any]) -> dict[str, Any]:
        return dict(value)

    def map_response_text(self, value: Any) -> LMTextPart:
        return LMTextPart(text=str(value))

    def forward(self, request: LMRequest) -> LMResponse:
        provider_request = self._to_text_request(request)
        return _completion_to_lm_response(self.text_completion(provider_request), request, lm=self)

    async def aforward(self, request: LMRequest) -> LMResponse:
        provider_request = self._to_text_request(request)
        return _completion_to_lm_response(await self.atext_completion(provider_request), request, lm=self)

    def text_completion(self, request: dict[str, Any]) -> Any:
        raise NotImplementedError("Subclasses must implement text_completion(request).")

    async def atext_completion(self, request: dict[str, Any]) -> Any:
        raise NotImplementedError("Subclasses must implement atext_completion(request).")

    def _to_text_request(self, request: LMRequest) -> dict[str, Any]:
        data = {"model": request.model, "prompt": self._messages_to_prompt(request.messages)}
        data.update(self._text_config_kwargs(request.config))
        return data

    def _text_config_kwargs(self, config: LMConfig) -> dict[str, Any]:
        data = self.map_request_provider_extensions(config.extensions)
        for key in ("temperature", "max_tokens", "top_p"):
            value = getattr(config, key)
            if value is not None:
                data[key] = value
        if config.stop:
            data["stop"] = config.stop
        if config.logprobs is not None:
            data["logprobs"] = self.map_request_logprobs(config.logprobs)
        if config.n is not None:
            data["n"] = self.map_request_multiple_outputs(config.n)
        if config.cache is not None and config.cache.rollout_id is not None:
            data["rollout_id"] = config.cache.rollout_id
        return data

    def _messages_to_prompt(self, messages: list[LMMessage]) -> str:
        chunks = []
        for message in messages:
            texts = []
            for part in message.parts:
                if not isinstance(part, LMTextPart):
                    raise ValueError(f"{type(self).__name__} only supports text parts, but received {type(part).__name__}.")
                texts.append(self.map_request_text(part.text))
            chunks.append("".join(texts))
        return "\n\n".join(chunks + ["BEGIN RESPONSE:"])


class TextCompletionLM(OpenAITextLM):
    """Call a provider that accepts OpenAI-style text completion requests."""

    def __init__(
        self,
        model: str,
        *,
        completion: Callable[..., Any] | None = None,
        acompletion: Callable[..., Any] | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self._completion_callable = completion
        self._acompletion_callable = acompletion

    def text_completion(self, request: dict[str, Any]) -> Any:
        if self._completion_callable is None:
            raise NotImplementedError("Pass completion=... or override text_completion(request).")
        return self._completion_callable(**request)

    async def atext_completion(self, request: dict[str, Any]) -> Any:
        if self._acompletion_callable is None:
            raise NotImplementedError("Pass acompletion=... or override atext_completion(request).")
        return await self._acompletion_callable(**request)


def _completion_to_lm_response(response: Any, request: LMRequest, *, lm: LanguageModel | None = None) -> LMResponse:
    choices = _get_value(response, "choices", []) or []
    return LMResponse(
        model=_get_value(response, "model") or request.model,
        outputs=[_choice_to_lm_output(choice, lm=lm) for choice in choices],
        usage=_usage_from_response(response),
        cache_hit=bool(_get_value(response, "cache_hit", False)),
        response_id=_get_value(response, "id"),
        provider_response=response,
    )


def _choice_to_lm_output(choice: Any, *, lm: LanguageModel | None = None) -> LMOutput:
    message = _get_value(choice, "message")
    parts = []
    if message is not None:
        reasoning = _get_value(message, "reasoning_content")
        if reasoning:
            parts.append(lm.map_response_reasoning(reasoning) if lm is not None else LMThinkingPart(text=reasoning))
        content = _get_value(message, "content")
        if content:
            parts.extend(_message_content_to_parts(content, lm=lm))
        for tool_call in _get_value(message, "tool_calls") or []:
            parts.append(lm.map_response_tool_calls(tool_call) if lm is not None else _provider_tool_call_to_part(tool_call))
        parts.extend(_extract_citations_from_choice(choice, lm=lm))
    else:
        text = _get_value(choice, "text")
        if text:
            parts.extend(_message_content_to_parts(text, lm=lm))
    finish_reason = _get_value(choice, "finish_reason")
    return LMOutput(parts=parts, finish_reason=finish_reason, truncated=finish_reason == "length", logprobs=_get_value(choice, "logprobs"), provider_output=choice)


def _responses_api_to_lm_response(response: Any, request: LMRequest, *, lm: LanguageModel | None = None) -> LMResponse:
    parts = []
    for output_item in _get_value(response, "output", []) or []:
        output_type = _get_value(output_item, "type")
        if output_type == "message":
            for content_item in _get_value(output_item, "content", []) or []:
                parts.extend(_response_content_item_to_parts(content_item, lm=lm))
                parts.extend(_responses_annotations_to_citations(content_item, lm=lm))
        elif output_type == "function_call":
            parts.append(lm.map_response_tool_calls(output_item) if lm is not None else _responses_function_call_to_part(output_item))
        elif output_type in {"image", "output_image", "image_generation_call"} and lm is not None:
            parts.append(lm.map_response_output_image(output_item))
        elif output_type in {"audio", "output_audio"} and lm is not None:
            parts.append(lm.map_response_output_audio(output_item))
        elif output_type in {"file", "output_file"} and lm is not None:
            parts.append(lm.map_response_output_file(output_item))
        elif output_type == "refusal" and lm is not None:
            parts.append(lm.map_response_refusal(output_item))
        elif output_type == "reasoning":
            for item in _get_value(output_item, "content") or _get_value(output_item, "summary") or []:
                text = _get_value(item, "text")
                if text:
                    parts.append(lm.map_response_reasoning(text) if lm is not None else LMThinkingPart(text=text))
    return LMResponse(
        model=_get_value(response, "model") or request.model,
        outputs=[LMOutput(parts=parts, provider_output=response)],
        usage=_usage_from_response(response),
        cache_hit=bool(_get_value(response, "cache_hit", False)),
        response_id=_get_value(response, "id"),
        provider_response=response,
    )


def _message_content_to_parts(content: Any, *, lm: LanguageModel | None = None) -> list[Any]:
    if isinstance(content, str):
        return [lm.map_response_text(content) if lm is not None else LMTextPart(text=content)]
    if not isinstance(content, list):
        return [lm.map_response_text(str(content)) if lm is not None else LMTextPart(text=str(content))]
    parts = []
    for item in content:
        parts.extend(_response_content_item_to_parts(item, lm=lm))
    return parts


def _response_content_item_to_parts(item: Any, *, lm: LanguageModel | None = None) -> list[Any]:
    item_type = _get_value(item, "type")
    text = _get_value(item, "text")
    if item_type in {"text", "output_text", "input_text"} or (text is not None and item_type is None):
        return [lm.map_response_text(text) if lm is not None else LMTextPart(text=text)]
    if item_type in {"refusal", "output_refusal"}:
        return [lm.map_response_refusal(item)] if lm is not None else []
    if item_type in {"image", "output_image", "image_url"}:
        return [lm.map_response_output_image(item)] if lm is not None else []
    if item_type in {"audio", "output_audio", "input_audio"}:
        return [lm.map_response_output_audio(item)] if lm is not None else []
    if item_type in {"file", "output_file", "input_file"}:
        return [lm.map_response_output_file(item)] if lm is not None else []
    if item_type in {"tool_call", "function_call"}:
        return [lm.map_response_tool_calls(item) if lm is not None else _provider_tool_call_to_part(item)]
    return []


def _provider_tool_call_to_part(tool_call: Any) -> LMToolCallPart:
    function = _get_value(tool_call, "function", {})
    name = _get_value(function, "name")
    arguments = _get_value(function, "arguments", "{}")
    provider_data = _model_dump(tool_call)
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except Exception as error:
        args = {}
        provider_data["raw_arguments"] = arguments
        provider_data["arguments_parse_error"] = str(error)
    return LMToolCallPart(id=_get_value(tool_call, "id"), name=name or "", args=args, provider_data=provider_data)


def _responses_function_call_to_part(output_item: Any) -> LMToolCallPart:
    args = _get_value(output_item, "arguments", {})
    provider_data = _model_dump(output_item)
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception as error:
            provider_data["raw_arguments"] = args
            provider_data["arguments_parse_error"] = str(error)
            args = {}
    return LMToolCallPart(id=_get_value(output_item, "call_id"), name=_get_value(output_item, "name", ""), args=args, provider_data=provider_data)


def _cost_from_response(response: Any) -> float | None:
    hidden = getattr(response, "_hidden_params", None) or {}
    if isinstance(hidden, dict):
        return hidden.get("response_cost")
    return None


def _usage_from_response(response: Any) -> LMUsage | None:
    usage = _get_value(response, "usage")
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump(exclude_none=True)
    elif not isinstance(usage, dict):
        usage = {key: value for key in dir(usage) if not key.startswith("_") for value in [getattr(usage, key)] if not callable(value)}
    return LMUsage(**dict(usage))


def _extract_citations_from_choice(choice: Any, *, lm: LanguageModel | None = None) -> list[LMCitationPart]:
    try:
        message = _get_value(choice, "message")
        provider_specific_fields = _get_value(message, "provider_specific_fields", {}) or {}
        citations_data = provider_specific_fields.get("citations")
        if isinstance(citations_data, list):
            citations = []
            for item in citations_data:
                citations.extend(item if isinstance(item, list) else [item])
            return [lm.map_response_citations(citation) if lm is not None else _citation_to_part(citation) for citation in citations]
    except Exception:
        return []
    return []


def _responses_annotations_to_citations(content_item: Any, *, lm: LanguageModel | None = None) -> list[LMCitationPart]:
    annotations = _get_value(content_item, "annotations", []) or []
    return [lm.map_response_citations(annotation) if lm is not None else _citation_to_part(annotation) for annotation in annotations]


def _citation_to_part(citation: Any) -> LMCitationPart:
    if hasattr(citation, "model_dump"):
        citation = citation.model_dump(exclude_none=True)
    if not isinstance(citation, dict):
        citation = {"text": str(citation)}
    citation_fields = {"cited_text", "text", "supported_text", "document_title", "title", "url"}
    return LMCitationPart(
        text=citation.get("cited_text") or citation.get("text") or citation.get("supported_text"),
        title=citation.get("document_title") or citation.get("title"),
        url=citation.get("url"),
        metadata={k: v for k, v in citation.items() if k not in citation_fields},
    )


def _media_source(part: LMImagePart | LMAudioPart | LMFilePart) -> str:
    if part.data is not None:
        return _data_uri(part.media_type, part.data)
    if part.url is not None:
        return part.url
    if part.file_id is not None:
        return part.file_id
    if part.path is not None:
        return part.path
    raise ValueError(f"{type(part).__name__} has no media source.")


def _data_uri(media_type: str, data: str) -> str:
    if data.startswith("data:"):
        return data
    return f"data:{media_type};base64,{data}"


def _split_data_uri(value: str) -> tuple[str, str]:
    if not value.startswith("data:") or "," not in value:
        return "application/octet-stream", value
    header, data = value.split(",", 1)
    media_type = header.removeprefix("data:").split(";", 1)[0]
    return media_type, data


def _media_format(media_type: str) -> str:
    return media_type.split("/", 1)[1] if "/" in media_type else media_type


def _part_text(value: Any) -> str:
    if isinstance(value, LMTextPart):
        return value.text
    return str(value)


def _get_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if isinstance(value, dict):
        return dict(value)
    data = {}
    for key in ("id", "call_id", "type", "name", "arguments", "status", "text", "refusal", "url", "data", "file_id", "filename", "media_type", "mime_type"):
        item = getattr(value, key, None)
        if item is not None:
            data[key] = item
    return data
