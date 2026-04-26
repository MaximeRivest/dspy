"""LiteLLM-backed language model implementation.

This module is the extraction point for LiteLLM-specific inference,
capability checks, errors, request conversion, and streaming helpers.
The public `dspy.LM` class can keep routing/backward-compatibility
while this class owns the LiteLLM wire behavior.
"""

import importlib
import logging
import os
import re
import warnings
from typing import Any, Literal, cast

import litellm
import pydantic
from anyio.streams.memory import MemoryObjectSendStream
from asyncer import syncify
from litellm import ContextWindowExceededError as LitellmContextWindowExceededError

import dspy
from dspy.clients.cache import request_cache
from dspy.clients.provider import Provider
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import ContextWindowExceededError

from .base_lm import BaseLM

logger = logging.getLogger(__name__)

__all__ = [
    "LiteLLMLM",
    "_add_dspy_identifier_to_headers",
    "_convert_chat_request_to_responses_request",
    "_convert_content_item_to_responses_format",
    "_get_stream_completion_fn",
    "alitellm_completion",
    "alitellm_responses_completion",
    "alitellm_text_completion",
    "litellm_completion",
    "litellm_responses_completion",
    "litellm_text_completion",
]


class LiteLLMLM(BaseLM):
    """
    A language model supporting chat or text completion requests for use with DSPy modules.
    """

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        use_developer_role: bool = False,
        **kwargs,
    ):
        """
        Create a new language model instance for use with DSPy modules and programs.

        Args:
            model: The model to use. This should be a string of the form ``"llm_provider/llm_name"``
                   supported by LiteLLM. For example, ``"openai/gpt-4o"``.
            model_type: The type of the model, either ``"chat"`` or ``"text"``.
            temperature: The sampling temperature to use when generating responses.
            max_tokens: The maximum number of tokens to generate per response.
            cache: Whether to cache the model responses for reuse to improve performance
                   and reduce costs.
            callbacks: A list of callback functions to run before and after each request.
            num_retries: The number of times to retry a request if it fails transiently due to
                         network error, rate limiting, etc. Requests are retried with exponential
                         backoff.
            provider: The provider to use. If not specified, the provider will be inferred from the model.
            finetuning_model: The model to finetune. In some providers, the models available for finetuning is different
                from the models available for inference.
            rollout_id: Optional integer used to differentiate cache entries for otherwise
                identical requests. Different values bypass DSPy's caches while still caching
                future calls with the same inputs and rollout ID. Note that `rollout_id`
                only affects generation when `temperature` is non-zero. This argument is
                stripped before sending requests to the provider.
        """
        # Remember to update LM.copy() if you modify the constructor!
        super().__init__(
            model=model,
            model_type=model_type,
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
            callbacks=callbacks,
            num_retries=num_retries,
            provider=provider,
            finetuning_model=finetuning_model,
            launch_kwargs=launch_kwargs,
            train_kwargs=train_kwargs,
            **kwargs,
        )
        self.use_developer_role = use_developer_role

        # Handle model-specific configuration for different model families
        model_family = model.split("/")[-1].lower() if "/" in model else model.lower()

        # Recognize OpenAI reasoning models (o1, o3, o4, gpt-5 family)
        # Exclude non-reasoning variants like gpt-5-chat this is in azure ai foundry
        # Allow date suffixes like -2023-01-01 after model name or mini/nano/pro
        # For gpt-5, use negative lookahead to exclude -chat and allow other suffixes
        model_pattern = re.match(
            r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
            model_family,
        )

        if model_pattern:
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise ValueError(
                    "OpenAI's reasoning models require passing temperature=1.0 or None and max_tokens >= 16000 or None to "
                    "`dspy.LM(...)`, e.g., dspy.LM('openai/gpt-5', temperature=1.0, max_tokens=16000)"
                )
            self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
            if self.kwargs.get("rollout_id") is None:
                self.kwargs.pop("rollout_id", None)

        self._completion_module_name = __name__
        self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))

    def _configure_public_lm_wrapper(self, module_name: str) -> None:
        """Use a public wrapper module for completion helper lookup.

        This keeps legacy patch paths like `dspy.clients.lm.litellm_completion`
        working when a `LiteLLMLM` is constructed through the public `LM`
        facade. Direct `LiteLLMLM` instances keep resolving helpers from this
        module.
        """
        self._completion_module_name = module_name

    def _completion_module(self):
        return importlib.import_module(self._completion_module_name)

    def infer_provider(self) -> Provider:
        from dspy.clients.openai import OpenAIProvider

        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        return Provider()

    @property
    def _provider_name(self) -> str:
        """Extract the provider name from the model string (e.g., 'openai' from 'openai/gpt-4o')."""
        if "/" in self.model:
            return self.model.split("/", 1)[0]
        return "openai"

    @property
    def supports_function_calling(self) -> bool:
        return litellm.supports_function_calling(model=self.model)

    @property
    def supports_reasoning(self) -> bool:
        return litellm.supports_reasoning(self.model)

    @property
    def supports_response_schema(self) -> bool:
        return litellm.supports_response_schema(model=self.model, custom_llm_provider=self._provider_name)

    @property
    def supported_params(self) -> set[str]:
        params = litellm.get_supported_openai_params(model=self.model, custom_llm_provider=self._provider_name)
        return set(params) if params else set()

    def _warn_zero_temp_rollout(self, temperature: float | None, rollout_id):
        if not self._warned_zero_temp_rollout and rollout_id is not None and temperature == 0:
            warnings.warn(
                "rollout_id has no effect when temperature=0; set temperature>0 to bypass the cache.",
                stacklevel=3,
            )
            self._warned_zero_temp_rollout = True

    def _get_cached_completion_fn(self, completion_fn, cache):
        ignored_args_for_cache_key = ["api_key", "api_base", "base_url"]
        if cache:
            completion_fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=ignored_args_for_cache_key,
            )(completion_fn)

        litellm_cache_args = {"no-cache": True, "no-store": True}

        return completion_fn, litellm_cache_args

    def _get_completion_fn(self):
        completion_module = self._completion_module()
        if self.model_type == "chat":
            return completion_module.litellm_completion
        elif self.model_type == "text":
            return completion_module.litellm_text_completion
        elif self.model_type == "responses":
            return completion_module.litellm_responses_completion
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _get_async_completion_fn(self):
        completion_module = self._completion_module()
        if self.model_type == "chat":
            return completion_module.alitellm_completion
        elif self.model_type == "text":
            return completion_module.alitellm_text_completion
        elif self.model_type == "responses":
            return completion_module.alitellm_responses_completion
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _extract_citations_from_response(self, choice):
        """Extract citations from LiteLLM response objects when available.

        Reference: https://docs.litellm.ai/docs/providers/anthropic#beta-citations-api
        """
        try:
            citations_data = choice.message.provider_specific_fields.get("citations")
            if isinstance(citations_data, list):
                return [citation for citations in citations_data for citation in citations]
        except Exception:
            return None

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        # Build the request.
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        completion = self._get_completion_fn()
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)

        try:
            results = completion(
                request=dict(model=self.model, messages=messages, **kwargs),
                num_retries=self.num_retries,
                cache=litellm_cache_args,
            )
        except LitellmContextWindowExceededError as e:
            raise ContextWindowExceededError(model=self.model) from e

        self._check_truncation(results)

        return results

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        # Build the request.
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)

        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)

        completion = self._get_async_completion_fn()
        completion, litellm_cache_args = self._get_cached_completion_fn(completion, cache)

        try:
            results = await completion(
                request=dict(model=self.model, messages=messages, **kwargs),
                num_retries=self.num_retries,
                cache=litellm_cache_args,
            )
        except LitellmContextWindowExceededError as e:
            raise ContextWindowExceededError(model=self.model) from e

        self._check_truncation(results)

        return results

def _get_stream_completion_fn(
    request: dict[str, Any],
    cache_kwargs: dict[str, Any],
    sync=True,
    headers: dict[str, Any] | None = None,
):
    stream = dspy.settings.send_stream
    caller_predict = dspy.settings.caller_predict

    if stream is None:
        return None

    # The stream is already opened, and will be closed by the caller.
    stream = cast(MemoryObjectSendStream, stream)
    caller_predict_id = id(caller_predict) if caller_predict else None

    if dspy.settings.track_usage:
        request["stream_options"] = {"include_usage": True}

    async def stream_completion(request: dict[str, Any], cache_kwargs: dict[str, Any]):
        response = await litellm.acompletion(
            cache=cache_kwargs,
            stream=True,
            headers=headers,
            **request,
        )
        chunks = []
        async for chunk in response:
            if caller_predict_id:
                # Add the predict id to the chunk so that the stream listener can identify which predict produces it.
                chunk.predict_id = caller_predict_id
            chunks.append(chunk)
            await stream.send(chunk)
        return litellm.stream_chunk_builder(chunks)

    def sync_stream_completion():
        syncified_stream_completion = syncify(stream_completion)
        return syncified_stream_completion(request, cache_kwargs)

    async def async_stream_completion():
        return await stream_completion(request, cache_kwargs)

    if sync:
        return sync_stream_completion
    else:
        return async_stream_completion


def litellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    stream_completion = _get_stream_completion_fn(request, cache, sync=True, headers=headers)
    if stream_completion is None:
        return litellm.completion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            headers=headers,
            **request,
        )

    return stream_completion()


def litellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    # Extract the provider and model from the model string.
    # TODO: Not all the models are in the format of "provider/model"
    model = request.pop("model").split("/", 1)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return litellm.text_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def alitellm_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = _add_dspy_identifier_to_headers(request.pop("headers", None))
    stream_completion = _get_stream_completion_fn(request, cache, sync=False, headers=headers)
    if stream_completion is None:
        return await litellm.acompletion(
            cache=cache,
            num_retries=num_retries,
            retry_strategy="exponential_backoff_retry",
            headers=headers,
            **request,
        )

    return await stream_completion()


async def alitellm_text_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    model = request.pop("model").split("/", 1)
    headers = request.pop("headers", None)
    provider, model = model[0] if len(model) > 1 else "openai", model[-1]

    # Use the API key and base from the request, or from the environment.
    api_key = request.pop("api_key", None) or os.getenv(f"{provider}_API_KEY")
    api_base = request.pop("api_base", None) or os.getenv(f"{provider}_API_BASE")

    # Build the prompt from the messages.
    prompt = "\n\n".join([x["content"] for x in request.pop("messages")] + ["BEGIN RESPONSE:"])

    return await litellm.atext_completion(
        cache=cache,
        model=f"text-completion-openai/{model}",
        api_key=api_key,
        api_base=api_base,
        prompt=prompt,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def litellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)

    return litellm.responses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


async def alitellm_responses_completion(request: dict[str, Any], num_retries: int, cache: dict[str, Any] | None = None):
    cache = cache or {"no-cache": True, "no-store": True}
    request = dict(request)
    request.pop("rollout_id", None)
    headers = request.pop("headers", None)
    request = _convert_chat_request_to_responses_request(request)

    return await litellm.aresponses(
        cache=cache,
        num_retries=num_retries,
        retry_strategy="exponential_backoff_retry",
        headers=_add_dspy_identifier_to_headers(headers),
        **request,
    )


def _convert_chat_request_to_responses_request(request: dict[str, Any]):
    """
    Convert a chat request to a responses request
    See https://platform.openai.com/docs/api-reference/responses/create for the responses API specification.
    Also see https://platform.openai.com/docs/api-reference/chat/create for the chat API specification.
    """
    request = dict(request)
    if "messages" in request:
        input_items = []
        for msg in request.pop("messages"):
            content_blocks = []
            c = msg.get("content")
            if isinstance(c, str):
                content_blocks.append({"type": "input_text", "text": c})
            elif isinstance(c, list):
                # Convert each content item from Chat API format to Responses API format
                for item in c:
                    content_blocks.append(_convert_content_item_to_responses_format(item))
            input_items.append({"role": msg.get("role", "user"), "content": content_blocks})
        request["input"] = input_items
    # Convert `reasoning_effort` to reasoning format supported by the Responses API
    if "reasoning_effort" in request:
        effort = request.pop("reasoning_effort")
        request["reasoning"] = {"effort": effort, "summary": "auto"}

    # Convert `response_format` to `text.format` for Responses API
    if "response_format" in request:
        response_format = request.pop("response_format")
        if isinstance(response_format, type) and issubclass(response_format, pydantic.BaseModel):
            response_format = {
                "name": response_format.__name__,
                "type": "json_schema",
                "schema": response_format.model_json_schema(),
            }
        text = request.pop("text", {})
        request["text"] = {**text, "format": response_format}

    return request


def _convert_content_item_to_responses_format(item: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a content item from Chat API format to Responses API format.

    For images, converts from:
        {"type": "image_url", "image_url": {"url": "..."}}
    To:
        {"type": "input_image", "image_url": "..."}

    For text, converts from:
        {"type": "text", "text": "..."}
    To:
        {"type": "input_text", "text": "..."}

    For other types, passes through as-is.
    """
    if item.get("type") == "image_url":
        image_url = item.get("image_url", {}).get("url", "")
        return {
            "type": "input_image",
            "image_url": image_url,
        }
    elif item.get("type") == "text":
        return {
            "type": "input_text",
            "text": item.get("text", ""),
        }
    elif item.get("type") == "file":
        file = item.get("file", {})
        return {
            "type": "input_file",
            "file_data": file.get("file_data"),
            "filename": file.get("filename"),
            "file_id": file.get("file_id"),
        }

    # For other items, return as-is
    return item


def _add_dspy_identifier_to_headers(headers: dict[str, Any] | None = None):
    headers = headers or {}
    return {
        "User-Agent": f"DSPy/{dspy.__version__}",
        **headers,
    }
