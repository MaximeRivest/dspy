"""Factory functions for built-in SDK language model integrations."""

from __future__ import annotations

from typing import Any, Literal

from dspy.clients.language_models.sdk import SDKLanguageModel
from dspy.clients.language_models.support import ImageSupport, LMSupport, ToolSupport


def openai_chat_support(*, streaming: bool = False, async_streaming: bool = False) -> LMSupport:
    return LMSupport(
        images=ImageSupport(urls=True, base64=True, file_ids=True, paths=True, placement="any_message"),
        audio=True,
        files=True,
        tools=ToolSupport(schemas=True, calls=True, results=True),
        response_schema=True,
        prompt_cache=True,
        logprobs=True,
        multiple_outputs=True,
        provider_extensions=True,
        streaming=streaming,
        async_streaming=async_streaming,
        output_images=True,
        output_audio=True,
        output_files=True,
        refusal=True,
    )


def openai_responses_support(*, streaming: bool = False, async_streaming: bool = False) -> LMSupport:
    return LMSupport(
        images=ImageSupport(urls=True, base64=True, file_ids=True, paths=True, placement="any_message"),
        files=True,
        tools=ToolSupport(schemas=True, calls=True, results=True),
        response_schema=True,
        reasoning=True,
        prompt_cache=True,
        logprobs=True,
        multiple_outputs=True,
        provider_extensions=True,
        streaming=streaming,
        async_streaming=async_streaming,
        citations=True,
        output_images=True,
        output_audio=True,
        output_files=True,
        refusal=True,
    )


def openai_text_support(*, streaming: bool = False, async_streaming: bool = False) -> LMSupport:
    return LMSupport(
        logprobs=True,
        multiple_outputs=True,
        provider_extensions=True,
        streaming=streaming,
        async_streaming=async_streaming,
    )


def openai_responses_lm(model: str, *, responses, cache: bool = True, callbacks: list[Any] | None = None, **kwargs: Any) -> SDKLanguageModel:
    from dspy.clients.language_models.openai_format import responses_to_lm_response, to_openai_responses_provider_request

    return SDKLanguageModel(
        model=model,
        support=openai_responses_support(),
        map_request=to_openai_responses_provider_request,
        call=lambda provider_request: responses(**provider_request.kwargs),
        map_response=responses_to_lm_response,
        metadata={"provider": "callable", "protocol": "openai_responses"},
        cache=cache,
        callbacks=callbacks,
        **kwargs,
    )


def litellm_chat_lm(model: str, *, cache: bool = True, callbacks: list[Any] | None = None, num_retries: int = 3, **kwargs: Any) -> SDKLanguageModel:
    from dspy.clients.language_models import litellm as litellm_impl
    from dspy.clients.language_models.openai_format import completion_to_lm_response, to_openai_chat_provider_request, to_openai_chat_request

    lm = SDKLanguageModel(
        model=model,
        support=openai_chat_support(streaming=True, async_streaming=True),
        map_request=to_openai_chat_provider_request,
        call=lambda provider_request: litellm_impl.litellm_chat_completion(
            request=provider_request.kwargs,
            num_retries=num_retries,
            cache={"no-cache": True, "no-store": True},
        ),
        map_response=lambda response, request: _with_litellm_cost(completion_to_lm_response(response, request)),
        metadata={"provider": "litellm", "protocol": "openai_chat", "num_retries": num_retries},
        cache=cache,
        callbacks=callbacks,
        **kwargs,
    )
    lm._stream_fn = lambda request: litellm_impl.completion_stream_to_events(
        litellm_impl.litellm_chat_stream(
            request=to_openai_chat_request(request),
            num_retries=num_retries,
            cache={"no-cache": True, "no-store": True},
        ),
        model=request.model,
    )
    return lm


def litellm_responses_lm(model: str, *, cache: bool = True, callbacks: list[Any] | None = None, num_retries: int = 3, **kwargs: Any) -> SDKLanguageModel:
    from dspy.clients.language_models import litellm as litellm_impl
    from dspy.clients.language_models.openai_format import responses_to_lm_response, to_openai_responses_provider_request, to_openai_responses_request

    lm = SDKLanguageModel(
        model=model,
        support=openai_responses_support(streaming=True, async_streaming=True),
        map_request=to_openai_responses_provider_request,
        call=lambda provider_request: litellm_impl.litellm_responses_completion(
            request=provider_request.kwargs,
            num_retries=num_retries,
            cache={"no-cache": True, "no-store": True},
        ),
        map_response=lambda response, request: _with_litellm_cost(responses_to_lm_response(response, request)),
        metadata={"provider": "litellm", "protocol": "openai_responses", "num_retries": num_retries},
        cache=cache,
        callbacks=callbacks,
        **kwargs,
    )
    lm._stream_fn = lambda request: litellm_impl.responses_stream_to_events(
        litellm_impl.litellm_responses_stream(
            request=to_openai_responses_request(request),
            num_retries=num_retries,
            cache={"no-cache": True, "no-store": True},
        ),
        model=request.model,
    )
    return lm


def litellm_text_lm(model: str, *, cache: bool = True, callbacks: list[Any] | None = None, num_retries: int = 3, **kwargs: Any) -> SDKLanguageModel:
    from dspy.clients.language_models import litellm as litellm_impl
    from dspy.clients.language_models.openai_format import completion_to_lm_response, to_openai_text_provider_request, to_openai_text_request

    lm = SDKLanguageModel(
        model=model,
        support=openai_text_support(streaming=True, async_streaming=True),
        map_request=to_openai_text_provider_request,
        call=lambda provider_request: litellm_impl.litellm_text_completion(
            request=provider_request.kwargs,
            num_retries=num_retries,
            cache={"no-cache": True, "no-store": True},
        ),
        map_response=lambda response, request: _with_litellm_cost(completion_to_lm_response(response, request)),
        metadata={"provider": "litellm", "protocol": "openai_text", "num_retries": num_retries},
        cache=cache,
        callbacks=callbacks,
        **kwargs,
    )
    lm._stream_fn = lambda request: litellm_impl.completion_stream_to_events(
        litellm_impl.litellm_text_stream(
            request=to_openai_text_request(request),
            num_retries=num_retries,
            cache={"no-cache": True, "no-store": True},
        ),
        model=request.model,
    )
    return lm


def lm15_lm(
    model: str,
    *,
    provider: Literal["openai", "anthropic", "gemini"] | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    cache: bool = True,
    callbacks: list[Any] | None = None,
    **kwargs: Any,
) -> SDKLanguageModel:
    from dspy.clients.language_models.lm15 import lm15_lm as make_lm15_lm

    return make_lm15_lm(
        model=model,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        cache=cache,
        callbacks=callbacks,
        **kwargs,
    )


def _with_litellm_cost(response):
    from dspy.clients.language_models.litellm import cost_from_response

    response.cost = cost_from_response(response.provider_response)
    return response
