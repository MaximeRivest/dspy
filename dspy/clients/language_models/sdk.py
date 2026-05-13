"""Function-first SDK language model wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.provider import ProviderRequest
from dspy.clients.language_models.support import ImageSupport, LMSupport, ToolSupport
from dspy.clients.language_models.types import (
    LMImagePart,
    LMOutput,
    LMRequest,
    LMResponse,
    LMStreamEvent,
    LMTextPart,
    LMToolResultPart,
    LMToolSpec,
    LMUsage,
)


class SDKLanguageModel(LanguageModel):
    """Wrap an arbitrary SDK using small mapping and extraction functions."""

    def __init__(
        self,
        model: str,
        *,
        call: Callable[[ProviderRequest], Any],
        map_request: Callable[[LMRequest], ProviderRequest] | None = None,
        map_response: Callable[[Any, LMRequest], LMResponse] | None = None,
        extract_text: Callable[[Any], str] | None = None,
        support: LMSupport | None = None,
        metadata: dict[str, Any] | None = None,
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        super().__init__(model=model, cache=cache, callbacks=callbacks, **kwargs)
        self._call = call
        self._map_request_fn = map_request
        self._map_response_fn = map_response
        self._extract_text = extract_text
        self.support = support or LMSupport()
        self.metadata = metadata or {}
        self._map_image: Callable[[LMImagePart], Any] | None = None
        self._place_images: Callable[[LMRequest, list[Any], ProviderRequest], ProviderRequest] | None = None
        self._extract_usage: Callable[[Any], LMUsage | dict[str, Any] | None] | None = None
        self._normalize_error_fn: Callable[[Exception, LMRequest], Exception] | None = None
        self._stream_fn: Callable[[LMRequest], Iterator[LMStreamEvent]] | None = None
        self._map_tool: Callable[[LMToolSpec], Any] | None = None
        self._map_tool_result: Callable[[LMToolResultPart], Any] | None = None
        self._extract_tool_calls: Callable[[Any], list[Any]] | None = None

    @classmethod
    def from_text(
        cls,
        *,
        model: str,
        generate: Callable[[LMRequest], Any],
        extract_text: Callable[[Any], str],
        **kwargs: Any,
    ) -> "SDKLanguageModel":
        return cls(model=model, call=lambda provider_request: generate(provider_request.data), extract_text=extract_text, **kwargs)

    @classmethod
    def from_messages(
        cls,
        *,
        model: str,
        generate: Callable[[LMRequest], Any],
        extract_text: Callable[[Any], str],
        **kwargs: Any,
    ) -> "SDKLanguageModel":
        return cls(model=model, call=lambda provider_request: generate(provider_request.data), extract_text=extract_text, **kwargs)

    def map_request(self, request: LMRequest) -> ProviderRequest:
        if self._map_request_fn is not None:
            provider_request = self._map_request_fn(request)
        else:
            provider_request = ProviderRequest(args=(_request_text_prompt(request),), data=request)
        if self._map_image is not None and request_has_images(request):
            images = [self._map_image(part) for part in iter_image_parts(request)]
            if self._place_images is not None:
                provider_request = _call_place_images(self._place_images, request, images, provider_request)
        return provider_request

    def call_provider(self, provider_request: ProviderRequest) -> Any:
        return self._call(provider_request)

    def map_response(self, provider_response: Any, request: LMRequest) -> LMResponse:
        if self._map_response_fn is not None:
            response = self._map_response_fn(provider_response, request)
        elif self._extract_text is not None:
            response = LMResponse.from_text(self._extract_text(provider_response), model=request.model)
        else:
            raise NotImplementedError("Pass map_response=... or extract_text=... to LM.from_sdk().")
        if self._extract_usage is not None and response.usage is None:
            response.usage = self._extract_usage(provider_response)
        if response.provider_response is None:
            response.provider_response = provider_response
        return response

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        if self._stream_fn is None:
            return super().forward_stream(request)
        yield from self._stream_fn(request)

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        if self._normalize_error_fn is not None:
            return self._normalize_error_fn(error, request)
        return error

    def with_images(
        self,
        *,
        map_image: Callable[[LMImagePart], Any],
        place_images: Callable[[LMRequest, list[Any], ProviderRequest], ProviderRequest] | None = None,
        support: ImageSupport | None = None,
    ) -> "SDKLanguageModel":
        new = self.copy()
        new._map_image = map_image
        new._place_images = place_images
        new.support = new.support.with_updates(images=support or ImageSupport(urls=True, base64=True))
        return new

    def with_usage(self, *, extract_usage: Callable[[Any], LMUsage | dict[str, Any] | None]) -> "SDKLanguageModel":
        new = self.copy()
        new._extract_usage = extract_usage
        new.support = new.support.with_updates(usage=True)
        return new

    def with_streaming(self, *, stream: Callable[[LMRequest], Iterator[LMStreamEvent]]) -> "SDKLanguageModel":
        new = self.copy()
        new._stream_fn = stream
        new.support = new.support.with_updates(streaming=True)
        return new

    def with_errors(self, *, normalize: Callable[[Exception, LMRequest], Exception]) -> "SDKLanguageModel":
        new = self.copy()
        new._normalize_error_fn = normalize
        return new

    def with_request_mapper(self, map_request: Callable[[LMRequest], ProviderRequest], *, support: LMSupport | None = None) -> "SDKLanguageModel":
        new = self.copy()
        new._map_request_fn = map_request
        if support is not None:
            new.support = support
        return new

    def with_citations(self, *, support: bool = True) -> "SDKLanguageModel":
        new = self.copy()
        new.support = new.support.with_updates(citations=support)
        return new

    def with_reasoning(self, *, support: bool = True) -> "SDKLanguageModel":
        new = self.copy()
        new.support = new.support.with_updates(reasoning=support)
        return new

    def with_tools(
        self,
        *,
        map_tool: Callable[[LMToolSpec], Any] | None = None,
        extract_tool_calls: Callable[[Any], list[Any]] | None = None,
        map_tool_result: Callable[[LMToolResultPart], Any] | None = None,
        support: ToolSupport | None = None,
    ) -> "SDKLanguageModel":
        new = self.copy()
        new._map_tool = map_tool
        new._extract_tool_calls = extract_tool_calls
        new._map_tool_result = map_tool_result
        new.support = new.support.with_updates(
            tools=support or ToolSupport(schemas=map_tool is not None, calls=extract_tool_calls is not None, results=map_tool_result is not None)
        )
        return new

    def copy(self, **overrides: Any):
        new = super().copy(**overrides)
        # SDKLanguageModel intentionally shares mapper/extractor functions and
        # SDK client closures with the copy. Only DSPy-owned runtime state is
        # isolated by LanguageModel.copy().
        return new


def _call_place_images(place_images: Callable[..., ProviderRequest], request: LMRequest, images: list[Any], provider_request: ProviderRequest) -> ProviderRequest:
    try:
        return place_images(request, images, provider_request)
    except TypeError:
        return place_images(request, images)


def iter_image_parts(request: LMRequest):
    for message in request.messages:
        for part in message.parts:
            if isinstance(part, LMImagePart):
                yield part


def request_has_images(request: LMRequest) -> bool:
    return any(True for _ in iter_image_parts(request))


def _request_text_prompt(request: LMRequest) -> str:
    chunks = []
    for message in request.messages:
        text = message.text
        if text:
            chunks.append(text)
    return "\n\n".join(chunks)
