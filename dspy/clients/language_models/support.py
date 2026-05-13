"""Explicit support declarations for DSPy language model integrations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal

ImagePlacement = Literal["any_message", "latest_user_message_only", "provider_request"]


@dataclass(frozen=True)
class ImageSupport:
    urls: bool = False
    base64: bool = False
    file_ids: bool = False
    paths: bool = False
    media_types: tuple[str, ...] | None = None
    placement: ImagePlacement = "any_message"


@dataclass(frozen=True)
class ToolSupport:
    schemas: bool = False
    calls: bool = False
    results: bool = False
    parallel: bool | None = None
    provider_tools: bool = False


@dataclass(frozen=True)
class LMSupport:
    text: bool = True
    messages: bool = True
    images: ImageSupport | None = None
    audio: bool = False
    files: bool = False
    tools: ToolSupport | None = None
    response_schema: bool = False
    reasoning: bool = False
    prompt_cache: bool = False
    logprobs: bool = False
    multiple_outputs: bool = False
    provider_extensions: bool = True
    streaming: bool = False
    async_streaming: bool = False
    native_async: bool = False
    usage: bool = False
    citations: bool = False
    output_images: bool = False
    output_audio: bool = False
    output_files: bool = False
    refusal: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs: Any) -> "LMSupport":
        return replace(self, **kwargs)

    def report(self, *, format: Literal["text", "json"] = "text") -> str | dict[str, Any]:
        data = self.to_dict()
        if format == "json":
            return data
        if format != "text":
            raise ValueError("format must be 'text' or 'json'.")
        lines = ["DSPy LM support"]
        for key, value in data.items():
            if key == "metadata":
                continue
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if hasattr(value, "__dataclass_fields__"):
                return {key: convert(getattr(value, key)) for key in value.__dataclass_fields__}
            if isinstance(value, tuple):
                return list(value)
            return value

        return {key: convert(getattr(self, key)) for key in self.__dataclass_fields__}
