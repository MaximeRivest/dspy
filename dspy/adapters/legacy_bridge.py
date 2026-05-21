"""Compatibility bridge from normalized LM messages to legacy message dicts."""

from __future__ import annotations

from typing import Any

from dspy.clients.language_models.types import (
    LMAudioPart,
    LMBinaryPart,
    LMDocumentPart,
    LMImagePart,
    LMMessage,
    LMPart,
    LMTextPart,
    LMVideoPart,
)


def legacy_messages_from_typed_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert normalized adapter messages to legacy OpenAI-style message dicts."""
    converted = []
    for message in messages:
        if isinstance(message, LMMessage):
            item = {"role": message.role}
            if message.name is not None:
                item["name"] = message.name
            content = list(message.parts)
        else:
            item = dict(message)
            content = item.get("content")
        if isinstance(content, str):
            item["content"] = content
        elif isinstance(content, list):
            if len(content) == 1 and isinstance(content[0], LMTextPart):
                item["content"] = content[0].text
            else:
                item["content"] = [_part_to_legacy_block(part) for part in content]
        else:
            item["content"] = content
        converted.append(item)
    return converted


def _part_to_legacy_block(part: LMPart | Any) -> dict[str, Any]:
    if isinstance(part, LMTextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, LMImagePart):
        source = part.url or part.file_id or part.path
        if source is None and part.data is not None:
            source = f"data:{part.media_type};base64,{part.data}"
        block = {"type": "image_url", "image_url": {"url": source or ""}}
        if part.detail is not None:
            block["image_url"]["detail"] = part.detail
        return block
    if isinstance(part, LMAudioPart):
        return {"type": "input_audio", "input_audio": {"data": part.data, "format": part.media_type.split("/", 1)[-1]}}
    if isinstance(part, LMVideoPart):
        return {"type": "file", "file": _binary_like_part_to_legacy_file(part)}
    if isinstance(part, LMDocumentPart):
        data = {"type": "document"}
        if part.source is not None:
            data["source"] = part.source
        else:
            data["source"] = _media_part_source(part)
            data["media_type"] = part.media_type
        if part.citations:
            data["citations"] = part.citations
        if part.title is not None:
            data["title"] = part.title
        if part.context is not None:
            data["context"] = part.context
        return data
    if isinstance(part, LMBinaryPart):
        return {"type": "file", "file": _binary_like_part_to_legacy_file(part)}
    if hasattr(part, "model_dump"):
        return part.model_dump(exclude_none=True)
    return {"type": "text", "text": str(part)}


def _binary_like_part_to_legacy_file(part: LMBinaryPart | LMVideoPart) -> dict[str, Any]:
    file: dict[str, Any] = {}
    source = _media_part_source(part)
    if source is not None:
        file["file_data"] = source
    if part.file_id is not None:
        file["file_id"] = part.file_id
    filename = getattr(part, "filename", None)
    if filename is not None:
        file["filename"] = filename
    return file


def _media_part_source(part: LMAudioPart | LMBinaryPart | LMDocumentPart | LMImagePart | LMVideoPart) -> str | None:
    if part.data is not None:
        return part.data if part.data.startswith("data:") else f"data:{part.media_type};base64,{part.data}"
    return part.url or part.file_id or part.path
