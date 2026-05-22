"""Boundary conversion between normalized LM types and legacy BaseLM calls."""

from __future__ import annotations

from typing import Any

from dspy.clients.openai_format import citation_to_part, provider_tool_call_to_part, to_openai_chat_request
from dspy.core.types import LMOutput, LMRequest, LMResponse, LMTextPart, LMThinkingPart


def legacy_call_kwargs_from_lm_request(request: LMRequest) -> dict[str, Any]:
    """Return keyword arguments for calling a legacy `BaseLM` from an `LMRequest`."""
    data = to_openai_chat_request(request)
    data.pop("model", None)
    if request.config.cache is not None:
        if request.config.cache.enabled is not None:
            data["cache"] = request.config.cache.enabled
        if request.config.cache.rollout_id is not None:
            data["rollout_id"] = request.config.cache.rollout_id
    return data


def lm_response_from_legacy_outputs(outputs: list[dict[str, Any] | str | None], request: LMRequest) -> LMResponse:
    """Normalize legacy `BaseLM` outputs immediately after the LM call."""
    if not outputs:
        return LMResponse(model=request.model, outputs=[LMOutput(parts=[], metadata={"empty_legacy_outputs": True})])
    return LMResponse(model=request.model, outputs=[lm_output_from_legacy_output(output) for output in outputs])


def lm_output_from_legacy_output(output: dict[str, Any] | str | None) -> LMOutput:
    """Normalize one legacy output item into an `LMOutput`."""
    if isinstance(output, str):
        return LMOutput(parts=[LMTextPart(text=output)])
    if output is None:
        return LMOutput(parts=[])

    parts = []
    text = output.get("text")
    if text:
        parts.append(LMTextPart(text=text))
    reasoning = output.get("reasoning_content")
    if reasoning:
        parts.append(LMThinkingPart(text=str(reasoning)))
    for tool_call in output.get("tool_calls") or []:
        parts.append(provider_tool_call_to_part(tool_call))
    for citation in output.get("citations") or []:
        parts.append(citation_to_part(citation))
    return LMOutput(parts=parts, logprobs=output.get("logprobs"))
