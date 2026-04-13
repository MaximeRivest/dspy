"""lm15 backend for DSPy.

Translates DSPy's OpenAI-format requests to lm15 calls and converts
lm15 responses back.  This is the **only** module that imports lm15.
All provider routing, retries, streaming, multimodal handling, and
cost tracking are delegated to lm15.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import lm15
from lm15 import UniversalLM, FunctionTool as LM15FunctionTool
from lm15.types import (
    Config,
    LMRequest,
    LMResponse,
    Message as LM15Message,
    Part,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ReasoningConfig,
    Usage as LM15Usage,
)
from lm15.errors import ContextWindowError as _LM15ContextWindowError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend protocol exports
# ---------------------------------------------------------------------------

ContextWindowError = _LM15ContextWindowError

# ---------------------------------------------------------------------------
# Singleton client
# ---------------------------------------------------------------------------

_client: UniversalLM | None = None


def _get_client() -> UniversalLM:
    global _client
    if _client is None:
        _client = lm15.build_default(use_pycurl=False, hydrate_models_dev_catalog=False)
    return _client


# ---------------------------------------------------------------------------
# Capability queries
# ---------------------------------------------------------------------------

_NO_TOOLS = {"davinci", "babbage", "ada", "whisper", "tts", "dall-e", "embedding"}


def _family(model: str) -> str:
    return model.split("/")[-1].lower() if "/" in model else model.lower()


def supports_function_calling(model: str) -> bool:
    return not any(x in _family(model) for x in _NO_TOOLS)


def supports_reasoning(model: str) -> bool:
    f = _family(model)
    return bool(re.match(r"^(o[1345]|gpt-5)(-|$)", f)) or "claude" in f


def supports_response_schema(model: str) -> bool:
    f = _family(model)
    return any(x in f for x in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4", "claude", "gemini"))


def supported_params(model: str) -> set[str]:
    return {
        "temperature", "max_tokens", "max_completion_tokens", "top_p",
        "frequency_penalty", "presence_penalty", "stop", "n",
        "response_format", "tools", "tool_choice", "reasoning_effort",
    }


# ---------------------------------------------------------------------------
# Message conversion: OpenAI dict format → lm15 types
# ---------------------------------------------------------------------------


def _content_to_parts(content: str | list[dict]) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (Part.text_part(content),)
    parts: list[Part] = []
    for item in content:
        t = item.get("type", "text")
        if t == "text":
            parts.append(Part.text_part(item.get("text", "")))
        elif t == "image_url":
            url = item.get("image_url", {}).get("url", "")
            m = re.match(r"data:([^;]+);base64,(.+)", url)
            parts.append(Part.image(data=m.group(2), media_type=m.group(1)) if m else Part.image(url=url))
        elif t == "input_audio":
            audio = item.get("input_audio", {})
            parts.append(Part.audio(data=audio.get("data", ""), media_type=f"audio/{audio.get('format', 'wav')}"))
        elif t == "document":
            src = item.get("source", {})
            parts.append(Part.document(data=src.get("data", ""), media_type=src.get("media_type", "text/plain")))
        elif t == "file":
            fi = item.get("file", {})
            if fi.get("file_data"):
                parts.append(Part.document(data=fi["file_data"]))
            elif fi.get("file_id"):
                parts.append(Part.document(file_id=fi["file_id"]))
        else:
            parts.append(Part.text_part(str(item)))
    return tuple(parts) if parts else (Part.text_part(""),)


def _convert_messages(messages: list[dict]) -> tuple[str | None, tuple[LM15Message, ...]]:
    """Extract system prompt and convert messages to lm15 format."""
    system_parts: list[str] = []
    out: list[LM15Message] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")
        tool_call_id = msg.get("tool_call_id")

        if role in ("system", "developer"):
            if content:
                system_parts.append(content if isinstance(content, str) else str(content))
            continue

        if role == "assistant":
            parts: list[Part] = []
            if content:
                parts.extend(_content_to_parts(content))
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        parsed = {"raw": args}
                    parts.append(Part.tool_call(id=tc.get("id", ""), name=func.get("name", ""), input=parsed))
            if parts:
                out.append(LM15Message(role="assistant", parts=tuple(parts)))
            continue

        if role == "tool":
            text = content if isinstance(content, str) else json.dumps(content) if content else ""
            out.append(LM15Message(
                role="tool",
                parts=(Part.tool_result(id=tool_call_id or "", content=[Part.text_part(text)]),),
            ))
            continue

        # user
        if content is not None:
            out.append(LM15Message(role="user", parts=_content_to_parts(content)))

    system = "\n\n".join(system_parts) if system_parts else None
    return system, tuple(out)


def _convert_tools(tools: list[dict]) -> tuple[LM15FunctionTool, ...]:
    out: list[LM15FunctionTool] = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
        else:
            func = tool
        out.append(LM15FunctionTool(
            name=func.get("name", ""),
            description=func.get("description"),
            parameters=func.get("parameters"),
        ))
    return tuple(out)


def _build_config(kwargs: dict) -> Config:
    cfg: dict[str, Any] = {}
    if kwargs.get("temperature") is not None:
        cfg["temperature"] = kwargs["temperature"]
    for key in ("max_tokens", "max_completion_tokens"):
        if kwargs.get(key) is not None:
            cfg["max_tokens"] = kwargs[key]
            break
    if kwargs.get("top_p") is not None:
        cfg["top_p"] = kwargs["top_p"]
    if kwargs.get("stop") is not None:
        s = kwargs["stop"]
        cfg["stop"] = tuple(s) if isinstance(s, list) else (s,)
    if kwargs.get("response_format") is not None:
        rf = kwargs["response_format"]
        import pydantic
        if isinstance(rf, type) and issubclass(rf, pydantic.BaseModel):
            cfg["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": rf.__name__, "schema": rf.model_json_schema()},
            }
        elif isinstance(rf, dict):
            cfg["response_format"] = rf
    if kwargs.get("reasoning_effort") is not None:
        cfg["reasoning"] = ReasoningConfig(effort=kwargs["reasoning_effort"])
    return Config(**cfg)


# ---------------------------------------------------------------------------
# Response conversion: lm15 → OpenAI ChatCompletion-shaped object
# ---------------------------------------------------------------------------


class _NS:
    """Tiny namespace for attribute access on dicts."""
    def __init__(self, d):
        self.__dict__.update(d)
    def __repr__(self):
        return repr(self.__dict__)
    def __iter__(self):
        return iter(self.__dict__)
    def __getitem__(self, k):
        return self.__dict__[k]
    def get(self, k, d=None):
        return self.__dict__.get(k, d)


def _to_chat_completion(resp: LMResponse, model: str) -> Any:
    """Convert lm15 LMResponse → OpenAI ChatCompletion shaped object."""
    texts = [p.text for p in resp.message.parts if isinstance(p, TextPart) and p.text]
    content = "\n".join(texts) if texts else None

    thinking = [p.text for p in resp.message.parts if isinstance(p, ThinkingPart) and p.text]
    reasoning_content = "\n".join(thinking) if thinking else None

    tc_parts = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    tool_calls = [
        _NS({"id": tc.id, "type": "function", "function": _NS({
            "name": tc.name,
            "arguments": json.dumps(tc.input) if isinstance(tc.input, dict) else str(tc.input),
        })})
        for tc in tc_parts
    ] or None

    msg = {"role": "assistant", "content": content}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
    if tool_calls:
        msg["tool_calls"] = tool_calls

    citations = resp.citations
    if citations:
        msg["provider_specific_fields"] = {
            "citations": [[{"text": c.text, "url": c.url, "title": c.title} for c in citations]]
        }

    finish = resp.finish_reason or "stop"
    if finish == "tool_call":
        finish = "tool_calls"

    return _NS({
        "id": resp.id,
        "object": "chat.completion",
        "created": 0,
        "model": resp.model or model,
        "choices": [_NS({"index": 0, "message": _NS(msg), "finish_reason": finish})],
        "usage": _NS({
            "prompt_tokens": resp.usage.input_tokens,
            "completion_tokens": resp.usage.output_tokens,
            "total_tokens": resp.usage.total_tokens,
        }),
    })


# ---------------------------------------------------------------------------
# Backend protocol entry points
# ---------------------------------------------------------------------------

# Keys to pop from the request dict before building lm15 Config
_POP_KEYS = (
    "tools", "tool_choice", "parallel_tool_calls",
    "n", "logprobs", "top_logprobs", "seed",
    "frequency_penalty", "presence_penalty",
    "stream", "stream_options",
    "api_key", "api_base", "base_url", "headers", "rollout_id",
)


def _do_complete(request: dict, model_type: str, num_retries: int) -> Any:
    request = dict(request)
    model = request.pop("model")
    messages = request.pop("messages", [])

    # Text completions: flatten messages into a single prompt
    if model_type == "text":
        prompt = "\n\n".join([m.get("content", "") for m in messages] + ["BEGIN RESPONSE:"])
        messages = [{"role": "user", "content": prompt}]

    tools_raw = request.pop("tools", None)
    for k in _POP_KEYS:
        request.pop(k, None)

    system, lm15_msgs = _convert_messages(messages)
    if not lm15_msgs:
        lm15_msgs = (LM15Message(role="user", parts=(Part.text_part(""),)),)

    lm15_tools = _convert_tools(tools_raw) if tools_raw else ()
    config = _build_config(request)

    lm_request = LMRequest(model=model, messages=lm15_msgs, system=system, tools=lm15_tools, config=config)

    provider = None
    if "/" in model:
        p = model.split("/", 1)[0]
        provider = {"azure": "openai"}.get(p, p)

    client = _get_client()
    last_err = None
    for attempt in range(num_retries + 1):
        try:
            resp = client.complete(lm_request, provider=provider)
            return _to_chat_completion(resp, model)
        except _LM15ContextWindowError:
            raise
        except Exception as e:
            last_err = e
            if attempt < num_retries:
                time.sleep(2 ** attempt)
    raise last_err


def complete_request(request: dict, model_type: str, num_retries: int):
    return _do_complete(request, model_type, num_retries)


async def acomplete_request(request: dict, model_type: str, num_retries: int):
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _do_complete, request, model_type, num_retries)
