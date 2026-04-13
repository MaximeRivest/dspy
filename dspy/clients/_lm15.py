"""lm15 backend for DSPy.

Translates DSPy's OpenAI-format messages to lm15 calls.  Returns
results in the shape that ``BaseLM._process_lm_response`` expects —
a thin namespace with ``.choices[0].message``, ``.usage``, and ``.model``.

This is the **only** module that imports lm15.
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
)
from lm15.errors import ContextWindowError as _LM15ContextWindowError

logger = logging.getLogger(__name__)

# -- Backend protocol export ------------------------------------------------
ContextWindowError = _LM15ContextWindowError

# -- Singleton client -------------------------------------------------------
_client: UniversalLM | None = None


def _get_client() -> UniversalLM:
    global _client
    if _client is None:
        _client = lm15.build_default(use_pycurl=False, hydrate_models_dev_catalog=False)
    return _client


# -- Capability queries -----------------------------------------------------

_NO_TOOLS = {"davinci", "babbage", "ada", "whisper", "tts", "dall-e", "embedding"}

def _family(m: str) -> str:
    return m.split("/")[-1].lower() if "/" in m else m.lower()

def supports_function_calling(m: str) -> bool:
    return not any(x in _family(m) for x in _NO_TOOLS)

def supports_reasoning(m: str) -> bool:
    f = _family(m)
    return bool(re.match(r"^(o[1345]|gpt-5)(-|$)", f)) or "claude" in f

def supports_response_schema(m: str) -> bool:
    f = _family(m)
    return any(x in f for x in ("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4", "claude", "gemini"))

def supported_params(m: str) -> set[str]:
    return {"temperature", "max_tokens", "max_completion_tokens", "top_p", "stop",
            "response_format", "tools", "tool_choice", "reasoning_effort"}


# -- Message conversion: OpenAI dicts → lm15 types -------------------------

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
            m = re.match(r"data:([^;]+);base64,(.+)", url, re.DOTALL)
            parts.append(Part.image(data=m.group(2), media_type=m.group(1)) if m else Part.image(url=url))
        elif t == "input_audio":
            a = item.get("input_audio", {})
            parts.append(Part.audio(data=a.get("data", ""), media_type=f"audio/{a.get('format','wav')}"))
        elif t == "document":
            s = item.get("source", {})
            parts.append(Part.document(data=s.get("data", ""), media_type=s.get("media_type", "text/plain")))
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
    system_parts: list[str] = []
    out: list[LM15Message] = []
    for msg in messages:
        role, content = msg.get("role", "user"), msg.get("content")
        tc, tc_id = msg.get("tool_calls"), msg.get("tool_call_id")

        if role in ("system", "developer"):
            if content:
                system_parts.append(content if isinstance(content, str) else str(content))
            continue

        if role == "assistant":
            parts: list[Part] = list(_content_to_parts(content)) if content else []
            if tc:
                for c in tc:
                    fn = c.get("function", {})
                    args = fn.get("arguments", "{}")
                    try:
                        parsed = json.loads(args) if isinstance(args, str) else args
                    except json.JSONDecodeError:
                        parsed = {"raw": args}
                    parts.append(Part.tool_call(id=c.get("id", ""), name=fn.get("name", ""), input=parsed))
            if parts:
                out.append(LM15Message(role="assistant", parts=tuple(parts)))
            continue

        if role == "tool":
            text = content if isinstance(content, str) else json.dumps(content) if content else ""
            out.append(LM15Message(role="tool", parts=(
                Part.tool_result(id=tc_id or "", content=[Part.text_part(text)]),)))
            continue

        if content is not None:
            out.append(LM15Message(role="user", parts=_content_to_parts(content)))

    return ("\n\n".join(system_parts) if system_parts else None, tuple(out))


def _convert_tools(tools: list[dict]) -> tuple[LM15FunctionTool, ...]:
    out = []
    for t in tools:
        fn = t.get("function", t) if t.get("type") == "function" else t
        out.append(LM15FunctionTool(name=fn.get("name", ""), description=fn.get("description"),
                                     parameters=fn.get("parameters")))
    return tuple(out)


def _build_config(kw: dict) -> Config:
    cfg: dict[str, Any] = {}
    if kw.get("temperature") is not None: cfg["temperature"] = kw["temperature"]
    for k in ("max_tokens", "max_completion_tokens"):
        if kw.get(k) is not None: cfg["max_tokens"] = kw[k]; break
    if kw.get("top_p") is not None: cfg["top_p"] = kw["top_p"]
    if kw.get("stop") is not None:
        s = kw["stop"]; cfg["stop"] = tuple(s) if isinstance(s, list) else (s,)
    if kw.get("response_format") is not None:
        rf = kw["response_format"]
        import pydantic
        if isinstance(rf, type) and issubclass(rf, pydantic.BaseModel):
            cfg["response_format"] = {"type": "json_schema",
                "json_schema": {"name": rf.__name__, "schema": rf.model_json_schema()}}
        elif isinstance(rf, dict):
            cfg["response_format"] = rf
    if kw.get("reasoning_effort") is not None:
        cfg["reasoning"] = ReasoningConfig(effort=kw["reasoning_effort"])
    return Config(**cfg)


# -- Response → thin namespace matching ChatCompletion shape ----------------

class _NS:
    __slots__ = ("__dict__",)
    def __init__(self, d): self.__dict__.update(d)
    def __repr__(self): return repr(self.__dict__)
    def __iter__(self): return iter(self.__dict__)
    def __getitem__(self, k): return self.__dict__[k]
    def get(self, k, d=None): return self.__dict__.get(k, d)


def _to_completion(resp: LMResponse, model: str):
    texts = [p.text for p in resp.message.parts if isinstance(p, TextPart) and p.text]
    content = "\n".join(texts) if texts else None

    thinking = [p.text for p in resp.message.parts if isinstance(p, ThinkingPart) and p.text]

    tc_parts = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    tool_calls = [_NS({"id": tc.id, "type": "function",
        "function": _NS({"name": tc.name,
            "arguments": json.dumps(tc.input) if isinstance(tc.input, dict) else str(tc.input)})})
        for tc in tc_parts] or None

    msg = {"role": "assistant", "content": content}
    if thinking: msg["reasoning_content"] = "\n".join(thinking)
    if tool_calls: msg["tool_calls"] = tool_calls
    cits = resp.citations
    if cits:
        msg["provider_specific_fields"] = {
            "citations": [[{"text": c.text, "url": c.url, "title": c.title} for c in cits]]}

    finish = resp.finish_reason or "stop"
    if finish == "tool_call": finish = "tool_calls"

    return _NS({
        "id": resp.id, "object": "chat.completion", "created": 0,
        "model": resp.model or model,
        "choices": [_NS({"index": 0, "message": _NS(msg), "finish_reason": finish})],
        "usage": _NS({"prompt_tokens": resp.usage.input_tokens,
                       "completion_tokens": resp.usage.output_tokens,
                       "total_tokens": resp.usage.total_tokens}),
    })


# -- Backend protocol entry points -----------------------------------------

_POP = ("tools", "tool_choice", "parallel_tool_calls", "n", "logprobs",
        "top_logprobs", "seed", "frequency_penalty", "presence_penalty",
        "stream", "stream_options", "api_key", "api_base", "base_url",
        "headers", "rollout_id")


def _do_complete(request: dict, model_type: str, num_retries: int):
    request = dict(request)
    model = request.pop("model")
    messages = request.pop("messages", [])

    if model_type == "text":
        prompt = "\n\n".join([m.get("content", "") for m in messages] + ["BEGIN RESPONSE:"])
        messages = [{"role": "user", "content": prompt}]

    tools_raw = request.pop("tools", None)
    for k in _POP: request.pop(k, None)

    system, lm15_msgs = _convert_messages(messages)
    if not lm15_msgs:
        lm15_msgs = (LM15Message(role="user", parts=(Part.text_part(""),)),)

    lm_request = LMRequest(
        model=model, messages=lm15_msgs, system=system,
        tools=_convert_tools(tools_raw) if tools_raw else (),
        config=_build_config(request),
    )

    provider = None
    if "/" in model:
        p = model.split("/", 1)[0]
        provider = {"azure": "openai"}.get(p, p)

    client = _get_client()
    last_err = None
    for attempt in range(num_retries + 1):
        try:
            return _to_completion(client.complete(lm_request, provider=provider), model)
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
    return await asyncio.get_event_loop().run_in_executor(None, _do_complete, request, model_type, num_retries)
