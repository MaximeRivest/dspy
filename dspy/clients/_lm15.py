"""lm15 backend for DSPy.

Translates DSPy's OpenAI-format messages to lm15 calls and extracts
outputs.  This is the **only** module that imports lm15.
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
    Config, LMRequest, LMResponse,
    Message as LM15Message, Part,
    TextPart, ThinkingPart, ToolCallPart,
    ReasoningConfig,
)
from lm15.errors import ContextWindowError as _LM15CWE

logger = logging.getLogger(__name__)

ContextWindowError = _LM15CWE

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

def supports_function_calling(m): return not any(x in _family(m) for x in _NO_TOOLS)
def supports_reasoning(m):
    f = _family(m); return bool(re.match(r"^(o[1345]|gpt-5)(-|$)", f)) or "claude" in f
def supports_response_schema(m):
    return any(x in _family(m) for x in ("gpt-4o","gpt-4.1","gpt-5","o1","o3","o4","claude","gemini"))
def supported_params(m):
    return {"temperature","max_tokens","max_completion_tokens","top_p","stop",
            "response_format","tools","tool_choice","reasoning_effort"}

# -- Message conversion: OpenAI dicts → lm15 --------------------------------

def _content_to_parts(content) -> tuple[Part, ...]:
    if isinstance(content, str):
        return (Part.text_part(content),)
    parts = []
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
            parts.append(Part.audio(data=a.get("data",""), media_type=f"audio/{a.get('format','wav')}"))
        elif t == "document":
            s = item.get("source", {})
            parts.append(Part.document(data=s.get("data",""), media_type=s.get("media_type","text/plain")))
        elif t == "file":
            fi = item.get("file", {})
            if fi.get("file_data"): parts.append(Part.document(data=fi["file_data"]))
            elif fi.get("file_id"): parts.append(Part.document(file_id=fi["file_id"]))
        else:
            parts.append(Part.text_part(str(item)))
    return tuple(parts) if parts else (Part.text_part(""),)


def messages_to_lm15(messages: list[dict], extra_parts: list | None = None):
    """Convert OpenAI dict messages → (system_str, lm15_messages).

    ``extra_parts`` are appended to the last user message (multimodal).
    """
    system_parts, out = [], []
    for msg in messages:
        role, content = msg.get("role","user"), msg.get("content")
        tc, tc_id = msg.get("tool_calls"), msg.get("tool_call_id")
        if role in ("system","developer"):
            if content: system_parts.append(content if isinstance(content,str) else str(content))
            continue
        if role == "assistant":
            parts = list(_content_to_parts(content)) if content else []
            if tc:
                for c in tc:
                    fn = c.get("function",{})
                    args = fn.get("arguments","{}")
                    try: parsed = json.loads(args) if isinstance(args,str) else args
                    except json.JSONDecodeError: parsed = {"raw": args}
                    parts.append(Part.tool_call(id=c.get("id",""), name=fn.get("name",""), input=parsed))
            if parts: out.append(LM15Message(role="assistant", parts=tuple(parts)))
            continue
        if role == "tool":
            text = content if isinstance(content,str) else json.dumps(content) if content else ""
            out.append(LM15Message(role="tool", parts=(
                Part.tool_result(id=tc_id or "", content=[Part.text_part(text)]),)))
            continue
        if content is not None:
            out.append(LM15Message(role="user", parts=_content_to_parts(content)))

    # Attach extra multimodal parts to last user message
    if extra_parts and out:
        for i in range(len(out)-1, -1, -1):
            if out[i].role == "user":
                out[i] = LM15Message(role="user", parts=out[i].parts + tuple(extra_parts))
                break

    system = "\n\n".join(system_parts) if system_parts else None
    return system, tuple(out)


def _convert_tools(tools):
    out = []
    for t in tools:
        fn = t.get("function",t) if t.get("type")=="function" else t
        out.append(LM15FunctionTool(name=fn.get("name",""), description=fn.get("description"),
                                     parameters=fn.get("parameters")))
    return tuple(out)


def _build_config(kw):
    cfg = {}
    if kw.get("temperature") is not None: cfg["temperature"] = kw["temperature"]
    for k in ("max_tokens","max_completion_tokens"):
        if kw.get(k) is not None: cfg["max_tokens"] = kw[k]; break
    if kw.get("top_p") is not None: cfg["top_p"] = kw["top_p"]
    if kw.get("stop") is not None:
        s = kw["stop"]; cfg["stop"] = tuple(s) if isinstance(s,list) else (s,)
    if kw.get("response_format") is not None:
        rf = kw["response_format"]
        import pydantic
        if isinstance(rf, type) and issubclass(rf, pydantic.BaseModel):
            cfg["response_format"] = {"type":"json_schema",
                "json_schema":{"name":rf.__name__,"schema":rf.model_json_schema()}}
        elif isinstance(rf, dict): cfg["response_format"] = rf
    if kw.get("reasoning_effort") is not None:
        cfg["reasoning"] = ReasoningConfig(effort=kw["reasoning_effort"])
    return Config(**cfg)


# -- Extract outputs from LMResponse ----------------------------------------

def extract_outputs(resp: LMResponse) -> list[dict[str, Any] | str]:
    """Convert an LMResponse to the list DSPy adapters expect."""
    out: dict[str, Any] = {}
    texts = [p.text for p in resp.message.parts if isinstance(p, TextPart) and p.text]
    out["text"] = "\n".join(texts) if texts else None

    thinking = [p.text for p in resp.message.parts if isinstance(p, ThinkingPart) and p.text]
    if thinking:
        out["reasoning_content"] = "\n".join(thinking)

    tc = [p for p in resp.message.parts if isinstance(p, ToolCallPart)]
    if tc:
        # Match the shape BaseLM._extract_outputs produces for tool_calls
        class _TC:
            def __init__(self, p):
                self.id = p.id; self.type = "function"
                self.function = type("F", (), {
                    "name": p.name,
                    "arguments": json.dumps(p.input) if isinstance(p.input,dict) else str(p.input)
                })()
        out["tool_calls"] = [_TC(p) for p in tc]

    cits = resp.citations
    if cits:
        out["citations"] = [{"text":c.text,"url":c.url,"title":c.title} for c in cits]

    if len(out) == 1 and "text" in out:
        return [out["text"]]
    return [out]


# -- Main completion function -----------------------------------------------

_POP = ("tools","tool_choice","parallel_tool_calls","n","logprobs",
        "top_logprobs","seed","frequency_penalty","presence_penalty",
        "stream","stream_options","api_key","api_base","base_url",
        "headers","rollout_id","_multimodal_parts")


def complete(
    model: str,
    messages: list[dict],
    num_retries: int = 3,
    multimodal_parts: list | None = None,
    **kwargs,
) -> tuple[LMResponse, list[dict[str, Any] | str]]:
    """Call lm15 and return (raw_response, extracted_outputs)."""
    kw = dict(kwargs)
    tools_raw = kw.pop("tools", None)
    for k in _POP:
        kw.pop(k, None)

    # Text completion mode
    model_type = kw.pop("_model_type", "chat")
    if model_type == "text":
        prompt = "\n\n".join([m.get("content","") for m in messages] + ["BEGIN RESPONSE:"])
        messages = [{"role":"user","content":prompt}]

    system, lm15_msgs = messages_to_lm15(messages, extra_parts=multimodal_parts)
    if not lm15_msgs:
        lm15_msgs = (LM15Message(role="user", parts=(Part.text_part(""),)),)

    lm_request = LMRequest(
        model=model, messages=lm15_msgs, system=system,
        tools=_convert_tools(tools_raw) if tools_raw else (),
        config=_build_config(kw),
    )

    provider = None
    if "/" in model:
        p = model.split("/",1)[0]
        provider = {"azure":"openai"}.get(p, p)

    client = _get_client()
    last_err = None
    for attempt in range(num_retries + 1):
        try:
            resp = client.complete(lm_request, provider=provider)
            return resp, extract_outputs(resp)
        except _LM15CWE:
            raise
        except Exception as e:
            last_err = e
            if attempt < num_retries:
                time.sleep(2 ** attempt)
    raise last_err
