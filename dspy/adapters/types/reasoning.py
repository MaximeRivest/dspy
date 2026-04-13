"""Reasoning type for DSPy — wraps lm15's native extended thinking."""

from typing import Any, Optional

import pydantic

from dspy.adapters.types.base_type import Type


class Reasoning(Type):
    """str-like type that captures LM reasoning. lm15 surfaces this via resp.thinking."""

    content: str

    def format(self):
        return self.content

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_input(cls, data: Any):
        if isinstance(data, cls): return data
        if isinstance(data, str): return {"content": data}
        if isinstance(data, dict) and "content" in data: return data
        raise ValueError(f"Invalid Reasoning: {data}")

    @classmethod
    def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
        effort = lm_kwargs.get("reasoning_effort") or lm.kwargs.get("reasoning_effort") or "low"
        if effort is None or not lm.supports_reasoning:
            return signature
        lm_kwargs["reasoning_effort"] = effort
        return signature.delete(field_name)

    @classmethod
    def parse_lm_response(cls, response) -> Optional["Reasoning"]:
        if isinstance(response, dict) and "reasoning_content" in response:
            return Reasoning(content=response["reasoning_content"])
        return None

    @classmethod
    def parse_stream_chunk(cls, chunk):
        try: return getattr(chunk.choices[0].delta, "reasoning_content", None)
        except Exception: return None

    @classmethod
    def is_streamable(cls): return True

    def __repr__(self): return repr(self.content)
    def __str__(self): return self.content
    def __eq__(self, o): return self.content == (o.content if isinstance(o, Reasoning) else o)
    def __len__(self): return len(self.content)
    def __getitem__(self, k): return self.content[k]
    def __contains__(self, x): return x in self.content
    def __iter__(self): return iter(self.content)
    def __add__(self, o):
        if isinstance(o, Reasoning): return Reasoning(content=self.content + o.content)
        if isinstance(o, str): return self.content + o
        return NotImplemented
    def __radd__(self, o):
        if isinstance(o, str): return o + self.content
        return NotImplemented
    def __getattr__(self, name):
        if hasattr(str, name): return getattr(self.content, name)
        raise AttributeError(f"Reasoning has no attribute '{name}'")
