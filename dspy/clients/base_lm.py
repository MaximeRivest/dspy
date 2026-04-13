"""Base language model stub — kept for isinstance checks and custom subclasses.

``dspy.LM`` inherits this but calls lm15 directly, bypassing forward().
Custom subclasses can still override ``forward()`` to return a
ChatCompletion-shaped object.
"""

import datetime
import uuid
from typing import Any, TextIO

from dspy.dsp.utils import settings
from dspy.utils.callback import with_callbacks
from dspy.utils.inspect_history import pretty_print_history

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []


class BaseLM:
    """Minimal base for DSPy language models."""

    def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    @property
    def supports_function_calling(self): return False
    @property
    def supports_reasoning(self): return False
    @property
    def supports_response_schema(self): return False
    @property
    def supported_params(self): return set()

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs):
        resp = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process(resp, prompt, messages, **kwargs)

    @with_callbacks
    async def acall(self, prompt=None, messages=None, **kwargs):
        resp = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process(resp, prompt, messages, **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):
        raise NotImplementedError

    async def aforward(self, prompt=None, messages=None, **kwargs):
        raise NotImplementedError

    def _process(self, response, prompt, messages, **kwargs):
        """Extract outputs from ChatCompletion-shaped response + log history."""
        outputs = []
        for c in response.choices:
            o = {"text": c.message.content if hasattr(c, "message") else c["text"]}
            if hasattr(c, "message"):
                if getattr(c.message, "reasoning_content", None):
                    o["reasoning_content"] = c.message.reasoning_content
                if getattr(c.message, "tool_calls", None):
                    o["tool_calls"] = c.message.tool_calls
            outputs.append(o)
        if all(len(o) == 1 for o in outputs):
            outputs = [o["text"] for o in outputs]

        if not settings.disable_history:
            _update_history(self, {
                "prompt": prompt, "messages": messages,
                "kwargs": {k: v for k, v in kwargs.items() if not k.startswith("api_")},
                "response": response, "outputs": outputs,
                "usage": dict(response.usage) if hasattr(response, "usage") else {},
                "timestamp": datetime.datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model": self.model, "response_model": getattr(response, "model", self.model),
                "model_type": self.model_type,
            })
        return outputs

    def copy(self, **kwargs):
        import copy as _c
        new = _c.deepcopy(self)
        new.history = []
        for k, v in kwargs.items():
            if hasattr(self, k): setattr(new, k, v)
            if (k in self.kwargs) or (not hasattr(self, k)):
                if v is None: new.kwargs.pop(k, None)
                else: new.kwargs[k] = v
        if hasattr(new, "_warned_zero_temp_rollout"):
            new._warned_zero_temp_rollout = False
        return new

    def inspect_history(self, n=1, file=None):
        pretty_print_history(self.history, n, file=file)

    def dump_state(self):
        return {}


def _update_history(lm, entry):
    if settings.disable_history:
        return
    if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
        GLOBAL_HISTORY.pop(0)
    GLOBAL_HISTORY.append(entry)
    if settings.max_history_size == 0:
        return
    if len(lm.history) >= settings.max_history_size:
        lm.history.pop(0)
    lm.history.append(entry)
    for mod in (settings.caller_modules or []):
        if len(mod.history) >= settings.max_history_size:
            mod.history.pop(0)
        mod.history.append(entry)


def inspect_history(n=1, file=None):
    pretty_print_history(GLOBAL_HISTORY, n, file=file)
