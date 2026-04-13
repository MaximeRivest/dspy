"""Base language model interface.

``BaseLM`` is a minimal stub kept for backward compatibility —
type-hints, ``isinstance`` checks, custom subclasses, and
``inspect_history`` all still work.  All real logic lives in
``dspy.LM``, which calls lm15 directly.
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
    """Minimal base class for DSPy language models.

    Most users should use ``dspy.LM`` directly. If you need a fully
    custom provider, subclass ``BaseLM`` and implement ``forward()``.
    """

    def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    # -- Capability queries (overridden by LM) ----------------------------

    @property
    def supports_function_calling(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supported_params(self) -> set[str]:
        return set()

    # -- Call interface ----------------------------------------------------

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs) -> list[dict[str, Any] | str]:
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        return self._process_lm_response(response, prompt, messages, **kwargs)

    @with_callbacks
    async def acall(self, prompt=None, messages=None, **kwargs) -> list[dict[str, Any] | str]:
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process_lm_response(response, prompt, messages, **kwargs)

    def forward(self, prompt=None, messages=None, **kwargs):
        raise NotImplementedError

    async def aforward(self, prompt=None, messages=None, **kwargs):
        raise NotImplementedError

    # -- Response processing (legacy — used by custom BaseLM subclasses) ---

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        """Extract outputs from an OpenAI-shaped response and log history."""
        merged = {**self.kwargs, **kwargs}
        outputs = self._extract_outputs(response, merged)

        if not settings.disable_history:
            safe_kw = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
            entry = {
                "prompt": prompt, "messages": messages, "kwargs": safe_kw,
                "response": response, "outputs": outputs,
                "usage": dict(response.usage) if hasattr(response, "usage") else {},
                "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
                "timestamp": datetime.datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model": self.model,
                "response_model": getattr(response, "model", self.model),
                "model_type": self.model_type,
            }
            _update_history(self, entry)
        return outputs

    def _extract_outputs(self, response, merged_kwargs):
        """Extract text/tool_calls/reasoning from a ChatCompletion-shaped response."""
        outputs = []
        for c in response.choices:
            out = {}
            out["text"] = c.message.content if hasattr(c, "message") else c["text"]
            if hasattr(c, "message"):
                if getattr(c.message, "reasoning_content", None):
                    out["reasoning_content"] = c.message.reasoning_content
                if getattr(c.message, "tool_calls", None):
                    out["tool_calls"] = c.message.tool_calls
                try:
                    cits = c.message.provider_specific_fields.get("citations")
                    if isinstance(cits, list):
                        out["citations"] = [ci for group in cits for ci in group]
                except Exception:
                    pass
            if merged_kwargs.get("logprobs") and hasattr(c, "logprobs"):
                out["logprobs"] = c.logprobs
            outputs.append(out)
        if all(len(o) == 1 for o in outputs):
            outputs = [o["text"] for o in outputs]
        return outputs

    # -- Copy, history, inspection ----------------------------------------

    def copy(self, **kwargs):
        import copy as _copy
        new = _copy.deepcopy(self)
        new.history = []
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(new, k, v)
            if (k in self.kwargs) or (not hasattr(self, k)):
                if v is None:
                    new.kwargs.pop(k, None)
                else:
                    new.kwargs[k] = v
        if hasattr(new, "_warned_zero_temp_rollout"):
            new._warned_zero_temp_rollout = False
        return new

    def inspect_history(self, n: int = 1, file: "TextIO | None" = None):
        pretty_print_history(self.history, n, file=file)

    def dump_state(self):
        return {}


# -- History management (shared) -------------------------------------------

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
    for module in (settings.caller_modules or []):
        if len(module.history) >= settings.max_history_size:
            module.history.pop(0)
        module.history.append(entry)


def inspect_history(n: int = 1, file: "TextIO | None" = None):
    """Display the global history shared across all LMs."""
    pretty_print_history(GLOBAL_HISTORY, n, file=file)
