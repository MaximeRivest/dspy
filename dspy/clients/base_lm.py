"""Base language model class.

``BaseLM`` defines the interface that all DSPy language models implement.
The default ``dspy.LM`` subclass routes calls through lm15; custom
subclasses can override ``forward()`` directly.
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
    """Base class for handling LLM calls.

    Most users should use ``dspy.LM`` directly.  Override ``forward()`` to
    plug in a custom provider — the return value must be shaped like an
    `OpenAI ChatCompletion <https://platform.openai.com/docs/api-reference/chat/object>`_.

    Examples:

    ```python
    import dspy

    class MyLM(dspy.BaseLM):
        def forward(self, prompt, messages=None, **kwargs):
            import lm15
            resp = lm15.call(self.model, prompt)
            # Build a ChatCompletion-shaped return value
            ...
    ```
    """

    def __init__(self, model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self.history = []

    # ------------------------------------------------------------------
    # Capability queries — overridden by LM to delegate to lm15
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Response processing
    # ------------------------------------------------------------------

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}

        if self.model_type == "responses":
            outputs = self._process_response(response)
        else:
            outputs = self._process_completion(response, merged_kwargs)

        if settings.disable_history:
            return outputs

        safe_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": safe_kwargs,
            "response": response,
            "outputs": outputs,
            "usage": dict(response.usage),
            "cost": getattr(response, "_hidden_params", {}).get("response_cost"),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "response_model": response.model,
            "model_type": self.model_type,
        }
        self.update_history(entry)
        return outputs

    def _process_completion(self, response, merged_kwargs):
        """Extract outputs from a ChatCompletion-shaped response."""
        outputs = []
        for c in response.choices:
            output = {}
            output["text"] = c.message.content if hasattr(c, "message") else c["text"]

            if hasattr(c, "message") and hasattr(c.message, "reasoning_content") and c.message.reasoning_content:
                output["reasoning_content"] = c.message.reasoning_content

            if merged_kwargs.get("logprobs"):
                output["logprobs"] = c.logprobs if hasattr(c, "logprobs") else c["logprobs"]

            if hasattr(c, "message") and getattr(c.message, "tool_calls", None):
                output["tool_calls"] = c.message.tool_calls

            # Citations from provider_specific_fields
            try:
                citations_data = c.message.provider_specific_fields.get("citations")
                if isinstance(citations_data, list):
                    output["citations"] = [cit for group in citations_data for cit in group]
            except Exception:
                pass

            outputs.append(output)

        if all(len(o) == 1 for o in outputs):
            outputs = [o["text"] for o in outputs]
        return outputs

    def _process_response(self, response):
        """Extract outputs from an OpenAI Responses API response."""
        text_outputs, tool_calls, reasoning = [], [], []
        for item in response.output:
            if item.type == "message":
                for c in item.content:
                    text_outputs.append(c.text)
            elif item.type == "function_call":
                tool_calls.append(item.model_dump())
            elif item.type == "reasoning":
                for c in getattr(item, "content", None) or getattr(item, "summary", None) or []:
                    reasoning.append(c.text)

        result = {}
        if text_outputs:
            result["text"] = "".join(text_outputs)
        if tool_calls:
            result["tool_calls"] = tool_calls
        if reasoning:
            result["reasoning_content"] = "".join(reasoning)
        return [result]

    # ------------------------------------------------------------------
    # Call interface
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Copy, history, inspection
    # ------------------------------------------------------------------

    def copy(self, **kwargs):
        import copy
        new = copy.deepcopy(self)
        new.history = []
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new.kwargs.pop(key, None)
                else:
                    new.kwargs[key] = value
        if hasattr(new, "_warned_zero_temp_rollout"):
            new._warned_zero_temp_rollout = False
        return new

    def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
        pretty_print_history(self.history, n, file=file)

    def update_history(self, entry):
        if settings.disable_history:
            return

        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)
        GLOBAL_HISTORY.append(entry)

        if settings.max_history_size == 0:
            return

        if len(self.history) >= settings.max_history_size:
            self.history.pop(0)
        self.history.append(entry)

        for module in (settings.caller_modules or []):
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)


def inspect_history(n: int = 1, file: "TextIO | None" = None) -> None:
    """Display the global history shared across all LMs."""
    pretty_print_history(GLOBAL_HISTORY, n, file=file)
