"""One LM class: litellm-backed chat completions with declared capabilities.

Deliberately minimal (stage A1): synchronous chat completions only — no
streaming, no callbacks, no caching, no retries. Failures surface as the
typed `LMError` from the contract table, never as provider-specific
exceptions.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from dspy.core.errors import LMError

__all__ = ["LM", "LMCapabilities"]


@dataclass(frozen=True)
class LMCapabilities:
    """Declared capability facts of a bound model.

    These are constructor data, stated by whoever binds the model.
    Adapter strategies predicate on these facts; nothing in the system
    sniffs a live client object to guess them.

    Attributes:
        instruct: True for instruction/chat-tuned models; False for raw
            base (completion) models.
        native_reasoning: The API surfaces a native reasoning channel.
        native_fc: The API supports native function calling.
        native_citations: The API surfaces native citations.
    """

    instruct: bool = True
    native_reasoning: bool = False
    native_fc: bool = False
    native_citations: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Return the facts as a plain dict (serialization-friendly)."""
        return asdict(self)


class LM:
    """A litellm-backed chat-completions language model.

    Args:
        model: The litellm model identifier, e.g. `"openai/gpt-4o-mini"`.
        instruct: Capability fact — instruction-tuned (True) or base model.
        native_reasoning: Capability fact — native reasoning channel.
        native_fc: Capability fact — native function calling.
        native_citations: Capability fact — native citations.
        temperature: Default sampling temperature for every request.
        max_tokens: Default completion-token cap for every request.
        **kwargs: Extra default request kwargs forwarded to litellm on
            every call (per-call kwargs override them).

    Examples:
        ```python
        lm = dspy.LM("openai/gpt-4o-mini", native_fc=True)
        dspy.configure(lm=lm)
        outputs = lm(prompt="Say hello.")
        ```
    """

    def __init__(
        self,
        model: str,
        *,
        instruct: bool = True,
        native_reasoning: bool = False,
        native_fc: bool = False,
        native_citations: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 4000,
        **kwargs: Any,
    ):
        self.model = model
        self.capabilities = LMCapabilities(
            instruct=instruct,
            native_reasoning=native_reasoning,
            native_fc=native_fc,
            native_citations=native_citations,
        )
        self.kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens, **kwargs}
        self.history: list[dict[str, Any]] = []

    def __call__(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run one chat completion and return the completion texts.

        Args:
            messages: Chat messages, `[{"role": ..., "content": ...}, ...]`.
            prompt: Sugar for a single user message; exclusive with `messages`.
            **kwargs: Per-call request kwargs; override the constructor defaults.

        Returns:
            One string per returned choice (usually one).

        Raises:
            LMError: On any transport or provider failure, and on a
                response with no text content.
        """
        messages = self._coerce_messages(messages, prompt)
        request_kwargs = {**self.kwargs, **kwargs}
        outputs = self._request(messages, request_kwargs)
        self.history.append(
            {
                "model": self.model,
                "messages": messages,
                "kwargs": request_kwargs,
                "outputs": outputs,
                "timestamp": time.time(),
            }
        )
        return outputs

    def _coerce_messages(
        self, messages: list[dict[str, Any]] | None, prompt: str | None
    ) -> list[dict[str, Any]]:
        if (messages is None) == (prompt is None):
            raise ValueError("Pass exactly one of `messages` or `prompt`.")
        if prompt is not None:
            return [{"role": "user", "content": prompt}]
        return messages

    def _request(self, messages: list[dict[str, Any]], request_kwargs: dict[str, Any]) -> list[str]:
        import litellm

        try:
            response = litellm.completion(model=self.model, messages=messages, **request_kwargs)
        except Exception as e:
            raise LMError(f"LM request to {self.model!r} failed: {e}") from e

        outputs: list[str] = []
        for choice in response.choices:
            content = choice.message.content
            if content is None:
                raise LMError(
                    f"LM {self.model!r} returned a choice with no text content "
                    f"(finish_reason={choice.finish_reason!r}). The minimal LM speaks "
                    "text-only chat completions; richer channels arrive with adapters v2."
                )
            outputs.append(content)
        return outputs

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, capabilities={self.capabilities})"
