"""A scripted LM for tests and demos: fixed answers, recorded calls."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from dspy.core.errors import LMError
from dspy.lm.lm import LM

__all__ = ["DummyLM"]


class DummyLM(LM):
    """An LM that replays a script instead of calling a provider.

    Args:
        outputs: Either a sequence of completion strings — one per call,
            in order — or a callable `f(messages) -> str` computed per
            call. A sequence that runs out refuses loudly with `LMError`;
            nothing silently repeats.
        **kwargs: Capability facts and default kwargs, as for `LM`.

    Attributes:
        calls: Every call made, in order, as
            `{"messages": ..., "kwargs": ...}` records.

    Examples:
        ```python
        lm = DummyLM(["Paris", "Berlin"])
        assert lm(prompt="Capital of France?") == ["Paris"]
        assert lm.calls[0]["messages"][0]["content"] == "Capital of France?"
        ```
    """

    def __init__(self, outputs: Sequence[str] | Callable[[list[dict[str, Any]]], str], **kwargs: Any):
        super().__init__(model="dummy", **kwargs)
        if callable(outputs):
            self._script: list[str] | None = None
            self._fn: Callable[[list[dict[str, Any]]], str] | None = outputs
        else:
            self._script = list(outputs)
            self._fn = None
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    def _request(self, messages: list[dict[str, Any]], request_kwargs: dict[str, Any]) -> list[str]:
        self.calls.append({"messages": messages, "kwargs": request_kwargs, "timestamp": time.time()})
        if self._fn is not None:
            return [self._fn(messages)]
        if self._cursor >= len(self._script):
            raise LMError(
                f"DummyLM script exhausted: {len(self._script)} scripted output(s), "
                f"but call #{self._cursor + 1} arrived. Script more outputs or fix the "
                "program making extra calls."
            )
        output = self._script[self._cursor]
        self._cursor += 1
        return [output]
