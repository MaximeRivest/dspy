"""A scripted LM for tests and demos: fixed answers, recorded calls."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from lm15 import Message, Request, Response, Usage, response_to_events

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
        return [self._next_output(messages)]

    def _complete(self, engine: Any, request: Request) -> Response:
        """Script the typed face too: every `Request` replays the same script."""
        messages = [{"role": "system", "content": request.system}] if request.system else []
        messages += [{"role": m.role, "content": m.text} for m in request.messages]
        self.calls.append({"messages": messages, "request": request, "timestamp": time.time()})
        output = self._next_output(messages)
        return Response(
            id=None,
            model=self.model,
            message=Message.assistant(output),
            finish_reason="stop",
            usage=Usage(),
        )

    def _stream(self, engine: Any, request: Request) -> Any:
        """Replay the scripted response as canonical stream events."""
        return response_to_events(self._complete(engine, request))

    def _next_output(self, messages: list[dict[str, Any]]) -> str:
        if self._fn is not None:
            return self._fn(messages)
        if self._cursor >= len(self._script):
            raise LMError(
                f"DummyLM script exhausted: {len(self._script)} scripted output(s), "
                f"but call #{self._cursor + 1} arrived. Script more outputs or fix the "
                "program making extra calls."
            )
        output = self._script[self._cursor]
        self._cursor += 1
        return output
