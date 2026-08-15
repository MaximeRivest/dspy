"""A scripted LM for tests and demos: fixed answers, recorded calls.

`DummyLM` is an ordinary `dspy.LM` bound to a scripted engine at the
same seam every real backend uses — `complete(Request) -> Response`.
Nothing is overridden on the LM itself, so the two call faces, history,
streaming, and error wrapping are exactly the production code paths.

Two scripting levels, one seam:

- `DummyLM(["a", "b"])` — dspy-level convenience: strings out, plus a
  legacy-shaped `calls` record for assertions.
- `dspy.LM("fake", router=lm15.testing.FakeLM([...]))` — canonical-level
  scripting: full `Response` objects, scripted exceptions, recorded
  `Request`s. Use it when a test asserts on the wire shape.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence

from lm15 import Message, Request, Response, Usage, response_to_events

from dspy.core.errors import LMError
from dspy.lm.lm import LM

__all__ = ["DummyLM"]


class _ScriptEngine:
    """A structural lm15 engine that replays a script and records calls."""

    def __init__(self, outputs: Sequence[str] | Callable[[list[dict[str, Any]]], str]):
        if callable(outputs):
            self._script: list[str] | None = None
            self._fn: Callable[[list[dict[str, Any]]], str] | None = outputs
        else:
            self._script = list(outputs)
            self._fn = None
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    def complete(self, request: Request) -> Response:
        messages = [{"role": "system", "content": request.system}] if request.system else []
        messages += [{"role": m.role, "content": m.text} for m in request.messages]
        self.calls.append(
            {
                "messages": messages,
                "kwargs": _config_kwargs(request),
                "request": request,
                "timestamp": time.time(),
            }
        )
        output = self._next_output(messages)
        return Response(
            id=None,
            model="dummy",
            message=Message.assistant(output),
            finish_reason="stop",
            usage=Usage(),
        )

    def stream(self, request: Request) -> Any:
        """Replay the scripted response as canonical stream events."""
        return response_to_events(self.complete(request))

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


def _config_kwargs(request: Request) -> dict[str, Any]:
    """The request's generation kwargs as a plain dict, for assertions."""
    from dspy.lm.lm import _CONFIG_FIELDS

    config = request.config
    kwargs = {
        k: value
        for k in _CONFIG_FIELDS
        if (value := getattr(config, k)) is not None and value != ()
    }
    if config.extensions:
        kwargs.update(config.extensions)
    return kwargs


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
            `{"messages": ..., "kwargs": ..., "request": ...}` records.

    Examples:
        ```python
        lm = DummyLM(["Paris", "Berlin"])
        assert lm(prompt="Capital of France?") == ["Paris"]
        assert lm.calls[0]["messages"][0]["content"] == "Capital of France?"
        ```
    """

    def __init__(self, outputs: Sequence[str] | Callable[[list[dict[str, Any]]], str], **kwargs: Any):
        engine = _ScriptEngine(outputs)
        super().__init__(model="dummy", router=engine, **kwargs)
        self.calls = engine.calls
