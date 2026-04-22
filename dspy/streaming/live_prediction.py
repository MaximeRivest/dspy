from __future__ import annotations

import asyncio
import threading
from typing import Any, AsyncIterator, Iterator

from dspy.primitives.prediction import Prediction
from dspy.streaming.buffer import StreamBuffer
from dspy.streaming.chunks import StreamChunk


class LivePrediction(Prediction):
    """A Prediction whose LM call runs eagerly in the background.

    Tokens start arriving the moment the object is created. Users consume them
    in two ways — both of which benefit from the head start:

    * **Iterate** (``for chunk in result``) to get real-time
      :class:`StreamChunk` objects. Already-buffered chunks arrive instantly.
    * **Access a field** (``result.answer``) to wait for the final parsed
      value — unless the prediction was cancelled, in which case field access
      returns immediately with the best partial value collected so far.

    Cancellation is intentionally UX-first: it stops iteration and unblocks
    field access as fast as possible. The underlying provider request may
    still be winding down in the background, but the ``LivePrediction`` itself
    becomes immediately usable.

    Examples:
        Stream tokens to the console::

            result = predict(question="What is 2+2?")
            for chunk in result:
                if chunk.field == "answer":
                    print(chunk.text, end="", flush=True)

        Or just grab the answer (still benefits from eager start)::

            result = predict(question="What is 2+2?")
            # ... do other work while tokens arrive in the background ...
            print(result.answer)   # waits only for the remaining time

        Cancel early::

            result = predict(question="Write a 10 000-word essay")
            for chunk in result:
                if "conclusion" in chunk.text:
                    result.cancel()
                    break
            print(result.answer)   # returns partial text immediately

        Async iteration::

            result = await predict.acall(question="What is 2+2?")
            async for chunk in result:
                print(chunk.text, end="", flush=True)
    """

    def __init__(self, *args: Any, **kwargs: Any):
        self._buffer: StreamBuffer | None = kwargs.pop("_buffer", None)
        self._producer_thread: threading.Thread | None = kwargs.pop("_thread", None)

        super().__init__(*args, **kwargs)

    # ── Constructors ────────────────────────────────────────

    @classmethod
    def from_producer(cls, producer) -> LivePrediction:
        """Create a LivePrediction backed by a background thread.

        ``producer(buffer)`` is called in a daemon thread.  It should:

        1. Put :class:`StreamChunk` objects into *buffer* via ``buffer.put()``.
        2. Call ``buffer.set_parsed(dict)`` with the final parsed fields.

        Lifecycle (``mark_done`` / error capture) is handled automatically.
        """
        buffer = StreamBuffer()

        from dspy.dsp.utils.settings import thread_local_overrides

        parent_overrides = thread_local_overrides.get().copy()

        def _run() -> None:
            token = thread_local_overrides.set(parent_overrides.copy())
            try:
                producer(buffer)
            except Exception as exc:
                if not buffer.is_done:
                    buffer.set_error(exc)
            finally:
                thread_local_overrides.reset(token)
                if not buffer.is_done:
                    buffer.mark_done()

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return cls(_buffer=buffer, _thread=thread)

    @classmethod
    def from_completed(
        cls,
        completions: list[dict[str, Any]],
        signature: Any = None,
    ) -> LivePrediction:
        """Wrap an already-complete result as a ``LivePrediction``.

        Useful for cache hits or ``n > 1`` fallbacks so that callers always
        get back the same type.
        """
        pred = Prediction.from_completions(completions, signature=signature)

        live = cls.__new__(cls)
        live._store = pred._store
        live._completions = pred._completions
        live._lm_usage = None

        buf = StreamBuffer()
        # Emit one synthetic chunk per field so iteration still yields something
        for key, value in pred._store.items():
            buf.put(
                StreamChunk(
                    type="output_field",
                    field=key,
                    text=str(value),
                    is_last=True,
                )
            )
        buf.set_parsed(dict(pred._store))
        buf.mark_done()
        live._buffer = buf
        live._producer_thread = None
        return live

    # ── Sync iteration ──────────────────────────────────────

    def __iter__(self) -> Iterator[StreamChunk]:
        if self._buffer is None:
            return
        yield from self._buffer
        self._ensure_parsed()

    def stream(self, *fields: str) -> Iterator[StreamChunk]:
        """Iterate only chunks whose ``field`` is in *fields*.

        If no field names are given, yields every chunk (same as iterating
        the prediction directly).
        """
        for chunk in self:
            if not fields or chunk.field in fields:
                yield chunk

    # ── Async iteration ─────────────────────────────────────

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        if self._buffer is None:
            return
        async for chunk in self._buffer:
            yield chunk
        self._ensure_parsed()

    async def astream(self, *fields: str) -> AsyncIterator[StreamChunk]:
        """Async version of :meth:`stream`."""
        async for chunk in self:
            if not fields or chunk.field in fields:
                yield chunk

    # ── Field access (blocking) ─────────────────────────────

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)

        self._ensure_parsed()

        if key in self._store:
            return self._store[key]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")

    def items(self, include_dspy: bool = False):
        self._ensure_parsed()
        return super().items(include_dspy=include_dspy)

    def keys(self, include_dspy: bool = False):
        self._ensure_parsed()
        return super().keys(include_dspy=include_dspy)

    def values(self, include_dspy: bool = False):
        self._ensure_parsed()
        return super().values(include_dspy=include_dspy)

    def toDict(self):
        self._ensure_parsed()
        return super().toDict()

    def _ensure_parsed(self) -> None:
        """Populate ``_store`` with the best available parsed result.

        Before cancellation this waits for completion. After cancellation it
        returns immediately with the partial parsed state collected so far.
        """
        if self._buffer is None:
            return
        if self._buffer.is_done and self._store:
            return

        parsed = self._buffer.wait_for_result()
        if parsed is not None:
            self._store.update(parsed)

    # ── Cancellation ────────────────────────────────────────

    def cancel(self) -> None:
        """Cancel the prediction immediately.

        Already-buffered chunks remain accessible. Iteration stops quickly and
        field access returns the best partial value collected so far instead of
        waiting for the provider request to finish.
        """
        if self._buffer:
            self._buffer.cancel()

    @property
    def is_cancelled(self) -> bool:
        return self._buffer.is_cancelled if self._buffer else False

    @property
    def is_done(self) -> bool:
        return self._buffer.is_done if self._buffer else True

    # ── Representation ──────────────────────────────────────

    def __repr__(self) -> str:
        if self._buffer and self._buffer.is_cancelled:
            return "LivePrediction(cancelled)"
        if self._buffer and not self._buffer.is_done:
            return "LivePrediction(streaming…)"
        return super().__repr__()
