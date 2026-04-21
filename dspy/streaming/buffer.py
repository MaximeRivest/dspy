import asyncio
import threading
from collections import deque
from typing import Any


class StreamBuffer:
    """Thread-safe buffer for streaming chunks between a producer and consumer.

    The producer (background thread) calls :meth:`put` and :meth:`mark_done`.
    The consumer (main thread) iterates with ``for chunk in buffer`` or blocks
    with :meth:`wait_for_result`.
    """

    def __init__(self):
        self._chunks: deque = deque()
        self._lock = threading.Lock()
        self._new_data = threading.Event()
        self._done = threading.Event()
        self._cancelled = threading.Event()
        self._parsed: dict[str, Any] | None = None
        self._error: BaseException | None = None
        self._chunk_count: int = 0

    # ── Producer API ────────────────────────────────────────

    def put(self, chunk: Any) -> None:
        """Append a chunk to the buffer.  No-op after cancellation."""
        if self._cancelled.is_set():
            return
        with self._lock:
            self._chunks.append(chunk)
            self._chunk_count += 1
        self._new_data.set()

    def set_parsed(self, parsed: dict[str, Any]) -> None:
        """Store the authoritative parsed result from the adapter."""
        self._parsed = parsed

    def set_error(self, error: BaseException) -> None:
        """Store an exception that will be raised on the consumer side."""
        self._error = error

    def mark_done(self) -> None:
        """Signal that the producer has finished (success or failure)."""
        self._done.set()
        self._new_data.set()

    # ── Sync Consumer API ───────────────────────────────────

    def __iter__(self):
        """Yield chunks as they arrive.  Already-buffered chunks are instant."""
        while True:
            # Drain everything currently buffered (no blocking)
            while True:
                with self._lock:
                    if self._chunks:
                        yield self._chunks.popleft()
                        continue
                break

            # Terminal check
            if self._done.is_set() or self._cancelled.is_set():
                # Final drain
                with self._lock:
                    while self._chunks:
                        yield self._chunks.popleft()
                if self._error and not self._cancelled.is_set():
                    raise self._error
                return

            # Block until producer signals new data or completion
            self._new_data.wait()
            self._new_data.clear()

    async def __aiter__(self):
        """Async iteration with cooperative yielding (5 ms poll)."""
        while True:
            # Drain buffered chunks
            while True:
                with self._lock:
                    if self._chunks:
                        yield self._chunks.popleft()
                        continue
                break

            if self._done.is_set() or self._cancelled.is_set():
                with self._lock:
                    while self._chunks:
                        yield self._chunks.popleft()
                if self._error and not self._cancelled.is_set():
                    raise self._error
                return

            await asyncio.sleep(0.005)

    def wait_for_result(self) -> dict[str, Any] | None:
        """Block until the stream completes, then return the parsed result."""
        self._done.wait()
        if self._error:
            raise self._error
        return self._parsed

    # ── Cancellation ────────────────────────────────────────

    def cancel(self) -> None:
        """Stop the producer and unblock a waiting consumer."""
        self._cancelled.set()
        self._new_data.set()

    # ── Introspection ───────────────────────────────────────

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def chunk_count(self) -> int:
        return self._chunk_count
