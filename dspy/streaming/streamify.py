"""Streaming wrapper for DSPy programs.

Wraps a DSPy program so that intermediate LM chunks are yielded
incrementally.  The streaming plumbing uses ``anyio`` memory channels
and is backend-agnostic — chunks are passed through as opaque objects
whose shape is determined by the ``StreamListener``.
"""

import asyncio
import contextvars
import logging
import threading
from asyncio import iscoroutinefunction
from queue import Queue
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Generator

import orjson
from anyio import create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectSendStream

from dspy.dsp.utils.settings import settings
from dspy.primitives.prediction import Prediction
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StatusStreamingCallback
from dspy.streaming.streaming_listener import StreamListener, find_predictor_for_stream_listeners
from dspy.utils.asyncify import asyncify

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dspy.primitives.module import Module


def _is_stream_chunk(value: Any) -> bool:
    """Return True if *value* looks like an LM streaming chunk.

    Works with any chunk type that exposes ``.choices[0].delta`` — this
    covers litellm ``ModelResponseStream``, OpenAI ``ChatCompletionChunk``,
    and DSPy's own ``StreamChunk`` shim.
    """
    return hasattr(value, "choices") and not isinstance(value, (Prediction, StatusMessage, str))


def streamify(
    program: "Module",
    status_message_provider: StatusMessageProvider | None = None,
    stream_listeners: list[StreamListener] | None = None,
    include_final_prediction_in_output_stream: bool = True,
    is_async_program: bool = False,
    async_streaming: bool = True,
) -> Callable[[Any, Any], Awaitable[Any]]:
    """Wrap a DSPy program so it streams outputs incrementally.

    See the docstring on the original ``streamify`` for full usage examples.
    """
    stream_listeners = stream_listeners or []
    if len(stream_listeners) > 0:
        predict_id_to_listener = find_predictor_for_stream_listeners(program, stream_listeners)
    else:
        predict_id_to_listener = {}

    if is_async_program:
        program = program.acall
    elif not iscoroutinefunction(program):
        program = asyncify(program)

    callbacks = list(settings.callbacks)
    status_streaming_callback = StatusStreamingCallback(status_message_provider)
    if not any(isinstance(c, StatusStreamingCallback) for c in callbacks):
        callbacks.append(status_streaming_callback)

    async def generator(args, kwargs, stream: MemoryObjectSendStream):
        with settings.context(send_stream=stream, callbacks=callbacks, stream_listeners=stream_listeners):
            prediction = await program(*args, **kwargs)
        await stream.send(prediction)

    async def async_streamer(*args, **kwargs):
        send_stream, receive_stream = create_memory_object_stream(16)
        async with create_task_group() as tg, send_stream, receive_stream:
            tg.start_soon(generator, args, kwargs, send_stream)

            async for value in receive_stream:
                if _is_stream_chunk(value):
                    if len(predict_id_to_listener) == 0:
                        yield value
                    else:
                        pid = getattr(value, "predict_id", None)
                        for listener in predict_id_to_listener.get(pid, []):
                            if output := listener.receive(value):
                                yield output
                elif isinstance(value, StatusMessage):
                    yield value
                elif isinstance(value, Prediction):
                    for listener in stream_listeners:
                        if final_chunk := listener.finalize():
                            yield final_chunk
                    if include_final_prediction_in_output_stream:
                        yield value
                    elif (
                        len(stream_listeners) == 0
                        or any(listener.cache_hit for listener in stream_listeners)
                        or not any(listener.stream_start for listener in stream_listeners)
                    ):
                        yield value
                    return
                else:
                    yield value

    if async_streaming:
        return async_streamer
    else:

        def sync_streamer(*args, **kwargs):
            return apply_sync_streaming(async_streamer(*args, **kwargs))

        return sync_streamer


def apply_sync_streaming(async_generator: AsyncGenerator) -> Generator:
    """Convert an async streaming generator to a sync generator."""
    queue: Queue = Queue()
    stop = object()
    context = contextvars.copy_context()

    def producer():
        async def runner():
            try:
                async for item in async_generator:
                    queue.put(item)
            finally:
                queue.put(stop)

        context.run(asyncio.run, runner())

    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    while True:
        item = queue.get()
        if item is stop:
            break
        yield item


async def streaming_response(streamer: AsyncGenerator) -> AsyncGenerator:
    """Convert a DSPy stream to an OpenAI-compatible SSE stream."""
    async for value in streamer:
        if isinstance(value, Prediction):
            data = {"prediction": dict(value.items(include_dspy=False))}
            yield f"data: {orjson.dumps(data).decode()}\n\n"
        elif _is_stream_chunk(value):
            chunk_json = value.json() if hasattr(value, "json") else orjson.dumps(value).decode()
            yield f"data: {orjson.dumps({'chunk': chunk_json}).decode()}\n\n"
        elif isinstance(value, str) and value.startswith("data:"):
            yield value
        else:
            raise ValueError(f"Unknown chunk value type: {value}")
    yield "data: [DONE]\n\n"
