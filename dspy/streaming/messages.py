"""Status messages and stream response types for DSPy streaming."""

import asyncio
import concurrent.futures
from dataclasses import dataclass
from typing import Any

from asyncer import syncify

from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback


@dataclass
class StreamResponse:
    predict_name: str
    signature_field_name: str
    chunk: str
    is_last_chunk: bool


@dataclass
class StatusMessage:
    message: str


def sync_send_to_stream(stream, message):
    """Send message to stream, works whether or not an event loop is running."""
    async def _send():
        await stream.send(message)

    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(lambda: asyncio.run(_send()))
            return future.result()
    except RuntimeError:
        return syncify(_send)()


class StatusMessageProvider:
    """Override methods to customize status messages during streaming."""

    def tool_start_status_message(self, instance, inputs):
        return f"Calling tool {instance.name}..."

    def tool_end_status_message(self, outputs):
        return "Tool calling finished! Querying the LLM with tool calling results..."

    def module_start_status_message(self, instance, inputs): pass
    def module_end_status_message(self, outputs): pass
    def lm_start_status_message(self, instance, inputs): pass
    def lm_end_status_message(self, outputs): pass


class StatusStreamingCallback(BaseCallback):
    def __init__(self, provider=None):
        self.p = provider or StatusMessageProvider()

    def _send(self, msg):
        stream = settings.send_stream
        if stream and msg:
            sync_send_to_stream(stream, StatusMessage(msg))

    def on_tool_start(self, call_id, instance, inputs):
        if instance.name != "finish":
            self._send(self.p.tool_start_status_message(instance, inputs))

    def on_tool_end(self, call_id, outputs=None, exception=None):
        if outputs != "Completed.":
            self._send(self.p.tool_end_status_message(outputs))

    def on_lm_start(self, call_id, instance, inputs):
        self._send(self.p.lm_start_status_message(instance, inputs))

    def on_lm_end(self, call_id, outputs=None, exception=None):
        self._send(self.p.lm_end_status_message(outputs))

    def on_module_start(self, call_id, instance, inputs):
        self._send(self.p.module_start_status_message(instance, inputs))

    def on_module_end(self, call_id, outputs=None, exception=None):
        self._send(self.p.module_end_status_message(outputs))
