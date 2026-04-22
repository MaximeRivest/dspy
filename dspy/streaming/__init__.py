# ── Incremental prediction API ───────────────────────────────
from dspy.streaming.chunks import StreamChunk
from dspy.streaming.live_prediction import LivePrediction

# ── Existing streamify API ───────────────────────────────────
from dspy.streaming.messages import StatusMessage, StatusMessageProvider, StreamResponse
from dspy.streaming.streamify import apply_sync_streaming, streamify, streaming_response
from dspy.streaming.streaming_listener import StreamListener

__all__ = [
    # New
    "LivePrediction",
    "StreamChunk",
    # Legacy
    "StatusMessage",
    "StatusMessageProvider",
    "streamify",
    "StreamListener",
    "StreamResponse",
    "streaming_response",
    "apply_sync_streaming",
]
