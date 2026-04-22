import threading
import time

import pytest

import dspy
from dspy.streaming.chunks import StreamChunk
from dspy.streaming.live_prediction import LivePrediction


class Sig(dspy.Signature):
    q: str = dspy.InputField()
    a: str = dspy.OutputField()


class TestLivePredictionCancellation:
    def test_cancel_returns_partial_result_immediately(self):
        started = threading.Event()

        def producer(buf):
            buf.put(StreamChunk(type="output_field", field="a", text="Par"))
            started.set()
            time.sleep(1.0)
            buf.put(StreamChunk(type="output_field", field="a", text="is", is_last=True))
            buf.set_parsed({"a": "Paris"})

        prediction = LivePrediction.from_producer(producer)
        assert started.wait(timeout=1.0)

        deadline = time.monotonic()
        prediction.cancel()
        assert prediction.a == "Par"
        assert time.monotonic() - deadline < 0.2
        assert prediction.is_cancelled is True
        assert repr(prediction) == "LivePrediction(cancelled)"

    def test_cancelled_iteration_stops_quickly(self):
        first_chunk_ready = threading.Event()

        def producer(buf):
            buf.put(StreamChunk(type="output_field", field="a", text="Par"))
            first_chunk_ready.set()
            time.sleep(1.0)
            buf.put(StreamChunk(type="output_field", field="a", text="is", is_last=True))
            buf.set_parsed({"a": "Paris"})

        prediction = LivePrediction.from_producer(producer)
        assert first_chunk_ready.wait(timeout=1.0)

        chunks = iter(prediction)
        first = next(chunks)
        assert first.text == "Par"

        prediction.cancel()
        assert list(chunks) == []


class TestPredictEagerPath:
    def test_eager_path_surfaces_error_without_sync_retry(self):
        program = dspy.Predict(Sig)

        class FailingAdapter:
            def __init__(self):
                self.sync_calls = 0

            async def acall(self, lm, lm_kwargs, signature, demos, inputs):
                raise RuntimeError("boom")

            def __call__(self, lm, lm_kwargs, signature, demos, inputs):
                self.sync_calls += 1
                return [{"a": "should not happen"}]

        adapter = FailingAdapter()
        prediction = program._make_live_prediction(adapter, object(), {}, Sig, [], {"q": "hi"})

        with pytest.raises(RuntimeError, match="boom"):
            _ = prediction.a

        assert adapter.sync_calls == 0
