import inspect
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from dspy.adapters.types import Type
from dspy.streaming.chunks import StreamChunk
from dspy.streaming.messages import StreamResponse

if TYPE_CHECKING:
    from dspy.primitives.module import Module

class StreamListener:
    """Listens to normalized stream chunks for one output field of a predictor."""

    def __init__(
        self,
        signature_field_name: str,
        predict: Any = None,
        predict_name: str | None = None,
        allow_reuse: bool = False,
    ):
        """
        Args:
            signature_field_name: The name of the field to listen to.
            predict: The predictor to listen to. If None, when calling `streamify()` it will automatically look for
                the predictor that has the `signature_field_name` in its signature.
            predict_name: The name of the predictor to listen to. If None, when calling `streamify()` it will
                automatically look for the predictor that has the `signature_field_name` in its signature.
            allow_reuse: If True, the stream listener can be reused for multiple streams. Please note that this could
                hurt the performance because the same stream chunk is sent to multiple listeners.
        """
        self.signature_field_name = signature_field_name
        self.predict = predict
        self.predict_name = predict_name

        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False
        self.allow_reuse = allow_reuse
        self._pending_chunk: StreamChunk | None = None

    def reset(self):
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False
        self._pending_chunk = None

    def receive(self, chunk: StreamChunk):
        if self.stream_end:
            if self.allow_reuse:
                self.reset()
            else:
                return None

        if chunk.type != "output_field" or chunk.field != self.signature_field_name:
            return None

        self.stream_start = True

        if self._pending_chunk is None:
            self._pending_chunk = chunk
            if chunk.is_last and chunk.text:
                pending = self._pending_chunk
                self._pending_chunk = None
                self.stream_end = True
                return StreamResponse(
                    self.predict_name,
                    self.signature_field_name,
                    pending.text,
                    is_last_chunk=True,
                )
            return None

        pending = self._pending_chunk

        if chunk.is_last and not chunk.text:
            self._pending_chunk = None
            self.stream_end = True
            return StreamResponse(
                self.predict_name,
                self.signature_field_name,
                pending.text,
                is_last_chunk=True,
            )

        self._pending_chunk = chunk
        return StreamResponse(
            self.predict_name,
            self.signature_field_name,
            pending.text,
            is_last_chunk=False,
        )

    def receive_raw(self, chunk: Any):
        if self.stream_end:
            if self.allow_reuse:
                self.reset()
            else:
                return None

        if (
            self._output_type
            and inspect.isclass(self._output_type)
            and issubclass(self._output_type, Type)
            and self._output_type.is_streamable()
        ):
            parsed_chunk = self._output_type.parse_stream_chunk(chunk)
            if parsed_chunk:
                self.stream_start = True
                return StreamResponse(
                    self.predict_name,
                    self.signature_field_name,
                    parsed_chunk,
                    is_last_chunk=False,
                )
        return None

    def finalize(self) -> StreamResponse | None:
        if self._pending_chunk is None:
            return None

        pending = self._pending_chunk
        self._pending_chunk = None
        self.stream_end = True
        return StreamResponse(
            self.predict_name,
            self.signature_field_name,
            pending.text,
            is_last_chunk=True,
        )

    @property
    def uses_raw_stream(self) -> bool:
        return (
            self._output_type is not None
            and inspect.isclass(self._output_type)
            and issubclass(self._output_type, Type)
            and self._output_type.is_streamable()
        )

    def _could_form_end_identifier(self, concat_message: str, adapter_name: str) -> bool:
        suffix_checks = {
            "ChatAdapter": (["[", "[[", "[[ ", "[[ #", "[[ ##"], "[[ ##"),
            "JSONAdapter": (['"', '",', '" ', '"}'], "}"),
            "XMLAdapter": (["<", "</"], "</"),
        }
        prefixes, contains = suffix_checks.get(adapter_name, ([], None))
        if any(concat_message.endswith(prefix) for prefix in prefixes):
            return True
        return contains is not None and contains in concat_message

    @property
    def _output_type(self) -> type | None:
        try:
            return self.predict.signature.output_fields[self.signature_field_name].annotation
        except Exception:
            return None


def find_predictor_for_stream_listeners(
    program: "Module", stream_listeners: list[StreamListener]
) -> dict[int, list[StreamListener]]:
    """Find the predictor for each stream listener.

    This is a utility function to automatically find the predictor for each stream listener. It is used when some
    listeners don't specify the predictor they want to listen to. If a listener's `signature_field_name` is not
    unique in the program, this function will raise an error.
    """
    predictors = program.named_predictors()

    field_name_to_named_predictor = {}
    for listener in stream_listeners:
        if listener.predict:
            continue
        field_name_to_named_predictor[listener.signature_field_name] = None

    for name, predictor in predictors:
        for field_name, field_info in predictor.signature.output_fields.items():
            if field_name not in field_name_to_named_predictor:
                continue

            if field_name_to_named_predictor[field_name] is not None:
                raise ValueError(
                    f"Signature field {field_name} is not unique in the program, cannot automatically determine which "
                    "predictor to use for streaming. Please specify the predictor to listen to."
                )
            field_name_to_named_predictor[field_name] = (name, predictor)

    predict_id_to_listener = defaultdict(list)
    for listener in stream_listeners:
        if listener.predict:
            predict_id_to_listener[id(listener.predict)].append(listener)
            continue
        if listener.signature_field_name not in field_name_to_named_predictor:
            raise ValueError(
                f"Signature field {listener.signature_field_name} is not a field of any predictor in the program, "
                "cannot automatically determine which predictor to use for streaming. Please verify your field name or "
                "specify the predictor to listen to."
            )
        listener.predict_name, listener.predict = field_name_to_named_predictor[listener.signature_field_name]
        predict_id_to_listener[id(listener.predict)].append(listener)
    return predict_id_to_listener
