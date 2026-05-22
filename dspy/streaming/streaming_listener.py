from __future__ import annotations

import inspect
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from dspy.adapters.types import Type
from dspy.adapters.types.base_type import _AdapterTypeContext, _TypeStreamParser
from dspy.clients._streaming import litellm_chunk_text, litellm_chunk_to_lm_stream_event, stream_event_text
from dspy.core.types import LMStreamEvent
from dspy.dsp.utils.settings import settings
from dspy.streaming.messages import StreamResponse

if TYPE_CHECKING:
    from litellm import ModelResponseStream

    from dspy.primitives.module import Module


class _NativeTypeStreamListener:
    """Private bridge from normalized LM stream events to `dspy.Type` stream parsers."""

    def __init__(self, owner: "StreamListener"):
        self.owner = owner
        self.parser = self._make_parser()

    def _make_parser(self) -> _TypeStreamParser | None:
        output_type = self.owner._output_type
        if not (output_type and inspect.isclass(output_type) and issubclass(output_type, Type)):
            return None
        context = _AdapterTypeContext(
            field_name=self.owner.signature_field_name,
            field_info=self.owner._output_field_info,
            signature=self.owner.predict.signature if self.owner.predict else None,
            lm=getattr(settings, "lm", None),
            lm_kwargs={},
            adapter=settings.adapter,
            role="output",
        )
        return output_type.stream_parser(context)

    def receive(self, event: LMStreamEvent) -> StreamResponse | None:
        if self.parser is None:
            return None
        parsed = self.parser.receive(event)
        if parsed is not None:
            return StreamResponse(self.owner.predict_name, self.owner.signature_field_name, parsed, is_last_chunk=False)
        return None

    def finalize(self) -> StreamResponse | None:
        if self.parser is None:
            return None
        parsed = self.parser.finalize()
        if parsed is not None:
            return StreamResponse(self.owner.predict_name, self.owner.signature_field_name, parsed, is_last_chunk=True)
        return None


class StreamListener:
    """Listen to a stream and capture a specific output field of a predictor.

    The listener is now adapter-neutral: provider chunks are normalized to
    `LMStreamEvent` where possible, native `dspy.Type` streaming is handled by
    type stream parsers, and textual field slicing is delegated to the active
    adapter via `adapter.stream_parser(field_name)`.
    """

    def __init__(
        self,
        signature_field_name: str,
        predict: Any = None,
        predict_name: str | None = None,
        allow_reuse: bool = False,
    ):
        self.signature_field_name = signature_field_name
        self.predict = predict
        self.predict_name = predict_name
        self.stream_start = False
        self.stream_end = False
        self.cache_hit = False
        self.allow_reuse = allow_reuse
        self._native_type_listener: _NativeTypeStreamListener | None = None
        self._adapter_stream_parser = None

    def receive(self, chunk: ModelResponseStream | LMStreamEvent):
        if self._finished_and_not_reusable():
            return None

        event = chunk if isinstance(chunk, LMStreamEvent) else litellm_chunk_to_lm_stream_event(chunk)
        if event is not None:
            if native_output := self._native_listener.receive(event):
                return native_output
            if isinstance(chunk, LMStreamEvent):
                text = stream_event_text(event)
                if text is None:
                    return None
                return self._receive_adapter_text(text)

        if isinstance(chunk, LMStreamEvent):
            return None

        # Legacy fallback for provider-shaped chunks until providers emit normalized events directly.
        if self._native_listener.parser is None and (legacy_output := self._legacy_custom_type_receive(chunk)):
            return legacy_output

        text = litellm_chunk_text(chunk)
        if text is None:
            return None
        return self._receive_adapter_text(text)

    def _receive_adapter_text(self, text: str) -> StreamResponse | None:
        parser = self._adapter_parser
        if parser is None:
            return None
        parsed = parser.receive_text(text)
        self._sync_adapter_state(parser)
        if parsed is None:
            return None
        return StreamResponse(
            self.predict_name,
            self.signature_field_name,
            parsed.chunk,
            is_last_chunk=parsed.is_last_chunk,
        )

    def _finished_and_not_reusable(self) -> bool:
        if not self.stream_end:
            return False
        if not self.allow_reuse:
            return True
        self.stream_end = False
        self.cache_hit = False
        self.stream_start = False
        self._native_type_listener = None
        if self._adapter_stream_parser is not None and hasattr(self._adapter_stream_parser, "reset_for_reuse"):
            self._adapter_stream_parser.reset_for_reuse()
        return False

    def _legacy_custom_type_receive(self, chunk: Any) -> StreamResponse | None:
        output_type = self._output_type
        if not (output_type and inspect.isclass(output_type) and issubclass(output_type, Type)):
            return None
        try:
            if not output_type.is_streamable():
                return None
            parsed_chunk = output_type.parse_stream_chunk(chunk)
        except Exception:
            return None
        if parsed_chunk:
            return StreamResponse(self.predict_name, self.signature_field_name, parsed_chunk, is_last_chunk=self.stream_end)
        return None

    @property
    def _adapter_parser(self):
        if self._adapter_stream_parser is not None:
            return self._adapter_stream_parser
        adapter = settings.adapter
        if adapter is None:
            # DSPy's default adapter is ChatAdapter; keep listener behavior aligned
            # when no explicit adapter is configured.
            from dspy.adapters._streaming import _ChatAdapterStreamParser

            self._adapter_stream_parser = _ChatAdapterStreamParser(self.signature_field_name)
            return self._adapter_stream_parser
        if not hasattr(adapter, "stream_parser"):
            return None
        self._adapter_stream_parser = adapter.stream_parser(self.signature_field_name)
        return self._adapter_stream_parser

    @property
    def _native_listener(self) -> _NativeTypeStreamListener:
        if self._native_type_listener is None:
            self._native_type_listener = _NativeTypeStreamListener(self)
        return self._native_type_listener

    def _sync_adapter_state(self, parser) -> None:
        self.stream_start = bool(getattr(parser, "stream_start", self.stream_start))
        self.stream_end = bool(getattr(parser, "stream_end", self.stream_end))
        self.cache_hit = bool(getattr(parser, "cache_hit", self.cache_hit))

    def _could_form_end_identifier(self, concat_message: str, adapter_name: str) -> bool:
        # Backward-compatible test/helper shim. Delimiter logic now lives in adapter parsers.
        from dspy.adapters._streaming import _ChatAdapterStreamParser, _JSONAdapterStreamParser, _XMLAdapterStreamParser

        parser_by_name = {
            "ChatAdapter": _ChatAdapterStreamParser,
            "JSONAdapter": _JSONAdapterStreamParser,
            "XMLAdapter": _XMLAdapterStreamParser,
        }
        parser_cls = parser_by_name.get(adapter_name)
        if parser_cls is None:
            return False
        return parser_cls(self.signature_field_name)._could_form_end_identifier(concat_message)

    def flush(self) -> str:
        parser = self._adapter_parser
        if parser is None:
            return ""
        token = parser.flush()
        self._sync_adapter_state(parser)
        return token

    def finalize(self) -> StreamResponse | None:
        if native_final := self._native_listener.finalize():
            return native_final
        parser = self._adapter_parser
        if parser is None:
            return None
        parsed = parser.finalize()
        self._sync_adapter_state(parser)
        if parsed is None:
            return None
        return StreamResponse(
            self.predict_name,
            self.signature_field_name,
            parsed.chunk,
            is_last_chunk=parsed.is_last_chunk,
        )

    @property
    def _output_type(self) -> type | None:
        try:
            return self.predict.signature.output_fields[self.signature_field_name].annotation
        except Exception:
            return None

    @property
    def _output_field_info(self) -> Any:
        try:
            return self.predict.signature.output_fields[self.signature_field_name]
        except Exception:
            return None


def find_predictor_for_stream_listeners(
    program: Module, stream_listeners: list[StreamListener]
) -> dict[int, list[StreamListener]]:
    """Find the predictor for each stream listener."""
    predictors = program.named_predictors()

    field_name_to_named_predictor = {}
    for listener in stream_listeners:
        if listener.predict:
            continue
        field_name_to_named_predictor[listener.signature_field_name] = None

    for name, predictor in predictors:
        for field_name in predictor.signature.output_fields:
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
                "cannot automatically determine which predictor to use. Please verify your field name or "
                "specify the predictor to listen to."
            )
        listener.predict_name, listener.predict = field_name_to_named_predictor[listener.signature_field_name]
        predict_id_to_listener[id(listener.predict)].append(listener)
    return predict_id_to_listener
