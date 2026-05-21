import logging
from typing import Any, get_origin

import json_repair

from dspy.adapters.legacy_bridge import legacy_messages_from_typed_messages
from dspy.adapters.types import History, Type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.adapters.types.type_strategy import TypeStrategy
from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.types import LMConfig, LMMessage, LMRequest, LMRequestPatch, LMResponse
from dspy.dsp.utils.settings import settings
from dspy.experimental import Citations
from dspy.signatures.signature import Signature
from dspy.utils.asyncify import asyncify
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]


def _default_adapter_types() -> list[Any]:
    from dspy.adapters.types import (
        NativeAudio,
        NativeCitations,
        NativeDocument,
        NativeFile,
        NativeHistory,
        NativeImage,
        TextCode,
        TextReasoning,
        TextToolCalls,
    )

    return [
        TextReasoning(),
        TextCode(),
        NativeImage(),
        NativeAudio(),
        NativeFile(),
        NativeDocument(),
        NativeCitations(),
        NativeHistory(),
        TextToolCalls(),
    ]


def _uses_language_model_contract(lm: Any) -> bool:
    """Whether an LM should be treated as a normalized LanguageModel backend."""
    return isinstance(lm, LanguageModel)


def _lm_supports_function_calling(lm: Any) -> bool:
    if isinstance(lm, BaseLM):
        return lm.supports_function_calling
    return bool(getattr(lm, "capabilities", None) and lm.capabilities.function_calling)


def _lm_supports_reasoning(lm: Any) -> bool:
    if isinstance(lm, BaseLM):
        return lm.supports_reasoning
    return bool(getattr(lm, "capabilities", None) and lm.capabilities.reasoning)


def _lm_supports_response_schema(lm: Any) -> bool:
    if isinstance(lm, BaseLM):
        return lm.supports_response_schema
    return bool(getattr(lm, "capabilities", None) and lm.capabilities.response_schema)


def _lm_output_response_dict(output: Any) -> dict[str, Any]:
    data: dict[str, Any] = {"text": output.text}
    if output.reasoning_content is not None:
        data["reasoning_content"] = output.reasoning_content
    if output.tool_calls:
        data["tool_calls"] = output.tool_calls
    if output.citations:
        data["citations"] = [citation.model_dump(exclude_none=True) for citation in output.citations]
    if output.logprobs is not None:
        data["logprobs"] = output.logprobs
    return data


def _lm_streaming_enabled(lm: LanguageModel) -> bool:
    try:
        lm._require_stream_support(async_=False)
    except NotImplementedError:
        return False
    return True


def _lm_async_streaming_enabled(lm: LanguageModel) -> bool:
    try:
        lm._require_stream_support(async_=True)
    except NotImplementedError:
        return False
    return True


def _type_strategy_for(annotation: Any, strategies: list[Any]) -> Any | None:
    for strategy in strategies:
        if hasattr(strategy, "matches") and strategy.matches(annotation):
            return strategy
    return None


def _merge_lm_config_objects(left: LMConfig | None, right: LMConfig | None) -> LMConfig | None:
    if left is None:
        return right
    if right is None:
        return left
    data = left.model_dump()
    right_data = right.model_dump(exclude_none=True)
    data.update(right_data)
    data["extensions"] = {**left.extensions, **right.extensions}
    return LMConfig(**data)


def _apply_lm_request_patch_to_legacy_adapter_call(
    signature: type[Signature],
    lm_kwargs: dict[str, Any],
    patch: Any,
) -> type[Signature]:
    """Apply request patches to the legacy BaseLM kwargs/signature path."""
    for key, value in patch.as_lm_kwargs().items():
        lm_kwargs[key] = value
    for field_name in patch.delete_input_fields:
        signature = signature.delete(field_name)
    for field_name in patch.delete_output_fields:
        signature = signature.delete(field_name)
    return signature


def _text_from_lm_stream_event(event: Any) -> str | None:
    from dspy.clients.language_models.types import LMStreamDeltaEvent, LMTextDelta

    if isinstance(event, LMStreamDeltaEvent) and isinstance(event.delta, LMTextDelta):
        return event.delta.text
    return None


class _FieldStreamParser:
    def __init__(self, adapter: "Adapter", listener: Any):
        self.adapter = adapter
        self.listener = listener
        self.field_name = listener.signature_field_name
        self.start_identifier = adapter.stream_start_identifier(self.field_name)
        self.search_buffer = ""
        self.content_buffer = ""

    def receive(self, text: str):
        if self.listener.stream_end:
            return None

        if not self.listener.stream_start:
            self.search_buffer += text
            start_index = self.search_buffer.find(self.start_identifier)
            if start_index == -1:
                self.search_buffer = self.adapter.trim_stream_start_buffer(self.search_buffer, self.start_identifier)
                return None
            self.listener.stream_start = True
            text = self.search_buffer[start_index + len(self.start_identifier) :].lstrip()
            self.search_buffer = ""

        if not text:
            return None

        self.content_buffer += text
        token, self.content_buffer, ended = self.adapter.consume_stream_field_buffer(
            self.field_name,
            self.content_buffer,
            final=False,
        )
        if ended:
            self.listener.stream_end = True
        if token or ended:
            return self.adapter.make_stream_response(self.listener, token, is_last_chunk=ended)
        return None

    def finalize(self):
        if self.listener.stream_end or not self.listener.stream_start:
            return None
        self.listener.stream_end = True
        token, self.content_buffer, _ = self.adapter.consume_stream_field_buffer(
            self.field_name,
            self.content_buffer,
            final=True,
        )
        if token:
            return self.adapter.make_stream_response(self.listener, token, is_last_chunk=True)
        return None


class Adapter:
    """Base Adapter class.

    The Adapter serves as the interface layer between DSPy module/signature and Language Models (LMs). It handles the
    complete transformation pipeline from DSPy inputs to LM calls and back to structured outputs.

    Key responsibilities:
        - Transform user inputs and signatures into properly formatted LM prompts, which also instructs the LM to format
            the response in a specific format.
        - Parse LM outputs into dictionaries matching the signature's output fields.
        - Enable/disable native LM features (function calling, citations, etc.) based on configuration.
        - Handle conversation history, few-shot examples, and custom type processing.

    The adapter pattern allows DSPy to work with different LM interfaces while maintaining a consistent programming
    model for users.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
        adapter_types: list[Any] | None = None,
    ):
        """
        Args:
            callbacks: List of callback functions to execute during `format()` and `parse()` methods. Callbacks can be
                used for logging, monitoring, or custom processing. Defaults to None (empty list).
            use_native_function_calling: Whether to enable native function calling capabilities when the LM supports it.
                If True, the adapter will automatically configure function calling when input fields contain `dspy.Tool`
                or `list[dspy.Tool]` types. Defaults to False.
            native_response_types: List of output field types that should be handled by native LM features rather than
                adapter parsing. For example, `dspy.Citations` can be populated directly by citation APIs
                (e.g., Anthropic's citation feature). Defaults to `[Citations]`.
        """
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.native_response_types = native_response_types or _DEFAULT_NATIVE_RESPONSE_TYPES
        self.adapter_types = _default_adapter_types() if adapter_types is None else list(adapter_types)

    # ------------------------------------------------------------------
    # Normalized LanguageModel streaming.
    #
    # Adapters own field-level parsing because they own the response format.
    # The streaming package only transports `StreamResponse`, raw normalized
    # `LMStreamEvent`s, status messages, and final predictions.
    # ------------------------------------------------------------------

    def stream_start_identifier(self, field_name: str) -> str:
        raise NotImplementedError(f"{type(self).__name__} does not support field-level streaming.")

    def trim_stream_start_buffer(self, buffer: str, start_identifier: str) -> str:
        max_suffix = min(len(buffer), len(start_identifier) - 1)
        for length in range(max_suffix, 0, -1):
            if start_identifier.startswith(buffer[-length:]):
                return buffer[-length:]
        return ""

    def consume_stream_field_buffer(self, field_name: str, buffer: str, *, final: bool) -> tuple[str, str, bool]:
        raise NotImplementedError(f"{type(self).__name__} does not support field-level streaming.")

    def make_stream_response(self, listener: Any, token: str, *, is_last_chunk: bool):
        from dspy.streaming.messages import StreamResponse

        return StreamResponse(
            listener.predict_name,
            listener.signature_field_name,
            token,
            is_last_chunk=is_last_chunk,
        )

    def _stream_listeners_for_current_predict(self) -> list[Any]:
        listeners = settings.stream_listeners or []
        caller_predict = settings.caller_predict
        return [listener for listener in listeners if listener.predict == caller_predict]

    def _emit_lm_stream_event(self, event: Any, parsers: list[_FieldStreamParser], send) -> None:
        if not parsers or settings.stream_include_lm_events:
            send(event)
        if not parsers:
            return
        text = _text_from_lm_stream_event(event)
        if text is None:
            return
        for parser in parsers:
            response = parser.receive(text)
            if response is not None:
                send(response)

    def _stream_language_model_response(self, lm: LanguageModel, request: LMRequest) -> LMResponse:
        from dspy.streaming.messages import sync_send_to_stream

        send_stream = settings.send_stream
        parsers = [_FieldStreamParser(self, listener) for listener in self._stream_listeners_for_current_predict()]
        stream = lm.stream(request=request)

        def send(value: Any) -> None:
            if send_stream is not None:
                sync_send_to_stream(send_stream, value)

        for event in stream:
            self._emit_lm_stream_event(event, parsers, send)
        for parser in parsers:
            response = parser.finalize()
            if response is not None:
                send(response)
        return stream.result()

    async def _astream_language_model_response(self, lm: LanguageModel, request: LMRequest) -> LMResponse:
        send_stream = settings.send_stream
        parsers = [_FieldStreamParser(self, listener) for listener in self._stream_listeners_for_current_predict()]
        stream = lm.astream(request=request)

        async def send(value: Any) -> None:
            if send_stream is not None:
                await send_stream.send(value)

        async for event in stream:
            emitted = []
            if not parsers or settings.stream_include_lm_events:
                emitted.append(event)
            if parsers:
                text = _text_from_lm_stream_event(event)
                if text is not None:
                    for parser in parsers:
                        response = parser.receive(text)
                        if response is not None:
                            emitted.append(response)
            for value in emitted:
                await send(value)
        for parser in parsers:
            response = parser.finalize()
            if response is not None:
                await send(response)
        return stream.result()

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    def collect_type_strategy_patches(
        self,
        signature: type[Signature],
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        inputs: dict[str, Any],
    ) -> LMRequestPatch:
        """Collect normalized request patches contributed by adapter type strategies."""
        patch = LMRequestPatch()
        previous_signature = getattr(self, "_current_signature", None)
        self._current_signature = signature
        try:
            for adapter_type in self.adapter_types:
                if hasattr(adapter_type, "prepare"):
                    patch = patch.merge(
                        adapter_type.prepare(
                            signature=signature,
                            lm=lm,
                            lm_kwargs=lm_kwargs,
                            inputs=inputs,
                            adapter=self,
                        )
                    )
        finally:
            if previous_signature is None:
                try:
                    delattr(self, "_current_signature")
                except AttributeError:
                    pass
            else:
                self._current_signature = previous_signature
        return patch

    def signature_without_patch_fields(self, signature: type[Signature], patch: LMRequestPatch) -> type[Signature]:
        """Return the prompt-facing signature after patch-requested field deletions."""
        for field_name in patch.delete_input_fields:
            if field_name in signature.fields:
                signature = signature.delete(field_name)
        for field_name in patch.delete_output_fields:
            if field_name in signature.fields:
                signature = signature.delete(field_name)
        return signature

    def merge_patch_into_request(self, request: LMRequest, patch: LMRequestPatch) -> LMRequest:
        """Merge request-level patch fields into a normalized LMRequest."""
        config = _merge_lm_config_objects(request.config, patch.config) or request.config
        return request.model_copy(
            update={
                "tools": [*request.tools, *patch.tools],
                "config": config,
                "metadata": {**request.metadata, **patch.metadata},
            },
            deep=True,
        )

    def _signature_without_strategy_parsed_output_fields(self, signature: type[Signature]) -> type[Signature]:
        for name, field in list(signature.output_fields.items()):
            adapter_type = _type_strategy_for(field.annotation, self.adapter_types)
            if adapter_type is not None and type(adapter_type).parse_output is not TypeStrategy.parse_output:
                signature = signature.delete(name)
        return signature

    def parse_type_strategy_outputs(
        self,
        original_signature: type[Signature],
        output: Any,
        value: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge output fields parsed by adapter type strategies into `value`."""
        for name, field in original_signature.output_fields.items():
            adapter_type = _type_strategy_for(field.annotation, self.adapter_types)
            if adapter_type is not None:
                parsed_value = adapter_type.parse_output(field_name=name, field=field, output=output, adapter=self)
                if parsed_value is not None:
                    value[name] = parsed_value
        return value

    def _call_preprocess(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> type[Signature]:
        patch = self.collect_type_strategy_patches(signature, lm, lm_kwargs, inputs)
        signature = _apply_lm_request_patch_to_legacy_adapter_call(signature, lm_kwargs, patch)

        if self.use_native_function_calling:
            tool_call_input_field_name = self._get_tool_call_input_field_name(signature)
            tool_call_output_field_name = self._get_tool_call_output_field_name(signature)

            if tool_call_output_field_name and tool_call_input_field_name is None:
                raise ValueError(
                    f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                    "input field with type `list[dspy.Tool]`."
                )

            if tool_call_output_field_name and _lm_supports_function_calling(lm):
                tools = inputs[tool_call_input_field_name]
                tools = tools if isinstance(tools, list) else [tools]

                lm_kwargs["tools"] = (
                    [tool.format_as_litellm_function_call() for tool in tools]
                    if isinstance(lm, BaseLM)
                    else tools
                )

                signature_for_native_function_calling = signature.delete(tool_call_output_field_name)
                signature_for_native_function_calling = signature_for_native_function_calling.delete(
                    tool_call_input_field_name
                )

                return signature_for_native_function_calling

        # Handle custom types that use native LM features, e.g., reasoning, citations, etc.
        for name, field in signature.output_fields.items():
            if _type_strategy_for(field.annotation, self.adapter_types) is not None:
                continue
            if (
                isinstance(field.annotation, type)
                and field.annotation in self.native_response_types
                and issubclass(field.annotation, Type)
            ):
                signature = field.annotation.adapt_to_native_lm_feature(signature, name, lm, lm_kwargs)

        return signature

    def _call_postprocess(
        self,
        processed_signature: type[Signature],
        original_signature: type[Signature],
        outputs: list[dict[str, Any] | str],
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        values = []

        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for output in outputs:
            output_logprobs = None
            tool_calls = None
            text = output

            if isinstance(output, dict):
                text = output["text"]
                output_logprobs = output.get("logprobs")
                tool_calls = output.get("tool_calls")

            if text:
                parse_signature = self._signature_without_strategy_parsed_output_fields(processed_signature)
                value = self.parse(parse_signature, text)
                for field_name in original_signature.output_fields.keys():
                    if field_name not in value:
                        # We need to set the field not present in the processed signature to None for consistency.
                        value[field_name] = None
            elif tool_calls and tool_call_output_field_name:
                value = {}
                for field_name in original_signature.output_fields.keys():
                    value[field_name] = None
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            if tool_calls and tool_call_output_field_name:
                tool_calls = [
                    {
                        "name": v["function"]["name"],
                        "args": json_repair.loads(v["function"]["arguments"]),
                    }
                    for v in tool_calls
                ]
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

            self.parse_type_strategy_outputs(original_signature, output, value)

            # Parse custom types that does not rely on the `Adapter.parse()` method
            for name, field in original_signature.output_fields.items():
                if (
                    isinstance(field.annotation, type)
                    and field.annotation in self.native_response_types
                    and issubclass(field.annotation, Type)
                ):
                    parsed_value = field.annotation.parse_lm_response(output)
                    if parsed_value is not None:
                        value[name] = parsed_value

            if output_logprobs:
                value["logprobs"] = output_logprobs

            values.append(value)

        return values

    def _call_postprocess_language_model(
        self,
        processed_signature: type[Signature],
        original_signature: type[Signature],
        response: LMResponse,
        lm: LanguageModel,
        lm_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        values = []
        tool_call_output_field_name = self._get_tool_call_output_field_name(original_signature)

        for output in response.outputs:
            text = output.text
            tool_calls = output.tool_calls

            if text:
                parse_signature = self._signature_without_strategy_parsed_output_fields(processed_signature)
                value = self.parse(parse_signature, text)
                for field_name in original_signature.output_fields.keys():
                    if field_name not in value:
                        value[field_name] = None
            elif tool_calls and tool_call_output_field_name:
                value = dict.fromkeys(original_signature.output_fields.keys())
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__,
                    signature=original_signature,
                    lm_response=str(output),
                    message="The LM returned an empty or null response.",
                )

            if tool_calls and tool_call_output_field_name:
                value[tool_call_output_field_name] = ToolCalls.from_dict_list(
                    [{"name": call.name, "args": call.args} for call in tool_calls]
                )

            self.parse_type_strategy_outputs(original_signature, output, value)

            for name, field in original_signature.output_fields.items():
                if (
                    isinstance(field.annotation, type)
                    and field.annotation in self.native_response_types
                    and issubclass(field.annotation, Type)
                ):
                    parsed_value = field.annotation.parse_lm_response(_lm_output_response_dict(output))
                    if parsed_value is not None:
                        value[name] = parsed_value

            if output.logprobs is not None:
                value["logprobs"] = output.logprobs

            values.append(value)

        return values

    def _language_model_request(
        self,
        lm: LanguageModel,
        messages: list[dict[str, Any] | LMMessage],
        lm_kwargs: dict[str, Any],
    ) -> LMRequest:
        # Delegate request construction to the LM so module calls preserve the
        # same constructor defaults as direct calls (temperature, max_tokens,
        # provider extensions, cache defaults, etc.). Constructing LMRequest
        # directly here would bypass lm.kwargs because lm(request=...) treats
        # the request as already normalized.
        return lm.normalize_request(messages=messages, **lm_kwargs)

    def __call__(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Execute the adapter pipeline: format inputs, call LM, and parse outputs.

        Args:
            lm: The Language Model instance to use for generation. Must be an instance of `dspy.BaseLM`.
            lm_kwargs: Additional keyword arguments to pass to the LM call (e.g., temperature, max_tokens). These are
                passed directly to the LM.
            signature: The DSPy signature associated with this LM call.
            demos: List of few-shot examples to include in the prompt. Each dictionary should contain keys matching the
                signature's input and output field names. Examples are formatted as user/assistant message pairs.
            inputs: The current input values for this call. Keys must match the signature's input field names.

        Returns:
            List of dictionaries representing parsed LM responses. Each dictionary contains keys matching the
            signature's output field names. For multiple generations (n > 1), returns multiple dictionaries.
        """
        if _uses_language_model_contract(lm):
            patch = self.collect_type_strategy_patches(signature, lm, lm_kwargs, inputs)
            processed_signature = self.signature_without_patch_fields(signature, patch)
            processed_signature = self._call_preprocess_language_model_builtin_types(lm, lm_kwargs, processed_signature, inputs)
            messages = self.format(processed_signature, demos, inputs, patch=patch)
            request = self._language_model_request(lm, messages, lm_kwargs)
            request = self.merge_patch_into_request(request, patch)
            if settings.send_stream is not None and _lm_streaming_enabled(lm):
                response = self._stream_language_model_response(lm, request)
            else:
                response = lm(request=request)
            return self.postprocess_language_model(processed_signature, signature, response, lm, lm_kwargs)

        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        messages = self.format(processed_signature, demos, inputs)
        outputs = lm(messages=legacy_messages_from_typed_messages(messages), **lm_kwargs)
        return self.postprocess_legacy(processed_signature, signature, outputs, lm, lm_kwargs)

    async def acall(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if _uses_language_model_contract(lm):
            patch = self.collect_type_strategy_patches(signature, lm, lm_kwargs, inputs)
            processed_signature = self.signature_without_patch_fields(signature, patch)
            processed_signature = self._call_preprocess_language_model_builtin_types(lm, lm_kwargs, processed_signature, inputs)
            messages = self.format(processed_signature, demos, inputs, patch=patch)
            request = self._language_model_request(lm, messages, lm_kwargs)
            request = self.merge_patch_into_request(request, patch)
            if settings.send_stream is not None and _lm_async_streaming_enabled(lm):
                response = await self._astream_language_model_response(lm, request)
            elif settings.send_stream is not None and _lm_streaming_enabled(lm):
                response = await asyncify(self._stream_language_model_response)(lm, request)
            else:
                response = await lm.acall(request=request)
            return self.postprocess_language_model(processed_signature, signature, response, lm, lm_kwargs)

        processed_signature = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        messages = self.format(processed_signature, demos, inputs)
        outputs = await lm.acall(messages=legacy_messages_from_typed_messages(messages), **lm_kwargs)
        return self.postprocess_legacy(processed_signature, signature, outputs, lm, lm_kwargs)

    def _call_preprocess_language_model_builtin_types(
        self,
        lm: LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> type[Signature]:
        """Apply legacy built-in native features to the normalized branch during migration."""
        if self.use_native_function_calling:
            tool_call_input_field_name = self._get_tool_call_input_field_name(signature)
            tool_call_output_field_name = self._get_tool_call_output_field_name(signature)
            if tool_call_output_field_name and tool_call_input_field_name is None:
                raise ValueError(
                    f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                    "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                    "input field with type `list[dspy.Tool]`."
                )
            if tool_call_output_field_name and _lm_supports_function_calling(lm):
                tools = inputs[tool_call_input_field_name]
                lm_kwargs["tools"] = tools if isinstance(tools, list) else [tools]
                signature = signature.delete(tool_call_output_field_name).delete(tool_call_input_field_name)

        for name, field in list(signature.output_fields.items()):
            if _type_strategy_for(field.annotation, self.adapter_types) is not None:
                continue
            if (
                isinstance(field.annotation, type)
                and field.annotation in self.native_response_types
                and issubclass(field.annotation, Type)
            ):
                signature = field.annotation.adapt_to_native_lm_feature(signature, name, lm, lm_kwargs)
        return signature

    def postprocess_language_model(
        self,
        processed_signature: type[Signature],
        original_signature: type[Signature],
        response: LMResponse,
        lm: LanguageModel,
        lm_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._call_postprocess_language_model(processed_signature, original_signature, response, lm, lm_kwargs)

    def postprocess_legacy(
        self,
        processed_signature: type[Signature],
        original_signature: type[Signature],
        outputs: list[dict[str, Any] | str],
        lm: BaseLM,
        lm_kwargs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self._call_postprocess(processed_signature, original_signature, outputs, lm, lm_kwargs)

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        patch: LMRequestPatch | None = None,
    ) -> list[dict[str, Any] | LMMessage]:
        """Render a DSPy call into LM messages.

        Concrete adapters own prompt rendering. The base adapter only owns shared LM-call plumbing: preprocessing,
        calling legacy or normalized LMs, streaming, and postprocessing.
        """
        raise NotImplementedError

    def _get_history_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                return name
        return None

    def _get_tool_call_input_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.input_fields.items():
            # Look for annotation `list[dspy.Tool]` or `dspy.Tool`
            origin = get_origin(field.annotation)
            if origin is list and field.annotation.__args__[0] == Tool:
                return name
            if field.annotation == Tool:
                return name
        return None

    def _get_tool_call_output_field_name(self, signature: type[Signature]) -> bool:
        for name, field in signature.output_fields.items():
            if field.annotation == ToolCalls:
                return name
        return None

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        """Parse the LM output into a dictionary of the output fields.

        This method parses the LM output into a dictionary of the output fields.

        Args:
            signature: The DSPy signature for which to parse the LM output.
            completion: The LM output to be parsed.

        Returns:
            A dictionary of the output fields.
        """
        raise NotImplementedError
