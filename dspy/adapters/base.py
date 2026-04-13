"""Adapter base class — the bridge between DSPy signatures and LM calls.

The adapter pipeline: preprocess → format → LM call → postprocess → parse.
``_call_preprocess`` handles native tool calling and native response types.
``_call_postprocess`` handles output parsing, tool call extraction, and
native type parsing.

``format()`` and ``parse()`` are implemented by subclasses (TemplateAdapter).
"""

import logging
from typing import Any, get_origin

import json_repair

from dspy.adapters.types import History, Type
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.experimental import Citations
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)

_DEFAULT_NATIVE_RESPONSE_TYPES = [Citations, Reasoning]


class Adapter:
    """Base Adapter class — handles preprocess/postprocess around format/parse."""

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[Type]] | None = None,
    ):
        self.callbacks = callbacks or []
        self.use_native_function_calling = use_native_function_calling
        self.native_response_types = native_response_types or _DEFAULT_NATIVE_RESPONSE_TYPES

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    # ------------------------------------------------------------------
    # Preprocess: native tool calling + native response types
    # ------------------------------------------------------------------

    def _call_preprocess(self, lm, lm_kwargs, signature, inputs):
        if self.use_native_function_calling:
            tc_in = self._get_tool_call_input_field_name(signature)
            tc_out = self._get_tool_call_output_field_name(signature)

            if tc_out and tc_in is None:
                raise ValueError(
                    f"Output field {tc_out} expects tool calls but no tool input field found. "
                    "Add an input field with type `list[dspy.Tool]`."
                )

            if tc_out and lm.supports_function_calling:
                tools = inputs[tc_in]
                tools = tools if isinstance(tools, list) else [tools]
                lm_kwargs["tools"] = [t.format_as_function_call() for t in tools]
                signature = signature.delete(tc_out).delete(tc_in)
                return signature

        for name, field in signature.output_fields.items():
            if (isinstance(field.annotation, type)
                    and field.annotation in self.native_response_types
                    and issubclass(field.annotation, Type)):
                signature = field.annotation.adapt_to_native_lm_feature(signature, name, lm, lm_kwargs)

        return signature

    # ------------------------------------------------------------------
    # Postprocess: parse outputs, extract tool calls, native types
    # ------------------------------------------------------------------

    def _call_postprocess(self, processed_sig, original_sig, outputs, lm, lm_kwargs):
        values = []
        tc_out = self._get_tool_call_output_field_name(original_sig)

        for output in outputs:
            logprobs = None
            tool_calls = None
            text = output

            if isinstance(output, dict):
                text = output["text"]
                logprobs = output.get("logprobs")
                tool_calls = output.get("tool_calls")

            if text:
                value = self.parse(processed_sig, text)
                for f in original_sig.output_fields:
                    if f not in value:
                        value[f] = None
            elif tool_calls and tc_out:
                value = {f: None for f in original_sig.output_fields}
            else:
                raise AdapterParseError(
                    adapter_name=type(self).__name__, signature=original_sig,
                    lm_response=str(output), message="Empty or null LM response.")

            if tool_calls and tc_out:
                parsed_tc = [{"name": v["function"]["name"],
                              "args": json_repair.loads(v["function"]["arguments"])} for v in tool_calls]
                value[tc_out] = ToolCalls.from_dict_list(parsed_tc)

            for name, field in original_sig.output_fields.items():
                if (isinstance(field.annotation, type)
                        and field.annotation in self.native_response_types
                        and issubclass(field.annotation, Type)):
                    parsed = field.annotation.parse_lm_response(output)
                    if parsed is not None:
                        value[name] = parsed

            if logprobs:
                value["logprobs"] = logprobs
            values.append(value)

        return values

    # ------------------------------------------------------------------
    # Call pipeline: preprocess → format → LM → postprocess
    # ------------------------------------------------------------------

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        processed = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        multimodal = _collect_multimodal_parts(inputs)
        messages = self.format(processed, demos, inputs)
        if multimodal:
            lm_kwargs["_multimodal_parts"] = multimodal
        outputs = lm(messages=messages, **lm_kwargs)
        return self._call_postprocess(processed, signature, outputs, lm, lm_kwargs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        processed = self._call_preprocess(lm, lm_kwargs, signature, inputs)
        multimodal = _collect_multimodal_parts(inputs)
        messages = self.format(processed, demos, inputs)
        if multimodal:
            lm_kwargs["_multimodal_parts"] = multimodal
        outputs = await lm.acall(messages=messages, **lm_kwargs)
        return self._call_postprocess(processed, signature, outputs, lm, lm_kwargs)

    # ------------------------------------------------------------------
    # Abstract methods — implemented by TemplateAdapter
    # ------------------------------------------------------------------

    def format(self, signature, demos, inputs, **kw):
        raise NotImplementedError

    def parse(self, signature, completion, **kw):
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_history_field_name(sig):
        for name, field in sig.input_fields.items():
            if field.annotation == History:
                return name
        return None

    @staticmethod
    def _get_tool_call_input_field_name(sig):
        for name, field in sig.input_fields.items():
            origin = get_origin(field.annotation)
            if origin is list and field.annotation.__args__[0] == Tool:
                return name
            if field.annotation == Tool:
                return name
        return None

    @staticmethod
    def _get_tool_call_output_field_name(sig):
        for name, field in sig.output_fields.items():
            if field.annotation == ToolCalls:
                return name
        return None


def _collect_multimodal_parts(inputs: dict[str, Any]) -> list:
    """Extract lm15 Parts from any multimodal Type values in inputs."""
    parts = []
    for v in inputs.values():
        if isinstance(v, Type) and hasattr(v, "to_lm15_part"):
            p = v.to_lm15_part()
            if p is not None:
                parts.append(p)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, Type) and hasattr(item, "to_lm15_part"):
                    p = item.to_lm15_part()
                    if p is not None:
                        parts.append(p)
    return parts
