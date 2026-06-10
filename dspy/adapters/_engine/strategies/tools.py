"""Native function-calling as a call-level plan step.

Spans an input field (the available ``Tool``/``list[Tool]``) AND an output
field (``ToolCalls``) — which is exactly why it is a PlanStep, not a
per-field TypeStrategy. Lifted verbatim from the builder (originally
base.py's ``_call_preprocess``); the golden corpus pins every byte of its
``lm_kwargs`` effects.
"""

from dspy.adapters._engine.patch import AdapterPatch, DebugLink
from dspy.adapters._engine.transforms import HideInputField, HideOutputField
from dspy.core.types import LMRequestPatch

_TOOL_PROVIDER_KEYS = ("tools", "tool_choice", "parallel_tool_calls")


class NativeFunctionCallingStep:
    name = "tools.native"
    priority = 500
    exclusive_group = "tool_calls.representation"

    def applies(self, ctx) -> bool:
        # The step always participates: when native FC is OFF its job is to
        # strip provider tool params (the legacy contract).
        return True

    def contribute(self, ctx) -> AdapterPatch:
        adapter, lm, lm_kwargs, signature, inputs = ctx.adapter, ctx.lm, ctx.lm_kwargs, ctx.signature, ctx.inputs

        if not adapter.use_native_function_calling:
            for key in _TOOL_PROVIDER_KEYS:
                lm_kwargs.pop(key, None)
            return AdapterPatch()

        tool_call_input_field_name = adapter._get_tool_call_input_field_name(signature)
        tool_call_output_field_name = adapter._get_tool_call_output_field_name(signature)

        if tool_call_output_field_name and tool_call_input_field_name is None:
            raise ValueError(
                f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                "input field with type `list[dspy.Tool]`."
            )

        if not (tool_call_output_field_name and lm.supports_function_calling):
            return AdapterPatch()

        tools = inputs[tool_call_input_field_name]
        tools = tools if isinstance(tools, list) else [tools]

        lm_tools = [tool.format_as_litellm_function_call() for tool in tools]

        lm_kwargs["tools"] = lm_tools
        if adapter.parallel_tool_calls is not None and lm_kwargs.get("parallel_tool_calls") is None:
            lm_kwargs["parallel_tool_calls"] = adapter.parallel_tool_calls

        return AdapterPatch(
            request=LMRequestPatch(tools=lm_tools),
            field_transforms=[
                HideInputField(tool_call_input_field_name, reason="native_function_calling"),
                HideOutputField(tool_call_output_field_name, reason="native_function_calling"),
            ],
            debug_links=[
                DebugLink(
                    source=f"Signature.inputs.{tool_call_input_field_name}",
                    strategy=self.name,
                    destination="lm.tools",
                ),
                DebugLink(
                    source=f"Signature.outputs.{tool_call_output_field_name}",
                    strategy=self.name,
                    destination="lm.tools + native tool-call parsing",
                ),
            ],
        )
