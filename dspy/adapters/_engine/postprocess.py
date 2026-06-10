"""Engine postprocess: typed LMResponse to parsed values, via plan parsers.

Closes TODO #3 for the engine path: instead of converting the normalized
response back to legacy outputs and hardcoding the parse sequence, the
engine derives ResponseViews from the LMResponse (through the SAME canonical
conversion rules, shared not reimplemented) and runs the plan's parsers —
the format/extraction text hook in the legacy text-parse position, then the
strategy-contributed channel hooks and third-party compatibility hooks in
plan order, with the tool-call merge, None-backfill, and logprobs semantics
reproduced line-for-line from the legacy loop.

One deliberate non-reproduction, recorded here because the corpus cannot
see it: legacy ``Reasoning.parse_lm_response`` evaluates
``"reasoning_content" in response`` on STR outputs too — a substring check
that would crash with a TypeError on pathological completions containing
that literal substring. The channel hooks read structured channels only and
cannot crash that way.
"""

from typing import Any

from dspy.adapters._engine.parse import FormatParserHook
from dspy.adapters._engine.parser_hook import ParseContext, ResponseView


def run_engine_postprocess(
    adapter, processed_signature, original_signature, response, lm, plan
) -> list[dict[str, Any]]:
    from dspy.adapters._engine.formats.twostep import TwoStepExtractionParserHook
    from dspy.adapters.base import _provider_tool_call_to_tool_call_dict
    from dspy.adapters.types.tool import ToolCalls
    from dspy.utils.exceptions import AdapterParseError

    views = ResponseView.from_lm_response(response)
    # Text parsing executes through the PUBLIC adapter.parse() — its engine
    # branch reaches the same resolved Format, and the with_callbacks wrapper
    # fires the parse_start/parse_end events the callback corpus pins (the
    # plan's Format/extraction hook remains the IR record). The corpus caught
    # exactly this when an earlier draft called the hook directly.
    channel_hooks = [p for p in plan.parsers if not isinstance(p, (FormatParserHook, TwoStepExtractionParserHook))]
    backfill_field_names = [field.destination_name for field in plan.output_fields]
    tool_call_output_field_name = next(
        (field.destination_name for field in plan.output_fields if field.annotation == ToolCalls), None
    )
    ctx = ParseContext(plan=plan, signature=processed_signature, lm=lm)

    def parse_text(view):
        return adapter.parse(processed_signature, view.text)

    values = []
    for view in views:
        output = view.raw
        output_logprobs = None
        tool_calls = None
        text = output

        if isinstance(output, dict):
            text = output["text"]
            output_logprobs = output.get("logprobs")
            tool_calls = output.get("tool_calls")

        if text and not (tool_calls and tool_call_output_field_name):
            value = parse_text(view)
        elif tool_calls and tool_call_output_field_name:
            try:
                value = parse_text(view) if text and processed_signature.output_fields else {}
            except AdapterParseError:
                value = {}
        else:
            raise AdapterParseError(
                adapter_name=type(adapter).__name__,
                signature=original_signature,
                lm_response=str(output),
                message="The LM returned an empty or null response.",
            )

        # Fields removed for native features are absent from the processed parse.
        for field_name in backfill_field_names:
            value.setdefault(field_name, None)

        if tool_calls and tool_call_output_field_name:
            tool_calls = [_provider_tool_call_to_tool_call_dict(tool_call) for tool_call in tool_calls]
            value[tool_call_output_field_name] = ToolCalls.from_dict_list(tool_calls)

        for hook in channel_hooks:
            value.update(hook.parse(view, ctx))

        if output_logprobs:
            value["logprobs"] = output_logprobs

        values.append(value)

    return values
