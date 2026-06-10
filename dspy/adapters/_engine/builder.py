"""Plan builder: the engine-side replacement for ``Adapter._call_preprocess``.

This closes TODO #1 (base.py:90-95): instead of mutating ``lm_kwargs`` and
returning only a render signature — losing what was decided — the builder
performs the IDENTICAL mutations in the IDENTICAL order while recording
every decision onto an :class:`AdapterPlan`: native tool specs, hidden
fields (as Hide transforms with backfill metadata), and the lm_kwargs deltas
produced by native-response-type hooks.

Behavior parity is structural, not aspirational: the legacy logic was moved
here line-for-line (tool stripping order, the snapshot semantics of the
native-response-types loop, ``Type.adapt_to_native_lm_feature`` still being
the planning hook — TODO #2 stays open until the strategies PR). The golden
corpus adjudicates byte-identity.
"""

from dataclasses import dataclass
from typing import Any

from dspy.adapters._engine.ir import AdapterPlan
from dspy.adapters._engine.transforms import HideInputField, HideOutputField, apply_field_transforms

_TOOL_PROVIDER_KEYS = ("tools", "tool_choice", "parallel_tool_calls")


@dataclass
class BuiltCall:
    """The builder's result: the recorded plan plus the legacy contract.

    ``render_signature`` preserves ``_call_preprocess``'s return value — the
    signature ``format()`` receives after native-feature field deletions.
    The plan records the same information as Hide transforms; both views
    must stay consistent (asserted in tests).
    """

    plan: AdapterPlan
    render_signature: Any


def build_plan(adapter, lm, lm_kwargs: dict[str, Any], signature, inputs: dict[str, Any]) -> BuiltCall:
    """Build the plan for one attempt. Always build fresh per attempt — the
    ChatAdapter->JSONAdapter fallback retry constructs a new adapter whose
    call must re-plan against the (mutated) lm_kwargs it receives."""
    # Lazy import: dspy.adapters.base must never import the engine at module
    # level (importing dspy must not load the engine), so the dependency
    # points this way only.
    from dspy.adapters import base as adapter_base
    from dspy.adapters.types import Type

    plan = AdapterPlan.from_signature(signature, inputs)
    transforms = []
    render_signature = signature

    # --- Native function-calling planning (verbatim from base.py:96-121) ---
    if not adapter.use_native_function_calling:
        for key in _TOOL_PROVIDER_KEYS:
            lm_kwargs.pop(key, None)
    else:
        tool_call_input_field_name = adapter._get_tool_call_input_field_name(signature)
        tool_call_output_field_name = adapter._get_tool_call_output_field_name(signature)

        if tool_call_output_field_name and tool_call_input_field_name is None:
            raise ValueError(
                f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                "input field with type `list[dspy.Tool]`."
            )

        if tool_call_output_field_name and lm.supports_function_calling:
            tools = inputs[tool_call_input_field_name]
            tools = tools if isinstance(tools, list) else [tools]

            lm_tools = [tool.format_as_litellm_function_call() for tool in tools]

            lm_kwargs["tools"] = lm_tools
            if adapter.parallel_tool_calls is not None and lm_kwargs.get("parallel_tool_calls") is None:
                lm_kwargs["parallel_tool_calls"] = adapter.parallel_tool_calls

            render_signature = render_signature.delete(tool_call_output_field_name)
            render_signature = render_signature.delete(tool_call_input_field_name)

            plan.tools.extend(lm_tools)
            transforms.append(HideInputField(tool_call_input_field_name, reason="native_function_calling"))
            transforms.append(HideOutputField(tool_call_output_field_name, reason="native_function_calling"))

    # --- Native response-type planning (verbatim from base.py:127-133) -----
    # TODO #2 compatibility: Type.adapt_to_native_lm_feature remains the
    # planning hook until the strategies PR; the builder records its effects.
    # Loop semantics preserved exactly: iterate the snapshot of output fields
    # as of loop start, reassigning render_signature inside.
    #
    # `issubclass` deliberately resolves through the adapters.base module
    # namespace: tests/predict/test_react.py shadows it there to prove the
    # `isinstance(..., type)` guard keeps generic aliases away from it.
    for name, field in render_signature.output_fields.items():
        if (
            isinstance(field.annotation, type)
            and field.annotation in adapter.native_response_types
            and getattr(adapter_base, "issubclass", issubclass)(field.annotation, Type)
        ):
            fields_before = set(render_signature.output_fields)
            kwargs_before = dict(lm_kwargs)
            render_signature = field.annotation.adapt_to_native_lm_feature(render_signature, name, lm, lm_kwargs)

            for deleted in sorted(fields_before - set(render_signature.output_fields)):
                transforms.append(HideOutputField(deleted, reason=f"native:{field.annotation.__name__}"))
            delta = {
                key: lm_kwargs[key]
                for key in lm_kwargs
                if key not in kwargs_before or kwargs_before[key] is not lm_kwargs[key]
            }
            if delta:
                plan.metadata.setdefault("native_feature_kwargs", {})[field.annotation.__name__] = delta

    plan.field_transforms.extend(transforms)
    plan.input_fields, plan.output_fields, transform_warnings = apply_field_transforms(
        plan.input_fields, plan.output_fields, transforms
    )
    plan.warnings.extend(transform_warnings)

    _record_format_parser(adapter, plan)

    return BuiltCall(plan=plan, render_signature=render_signature)


def _record_format_parser(adapter, plan: AdapterPlan) -> None:
    """Record the resolved Format's text parser on the plan, so the IR is
    complete and rendering/parsing demonstrably share one Format instance
    (the coupling invariant). Consumed directly once engine postprocess
    parses LMResponse objects."""
    from dspy.adapters._engine.formats import resolve_format
    from dspy.adapters._engine.overrides import resolve_override_verdict

    if resolve_override_verdict(adapter).engine_eligible:
        fmt = resolve_format(adapter)
        if fmt is not None:
            plan.parsers.append(fmt.make_parser_hook(adapter))


def assert_unrendered(plan: AdapterPlan) -> None:
    """Tripwire until the renderer lands: the plan-boundary PR must not put
    request content on plans, so silent content drops are impossible."""
    if plan.system_parts or plan.user_parts or plan.history_messages or plan.assistant_prefill_parts:
        raise AssertionError("AdapterPlan carries request content but the engine renderer is not wired yet")
