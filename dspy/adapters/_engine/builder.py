"""Plan builder: the engine-side replacement for ``Adapter._call_preprocess``.

This closes TODO #1 (the explicit plan) and TODO #2 (provider-specific
planning out of semantic types): native function calling runs as a
call-level :class:`PlanStep`, and every native response type runs through
ONE uniform per-field :class:`TypeStrategy` loop. Built-ins (Reasoning,
Citations) resolve from the strategy registry, with gates in shared
predicates inside the type modules — the legacy hooks import the SAME
predicates, so the two paths cannot drift. Third-party types resolve to a
registered strategy when one exists, else to their documented
``Type.adapt_to_native_lm_feature`` hook auto-wrapped in
:class:`LegacyTypeHookStrategy` — silently honored (deprecation is a future
exposure epic's decision), with effects captured onto the plan either way.

Behavior parity is structural: identical ``lm_kwargs`` mutations in
identical order (step first, then the snapshot loop over output fields),
identical signature derivation, adjudicated by the golden corpus.
"""

from dataclasses import dataclass
from typing import Any

from dspy.adapters._engine.ir import AdapterPlan
from dspy.adapters._engine.patch import StrategyTrace
from dspy.adapters._engine.transforms import apply_field_transforms


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
    from dspy.adapters._engine.strategies import NativeFunctionCallingStep, field_strategy_for
    from dspy.adapters._engine.strategy import CallContext, FieldContext
    from dspy.adapters.types import Type

    plan = AdapterPlan.from_signature(signature, inputs)
    render_signature = signature

    # --- Native function-calling planning (call-level step) ----------------
    step = NativeFunctionCallingStep()
    step_ctx = CallContext(
        adapter=adapter, plan=plan, signature=render_signature, inputs=inputs, lm=lm, lm_kwargs=lm_kwargs
    )
    patch = step.contribute(step_ctx)
    render_signature = _apply_patch(plan, patch, render_signature, step.name)

    # --- Native response-type planning (per-field strategies + fallback) ---
    # Loop semantics preserved exactly from the legacy hook chain: iterate
    # the snapshot of output fields as of loop start, deriving the render
    # signature inside.
    #
    # `issubclass` deliberately resolves through the adapters.base module
    # namespace: tests/predict/test_react.py shadows it there to prove the
    # `isinstance(..., type)` guard keeps generic aliases away from it.
    for name, field in render_signature.output_fields.items():
        if not (
            isinstance(field.annotation, type)
            and field.annotation in adapter.native_response_types
            and getattr(adapter_base, "issubclass", issubclass)(field.annotation, Type)
        ):
            continue

        strategy = field_strategy_for(field.annotation)
        render_field = plan.find_field("output", name)
        ctx = FieldContext(
            adapter=adapter,
            plan=plan,
            field=render_field,
            role="output",
            lm=lm,
            lm_kwargs=lm_kwargs,
            signature=render_signature,
        )
        if strategy.applies(ctx):
            kwargs_before = dict(lm_kwargs)
            patch = strategy.contribute(ctx)
            self_traced = bool(patch.strategy_trace)
            render_signature = _apply_patch(plan, patch, render_signature, strategy.name)
            _record_kwargs_delta(plan, field.annotation.__name__, kwargs_before, lm_kwargs)
            # A strategy may self-report its trace via the patch (the legacy
            # wrapper does: its decision depends on observed effects); the
            # builder records the standard applies-based entry otherwise.
            if not self_traced:
                plan.strategy_trace.append(
                    StrategyTrace(strategy=strategy.name, field=name, decision="selected", reason="applies")
                )
        else:
            plan.strategy_trace.append(
                StrategyTrace(strategy=strategy.name, field=name, decision="skipped", reason="applies=False")
            )

    plan.input_fields, plan.output_fields, transform_warnings = apply_field_transforms(
        plan.input_fields, plan.output_fields, plan.field_transforms
    )
    plan.warnings.extend(transform_warnings)

    _record_format_parser(adapter, plan)

    return BuiltCall(plan=plan, render_signature=render_signature)


def _apply_patch(plan: AdapterPlan, patch, render_signature, source_name: str):
    """Merge a strategy/step patch into the plan AND derive the legacy
    render-signature deletions from its hide/delete channels: a field hidden
    by the engine is a field deleted from the signature format() receives."""
    from dspy.adapters._engine.transforms import HideInputField, HideOutputField

    if patch.replace_render_signature is not None:
        # Legacy-hook channel: the hook already returned the rewritten
        # signature (deletions applied, and possibly richer edits a
        # delete-only reconstruction would drop). Use it wholesale.
        patch.merge_into(plan)
        return patch.replace_render_signature

    for transform in patch.field_transforms:
        if isinstance(transform, (HideInputField, HideOutputField)):
            render_signature = render_signature.delete(transform.name)
    for name in (*patch.request.delete_output_fields, *patch.request.delete_input_fields):
        render_signature = render_signature.delete(name)
    patch.merge_into(plan)
    return render_signature


def _record_kwargs_delta(plan: AdapterPlan, type_name: str, before: dict, after: dict) -> None:
    delta = {key: after[key] for key in after if key not in before or before[key] is not after[key]}
    if delta:
        plan.metadata.setdefault("native_feature_kwargs", {})[type_name] = delta


def _record_format_parser(adapter, plan: AdapterPlan) -> None:
    """Record the resolved Format's parser on the plan, so the IR is
    complete and rendering/parsing demonstrably share one Format instance
    (the coupling invariant)."""
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
