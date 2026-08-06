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
    from dspy.adapters._engine.strategies import NativeFunctionCallingStep, strategy_for
    from dspy.adapters._engine.strategies.vocabulary import NATIVE_BINDINGS, TEXTUAL_BINDINGS
    from dspy.adapters._engine.strategy import CallContext, FieldContext
    from dspy.adapters.types import Type

    plan = AdapterPlan.from_signature(signature, inputs)
    render_signature = signature

    # Per-role strategy bindings (constructor-validated); every unbound role
    # is "auto". Declared bindings are recorded so the plan states them.
    bindings = getattr(adapter, "strategies", None) or {}
    if bindings:
        plan.metadata["strategy_bindings"] = dict(bindings)

    # --- Native function-calling planning (call-level step) ----------------
    if bindings.get("tools") == "native_fc":
        # An explicit native binding refuses where "auto" silently rendered
        # tools textually (declare-don't-discover, ADP-006).
        if adapter._get_tool_call_output_field_name(signature) and not lm.supports_function_calling:
            raise ValueError(
                f"strategies binding tools='native_fc' cannot be honored: lm {getattr(lm, 'model', lm)!r} "
                "does not support function calling — an explicit native binding refuses instead of "
                "silently rendering tools textually"
            )
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

        render_field = plan.find_field("output", name)
        semantic_role = (render_field.metadata or {}).get("semantic_role", "plain")
        binding = bindings.get(semantic_role, "auto")

        if binding in TEXTUAL_BINDINGS.get(semantic_role, ()):
            # Bound textual: the native strategy stands down and the field
            # stays visible for the template to host.
            plan.strategy_trace.append(
                StrategyTrace(
                    strategy=f"{semantic_role}.{binding}",
                    field=name,
                    decision="selected",
                    reason=f"bound: {binding}",
                    resolved_by="binding",
                )
            )
            _record_resolution(plan, name, semantic_role, binding, served="textual")
            continue

        strategy, resolved_by = strategy_for(semantic_role, field.annotation)
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
                    StrategyTrace(
                        strategy=strategy.name,
                        field=name,
                        decision="selected",
                        reason="applies",
                        resolved_by=resolved_by,
                    )
                )
                _record_resolution(plan, name, semantic_role, binding, served="native")
        else:
            if binding == NATIVE_BINDINGS.get(semantic_role):
                raise ValueError(
                    f"strategies binding {semantic_role}={binding!r} cannot be honored for field {name!r}: "
                    f"{strategy.name} does not apply for lm {getattr(lm, 'model', lm)!r} — an explicit "
                    "native binding refuses instead of silently falling back to the textual lane"
                )
            plan.strategy_trace.append(
                StrategyTrace(
                    strategy=strategy.name,
                    field=name,
                    decision="skipped",
                    reason="applies=False",
                    resolved_by=resolved_by,
                )
            )
            _record_resolution(plan, name, semantic_role, binding, served="textual")

    plan.input_fields, plan.output_fields, transform_warnings = apply_field_transforms(
        plan.input_fields, plan.output_fields, plan.field_transforms
    )
    plan.warnings.extend(transform_warnings)

    fmt = _record_format_parser(adapter, plan)
    if fmt is not None and fmt.preset_name:
        _check_template_capacity(fmt, lm, plan)

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


def _record_resolution(plan: AdapterPlan, field_name: str, role: str, binding: str, served: str) -> None:
    """Record how one field's role resolved at bake (declare-don't-discover):
    the declared binding and the lane it landed in. Self-traced legacy hooks
    record through their own trace instead."""
    plan.metadata.setdefault("strategy_resolution", {})[field_name] = {
        "role": role,
        "binding": binding,
        "served": served,
    }


def _record_kwargs_delta(plan: AdapterPlan, type_name: str, before: dict, after: dict) -> None:
    delta = {key: after[key] for key in after if key not in before or before[key] is not after[key]}
    if delta:
        plan.metadata.setdefault("native_feature_kwargs", {})[type_name] = delta


def _record_format_parser(adapter, plan: AdapterPlan):
    """Record the resolved Format's parser on the plan, so the IR is
    complete and rendering/parsing demonstrably share one Format instance
    (the coupling invariant). Returns the Format when the engine path will
    render (the bake capacity checks apply exactly then)."""
    from dspy.adapters._engine.formats import resolve_format
    from dspy.adapters._engine.overrides import resolve_override_verdict

    if resolve_override_verdict(adapter).engine_eligible:
        fmt = resolve_format(adapter)
        if fmt is not None:
            plan.parsers.append(fmt.make_parser_hook(adapter))
            return fmt
    return None


def _check_template_capacity(fmt, lm, plan: AdapterPlan) -> None:
    """The bake-time triple check: signature roles x LM capabilities x
    template capacity.

    ADP-006: every textually-served role-bearing field needs a textual lane
    in the template — media/tools are per-field questions; a missing lane
    refuses naming the field+role, the LM, and the template. ADP-007: an
    explicit template slot naming a natively-hidden field refuses at bake,
    in the live and example lanes both, never rendering empty. The built-in
    presets host every role and carry no explicit field slots, so existing
    programs never trip either check.
    """
    from dspy.adapters._engine.presets import effective_capacity

    capacity = effective_capacity(fmt)
    template_name = fmt.preset_name + (
        " (system override)" if getattr(fmt, "system_template_message", None) is not None else ""
    )
    lm_name = getattr(lm, "model", lm)

    for render_field in (*plan.input_fields, *plan.output_fields):
        name = render_field.name
        if render_field.hidden:
            if name in capacity.field_slots:
                raise ValueError(
                    f"template {template_name!r} places field {name!r} in an explicit slot, but the field "
                    f"is served natively for lm {lm_name!r} and leaves the token stream — an explicit slot "
                    "referencing a natively-served field refuses at bake, never renders empty (ADP-007)"
                )
            if name in capacity.directive_field_slots:
                raise ValueError(
                    f"template {template_name!r} places field {name!r} in an explicit slot of its "
                    f"demos/history patterns, but the field is served natively for lm {lm_name!r} — "
                    "example turns cannot show a field the live call hides (ADP-007)"
                )
            continue

        role = (render_field.metadata or {}).get("semantic_role", "plain")
        if role == "plain":
            continue
        if role in ("media", "tools"):
            hosted = capacity.hosts_role_textually(role, field=name)
        elif role == "history":
            hosted = capacity.hosts_role_textually(role)
        elif render_field.role == "input":
            hosted = capacity.iterates_inputs or name in capacity.field_slots
        else:
            hosted = capacity.hosts_role_textually(role)
        if not hosted:
            raise ValueError(
                f"no lane for field {name!r} (role {role!r}): lm {lm_name!r} serves it textually, but "
                f"template {template_name!r} declares no textual hosting for it — the "
                "(roles x capabilities x capacity) triple refuses at bake instead of degrading (ADP-006)"
            )


def assert_unrendered(plan: AdapterPlan) -> None:
    """Tripwire until the renderer lands: the plan-boundary PR must not put
    request content on plans, so silent content drops are impossible."""
    if plan.system_parts or plan.user_parts or plan.history_messages or plan.assistant_prefill_parts:
        raise AssertionError("AdapterPlan carries request content but the engine renderer is not wired yet")
