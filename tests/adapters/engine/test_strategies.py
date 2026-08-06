"""QC-10 gates: provider hacks live in strategies with shared predicates,
third-party Type hooks stay silently honored, and the strategy contracts
carry the shapes a future exposure epic would publish unchanged."""

import inspect as inspect_module

from golden.harness import Recorder, StubLM

import dspy
from dspy.adapters._engine.builder import build_plan
from dspy.adapters._engine.patch import AdapterPatch
from dspy.adapters._engine.strategies import (
    NativeCitationsStrategy,
    NativeFunctionCallingStep,
    NativeReasoningStrategy,
    builtin_field_strategy_for,
)
from dspy.adapters._engine.strategies.citations import NativeCitationsParserHook
from dspy.adapters._engine.strategies.reasoning import NativeReasoningParserHook
from dspy.adapters._engine.strategy import PlanStep, TypeStrategy
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.reasoning import Reasoning
from dspy.experimental import Citations


class ThoughtfulQA(dspy.Signature):
    """Think then answer."""

    question: str = dspy.InputField()
    reasoning: Reasoning = dspy.OutputField()
    answer: str = dspy.OutputField()


def _lm(**caps):
    return StubLM(Recorder(), **caps)


def test_strategy_contracts_match_exposure_shapes():
    """name/priority/exclusive_group + applies/contribute over a context —
    the strategies.md vocabulary, verifiable via the runtime protocols."""
    for strategy in (NativeReasoningStrategy(), NativeCitationsStrategy()):
        assert isinstance(strategy, TypeStrategy)
        assert strategy.priority == 500
        assert strategy.exclusive_group.endswith(".representation")
    assert isinstance(NativeFunctionCallingStep(), PlanStep)
    assert NativeFunctionCallingStep().name == "tools.native"


def test_builtin_mapping():
    assert isinstance(builtin_field_strategy_for(Reasoning), NativeReasoningStrategy)
    assert isinstance(builtin_field_strategy_for(Citations), NativeCitationsStrategy)
    assert builtin_field_strategy_for(str) is None


def test_reasoning_strategy_effects_and_trace():
    lm_kwargs = {}
    built = build_plan(ChatAdapter(), _lm(supports_reasoning=True), lm_kwargs, ThoughtfulQA, {"question": "Q?"})

    assert lm_kwargs == {"reasoning_effort": "low"}
    assert built.plan.metadata["native_feature_kwargs"]["Reasoning"] == {"reasoning_effort": "low"}
    assert "reasoning" not in built.render_signature.output_fields
    assert built.plan.find_field("output", "reasoning").hidden
    assert any(isinstance(p, NativeReasoningParserHook) for p in built.plan.parsers)
    selected = [t for t in built.plan.strategy_trace if t.strategy == "reasoning.native"]
    assert len(selected) == 1 and selected[0].decision == "selected"


def test_reasoning_strategy_skips_record_trace():
    built = build_plan(ChatAdapter(), _lm(), {}, ThoughtfulQA, {"question": "Q?"})
    skipped = [t for t in built.plan.strategy_trace if t.strategy == "reasoning.native"]
    assert len(skipped) == 1 and skipped[0].decision == "skipped"
    assert "reasoning" in built.render_signature.output_fields


def test_shared_predicates_are_the_single_source():
    """Both the legacy hooks and the strategies must read the SAME predicate
    functions — enforced structurally by source inspection."""
    from dspy.adapters.types import citation, reasoning

    assert "native_citations_supported(lm)" in inspect_module.getsource(citation.Citations.adapt_to_native_lm_feature)
    assert "resolve_native_reasoning_effort(lm, lm_kwargs)" in inspect_module.getsource(
        reasoning.Reasoning.adapt_to_native_lm_feature
    )
    assert "native_citations_supported" in inspect_module.getsource(NativeCitationsStrategy.applies)
    assert "resolve_native_reasoning_effort" in inspect_module.getsource(NativeReasoningStrategy.applies)


def test_types_never_import_the_engine():
    """Dependency direction: engine imports types; type modules must not
    import the engine anywhere in their source."""
    from dspy.adapters.types import base_type, citation, reasoning, tool

    for module in (citation, reasoning, tool, base_type):
        assert "_engine" not in inspect_module.getsource(module), module.__name__


def test_third_party_type_hook_still_silently_honored():
    """A custom Type in native_response_types with its own
    adapt_to_native_lm_feature must keep working on the engine path, with no
    deprecation warning."""
    calls = {}

    class CustomNative(dspy.Type):
        content: str = "x"

        @classmethod
        def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
            calls["hook"] = field_name
            lm_kwargs["custom_flag"] = True
            return signature.delete(field_name)

        @classmethod
        def description(cls):
            return "custom"

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        custom: CustomNative = dspy.OutputField()
        answer: str = dspy.OutputField()

    import warnings as warnings_module

    adapter = ChatAdapter(native_response_types=[CustomNative])
    lm_kwargs = {}
    with warnings_module.catch_warnings():
        warnings_module.simplefilter("error", DeprecationWarning)
        built = build_plan(adapter, _lm(), lm_kwargs, Sig, {"question": "Q?"})

    assert calls["hook"] == "custom"
    assert lm_kwargs == {"custom_flag": True}
    assert "custom" not in built.render_signature.output_fields
    assert built.plan.find_field("output", "custom").hidden
    assert built.plan.metadata["native_feature_kwargs"]["CustomNative"] == {"custom_flag": True}
    trace = [t for t in built.plan.strategy_trace if t.strategy == "type_hook:CustomNative"]
    assert len(trace) == 1 and trace[0].decision == "selected"


def test_native_types_excluded_when_not_configured():
    """native_response_types gates strategies exactly as it gated hooks."""
    adapter = ChatAdapter(native_response_types=[Citations])  # Reasoning excluded
    lm_kwargs = {}
    built = build_plan(adapter, _lm(supports_reasoning=True), lm_kwargs, ThoughtfulQA, {"question": "Q?"})
    assert lm_kwargs == {}
    assert "reasoning" in built.render_signature.output_fields
    assert not any(t.strategy == "reasoning.native" for t in built.plan.strategy_trace)


def test_citations_parser_hook_reads_channel():
    from golden.harness import canonicalize

    hook = NativeCitationsParserHook("citations")
    view_data = {
        "text": "x",
        "citations": [{"cited_text": "Water boils.", "document_index": 0, "start_char_index": 0, "end_char_index": 12}],
    }
    from dspy.adapters._engine.parser_hook import ParseContext, ResponseView

    values = hook.parse(ResponseView(view_data), ParseContext(plan=None))
    assert isinstance(values["citations"], Citations)
    assert canonicalize(hook.parse(ResponseView("plain"), ParseContext(plan=None))) == {}


def test_reasoning_parser_hook_reads_channel():
    from dspy.adapters._engine.parser_hook import ParseContext, ResponseView

    hook = NativeReasoningParserHook("reasoning")
    values = hook.parse(ResponseView({"text": "x", "reasoning_content": "thinking"}), ParseContext(plan=None))
    assert values == {"reasoning": Reasoning(content="thinking")}
    assert hook.parse(ResponseView("plain"), ParseContext(plan=None)) == {}


def test_contribute_returns_adapter_patch():
    lm_kwargs = {}
    adapter = ChatAdapter()
    built = build_plan(adapter, _lm(supports_reasoning=True), lm_kwargs, ThoughtfulQA, {"question": "Q?"})
    # Spot-check the patch type contract directly.
    from dspy.adapters._engine.ir import RenderField
    from dspy.adapters._engine.strategy import FieldContext

    ctx = FieldContext(
        adapter=adapter,
        plan=built.plan,
        field=RenderField(name="reasoning", original_name="reasoning", role="output", annotation=Reasoning),
        role="output",
        lm=_lm(supports_reasoning=True),
        lm_kwargs={},
    )
    strategy = NativeReasoningStrategy()
    assert strategy.applies(ctx)
    assert isinstance(strategy.contribute(ctx), AdapterPatch)


class _BadgeNative(dspy.Type):
    """Third-party fixture: a custom native type used both ways below."""

    content: str = "x"

    @classmethod
    def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
        lm_kwargs["badge_flag"] = True
        return signature.delete(field_name)

    @classmethod
    def parse_lm_response(cls, output):
        channel = output.get("badge") if isinstance(output, dict) else None
        return cls(content=channel) if channel is not None else None

    @classmethod
    def description(cls):
        return "badge"


class _BadgeSig(dspy.Signature):
    question: str = dspy.InputField()
    badge: _BadgeNative = dspy.OutputField()
    answer: str = dspy.OutputField()


class _BadgeStrategy:
    """The same behavior expressed as a proper TypeStrategy."""

    name = "badge.native"
    priority = 500
    exclusive_group = "badge.representation"

    def applies(self, ctx) -> bool:
        return True

    def contribute(self, ctx) -> AdapterPatch:
        from dspy.adapters._engine.parser_hook import ThirdPartyNativeParserHook
        from dspy.adapters._engine.transforms import HideOutputField

        ctx.lm_kwargs["badge_flag"] = True
        return AdapterPatch(
            field_transforms=[HideOutputField(ctx.field.name, reason="native:_BadgeNative")],
            parsers=[ThirdPartyNativeParserHook(_BadgeNative, ctx.field.name)],
        )


def _badge_plan_effects(built, lm_kwargs):
    """The observable plan effects both paths must agree on."""
    from dspy.adapters._engine.parser_hook import ThirdPartyNativeParserHook

    assert lm_kwargs == {"badge_flag": True}
    assert built.plan.metadata["native_feature_kwargs"]["_BadgeNative"] == {"badge_flag": True}
    assert "badge" not in built.render_signature.output_fields
    assert built.plan.find_field("output", "badge").hidden
    hooks = [p for p in built.plan.parsers if isinstance(p, ThirdPartyNativeParserHook)]
    assert len(hooks) == 1 and hooks[0].annotation is _BadgeNative and hooks[0].destination == "badge"


def test_third_party_both_ways_equivalent_plans():
    """The same custom type via the legacy hook (auto-wrap) and via direct
    strategy registration: identical plan effects, distinct trace names."""
    from dspy.adapters._engine.strategies import register_field_strategy, unregister_field_strategy

    adapter = ChatAdapter(native_response_types=[_BadgeNative])

    # Way 1: unregistered -> auto-wrapped legacy hook.
    legacy_kwargs = {}
    legacy_built = build_plan(adapter, _lm(), legacy_kwargs, _BadgeSig, {"question": "Q?"})
    _badge_plan_effects(legacy_built, legacy_kwargs)
    legacy_trace = [t for t in legacy_built.plan.strategy_trace if t.field == "badge"]
    assert len(legacy_trace) == 1
    assert legacy_trace[0].strategy == "type_hook:_BadgeNative"
    assert legacy_trace[0].decision == "selected"
    assert legacy_trace[0].reason == "third-party adapt_to_native_lm_feature"

    # Way 2: registered proper strategy.
    register_field_strategy(_BadgeNative, _BadgeStrategy())
    try:
        registered_kwargs = {}
        registered_built = build_plan(adapter, _lm(), registered_kwargs, _BadgeSig, {"question": "Q?"})
        _badge_plan_effects(registered_built, registered_kwargs)
        registered_trace = [t for t in registered_built.plan.strategy_trace if t.field == "badge"]
        assert len(registered_trace) == 1
        assert registered_trace[0].strategy == "badge.native"
        assert registered_trace[0].decision == "selected"
        assert registered_trace[0].reason == "applies"

        # Same hidden-field transforms either way.
        assert [type(t).__name__ for t in legacy_built.plan.field_transforms] == [
            type(t).__name__ for t in registered_built.plan.field_transforms
        ]
    finally:
        unregister_field_strategy(_BadgeNative)


def test_unregister_restores_the_auto_wrap():
    from dspy.adapters._engine.strategies import (
        LegacyTypeHookStrategy,
        field_strategy_for,
        register_field_strategy,
        unregister_field_strategy,
    )

    register_field_strategy(_BadgeNative, _BadgeStrategy())
    assert isinstance(field_strategy_for(_BadgeNative), _BadgeStrategy)
    unregister_field_strategy(_BadgeNative)
    assert isinstance(field_strategy_for(_BadgeNative), LegacyTypeHookStrategy)


def test_builtins_resolve_through_the_uniform_path():
    from dspy.adapters._engine.strategies import field_strategy_for

    assert isinstance(field_strategy_for(Reasoning), NativeReasoningStrategy)
    assert isinstance(field_strategy_for(Citations), NativeCitationsStrategy)


# --- The double-key registry (epic-C §6 stage 2) ---------------------------------


def test_double_key_role_first_annotation_fallback():
    from dspy.adapters._engine.strategies import field_strategy_for, strategy_for

    strategy, resolved_by = strategy_for("reasoning", Reasoning)
    assert resolved_by == "role"
    assert strategy is field_strategy_for(Reasoning)  # same instance under both keys

    strategy, resolved_by = strategy_for("citations", Citations)
    assert resolved_by == "role"
    assert strategy is field_strategy_for(Citations)

    # A plain role never hits the role table: annotation key resolves.
    strategy, resolved_by = strategy_for("plain", _BadgeNative)
    assert resolved_by == "annotation"
    assert strategy.name == "type_hook:_BadgeNative"


def test_registered_annotation_strategy_still_beats_the_builtin_role_entry():
    """Compat pin: register_field_strategy kept its today-semantics — a
    registered annotation entry outranks built-ins under either key."""
    from dspy.adapters._engine.strategies import (
        register_field_strategy,
        strategy_for,
        unregister_field_strategy,
    )

    custom = _BadgeStrategy()
    register_field_strategy(Reasoning, custom)
    try:
        strategy, resolved_by = strategy_for("reasoning", Reasoning)
        assert strategy is custom
        assert resolved_by == "annotation"
    finally:
        unregister_field_strategy(Reasoning)


def test_registered_role_strategy_wins_and_unknown_role_refuses():
    from dspy.adapters._engine.strategies import (
        register_role_strategy,
        strategy_for,
        unregister_role_strategy,
    )

    custom = _BadgeStrategy()
    register_role_strategy("reasoning", custom)
    try:
        strategy, resolved_by = strategy_for("reasoning", Reasoning)
        assert strategy is custom
        assert resolved_by == "role"
    finally:
        unregister_role_strategy("reasoning")

    import pytest

    with pytest.raises(ValueError, match="valid roles"):
        register_role_strategy("vibes", custom)


def test_builder_trace_records_which_key_resolved():
    lm_kwargs = {}
    built = build_plan(ChatAdapter(), _lm(supports_reasoning=True), lm_kwargs, ThoughtfulQA, {"question": "Q?"})
    trace = [t for t in built.plan.strategy_trace if t.field == "reasoning"]
    assert len(trace) == 1
    assert trace[0].resolved_by == "role"


def test_legacy_wrapper_no_effects_reports_skipped():
    """A hook that does nothing must trace 'skipped' — the observed-effects
    decision rule, byte-identical to the inline block it replaced."""

    class _InertNative(dspy.Type):
        content: str = "x"

        @classmethod
        def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
            return signature

        @classmethod
        def description(cls):
            return "inert"

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        inert: _InertNative = dspy.OutputField()

    adapter = ChatAdapter(native_response_types=[_InertNative])
    built = build_plan(adapter, _lm(), {}, Sig, {"question": "Q?"})
    trace = [t for t in built.plan.strategy_trace if t.strategy == "type_hook:_InertNative"]
    assert len(trace) == 1 and trace[0].decision == "skipped"
    assert "inert" in built.render_signature.output_fields
