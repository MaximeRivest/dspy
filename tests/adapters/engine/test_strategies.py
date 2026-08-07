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


def test_builtin_role_entry_never_shadows_an_admitted_subclass_hook():
    """A Reasoning SUBCLASS carrying its own adapt_to_native_lm_feature must
    resolve to its legacy hook exactly as the annotation-keyed path did —
    the builtin role entry answers only for its native annotation, never
    for an admitted class that overrides the hook."""
    from dspy.adapters._engine.strategies import strategy_for

    calls = {}

    class MyReasoning(Reasoning):
        @classmethod
        def adapt_to_native_lm_feature(cls, signature, field_name, lm, lm_kwargs):
            calls["field"] = field_name
            # This provider's reasoning must NOT be requested through
            # reasoning_effort; the field stays textual.
            return signature

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        thought: MyReasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    strategy, resolved_by = strategy_for("reasoning", MyReasoning)
    assert strategy.name == "type_hook:MyReasoning"
    assert resolved_by == "annotation"

    lm_kwargs = {}
    built = build_plan(
        ChatAdapter(native_response_types=[MyReasoning]),
        _lm(supports_reasoning=True),
        lm_kwargs,
        Sig,
        {"question": "Q?"},
    )
    assert calls == {"field": "thought"}
    assert lm_kwargs == {}
    assert built.plan.find_field("output", "thought").hidden is False
    assert "thought" in built.render_signature.output_fields
    parser_names = [type(parser).__name__ for parser in built.plan.parsers]
    assert "ThirdPartyNativeParserHook" in parser_names
    assert "NativeReasoningParserHook" not in parser_names


def test_double_key_resolution_matches_the_annotation_path_wherever_it_answered():
    """The restoration invariant: for every annotation the annotation-keyed
    path handled, strategy_for(role, annotation) resolves identically to
    field_strategy_for(annotation) — the role key only ADDS resolution
    where the annotation path had nothing."""
    from dspy.adapters._engine.roles import semantic_role_for
    from dspy.adapters._engine.strategies import field_strategy_for, strategy_for

    class SubReasoning(Reasoning):
        pass

    class SubCitations(Citations):
        pass

    for annotation in (Reasoning, Citations, SubReasoning, SubCitations, str):
        role = semantic_role_for(annotation)
        double_key, _ = strategy_for(role, annotation)
        annotation_only = field_strategy_for(annotation)
        assert type(double_key) is type(annotation_only)
        assert double_key.name == annotation_only.name


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


# --- The strategies={} binding surface at bake -----------------------------------


def test_textual_binding_stands_the_native_strategy_down():
    adapter = ChatAdapter(strategies={"reasoning": "textual_field"})
    lm_kwargs = {}
    built = build_plan(adapter, _lm(supports_reasoning=True), lm_kwargs, ThoughtfulQA, {"question": "Q?"})

    assert lm_kwargs == {}  # no reasoning_effort: the native channel stood down
    assert "reasoning" in built.render_signature.output_fields
    trace = [t for t in built.plan.strategy_trace if t.field == "reasoning"]
    assert len(trace) == 1
    assert trace[0].strategy == "reasoning.textual_field"
    assert trace[0].reason == "bound: textual_field"
    assert trace[0].resolved_by == "binding"
    assert built.plan.metadata["strategy_bindings"] == {"reasoning": "textual_field"}
    assert built.plan.metadata["strategy_resolution"]["reasoning"] == {
        "role": "reasoning",
        "binding": "textual_field",
        "served": "textual",
    }


def test_explicit_native_binding_refuses_when_the_lm_cannot_serve_it():
    import pytest

    adapter = ChatAdapter(strategies={"reasoning": "native_channel"})
    with pytest.raises(ValueError, match="cannot be honored"):
        build_plan(adapter, _lm(supports_reasoning=False), {}, ThoughtfulQA, {"question": "Q?"})


def test_explicit_native_binding_matches_auto_when_the_lm_serves_it():
    auto_kwargs, bound_kwargs = {}, {}
    auto = build_plan(ChatAdapter(), _lm(supports_reasoning=True), auto_kwargs, ThoughtfulQA, {"question": "Q?"})
    bound = build_plan(
        ChatAdapter(strategies={"reasoning": "native_channel"}),
        _lm(supports_reasoning=True),
        bound_kwargs,
        ThoughtfulQA,
        {"question": "Q?"},
    )
    assert auto_kwargs == bound_kwargs == {"reasoning_effort": "low"}
    assert list(auto.render_signature.output_fields) == list(bound.render_signature.output_fields)
    assert bound.plan.metadata["strategy_resolution"]["reasoning"]["served"] == "native"


def test_auto_resolution_is_recorded():
    built = build_plan(ChatAdapter(), _lm(supports_reasoning=True), {}, ThoughtfulQA, {"question": "Q?"})
    assert built.plan.metadata["strategy_resolution"]["reasoning"] == {
        "role": "reasoning",
        "binding": "auto",
        "served": "native",
    }
    built = build_plan(ChatAdapter(), _lm(supports_reasoning=False), {}, ThoughtfulQA, {"question": "Q?"})
    assert built.plan.metadata["strategy_resolution"]["reasoning"]["served"] == "textual"


def test_textual_strategies_contribute_template_fragments():
    """A strategy's fragments flow patch -> plan -> plan_scope -> the
    preset's positional {fragments(...)} slots (spec section 2: a textual
    strategy is a template-fragment provider)."""
    from dspy.adapters._engine.render import plan_scope
    from dspy.adapters._engine.strategies import register_field_strategy, unregister_field_strategy

    class _CiteNote(dspy.Type):
        content: str = ""

        @classmethod
        def description(cls):
            return "note"

    class _CiteStrategy:
        name = "citations.span_markers.test"
        priority = 500
        exclusive_group = None

        def applies(self, ctx):
            return True

        def contribute(self, ctx):
            return AdapterPatch(
                fragments={
                    "system": ["When quoting, wrap the quote in <cite>."],
                    "user": ["Cite your sources."],
                }
            )

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        note: _CiteNote = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = ChatAdapter(native_response_types=[_CiteNote])
    register_field_strategy(_CiteNote, _CiteStrategy())
    try:
        built = build_plan(adapter, _lm(), {}, Sig, {"question": "Q?"})
        assert built.plan.fragments == {
            "system": ["When quoting, wrap the quote in <cite>."],
            "user": ["Cite your sources."],
        }

        with plan_scope(built.plan):
            messages = adapter.format(built.render_signature, [], {"question": "Q?"})
        system, user = messages[0]["content"], messages[-1]["content"]
        assert "[[ ## completed ## ]]\nWhen quoting, wrap the quote in <cite>.\nIn adhering" in system
        assert "Cite your sources.\nRespond with the corresponding output fields" in user

        # Outside any scope the slots are empty and cost zero bytes.
        bare = adapter.format(built.render_signature, [], {"question": "Q?"})
        assert "<cite>" not in bare[0]["content"]
        assert "Cite your sources." not in bare[-1]["content"]
    finally:
        unregister_field_strategy(_CiteNote)


def test_fragments_reach_the_wire_through_the_full_pipeline():
    """__call__ enters plan_scope around format(), sync and async alike."""
    from golden.harness import run_adapter_capture

    from dspy.adapters._engine.strategies import register_field_strategy, unregister_field_strategy

    class _NoteType(dspy.Type):
        content: str = ""

        @classmethod
        def description(cls):
            return "note"

    class _FragStrategy:
        name = "test.fragments"
        priority = 500
        exclusive_group = None

        def applies(self, ctx):
            return True

        def contribute(self, ctx):
            return AdapterPatch(fragments={"system": ["FRAGMENT LINE."]})

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        note: _NoteType = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = ChatAdapter(native_response_types=[_NoteType])
    register_field_strategy(_NoteType, _FragStrategy())
    try:
        for mode in ("sync", "async"):
            recorder = Recorder()
            run_adapter_capture(adapter, StubLM(recorder), {}, Sig, [], {"question": "Q?"}, mode, recorder)
            system = recorder.calls[0]["messages"][0]["content"]
            assert "[[ ## completed ## ]]\nFRAGMENT LINE.\nIn adhering" in system
    finally:
        unregister_field_strategy(_NoteType)


def test_native_fc_binding_refuses_without_capability():
    import pytest

    def add(a: int, b: int) -> int:
        return a + b

    class ToolSig(dspy.Signature):
        question: str = dspy.InputField()
        tools: list[dspy.Tool] = dspy.InputField()
        tool_calls: dspy.ToolCalls = dspy.OutputField()

    adapter = ChatAdapter(use_native_function_calling=True, strategies={"tools": "native_fc"})
    inputs = {"question": "2+2?", "tools": [dspy.Tool(add)]}
    with pytest.raises(ValueError, match="does not support function calling"):
        build_plan(adapter, _lm(supports_function_calling=False), {}, ToolSig, inputs)
    # With the capability, the binding matches auto exactly.
    lm_kwargs = {}
    built = build_plan(adapter, _lm(supports_function_calling=True), lm_kwargs, ToolSig, inputs)
    assert "tools" in lm_kwargs
    assert "tool_calls" not in built.render_signature.output_fields


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


# --- Role-keyed admission: the D-delta role lane ---------------------------------


class _RolePolyfill:
    """A gate-shaped textual strategy: hides its field, carries a parser."""

    name = "cite_polyfill"
    priority = 500
    exclusive_group = None
    capability_requirements = ()

    class _Parser:
        def parse(self, response_view, ctx):
            return {"citations": "FILLED-BY-STRATEGY"}

    parser = _Parser()

    def applies(self, ctx):
        return True

    def contribute(self, ctx):
        from dspy.adapters._engine.transforms import HideOutputField

        return AdapterPatch(
            field_transforms=[HideOutputField(ctx.field.name, reason="cite_polyfill")],
            parsers=[self.parser],
        )


def test_split_role_spelling_engages_native_reasoning_under_auto():
    """Persona repro (strategies user, bug 1): all four role spellings are
    one meaning at delivery time — a `@reasoning` field on a plain shape
    engages the native channel exactly as the fused type does."""
    sig = dspy.Signature("question -> thoughts @reasoning, answer")
    lm_kwargs = {}
    built = build_plan(ChatAdapter(), _lm(supports_reasoning=True), lm_kwargs, sig, {"question": "Q?"})
    assert lm_kwargs == {"reasoning_effort": "low"}
    assert built.plan.find_field("output", "thoughts").hidden
    assert "thoughts" not in built.render_signature.output_fields
    assert any(isinstance(p, NativeReasoningParserHook) for p in built.plan.parsers)
    assert built.plan.metadata["strategy_resolution"]["thoughts"] == {
        "role": "reasoning",
        "binding": "auto",
        "served": "native",
    }


def test_split_role_spelling_stays_textual_on_an_incapable_lm():
    sig = dspy.Signature("question -> thoughts @reasoning, answer")
    lm_kwargs = {}
    built = build_plan(ChatAdapter(), _lm(), lm_kwargs, sig, {"question": "Q?"})
    assert lm_kwargs == {}
    assert built.plan.find_field("output", "thoughts").hidden is False
    assert built.plan.metadata["strategy_resolution"]["thoughts"]["served"] == "textual"


def test_split_role_explicit_native_binding_refuses_on_an_incapable_lm():
    """Persona repro: an explicit native binding on a role-spelled field must
    refuse, never silently render the textual marker and set no kwargs."""
    import pytest

    sig = dspy.Signature("question -> thoughts @reasoning, answer")
    adapter = ChatAdapter(strategies={"reasoning": "native_channel"})
    with pytest.raises(ValueError, match="cannot be honored"):
        build_plan(adapter, _lm(supports_reasoning=False), {}, sig, {"question": "Q?"})


def test_split_role_textual_binding_recorded():
    sig = dspy.Signature("question -> thoughts @reasoning, answer")
    adapter = ChatAdapter(strategies={"reasoning": "textual_field"})
    built = build_plan(adapter, _lm(supports_reasoning=True), {}, sig, {"question": "Q?"})
    assert built.plan.metadata["strategy_resolution"]["thoughts"] == {
        "role": "reasoning",
        "binding": "textual_field",
        "served": "textual",
    }


def test_role_lane_never_engages_fused_types_or_plain_fields():
    """The role lane only ADDS behavior for explicitly-declared roles on
    plain shapes: a fused type excluded from native_response_types stays
    excluded, and undeclared plain fields never consult strategies."""
    lm_kwargs = {}
    built = build_plan(
        ChatAdapter(native_response_types=[Citations]),  # Reasoning excluded
        _lm(supports_reasoning=True),
        lm_kwargs,
        ThoughtfulQA,
        {"question": "Q?"},
    )
    assert lm_kwargs == {}
    assert "reasoning" in built.render_signature.output_fields
    assert built.plan.strategy_trace == []


def test_registered_strategy_is_bindable_by_name_and_serializes():
    """Persona repro (strategy author, bug 1): register_strategy and
    strategies= compose — the registered name is bindable, rides the entry,
    and refuses at load when dangling (ADP-005)."""
    import pytest

    from dspy.adapters.serde import load_entry

    dspy.adapters.register_strategy(_RolePolyfill(), role="citations")
    try:
        adapter = ChatAdapter(strategies={"citations": "cite_polyfill"})
        sig = dspy.Signature("question -> citations @citations, answer")
        built = build_plan(adapter, _lm(), {}, sig, {"question": "Q?"})
        assert built.plan.find_field("output", "citations").hidden
        assert built.plan.metadata["strategy_resolution"]["citations"]["binding"] == "cite_polyfill"
        entry = adapter.dump_entry()
        assert entry["strategies"]["citations"] == "cite_polyfill"
        assert load_entry(entry).strategies["citations"] == "cite_polyfill"
    finally:
        dspy.adapters.unregister_strategy(role="citations")
    with pytest.raises(Exception, match="cite_polyfill"):
        load_entry(entry)  # dangling registered name is a link error


def test_registered_role_strategy_serves_under_auto_for_split_fields():
    dspy.adapters.register_strategy(_RolePolyfill(), role="citations")
    try:
        sig = dspy.Signature("question -> citations @citations, answer")
        built = build_plan(ChatAdapter(), _lm(), {}, sig, {"question": "Q?"})
        assert built.plan.find_field("output", "citations").hidden
        trace = [t for t in built.plan.strategy_trace if t.field == "citations"]
        assert trace and trace[0].strategy == "cite_polyfill" and trace[0].resolved_by == "role"
    finally:
        dspy.adapters.unregister_strategy(role="citations")


def test_registered_role_strategy_never_answers_an_explicit_native_binding():
    """Persona repro (strategy author, bug 2): the user demanded native; a
    registered textual strategy must not hijack it — the binding refuses on
    an incapable LM instead of silently substituting."""
    import pytest

    dspy.adapters.register_strategy(_RolePolyfill(), role="citations")
    try:

        class CitedQA(dspy.Signature):
            question: str = dspy.InputField()
            quotes: Citations = dspy.OutputField()
            answer: str = dspy.OutputField()

        adapter = ChatAdapter(strategies={"citations": "native"})
        with pytest.raises(ValueError, match="cannot be honored"):
            build_plan(adapter, _lm(), {}, CitedQA, {"question": "Q?"})
    finally:
        dspy.adapters.unregister_strategy(role="citations")


def test_capability_requirements_consulted_at_plan_time():
    """Persona repro (strategy author, bug 4): the declaration is enforced —
    unmet requirements skip the strategy under auto (recorded) and refuse
    an explicit binding naming the flag."""
    import pytest

    class Demanding(_RolePolyfill):
        name = "demanding_citations"
        capability_requirements = ("supports_teleportation",)

    dspy.adapters.register_strategy(Demanding(), role="citations")
    try:
        sig = dspy.Signature("question -> citations @citations, answer")
        built = build_plan(ChatAdapter(), _lm(), {}, sig, {"question": "Q?"})
        assert built.plan.find_field("output", "citations").hidden is False
        trace = [t for t in built.plan.strategy_trace if t.field == "citations"]
        assert trace and "capability_requirements unmet: supports_teleportation" in trace[0].reason
        assert built.plan.metadata["strategy_resolution"]["citations"]["served"] == "textual"

        adapter = ChatAdapter(strategies={"citations": "demanding_citations"})
        with pytest.raises(ValueError, match="supports_teleportation"):
            build_plan(adapter, _lm(), {}, sig, {"question": "Q?"})
    finally:
        dspy.adapters.unregister_strategy(role="citations")


def test_unknown_binding_error_lists_registered_names_not_unimplemented_ones():
    import pytest

    dspy.adapters.register_strategy(_RolePolyfill(), role="citations")
    try:
        with pytest.raises(ValueError) as excinfo:
            ChatAdapter(strategies={"citations": "bogus"})
        message = str(excinfo.value)
        assert "valid citations strategies: auto, native, cite_polyfill" in message
        assert "declared in the vocabulary but not yet implemented: span_markers, json_quotes" in message
    finally:
        dspy.adapters.unregister_strategy(role="citations")


def test_list_valued_binding_gets_a_teaching_error_not_a_type_error():
    import pytest

    with pytest.raises(ValueError, match="a binding names a strategy as a string"):
        ChatAdapter(strategies={"reasoning": ["textual_field"]})
