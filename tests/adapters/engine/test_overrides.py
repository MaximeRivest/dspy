"""Override-detection zoo: the staging mechanism for the whole migration.

Covers the real-world subclass patterns the codebase contains today:
refine.py's WrapperAdapter (overrides only __call__: engine-eligible),
SpyChatAdapter / CustomHistoryAdapter-style format overrides (legacy),
the easy-to-miss non-format_*-prefixed hooks, depth-2 with_callbacks
re-wrapping from Adapter.__init_subclass__, and the meta-test that forces
every core adapter method into an explicit routing decision.
"""

import pytest

from dspy.adapters._engine import overrides
from dspy.adapters._engine.overrides import (
    DETECTION_SURFACE,
    ORCHESTRATION_ALLOWLIST,
    register_engine_backed,
    resolve_override_verdict,
)
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.base import Adapter
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter
from dspy.adapters.xml_adapter import XMLAdapter

CORE_ADAPTER_CLASSES = (Adapter, ChatAdapter, JSONAdapter, XMLAdapter, BAMLAdapter, TwoStepAdapter)


@pytest.fixture(autouse=True)
def _clean_registry():
    overrides._reset_registry_for_tests()
    yield
    overrides._reset_registry_for_tests()
    # Resume lazy core migrations so tests running after this module see the
    # real engine state, regardless of test order.
    from dspy.adapters._engine import migrated

    migrated._reset_for_tests()


def test_empty_registry_routes_everything_legacy():
    verdict = resolve_override_verdict(ChatAdapter())
    assert not verdict.engine_eligible
    assert verdict.reasons == (("ChatAdapter", "<class not engine-backed>"),)


def test_registered_class_is_engine_eligible():
    register_engine_backed(ChatAdapter)
    assert resolve_override_verdict(ChatAdapter()).engine_eligible


def test_noop_subclass_stays_eligible_despite_callback_rewrapping():
    register_engine_backed(ChatAdapter)

    class NoOpSubclass(ChatAdapter):
        pass

    # __init_subclass__ re-wrapped format/parse into the subclass __dict__...
    assert "format" in vars(NoOpSubclass)
    # ...but recursive unwrap reaches the same raw functions: eligible.
    assert resolve_override_verdict(NoOpSubclass()).engine_eligible


def test_format_override_routes_legacy_with_reason():
    register_engine_backed(ChatAdapter)

    class SpyChatAdapter(ChatAdapter):
        def format(self, signature, demos, inputs):
            return super().format(signature, demos, inputs)

    verdict = resolve_override_verdict(SpyChatAdapter())
    assert not verdict.engine_eligible
    assert ("SpyChatAdapter", "format") in verdict.reasons


def test_history_hook_override_routes_legacy():
    register_engine_backed(ChatAdapter)

    class CustomHistoryAdapter(ChatAdapter):
        def format_conversation_history(self, signature, history_field_name, inputs):
            return []

    verdict = resolve_override_verdict(CustomHistoryAdapter())
    assert not verdict.engine_eligible
    assert ("CustomHistoryAdapter", "format_conversation_history") in verdict.reasons


def test_non_format_prefixed_hooks_are_detected():
    register_engine_backed(ChatAdapter)

    class CustomRequirements(ChatAdapter):
        def user_message_output_requirements(self, signature):
            return "custom sentence"

    class CustomFallback(ChatAdapter):
        def _make_json_adapter_fallback(self):
            return super()._make_json_adapter_fallback()

    assert not resolve_override_verdict(CustomRequirements()).engine_eligible
    assert not resolve_override_verdict(CustomFallback()).engine_eligible


def test_call_only_wrapper_stays_eligible():
    """refine.py's WrapperAdapter pattern: overrides __call__ and delegates
    via super() — must remain engine-eligible."""
    register_engine_backed(ChatAdapter)

    class WrapperAdapter(ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

        async def acall(self, lm, lm_kwargs, signature, demos, inputs):
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)

        def format_finetune_data(self, signature, demos, inputs, outputs):
            raise NotImplementedError

    assert resolve_override_verdict(WrapperAdapter()).engine_eligible


def test_unmigrated_core_subclasses_route_legacy():
    """Registering ChatAdapter must NOT make its core subclasses eligible:
    they define their own detection-surface methods and migrate in their own
    PRs."""
    register_engine_backed(ChatAdapter)
    for cls in (JSONAdapter, XMLAdapter, BAMLAdapter):
        verdict = resolve_override_verdict(cls())
        assert not verdict.engine_eligible, cls.__name__


def test_registering_subclass_makes_it_eligible():
    register_engine_backed(ChatAdapter)
    register_engine_backed(XMLAdapter)
    assert resolve_override_verdict(XMLAdapter()).engine_eligible
    # And a user override below the registered subclass still routes legacy.

    class CustomXML(XMLAdapter):
        def format_field_with_value(self, fields_with_values):
            return super().format_field_with_value(fields_with_values)

    assert not resolve_override_verdict(CustomXML()).engine_eligible


def test_verdicts_are_cached_per_class_and_invalidated_on_registration():
    first = resolve_override_verdict(ChatAdapter())
    assert resolve_override_verdict(ChatAdapter()) is first
    register_engine_backed(ChatAdapter)
    assert resolve_override_verdict(ChatAdapter()).engine_eligible


def test_detection_never_calls_issubclass(monkeypatch):
    """tests/predict/test_react.py shadows ``issubclass`` as a module
    attribute on adapter modules; detection must walk __mro__/__dict__
    directly. Shadow it in the overrides module with a tripwire and run a
    full verdict — module-global shadowing intercepts any bare call."""
    register_engine_backed(ChatAdapter)

    def _tripwire(*args, **kwargs):
        raise AssertionError("override detection must not call issubclass")

    monkeypatch.setattr(overrides, "issubclass", _tripwire, raising=False)
    overrides.clear_verdict_cache()
    assert resolve_override_verdict(ChatAdapter()).engine_eligible


def test_meta_every_core_method_has_an_explicit_routing_decision():
    """Any new render-participating method added to a core adapter must be
    added to DETECTION_SURFACE or ORCHESTRATION_ALLOWLIST — this test is the
    forcing function that keeps the detection surface from going stale."""
    undecided = []
    for cls in CORE_ADAPTER_CLASSES:
        for name, attr in vars(cls).items():
            if isinstance(attr, (staticmethod, classmethod)):
                attr = attr.__func__
            if not callable(attr):
                continue
            if name in DETECTION_SURFACE or name in ORCHESTRATION_ALLOWLIST:
                continue
            undecided.append(f"{cls.__name__}.{name}")
    assert not undecided, (
        "Methods with no routing decision (add to DETECTION_SURFACE if they "
        f"participate in rendering/parsing, else to ORCHESTRATION_ALLOWLIST): {undecided}"
    )
