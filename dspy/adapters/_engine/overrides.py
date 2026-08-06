"""Legacy-override detection: the engine's all-or-nothing staging mechanism.

User subclasses that override any render/parse-participating method must
keep flowing through the byte-untouched legacy pipeline — partial engine
application could silently drop a customization. Detection is therefore
ALL-OR-NOTHING per instance: one overridden method on the detection surface
routes the whole call legacy.

The registry is also the migration STAGING mechanism: a class is
engine-eligible only when a migration PR has explicitly registered it (e.g.
ChatAdapter in the renderer-cutover PR), so unmigrated core adapters —
XMLAdapter before its own PR, BAMLAdapter until migrated — route legacy
automatically with no flag day. The registry starts EMPTY: until the first
cutover lands, every instance routes legacy.

Mechanism (chosen for the realities of ``Adapter.__init_subclass__``, which
re-wraps ``format``/``parse`` with ``with_callbacks`` in EVERY subclass's
``__dict__``, so attribute presence alone proves nothing):

1. Find the most-derived REGISTERED class in ``type(adapter).__mro__``
   (walked directly — never via ``issubclass``, which at least one existing
   test monkeypatches at module scope). None registered => legacy.
2. For every detection-surface method, resolve the implementation the
   instance would actually dispatch to (``inspect.getattr_static``), then
   recursively ``inspect.unwrap`` it (``with_callbacks`` uses
   ``functools.wraps``, so ``__wrapped__`` chains reach the raw function at
   any wrapping depth).
3. Compare BY IDENTITY against the raw captured at registration time. Any
   mismatch — a user override, or a core method the registered class did not
   own — routes legacy, with machine-readable ``(class, method)`` reasons.

Overriding ``__call__``/``acall``/``format_finetune_data`` does NOT
disqualify: orchestration wrappers like refine.py's WrapperAdapter delegate
via ``super()`` and remain engine-eligible.
"""

import inspect
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: Method names that participate in rendering or parsing. Any override of one
#: of these routes the instance legacy. The companion meta-test
#: (tests/adapters/engine/test_overrides.py) diffs this surface against the
#: core classes' actual __dict__ so a future render-participating method
#: cannot silently dodge detection.
DETECTION_SURFACE = frozenset(
    {
        "format",
        "parse",
        "format_system_message",
        "format_field_description",
        "format_field_structure",
        "format_task_description",
        "format_user_message_content",
        "format_assistant_message_content",
        "format_demos",
        "format_conversation_history",
        "format_field_with_value",
        # Not format_*-prefixed and easy to miss:
        "user_message_output_requirements",
        "_make_json_adapter_fallback",
        "_create_extractor_signature",
        "_parse_field_value",
        "_legacy_async_quirks_postprocess",
    }
)

#: Methods that exist on core adapter classes but deliberately do NOT trigger
#: legacy routing: pure orchestration (the engine replaces the pipeline
#: AROUND them), planning/IO helpers the engine owns, or sanctioned wrapper
#: points. The meta-test forces every core-class method to appear in exactly
#: one of the two sets, so adding a method to dspy/adapters forces an
#: explicit routing decision here.
ORCHESTRATION_ALLOWLIST = frozenset(
    {
        "__init__",
        "__init_subclass__",
        "__call__",
        "acall",
        "format_finetune_data",
        "_call_preprocess",
        "_call_postprocess",
        "_render_request",
        "_call_lm",
        "_acall_lm",
        "_legacy_call_kwargs",
        "_coerce_lm_messages",
        "_chat_dict_to_lm_message",
        "_normalize_legacy_outputs",
        "_get_history_field_name",
        "_get_tool_call_input_field_name",
        "_get_tool_call_output_field_name",
        "_json_adapter_call_common",
        # Serde surface (D-5): reads preset data, participates in neither
        # rendering nor parsing.
        "_effective_preset_view",
        "dump_entry",
        "literal_table",
        # Fallback decision: consulted by the orchestration wrappers, which
        # run identically around both the engine and legacy pipelines.
        "_should_reraise_instead_of_fallback",
    }
)

# cls -> {method_name: raw function} captured at registration time.
_REGISTRY: dict[type, dict[str, object]] = {}

_VERDICT_CACHE: dict[type, "OverrideVerdict"] = {}


@dataclass(frozen=True)
class OverrideVerdict:
    """The routing decision for one adapter class."""

    engine_eligible: bool
    reasons: tuple = field(default_factory=tuple)  # of (class_name, method_name)


def register_engine_backed(cls) -> None:
    """Declare ``cls`` migrated: its CURRENT detection-surface resolution is
    the engine-backed reference. Called by migration PRs at import time of
    dspy.adapters modules, never by user code."""
    raws = {}
    for name in DETECTION_SURFACE:
        attr = inspect.getattr_static(cls, name, None)
        if attr is not None:
            raws[name] = _raw_function(attr)
    _REGISTRY[cls] = raws
    _VERDICT_CACHE.clear()


def registered_classes() -> tuple:
    """Read-only view for tests."""
    return tuple(_REGISTRY)


def resolve_override_verdict(adapter) -> OverrideVerdict:
    """Decide engine-vs-legacy routing for one adapter instance."""
    from dspy.adapters._engine.migrated import ensure_core_migrations

    ensure_core_migrations()

    adapter_cls = type(adapter)
    cached = _VERDICT_CACHE.get(adapter_cls)
    if cached is not None:
        return cached

    registered = None
    for klass in adapter_cls.__mro__:
        if klass in _REGISTRY:
            registered = klass
            break

    if registered is None:
        verdict = OverrideVerdict(engine_eligible=False, reasons=((adapter_cls.__name__, "<class not engine-backed>"),))
    else:
        reasons = []
        reference = _REGISTRY[registered]
        for name in DETECTION_SURFACE:
            attr = inspect.getattr_static(adapter_cls, name, None)
            actual = _raw_function(attr) if attr is not None else None
            expected = reference.get(name)
            if actual is not expected:
                reasons.append((_defining_class(adapter_cls, name, actual), name))
        verdict = OverrideVerdict(engine_eligible=not reasons, reasons=tuple(sorted(set(reasons))))

    _VERDICT_CACHE[adapter_cls] = verdict
    if not verdict.engine_eligible and os.environ.get("DSPY_ADAPTER_ENGINE_DEBUG"):
        logger.debug(
            "Adapter %s routes through the legacy pipeline; non-engine surface: %s",
            adapter_cls.__name__,
            verdict.reasons,
        )
    return verdict


def clear_verdict_cache() -> None:
    """Test hook: verdicts are cached by class identity."""
    _VERDICT_CACHE.clear()


def _reset_registry_for_tests() -> None:
    """Empty the registries AND suppress lazy core migrations, so tests
    control registration explicitly."""
    from dspy.adapters._engine.formats import _reset_formats_for_tests
    from dspy.adapters._engine.migrated import _suppress_for_tests

    _suppress_for_tests()
    _reset_formats_for_tests()
    _REGISTRY.clear()
    _VERDICT_CACHE.clear()


def _raw_function(attr):
    """Fully unwrap a class attribute to its raw function, or None."""
    if isinstance(attr, (staticmethod, classmethod)):
        attr = attr.__func__
    if not callable(attr):
        return None
    return inspect.unwrap(attr)


def _defining_class(adapter_cls, name, raw) -> str:
    for klass in adapter_cls.__mro__:
        attr = vars(klass).get(name)
        if attr is not None and _raw_function(attr) is raw:
            return klass.__name__
    return adapter_cls.__name__
