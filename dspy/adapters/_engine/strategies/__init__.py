"""Built-in engine strategies and the double-key strategy registry.

``strategy_for(role, annotation)`` is the builder's resolution path
(epic-C §6 stage 2): the semantic ROLE is consulted first, the annotation
type second, and the resolution records which key answered. Built-ins
register under BOTH keys as the same instances, so resolution is
byte-identical whichever key hits; a field with no registry entry falls
back to the annotation's documented ``adapt_to_native_lm_feature`` hook
auto-wrapped in :class:`LegacyTypeHookStrategy` — one uniform strategy
loop, no special-cased third-party branch.

Registered entries beat built-ins in either keying (today's
``register_field_strategy`` contract, preserved), and a registered ROLE
strategy beats a registered annotation strategy — the resolution order is
registered-role, registered-annotation, builtin-role, builtin-annotation,
legacy hook. Both registration seams stay engine-private until the
exposure epic publishes them with admission gates (spec section 9).
"""

from dspy.adapters._engine.strategies.citations import NativeCitationsStrategy
from dspy.adapters._engine.strategies.legacy import LegacyTypeHookStrategy
from dspy.adapters._engine.strategies.reasoning import NativeReasoningStrategy
from dspy.adapters._engine.strategies.tools import NativeFunctionCallingStep

_BUILTIN_FIELD_STRATEGIES = None
_BUILTIN_ROLE_STRATEGIES = None
_REGISTERED_FIELD_STRATEGIES: dict = {}
_REGISTERED_ROLE_STRATEGIES: dict = {}


def _ensure_builtins() -> tuple[dict, dict]:
    global _BUILTIN_FIELD_STRATEGIES, _BUILTIN_ROLE_STRATEGIES
    if _BUILTIN_FIELD_STRATEGIES is None:
        from dspy.adapters.types.reasoning import Reasoning
        from dspy.experimental import Citations

        citations = NativeCitationsStrategy()
        reasoning = NativeReasoningStrategy()
        # One instance per built-in, registered under both keys, so the
        # role lookup and the annotation lookup cannot diverge.
        _BUILTIN_FIELD_STRATEGIES = {Citations: citations, Reasoning: reasoning}
        _BUILTIN_ROLE_STRATEGIES = {"citations": citations, "reasoning": reasoning}
    return _BUILTIN_FIELD_STRATEGIES, _BUILTIN_ROLE_STRATEGIES


def builtin_field_strategy_for(annotation):
    """The built-in annotation mapping only — registered entries excluded."""
    return _ensure_builtins()[0].get(annotation)


def field_strategy_for(annotation):
    """Annotation-keyed resolution: registered entry first, then built-in,
    else the legacy hook auto-wrapped. The double-key fallback path."""
    field_builtins, _ = _ensure_builtins()
    strategy = _REGISTERED_FIELD_STRATEGIES.get(annotation) or field_builtins.get(annotation)
    if strategy is None:
        return LegacyTypeHookStrategy(annotation)
    return strategy


def strategy_for(role, annotation) -> tuple:
    """Resolve a field's strategy by role first, annotation second.

    Returns ``(strategy, resolved_by)`` where ``resolved_by`` is ``"role"``
    or ``"annotation"`` — recorded on the strategy trace so resolution is
    always explainable. Byte-identical to the annotation-only path by
    construction: built-ins are the same instances under both keys.
    """
    field_builtins, role_builtins = _ensure_builtins()
    registered_role = _REGISTERED_ROLE_STRATEGIES.get(role)
    if registered_role is not None:
        return registered_role, "role"
    registered_field = _REGISTERED_FIELD_STRATEGIES.get(annotation)
    if registered_field is not None:
        return registered_field, "annotation"
    builtin_role = role_builtins.get(role)
    if builtin_role is not None:
        return builtin_role, "role"
    builtin_field = field_builtins.get(annotation)
    if builtin_field is not None:
        return builtin_field, "annotation"
    return LegacyTypeHookStrategy(annotation), "annotation"


def register_field_strategy(annotation, strategy) -> None:
    """Bind a ``TypeStrategy`` to a semantic type (engine-private seam)."""
    _REGISTERED_FIELD_STRATEGIES[annotation] = strategy


def unregister_field_strategy(annotation) -> None:
    """Remove a registered strategy; the type falls back to built-in or hook."""
    _REGISTERED_FIELD_STRATEGIES.pop(annotation, None)


def register_role_strategy(role: str, strategy) -> None:
    """Bind a ``TypeStrategy`` to a semantic role (engine-private seam).

    The role must name an entry in the closed vocabulary; an unknown role
    refuses eagerly naming the valid set.
    """
    from dspy.signatures.field import SEMANTIC_ROLES

    if role not in SEMANTIC_ROLES:
        raise ValueError(f"unknown semantic role {role!r} — valid roles: {', '.join(sorted(SEMANTIC_ROLES))}")
    _REGISTERED_ROLE_STRATEGIES[role] = strategy


def unregister_role_strategy(role: str) -> None:
    """Remove a registered role strategy; the role falls back to built-ins."""
    _REGISTERED_ROLE_STRATEGIES.pop(role, None)


__all__ = [
    "LegacyTypeHookStrategy",
    "NativeCitationsStrategy",
    "NativeFunctionCallingStep",
    "NativeReasoningStrategy",
    "builtin_field_strategy_for",
    "field_strategy_for",
    "register_field_strategy",
    "register_role_strategy",
    "strategy_for",
    "unregister_field_strategy",
    "unregister_role_strategy",
]
