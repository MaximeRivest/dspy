"""Built-in engine strategies and the double-key strategy registry.

``strategy_for(role, annotation)`` is the builder's resolution path
(epic-C §6 stage 2): the semantic ROLE is consulted first, the annotation
type second, and the resolution records which key answered. Built-ins
register under BOTH keys as the same instances, so resolution is
byte-identical whichever key hits — and a builtin ROLE entry answers only
for the exact annotation registered beside it, so the double-key path
resolves identically to the annotation-only path wherever that path
answered (an admitted subclass keeps its hook; the role key only adds
resolution where the annotation path had none). A field with no registry
entry falls back to the annotation's documented ``adapt_to_native_lm_feature`` hook
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
#: role -> ordered {strategy name -> strategy}. Registration under a role
#: also extends that role's BINDING vocabulary with the strategy's name
#: (D-δ, spec section 5): `strategies={role: name}` binds it explicitly,
#: while "auto" consults the role's registrations in registration order.
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


def builtin_role_strategy_for(role):
    """The built-in role mapping only — registered entries excluded."""
    return _ensure_builtins()[1].get(role)


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
    construction: built-ins are the same instances under both keys, and a
    BUILTIN role entry answers only for the exact annotation that carries
    it natively — an admitted subclass (whose ``adapt_to_native_lm_feature``
    override the annotation path always honored) still resolves through the
    annotation key. The role key only ADDS resolution where the annotation
    path had nothing.
    """
    field_builtins, role_builtins = _ensure_builtins()
    registered_role = next(iter(_REGISTERED_ROLE_STRATEGIES.get(role, {}).values()), None)
    if registered_role is not None:
        return registered_role, "role"
    registered_field = _REGISTERED_FIELD_STRATEGIES.get(annotation)
    if registered_field is not None:
        return registered_field, "annotation"
    builtin_field = field_builtins.get(annotation)
    builtin_role = role_builtins.get(role)
    if builtin_role is not None and builtin_role is builtin_field:
        return builtin_role, "role"
    if builtin_field is not None:
        return builtin_field, "annotation"
    return LegacyTypeHookStrategy(annotation), "annotation"


def register_field_strategy(annotation, strategy) -> None:
    """Bind a ``TypeStrategy`` to a semantic type (engine-private seam)."""
    _REGISTERED_FIELD_STRATEGIES[annotation] = strategy


def unregister_field_strategy(annotation) -> None:
    """Remove a registered strategy; the type falls back to built-in or
    hook. Unregistering an annotation with nothing registered refuses (gate
    parity with the codec door)."""
    if _REGISTERED_FIELD_STRATEGIES.pop(annotation, None) is None:
        registered = ", ".join(getattr(key, "__name__", repr(key)) for key in _REGISTERED_FIELD_STRATEGIES)
        raise KeyError(
            f"no strategy registered under annotation {getattr(annotation, '__name__', repr(annotation))} — "
            f"annotations with registrations: {registered or '(none)'}"
        )


def register_role_strategy(role: str, strategy) -> None:
    """Bind a ``TypeStrategy`` to a semantic role (engine-private seam).

    The role must name an entry in the closed vocabulary; an unknown role
    refuses eagerly naming the valid set. The strategy's declared ``name``
    joins the role's binding vocabulary (D-δ), so it may not collide with
    the role's builtin strategy names or ``"auto"``, and a duplicate name
    refuses — ``unregister_strategy(role=...)`` first, deliberately (gate
    parity with the codec door).
    """
    from dspy.adapters._engine.strategies.vocabulary import STRATEGIES_VOCABULARY
    from dspy.signatures.field import SEMANTIC_ROLES

    if role not in SEMANTIC_ROLES:
        raise ValueError(f"unknown semantic role {role!r} — valid roles: {', '.join(sorted(SEMANTIC_ROLES))}")
    name = getattr(strategy, "name", None)
    builtin_names = ("auto", *STRATEGIES_VOCABULARY.get(role, ()))
    if name in builtin_names:
        raise ValueError(
            f"strategy name {name!r} collides with the {role} binding vocabulary "
            f"({', '.join(builtin_names)}) — a registered strategy's name must not shadow a builtin "
            "binding name"
        )
    entries = _REGISTERED_ROLE_STRATEGIES.setdefault(role, {})
    if name in entries:
        raise ValueError(
            f"a strategy named {name!r} is already registered under role {role!r} — "
            f"unregister_strategy(role={role!r}) first"
        )
    entries[name] = strategy


def unregister_role_strategy(role: str) -> None:
    """Remove a role's registered strategies; the role falls back to
    built-ins. Unregistering a role with nothing registered refuses naming
    what is registered (gate parity with the codec door)."""
    if not _REGISTERED_ROLE_STRATEGIES.pop(role, None):
        registered = [key for key, entries in _REGISTERED_ROLE_STRATEGIES.items() if entries]
        raise KeyError(
            f"no strategy registered under role {role!r} — roles with registrations: "
            f"{', '.join(registered) or '(none)'}"
        )


def registered_role_strategies(role: str) -> dict:
    """The role's registered strategies, name-keyed, in registration order."""
    return dict(_REGISTERED_ROLE_STRATEGIES.get(role, {}))


__all__ = [
    "LegacyTypeHookStrategy",
    "NativeCitationsStrategy",
    "NativeFunctionCallingStep",
    "NativeReasoningStrategy",
    "builtin_field_strategy_for",
    "builtin_role_strategy_for",
    "field_strategy_for",
    "register_field_strategy",
    "register_role_strategy",
    "registered_role_strategies",
    "strategy_for",
    "unregister_field_strategy",
    "unregister_role_strategy",
]
