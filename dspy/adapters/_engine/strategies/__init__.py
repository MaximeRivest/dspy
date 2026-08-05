"""Built-in engine strategies and the field-strategy registry.

``field_strategy_for(annotation)`` is the builder's single resolution path:
a registry entry (built-in or registered third-party) when one exists,
otherwise the annotation's documented ``adapt_to_native_lm_feature`` hook
auto-wrapped in :class:`LegacyTypeHookStrategy` — one uniform strategy loop,
no special-cased third-party branch. ``register_field_strategy`` is the
authored-strategy seam (spec §Adapter-semantics): a custom Type's
native-feature behavior expressed as a proper ``TypeStrategy``. The legacy
hook keeps working unchanged with zero deprecation signaling — that
decision belongs to a future exposure epic.
"""

from dspy.adapters._engine.strategies.citations import NativeCitationsStrategy
from dspy.adapters._engine.strategies.legacy import LegacyTypeHookStrategy
from dspy.adapters._engine.strategies.reasoning import NativeReasoningStrategy
from dspy.adapters._engine.strategies.tools import NativeFunctionCallingStep

_BUILTIN_FIELD_STRATEGIES = None
_REGISTERED_FIELD_STRATEGIES: dict = {}


def _builtins() -> dict:
    global _BUILTIN_FIELD_STRATEGIES
    if _BUILTIN_FIELD_STRATEGIES is None:
        from dspy.adapters.types.reasoning import Reasoning
        from dspy.experimental import Citations

        _BUILTIN_FIELD_STRATEGIES = {
            Citations: NativeCitationsStrategy(),
            Reasoning: NativeReasoningStrategy(),
        }
    return _BUILTIN_FIELD_STRATEGIES


def builtin_field_strategy_for(annotation):
    """The built-in mapping only — registered entries excluded."""
    return _builtins().get(annotation)


def field_strategy_for(annotation):
    """Resolve the strategy for a semantic type: registered entry first,
    then built-in, else the legacy hook auto-wrapped."""
    strategy = _REGISTERED_FIELD_STRATEGIES.get(annotation) or _builtins().get(annotation)
    if strategy is None:
        return LegacyTypeHookStrategy(annotation)
    return strategy


def register_field_strategy(annotation, strategy) -> None:
    """Bind a ``TypeStrategy`` to a semantic type (engine-private seam)."""
    _REGISTERED_FIELD_STRATEGIES[annotation] = strategy


def unregister_field_strategy(annotation) -> None:
    """Remove a registered strategy; the type falls back to built-in or hook."""
    _REGISTERED_FIELD_STRATEGIES.pop(annotation, None)


__all__ = [
    "LegacyTypeHookStrategy",
    "NativeCitationsStrategy",
    "NativeFunctionCallingStep",
    "NativeReasoningStrategy",
    "builtin_field_strategy_for",
    "field_strategy_for",
    "register_field_strategy",
    "unregister_field_strategy",
]
