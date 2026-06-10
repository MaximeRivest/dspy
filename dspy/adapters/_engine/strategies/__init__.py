"""Built-in engine strategies: the migrated provider-specific planning.

``builtin_field_strategy_for(annotation)`` maps a semantic type to its
built-in native strategy. Third-party types in ``native_response_types``
that have no built-in here flow through the silent compatibility step
(``Type.adapt_to_native_lm_feature`` / ``parse_lm_response``), preserving
the documented extension pattern with zero deprecation signaling — that
decision belongs to a future registry epic.
"""

from dspy.adapters._engine.strategies.citations import NativeCitationsStrategy
from dspy.adapters._engine.strategies.reasoning import NativeReasoningStrategy
from dspy.adapters._engine.strategies.tools import NativeFunctionCallingStep

_BUILTIN_FIELD_STRATEGIES = None


def builtin_field_strategy_for(annotation):
    global _BUILTIN_FIELD_STRATEGIES
    if _BUILTIN_FIELD_STRATEGIES is None:
        from dspy.adapters.types.reasoning import Reasoning
        from dspy.experimental import Citations

        _BUILTIN_FIELD_STRATEGIES = {
            Citations: NativeCitationsStrategy(),
            Reasoning: NativeReasoningStrategy(),
        }
    return _BUILTIN_FIELD_STRATEGIES.get(annotation)


__all__ = [
    "NativeCitationsStrategy",
    "NativeFunctionCallingStep",
    "NativeReasoningStrategy",
    "builtin_field_strategy_for",
]
