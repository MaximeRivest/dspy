"""Single source of truth for which core adapters are engine-backed.

Each migration PR adds exactly its class here, pairing the override-registry
entry (engine eligibility) with the Format registration (how to render).
Application is lazy and idempotent: triggered by the first override-verdict
resolution of a live call, never by ``import dspy``.
"""

_applied = False


def ensure_core_migrations() -> None:
    global _applied
    if _applied:
        return
    _applied = True

    from dspy.adapters._engine.formats import register_format
    from dspy.adapters._engine.formats.chat import ChatFormat
    from dspy.adapters._engine.formats.json import JSONFormat
    from dspy.adapters._engine.overrides import register_engine_backed
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.adapters.json_adapter import JSONAdapter

    # QC-05: ChatAdapter rendering + parsing.
    register_engine_backed(ChatAdapter)
    register_format(ChatAdapter, ChatFormat())

    # QC-07: JSONAdapter (BAMLAdapter, its overriding subclass, stays legacy
    # automatically until its own migration).
    register_engine_backed(JSONAdapter)
    register_format(JSONAdapter, JSONFormat())


def _suppress_for_tests() -> None:
    """Mark migrations applied without applying them, so tests control
    registry state explicitly."""
    global _applied
    _applied = True


def _reset_for_tests() -> None:
    global _applied
    _applied = False
