"""The strategy contracts: per-field TypeStrategy and call-level PlanStep.

These are the shapes a later exposure epic would publish unchanged (per the
design docs' strategies vocabulary): a strategy declares a stable ``name``,
a ``priority``, and an optional ``exclusive_group``; ``applies(ctx)`` is
side-effect-free; ``contribute(ctx)`` returns an
:class:`~dspy.adapters._engine.patch.AdapterPatch`. Selection decisions are
recorded as :class:`~dspy.adapters._engine.patch.StrategyTrace` entries on
the plan.

Legacy-compat channel, documented deviation: today's planning hooks mutate
``lm_kwargs`` (OpenAI-shaped, e.g. ``reasoning_effort``) rather than writing
``LMConfig``. Until the normalized-config epic, strategies receive
``ctx.lm_kwargs`` and may mutate it exactly as the hooks did — byte parity
over purity. The patch still carries everything else (field hides, tools,
parsers, provenance), so exposure later deprecates the channel without
re-signaturing anything.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from dspy.adapters._engine.ir import AdapterPlan, RenderField
from dspy.adapters._engine.patch import AdapterPatch


@dataclass
class FieldContext:
    """Everything a per-field strategy may consult."""

    adapter: Any
    plan: AdapterPlan
    field: RenderField
    role: str
    lm: Any
    lm_kwargs: dict[str, Any]
    signature: Any = None
    tools: list = field(default_factory=list)
    history_mode: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallContext:
    """Everything a call-level plan step may consult (concerns that span
    fields, like native tool plumbing pairing an input with an output)."""

    adapter: Any
    plan: AdapterPlan
    signature: Any
    inputs: dict[str, Any]
    lm: Any
    lm_kwargs: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TypeStrategy(Protocol):
    """Semantic representation rule for one field."""

    name: str
    priority: int

    def applies(self, ctx: FieldContext) -> bool: ...

    def contribute(self, ctx: FieldContext) -> AdapterPatch: ...


@runtime_checkable
class PlanStep(Protocol):
    """Call-level planning rule."""

    name: str
    priority: int

    def applies(self, ctx: CallContext) -> bool: ...

    def contribute(self, ctx: CallContext) -> AdapterPatch: ...


def resolution_order(strategies):
    """Deterministic candidate ordering: priority desc, then name."""
    return sorted(strategies, key=lambda s: (-s.priority, s.name))
