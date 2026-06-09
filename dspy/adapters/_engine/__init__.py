"""Private adapter-engine internals. No stability promise whatsoever.

Nothing in this package is public API: names, modules, and semantics may
change or disappear in any release without deprecation. It is never
re-exported from ``dspy`` or ``dspy.adapters`` (enforced by tests), and
external code must not import it.

The engine reifies adapter behavior as data: a ``SignatureCall`` is compiled
into an :class:`~dspy.adapters._engine.ir.AdapterPlan` (the intermediate
representation), strategies contribute :class:`~dspy.adapters._engine.patch.AdapterPatch`
objects into the plan's slots, field transforms produce the model-facing
field view, and parser hooks map the LM response back to prediction values.
Design vocabulary follows ``scratch/newadapter/*.md``; only the collapsed
slot schema is dataclass structure — the rich slot taxonomy from ``slots.md``
exists here as docstring/debug-link vocabulary.
"""
