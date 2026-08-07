"""Semantic-role resolution, re-exported from the signatures layer.

A field's ``semantic_role`` is what the field means to the LM exchange
(signature-level intent), separate from its data shape. Since D-δ the
resolution logic lives in ``dspy.signatures.roles`` — conflicts raise at
signature construction naming the field, which the signature metaclass can
only do if the checks live below the engine. This module keeps the engine's
historical import surface.

Roles are threaded onto ``RenderField.metadata`` by
``AdapterPlan.from_signature`` and are load-bearing for strategy
resolution: the builder resolves ``strategy_for(role, annotation)``
role-first with the annotation key as fallback
(roadmap/epic-C-semantic-roles.md §6, stage 2), and since D-δ consults the
role lane for explicitly-declared roles on plain shapes as well (spec
section 2, role-keyed admission).
"""

from dspy.signatures.roles import (  # noqa: F401
    _marker_roles,
    explicit_roles,
    resolve_semantic_role,
    semantic_role_for,
)

__all__ = [
    "explicit_roles",
    "resolve_semantic_role",
    "semantic_role_for",
]
