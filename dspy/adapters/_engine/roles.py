"""Semantic-role derivation from legacy type annotations.

A field's ``semantic_role`` is what the field means to the LM exchange
(signature-level intent), separate from its data shape. The closed
vocabulary lives in ``dspy.signatures.field.SEMANTIC_ROLES``; this module
derives the implied role for the legacy semantic-type annotations
(``Reasoning`` implies ``reasoning``, ``list[Tool]`` implies ``tools``, ...)
so existing programs participate in the role system unchanged.

Derive-and-record only: roles are threaded onto ``RenderField.metadata`` by
``AdapterPlan.from_signature`` and consulted by nothing yet. Strategy
resolution stays annotation-keyed until the cutover epic
(roadmap/epic-C-semantic-roles.md §6).
"""

from types import UnionType
from typing import Any, Union, get_args, get_origin

_DERIVED: dict[Any, str] | None = None


def _derivation_table() -> dict[Any, str]:
    global _DERIVED
    if _DERIVED is None:
        from dspy.adapters.types.audio import Audio
        from dspy.adapters.types.code import Code
        from dspy.adapters.types.document import Document
        from dspy.adapters.types.file import File
        from dspy.adapters.types.history import History
        from dspy.adapters.types.image import Image
        from dspy.adapters.types.reasoning import Reasoning
        from dspy.adapters.types.tool import Tool, ToolCalls
        from dspy.experimental import Citations

        _DERIVED = {
            Reasoning: "reasoning",
            Tool: "tools",
            ToolCalls: "tool_calls",
            Citations: "citations",
            History: "history",
            Image: "media",
            Audio: "media",
            File: "media",
            Document: "media",
            Code: "code",
        }
    return _DERIVED


def _core_annotations(annotation: Any) -> list[Any]:
    """Unwrap ``Optional``/unions and ``list`` down to candidate core types."""
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        cores: list[Any] = []
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            cores.extend(_core_annotations(arg))
        return cores
    if origin is list:
        args = get_args(annotation)
        return _core_annotations(args[0]) if args else []
    return [annotation]


def semantic_role_for(annotation: Any) -> str:
    """Derive the semantic role implied by a legacy type annotation.

    Unwraps ``Optional[...]`` and ``list[...]``; every non-None member of a
    union must agree on the same role, otherwise the field is ``plain``.
    ``Code`` subclasses (``dspy.Code["python"]``) derive ``code``.
    """
    table = _derivation_table()
    roles = set()
    for core in _core_annotations(annotation):
        role = table.get(core)
        if role is None and isinstance(core, type):
            for base, base_role in table.items():
                if issubclass(core, base):
                    role = base_role
                    break
        roles.add(role or "plain")
    if len(roles) == 1:
        return roles.pop()
    return "plain"


def resolve_semantic_role(field_info: Any) -> str:
    """A field's effective role: explicit ``role=`` declaration wins, else derived.

    An explicit role that contradicts a non-``plain`` derived role is a
    declaration bug and raises.
    """
    extra = getattr(field_info, "json_schema_extra", None) or {}
    declared = extra.get("semantic_role") if isinstance(extra, dict) else None
    derived = semantic_role_for(field_info.annotation)
    if declared is None:
        return derived
    if derived != "plain" and declared != derived:
        raise ValueError(
            f"Field declares role={declared!r} but its annotation implies role={derived!r}; "
            "drop one of the two declarations."
        )
    return declared
