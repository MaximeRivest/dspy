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
from typing import Annotated, Any, Union, get_args, get_origin

from dspy.signatures.roles import SemanticRole

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
    """Unwrap ``Optional``/unions, ``list``, and ``Annotated`` down to core types."""
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
    if origin is Annotated:
        return _core_annotations(get_args(annotation)[0])
    return [annotation]


def _marker_roles(annotation: Any) -> set[str]:
    """Collect role-marker declarations anywhere in the annotation tree.

    ``citations[list[str]]`` and ``list[citations[str]]`` both declare the
    field's participation in the citations role (per-item vs whole-value is
    strategy territory, not role territory — see the epic doc).
    """
    roles: set[str] = set()
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        roles.update(meta.name for meta in args[1:] if isinstance(meta, SemanticRole))
        roles.update(_marker_roles(args[0]))
        return roles
    if origin in (Union, UnionType) or origin is list:
        for arg in get_args(annotation):
            if arg is not type(None):
                roles.update(_marker_roles(arg))
    return roles


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
    """A field's effective role: explicit declarations win, else derived.

    Explicit declarations are the ``role=`` kwarg and ``Annotated`` role
    markers (pydantic hoists a top-level ``Annotated``'s metadata onto
    ``FieldInfo.metadata``; nested markers stay in the annotation tree). Any
    two declarations that disagree — marker vs kwarg, or either vs a
    non-``plain`` legacy-type derivation — are a declaration bug and raise.
    """
    declared: set[str] = set()
    extra = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extra, dict) and extra.get("semantic_role") is not None:
        declared.add(extra["semantic_role"])
    for meta in getattr(field_info, "metadata", None) or []:
        if isinstance(meta, SemanticRole):
            declared.add(meta.name)
    declared.update(_marker_roles(field_info.annotation))

    if len(declared) > 1:
        raise ValueError(
            f"Field declares conflicting semantic roles {sorted(declared)}; declare exactly one."
        )

    derived = semantic_role_for(field_info.annotation)
    if not declared:
        return derived
    (role,) = declared
    if derived != "plain" and role != derived:
        raise ValueError(
            f"Field declares role={role!r} but its annotation implies role={derived!r}; "
            "drop one of the two declarations."
        )
    return role
