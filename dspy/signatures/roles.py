"""Semantic-role markers and role resolution: the cross-surface primitive.

A role marker declares a field's relationship to the LM exchange inside the
type annotation itself, so every signature surface (class signatures, string
signatures, plain function signatures) speaks one vocabulary:

    answer: Annotated[str, citations] = dspy.OutputField()   # canonical
    answer: citations[str] = dspy.OutputField()              # sugar, same object
    dspy.OutputField(role="citations")                       # kwarg spelling

``citations[str]`` is exactly ``Annotated[str, citations]``: type checkers
see ``str``; dspy sees the marker. The legacy semantic types are the fused
spelling (``Reasoning`` is ``Annotated[str, reasoning]`` with the shape
pinned). See roadmap/epic-C-semantic-roles.md.

Resolution lives here too (D-δ): conflicting declarations raise at
signature construction naming the field, so the checks cannot depend on the
adapter engine — the engine re-exports these functions instead.
"""

from types import UnionType
from typing import Annotated, Any, Union, get_args, get_origin

from dspy.signatures.field import SEMANTIC_ROLES


class SemanticRole:
    """A semantic-role marker usable as ``Annotated`` metadata.

    ``marker[shape]`` returns ``Annotated[shape, marker]`` — subscript sugar
    so roles compose with any shape without hiding it from type checkers.
    """

    __slots__ = ("name",)

    def __init__(self, name: str):
        if name not in SEMANTIC_ROLES:
            raise ValueError(f"Unknown semantic role {name!r}. Valid roles: {sorted(SEMANTIC_ROLES)}.")
        self.name = name

    def __getitem__(self, shape: Any) -> Any:
        return Annotated[shape, self]

    def __repr__(self) -> str:
        return f"dspy.roles.{self.name}"


plain = SemanticRole("plain")
reasoning = SemanticRole("reasoning")
tools = SemanticRole("tools")
tool_calls = SemanticRole("tool_calls")
citations = SemanticRole("citations")
history = SemanticRole("history")
media = SemanticRole("media")
code = SemanticRole("code")

ALL_ROLE_MARKERS = {
    marker.name: marker
    for marker in (plain, reasoning, tools, tool_calls, citations, history, media, code)
}


# ---------------------------------------------------------------------------
# Role resolution (derivation + explicit declarations)
# ---------------------------------------------------------------------------

_DERIVED: dict[Any, str] | None = None


def _derivation_table() -> dict[Any, str]:
    # Lazy: the adapter types layer imports this module's markers, so the
    # table can only be built after dspy finishes importing. Fields with no
    # explicit role declaration never need it at class-construction time.
    global _DERIVED
    if _DERIVED is None:
        from dspy.adapters.types.audio import Audio
        from dspy.adapters.types.citation import Citations
        from dspy.adapters.types.code import Code
        from dspy.adapters.types.document import Document
        from dspy.adapters.types.file import File
        from dspy.adapters.types.history import History
        from dspy.adapters.types.image import Image
        from dspy.adapters.types.reasoning import Reasoning
        from dspy.adapters.types.tool import Tool, ToolCalls

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
    strategy territory, not role territory — see the epic doc). Since D-δ
    the scan descends into every generic container (dict values, tuples,
    sets, ...), so a marker is never silently invisible to the conflict
    check.
    """
    roles: set[str] = set()
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        roles.update(meta.name for meta in args[1:] if isinstance(meta, SemanticRole))
        roles.update(_marker_roles(args[0]))
        return roles
    if origin is not None:
        for arg in get_args(annotation):
            if arg is not type(None) and not isinstance(arg, (str, int, bytes, bool)):
                roles.update(_marker_roles(arg))
    return roles


def explicit_roles(field_info: Any) -> set[str]:
    """Every role explicitly declared on a field, across all spellings.

    The ``role=`` kwarg (stored as ``json_schema_extra['semantic_role']``),
    ``Annotated`` markers hoisted onto ``FieldInfo.metadata``, and markers
    nested anywhere in the annotation tree. Pure signature-layer data —
    consultable at class construction with no engine import.
    """
    declared: set[str] = set()
    extra = getattr(field_info, "json_schema_extra", None) or {}
    if isinstance(extra, dict) and extra.get("semantic_role") is not None:
        declared.add(extra["semantic_role"])
    for meta in getattr(field_info, "metadata", None) or []:
        if isinstance(meta, SemanticRole):
            declared.add(meta.name)
    declared.update(_marker_roles(field_info.annotation))
    return declared


def semantic_role_for(annotation: Any) -> str:
    """Derive the semantic role implied by a legacy type annotation.

    Unwraps ``Optional[...]`` and ``list[...]``; every non-None member of a
    union must agree on the same role, otherwise the field is ``plain``.
    ``Code`` subclasses (``dspy.Code["python"]``) derive ``code``.
    """
    table = _derivation_table()
    roles: set[str] = set()
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


def _field_label(field_name: str | None, owner: str | None) -> str:
    if field_name and owner:
        return f"Field {field_name!r} in {owner!r}"
    if field_name:
        return f"Field {field_name!r}"
    return "Field"


def resolve_semantic_role(field_info: Any, *, field_name: str | None = None, owner: str | None = None) -> str:
    """A field's effective role: explicit declarations win, else derived.

    Explicit declarations are the ``role=`` kwarg and ``Annotated`` role
    markers (pydantic hoists a top-level ``Annotated``'s metadata onto
    ``FieldInfo.metadata``; nested markers stay in the annotation tree). Any
    two declarations that disagree — marker vs kwarg, or either vs a
    non-``plain`` legacy-type derivation — are a declaration bug and raise,
    naming the field when the caller provides its name (signature
    construction does, so conflicts raise at declaration — D-δ).
    """
    declared = explicit_roles(field_info)

    if len(declared) > 1:
        raise ValueError(
            f"{_field_label(field_name, owner)} declares conflicting semantic roles "
            f"{sorted(declared)}; declare exactly one."
        )

    derived = semantic_role_for(field_info.annotation)
    if not declared:
        return derived
    (role,) = declared
    if derived != "plain" and role != derived:
        raise ValueError(
            f"{_field_label(field_name, owner)} declares role={role!r} but its annotation implies "
            f"role={derived!r}; drop one of the two declarations."
        )
    return role


def validate_declared_roles(field_info: Any, *, field_name: str, owner: str) -> None:
    """Refuse conflicting role declarations at signature construction (D-δ).

    Fields with no explicit declaration skip entirely — derivation alone
    cannot conflict, and skipping keeps class construction free of adapter
    imports for every legacy signature.
    """
    if not explicit_roles(field_info):
        return
    resolve_semantic_role(field_info, field_name=field_name, owner=owner)


__all__ = [
    "ALL_ROLE_MARKERS",
    "SemanticRole",
    "citations",
    "code",
    "explicit_roles",
    "history",
    "media",
    "plain",
    "reasoning",
    "resolve_semantic_role",
    "semantic_role_for",
    "tool_calls",
    "tools",
    "validate_declared_roles",
]
