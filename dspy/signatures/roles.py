"""Semantic-role markers: the cross-surface primitive for declaring roles.

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
"""

from typing import Annotated, Any

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

__all__ = [
    "ALL_ROLE_MARKERS",
    "SemanticRole",
    "citations",
    "code",
    "history",
    "media",
    "plain",
    "reasoning",
    "tool_calls",
    "tools",
]
