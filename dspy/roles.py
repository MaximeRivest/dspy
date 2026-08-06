"""Semantic-role markers, publicly importable as ``dspy.roles``.

Roles declare what a field means to the LM exchange, independent of its data
shape. Four spellings resolve to the same marker objects:

    import dspy

    answer: dspy.roles.citations[str] = dspy.OutputField()       # subscript sugar
    answer: Annotated[str, dspy.roles.citations] = dspy.OutputField()  # canonical
    answer: str = dspy.OutputField(role="citations")             # kwarg
    dspy.Signature("question -> answer: str @citations")         # string shorthand

The vocabulary is closed and versioned: ``plain``, ``reasoning``, ``tools``,
``tool_calls``, ``citations``, ``history``, ``media``, ``code``. The marker
objects live in ``dspy.signatures.roles``; this module is the public door.
"""

from dspy.signatures.roles import (
    ALL_ROLE_MARKERS,
    SemanticRole,
    citations,
    code,
    history,
    media,
    plain,
    reasoning,
    tool_calls,
    tools,
)

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
