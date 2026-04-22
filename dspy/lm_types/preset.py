"""Typed preset fragments and message elements.

A preset is a typed list of message elements.  Each element is either a
``Message`` with typed content, or a directive (``Demos``, ``History``)
that expands into multiple messages at render time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Union

# ─────────────────────────────────────────────────────────────
# Content fragments (renderable pieces of a message)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Text:
    """A literal text fragment."""
    text: str


@dataclass(frozen=True)
class Instruction:
    """Renders ``signature.instructions`` — the MIPRO/COPRO optimizable slot."""
    pass


@dataclass(frozen=True)
class InputFields:
    """Renders the numbered list of input field descriptions."""
    pass


@dataclass(frozen=True)
class OutputFields:
    """Renders the numbered list of output field descriptions."""
    pass


@dataclass(frozen=True)
class Structure:
    """Renders the full field-placeholder template for the given style.

    For ``style="chat"``::

        [[ ## field ## ]]
        {field}
        ...
        [[ ## completed ## ]]

    Fragments delegate style lookups through the registered ``Style``.
    """
    style: str = "chat"


@dataclass(frozen=True)
class Inputs:
    """Renders the current input values in the chosen style."""
    style: str = "chat"


@dataclass(frozen=True)
class Outputs:
    """Renders output field placeholders or schema."""
    style: str = "schema"


@dataclass(frozen=True)
class OutputRequest:
    """Renders the 'respond with…' instruction for the given style."""
    style: str = "chat"


@dataclass(frozen=True)
class ForEach:
    """Iterate a small template over the signature's input or output fields.

    Placeholders available in ``template``:
        - ``{name}``: field name
        - ``{type}``: field type name
        - ``{desc}``: field description (empty if unset)
        - ``{value}``: current formatted input value (inputs only)
    """
    fields: Literal["input", "output"]
    template: str
    separator: str = "\n"


@dataclass(frozen=True)
class CallableFragment:
    """Escape hatch: a callable that renders against the rendering context.

    The callable receives ``(context, **args)`` and returns a string.
    """
    fn: Callable[..., str]
    args: dict = field(default_factory=dict)


# The union of all content fragments (plus plain str for convenience)
Fragment = Union[
    str,
    Text,
    Instruction,
    InputFields,
    OutputFields,
    Structure,
    Inputs,
    Outputs,
    OutputRequest,
    ForEach,
    CallableFragment,
]


Content = Union[Fragment, list[Fragment], tuple[Fragment, ...]]
"""Message content: one fragment, or a sequence of them."""


# ─────────────────────────────────────────────────────────────
# Message envelope and directives
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Message:
    """A regular chat message with typed content."""
    role: Literal["system", "user", "assistant"]
    content: Content


@dataclass(frozen=True)
class Demos:
    """Directive: inject few-shot examples here.

    Without arguments, demos render as user/assistant pairs matching
    the adapter's style.  With ``user`` and ``assistant`` templates,
    each demo is rendered with custom formatting.
    """
    user: Content | None = None
    assistant: Content | None = None


@dataclass(frozen=True)
class History:
    """Directive: inject conversation history here."""
    user: Content | None = None
    assistant: Content | None = None


Element = Union[Message, Demos, History]
"""One entry in a preset's message list."""


# ─────────────────────────────────────────────────────────────
# Preset container
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Preset:
    """A typed adapter preset: message templates + style name.

    ``messages`` is a tuple of ``Element`` values.  ``style`` is a
    registered style name (or pass a ``Style`` instance to ``AdapterV2``
    directly).
    """
    messages: tuple[Element, ...]
    style: str = "chat"


# ─────────────────────────────────────────────────────────────
# Built-in presets
# ─────────────────────────────────────────────────────────────

CHAT_PRESET = Preset(
    messages=(
        Message(role="system", content=[
            InputFields(),
            Text("\n"),
            OutputFields(),
            Text("\n"),
            Structure(style="chat"),
            Text("\n"),
            Instruction(),
        ]),
        Demos(),
        History(),
        Message(role="user", content=[
            Inputs(style="chat"),
            Text("\n\n"),
            OutputRequest(style="chat"),
        ]),
    ),
    style="chat",
)


JSON_PRESET = Preset(
    messages=(
        Message(role="system", content=[
            InputFields(),
            Text("\n"),
            OutputFields(),
            Text("\n"),
            Text("Inputs will have the following structure:\n"),
            Inputs(style="chat"),
            Text("\n\nOutputs will be a JSON object with the following fields.\n"),
            Outputs(style="schema"),
            Text("\n\n"),
            Instruction(),
        ]),
        Demos(),
        History(),
        Message(role="user", content=[
            Inputs(style="chat"),
            Text("\n\n"),
            OutputRequest(style="json"),
        ]),
    ),
    style="json",
)


XML_PRESET = Preset(
    messages=(
        Message(role="system", content=[
            InputFields(),
            Text("\n"),
            OutputFields(),
            Text("\n"),
            Structure(style="xml"),
            Text("\n"),
            Instruction(),
        ]),
        Demos(),
        History(),
        Message(role="user", content=[
            Inputs(style="xml"),
            Text("\n\n"),
            OutputRequest(style="xml"),
        ]),
    ),
    style="xml",
)


CSV_PRESET = Preset(
    messages=(
        Message(role="system", content=[
            InputFields(),
            Text("\n"),
            OutputFields(),
            Text("\n"),
            Instruction(),
        ]),
        Demos(),
        History(),
        Message(role="user", content=[
            Inputs(style="chat"),
            Text("\n\n"),
            OutputRequest(style="csv"),
        ]),
    ),
    style="csv",
)


PRESETS: dict[str, Preset] = {
    "chat": CHAT_PRESET,
    "json": JSON_PRESET,
    "xml": XML_PRESET,
    "csv": CSV_PRESET,
}


__all__ = [
    "Text",
    "Instruction",
    "InputFields",
    "OutputFields",
    "Structure",
    "Inputs",
    "Outputs",
    "OutputRequest",
    "ForEach",
    "CallableFragment",
    "Fragment",
    "Content",
    "Message",
    "Demos",
    "History",
    "Element",
    "Preset",
    "CHAT_PRESET",
    "JSON_PRESET",
    "XML_PRESET",
    "CSV_PRESET",
    "PRESETS",
]
