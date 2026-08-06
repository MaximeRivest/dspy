"""The template language's entire vocabulary, as one data structure.

The language is closed (spec section 3), so everything in it is enumerable —
and the contract requires that it be enumerable *as data, from one source*.
This module is that source: the parser validates against it, error messages
enumerate valid sets from it, and ``describe_template_language()`` returns
it. Nothing else may hardcode a slot, style, variable, or directive name.

Every entry carries a one-line human description, so the structure doubles
as the documentation table; entries that need machine-read facts beyond
the description (loop options declare a ``takes_value`` arity bit) carry
them alongside it — the parser's acceptance reads the same data the error
messages enumerate.
"""

from copy import deepcopy

#: Version of the template language's vocabulary; carried in every
#: serialized preset's versions block (D-024). Extending the language is a
#: versioned act.
TEMPLATE_LANGUAGE_VERSION = "1.0.0"

VOCABULARY: dict = {
    "value_slots": {
        "{field_name}": "a signature field's value, spelled by the direction's bound codec; unknown names refuse at render, missing values render empty",
        "{field('name')}": "the escape spelling of a field's value slot — unambiguous for any name, and the only spelling for a field carrying a reserved name",
        "{instruction}": "the signature instructions (ProgramIR 3a); accepts style=",
    },
    "instruction_styles": {
        "raw": "the instructions string verbatim",
        "indented": "dedent, then prefix every line with a newline and eight spaces (the historical ChatAdapter objective block)",
    },
    "aggregate_slots": {
        "inputs": "all input-field values in one block; accepts style=",
        "outputs": "the output-field schema (or values, in assistant position); accepts style= and, for style='xml', wrap=",
        "demos": "all few-shot demos inline in one block; accepts style=",
        "history": "all conversation-history turns inline in one block; accepts style=",
    },
    "aggregate_styles": {
        "inputs": {
            "default": "name: value lines",
            "yaml": "name: value lines (same as default)",
            "json": "one JSON object of the rendered values",
            "xml": "<name>value</name> lines",
            "chat": "[[ ## name ## ]] marker blocks",
        },
        "outputs": {
            "default": "the numbered field-description list",
            "schema": "one JSON object of per-field JSON schemas",
            "xml": "<name>description</name> placeholder tags, optionally wrapped",
            "chat": "[[ ## name ## ]] marker blocks with typed placeholders",
            "json_object": "one JSON object: typed placeholders as values, or the call's output values in assistant position",
        },
        "demos": {
            "default": "numbered 'Example N:' text blocks",
            "json": "one JSON object per demo",
            "yaml": "name: value lines per demo",
            "xml": "<name>value</name> lines per demo",
            "chat": "[[ ## name ## ]] marker blocks per demo",
        },
        "history": {
            "default": "numbered 'Turn N:' text blocks",
            "json": "one JSON array of turn objects",
            "yaml": "yaml-style turn list",
            "xml": "<turn> blocks",
            "chat": "[[ ## name ## ]] marker blocks per turn",
        },
    },
    "aggregate_kwargs": {
        "inputs": ("style",),
        "outputs": ("style", "wrap"),
        "demos": ("style",),
        "history": ("style",),
    },
    "fragment_targets": {
        "system": "the system-message slot textual strategies target",
        "user": "the user-message slot textual strategies target",
    },
    "loop_collections": {
        "inputs": "the plan's visible input fields",
        "outputs": "the plan's visible output fields",
    },
    "loop_variables": {
        "i": "1-based position in the iteration",
        "index": "alias of i",
        "name": "the model-facing field name",
        "type": "the annotation name (str, list[str], ...)",
        "desc": "the bare field description, empty when unset",
        "desc_suffix": "the field-description remainder after `name` (type): — description, custom-type notes, constraints",
        "value": "the field's value for the current call/demo/turn, spelled by the direction's codec",
        "placeholder": "the literal {field_name} token",
        "typed_placeholder": "the field's schema, spelled by the direction's codec (the shared text codec renders the placeholder plus the historical type note)",
        "marker": "the [[ ## field_name ## ]] marker",
        "chat_type_hint": "the historical non-str output hint suffix, empty for str",
    },
    "loop_options": {
        "separator": {
            "takes_value": True,
            "description": "string joined between iterations (quoted; \\n \\t \\r \\\\ \\' \\\" escapes); default newline",
        },
        "strip": {
            "takes_value": False,
            "description": "bare flag: str.strip() the joined result (the historical section-strip shape)",
        },
    },
    "directive_roles": {
        "demos": "expands into user/assistant message pairs per demo; optional user=/assistant= content templates",
        "history": "expands prior conversation turns; optional user=/assistant= content templates",
    },
    "message_roles": {
        "system": "one rendered system message",
        "user": "one rendered user message",
        "assistant": "one rendered assistant message",
    },
    "parsers": {
        "chat": "[[ ## field ## ]] labeled sections",
        "json": "one JSON object with output-field keys",
        "xml": "<field>value</field> tags",
        "full_text": "the whole completion into exactly one plain output field",
    },
    "escapes": {
        "{{": "a literal {",
        "}}": "a literal }",
        "{{{f.attr}}}": "the loop variable's value wrapped in literal braces (e.g. {question})",
    },
}


#: The names the language reserves for its own slots (spec section 3). A
#: signature field may carry one — loops and aggregates render it like any
#: other field — but it has no bare value-slot spelling, and shadowing is
#: never silent: the parser and renderer refuse naming the collision and
#: the {field('name')} escape spelling as the way out.
RESERVED_SLOT_NAMES: tuple = ("instruction", *VOCABULARY["aggregate_slots"], "fragments", "field")


def describe_template_language() -> dict:
    """The whole template-language vocabulary, as data.

    Returns a deep copy of the one structure the validator and every error
    message read, so callers can introspect (or render docs from) exactly
    the language the parser enforces.
    """
    return deepcopy(VOCABULARY)


def spell_out(names) -> str:
    """Comma-join names for teaching error messages, in declared order."""
    return ", ".join(names)
