"""The per-role strategy vocabulary, as one data structure (spec section 5).

Every role that admits ≥2 materially different strategies has a closed,
versioned set of strategy names; ``"auto"`` is always legal and resolves
against LM capabilities at plan build with the resolution recorded
(declare-don't-discover). The adapter constructors' ``strategies=``
binding surface validates against this data eagerly, and the D-5 serde
carries ``STRATEGIES_VERSION`` in its versions block — the vocabulary and
the validator cannot drift because there is only the one structure.

``IMPLEMENTED_STRATEGIES`` is the subset a binding may name today; naming
a declared-but-unimplemented entry refuses eagerly with the implemented
set, never a silent acceptance.
"""

#: Version of the strategy vocabulary (D-024 serde versions block).
STRATEGIES_VERSION = "1.0.0"

STRATEGIES_VOCABULARY: dict = {
    "reasoning": {
        "native_channel": "the provider reasoning channel; the field leaves the token stream",
        "textual_field": "the visible marker-field polyfill (the historical CoT shape)",
        "prefill": "assistant-prefill steering of the reasoning text",
    },
    "tools": {
        "native_fc": "provider function calling; tools leave the token stream",
        "textual_json": "tools rendered textually, calls parsed from the completion",
        "xml_dispatch": "XML-tagged textual tool dispatch",
    },
    "citations": {
        "native": "the provider citation channel, assembled outside the token stream",
        "span_markers": "inline span-marker citations resolved against the sources",
        "json_quotes": "quote-object citations parsed from the completion",
    },
    "media": {
        "native_parts": "media rendered as native content parts at the field's slot",
        "url_reference": "media passed as URL references in text",
    },
    "history": {
        "directive_turns": "prior turns expand through the template's history directive",
        "inline": "history rendered inline in one block",
    },
}

#: What a binding may name today, per role ("auto" is always legal).
IMPLEMENTED_STRATEGIES: dict = {
    "reasoning": ("native_channel", "textual_field"),
    "tools": ("native_fc", "textual_json"),
    "citations": ("native",),
    "media": ("native_parts",),
    "history": ("directive_turns",),
}

#: The binding name that means "serve this role natively", per role.
NATIVE_BINDINGS: dict = {
    "reasoning": "native_channel",
    "citations": "native",
    "tools": "native_fc",
}

#: Binding names that force the textual lane (the field stays visible and
#: the template hosts it), per role.
TEXTUAL_BINDINGS: dict = {
    "reasoning": ("textual_field",),
    "tools": ("textual_json",),
}


def check_binding_name(role: str, name: str) -> None:
    """Refuse an unknown role, unknown name, or unimplemented name.

    ``"auto"`` is always legal. The one validation every binding surface
    shares — constructor kwarg, entry load, preset registration — so the
    admitted set cannot drift between them.
    """
    if role not in STRATEGIES_VOCABULARY:
        raise ValueError(
            f"unknown strategy-binding role {role!r} — bindable roles: {', '.join(STRATEGIES_VOCABULARY)}"
        )
    if name == "auto":
        return
    valid = STRATEGIES_VOCABULARY[role]
    if name not in valid:
        raise ValueError(f"unknown {role} strategy {name!r} — valid {role} strategies: auto, {', '.join(valid)}")
    implemented = IMPLEMENTED_STRATEGIES[role]
    if name not in implemented:
        raise ValueError(
            f"{role} strategy {name!r} is declared in the vocabulary but not implemented — "
            f"implemented {role} strategies: auto, {', '.join(implemented)}"
        )


def validate_strategy_bindings(bindings, *, use_native_function_calling: bool) -> dict:
    """Validate a ``strategies={...}`` constructor binding eagerly.

    Returns the normalized dict. Unknown roles, unknown strategy names,
    declared-but-unimplemented names, and a tools binding disagreeing with
    ``use_native_function_calling`` all refuse here with teaching errors —
    a binding never fails later or degrades silently.
    """
    if not isinstance(bindings, dict):
        raise ValueError(
            f"strategies= takes a dict of role -> strategy-name bindings, got {type(bindings).__name__} — "
            f"e.g. strategies={{'reasoning': 'textual_field'}}"
        )
    normalized: dict = {}
    for role, name in bindings.items():
        check_binding_name(role, name)
        normalized[role] = name

    # The tools binding and the legacy kwarg say the same thing; until the
    # kwarg deprecates, the two declarations must agree (L5: refuse over
    # silently preferring either).
    tools_binding = normalized.get("tools")
    if tools_binding == "native_fc" and not use_native_function_calling:
        raise ValueError(
            "strategies={'tools': 'native_fc'} conflicts with use_native_function_calling=False — "
            "pass use_native_function_calling=True; the two declarations must agree"
        )
    if tools_binding == "textual_json" and use_native_function_calling:
        raise ValueError(
            "strategies={'tools': 'textual_json'} conflicts with use_native_function_calling=True — "
            "pass use_native_function_calling=False; the two declarations must agree"
        )
    return normalized
