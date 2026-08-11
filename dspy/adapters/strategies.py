"""The strategy engine: rule validation, resolution, and the role registry.

A `strategies` entry value is either a NAME (a binding into the role's
registry — `"auto"` always legal) or a RULE object (`{"kind": "rule"}`,
the four-plus-two faces as data). Predicates test DECLARED LM-capability
facts (`dspy.lm.LMCapabilities`), never live objects; an explicit binding
whose predicate fails refuses loudly naming the capability, and `"auto"`
resolves to the first registered rule whose predicate passes — recorded,
so resolution is always explainable.

The builtin rules are the adapter-ir-stage trios verbatim: reasoning
native / prefix_cot / interleaved (example 05), tools native_fc /
cli_text / xml_blocks (example 06), citations native / inline
(example 07). `register_strategy(role, name, rule)` is the public door.
"""

import re
from dataclasses import dataclass, field
from typing import Any

from dspy.adapters.codecs import COERCE_SHAPES
from dspy.adapters.errors import AdapterError, EntryError
from dspy.adapters.parse import validate_pipeline

#: Version of the name-binding strategy vocabulary.
STRATEGIES_VERSION = "1.0.0"

#: Version an entry carries when any strategy value is a rule object.
STRATEGIES_RULES_VERSION = "1.1.0-draft"

#: Version of the LM-capability vocabulary (predicates and `requires`).
LM_CAPABILITIES_VERSION = "0.1.0"

#: The declared capability facts a predicate may test.
LM_CAPABILITY_FACTS = (
    "instruct",
    "completion",
    "native_reasoning",
    "native_function_calling",
    "native_citations",
    "image_input",
)

#: Fact name -> LMCapabilities attribute.
_FACT_ATTRS = {
    "instruct": "instruct",
    "native_reasoning": "native_reasoning",
    "native_function_calling": "native_fc",
    "native_citations": "native_citations",
    "image_input": "image_input",
}

#: Roles a strategies block may bind.
BINDABLE_ROLES = ("reasoning", "tools", "citations", "media", "history", "code")

#: Name bindings with no rule behind them: the template (or the codec
#: layer) already conducts them structurally.
STRUCTURAL_BINDINGS: dict[str, tuple[str, ...]] = {
    "media": ("native_parts", "url_reference"),
    "history": ("directive_turns", "inline"),
}


# ---------------------------------------------------------------------------
# Capability facts and predicates
# ---------------------------------------------------------------------------


def capability_fact(capabilities, name: str) -> bool:
    """One declared fact, read off an `LMCapabilities` (None = no facts)."""
    if name not in LM_CAPABILITY_FACTS:
        raise AdapterError(f"unknown LM capability {name!r} — capability vocabulary: {', '.join(LM_CAPABILITY_FACTS)}")
    if capabilities is None:
        return False
    if name == "completion":
        return not capabilities.instruct
    return bool(getattr(capabilities, _FACT_ATTRS[name], False))


def evaluate_predicate(predicate: dict, capabilities) -> bool:
    """Evaluate a predicate tree over declared capability facts."""
    if "capability" in predicate:
        return capability_fact(capabilities, predicate["capability"])
    if "all" in predicate:
        return all(evaluate_predicate(p, capabilities) for p in predicate["all"])
    if "any" in predicate:
        return any(evaluate_predicate(p, capabilities) for p in predicate["any"])
    if "not" in predicate:
        return not evaluate_predicate(predicate["not"], capabilities)
    raise AdapterError(f"malformed predicate {predicate!r}")  # pragma: no cover - validation prevents


def validate_predicate(predicate: Any, *, where: str) -> None:
    """Refuse a malformed predicate eagerly, naming the valid atoms."""
    if not isinstance(predicate, dict) or len(predicate) != 1:
        raise EntryError(
            f"{where}: a predicate is one of {{'capability': ...}}, {{'all': [...]}}, "
            f"{{'any': [...]}}, {{'not': ...}} — got {predicate!r}"
        )
    key = next(iter(predicate))
    if key == "capability":
        name = predicate[key]
        if name not in LM_CAPABILITY_FACTS:
            raise EntryError(
                f"{where}: unknown LM capability {name!r} — capability vocabulary: {', '.join(LM_CAPABILITY_FACTS)}"
            )
        return
    if key in ("all", "any"):
        branches = predicate[key]
        if not isinstance(branches, list) or not branches:
            raise EntryError(f"{where}: {key!r} takes a non-empty list of predicates")
        for index, branch in enumerate(branches):
            validate_predicate(branch, where=f"{where}.{key}[{index}]")
        return
    if key == "not":
        validate_predicate(predicate[key], where=f"{where}.not")
        return
    raise EntryError(f"{where}: unknown predicate form {key!r} — valid forms: capability, all, any, not")


def predicate_capabilities(predicate: dict) -> list[str]:
    """Every capability atom a predicate mentions, in order (for `requires`)."""
    if "capability" in predicate:
        return [predicate["capability"]]
    for key in ("all", "any"):
        if key in predicate:
            atoms: list[str] = []
            for branch in predicate[key]:
                atoms.extend(predicate_capabilities(branch))
            return atoms
    if "not" in predicate:
        return predicate_capabilities(predicate["not"])
    return []


def describe_predicate(predicate: dict) -> str:
    if "capability" in predicate:
        return f"capability {predicate['capability']!r}"
    key = next(iter(predicate))
    return f"{key}(...) over capabilities {predicate_capabilities(predicate)}"


# ---------------------------------------------------------------------------
# Rule validation
# ---------------------------------------------------------------------------

_RULE_FACES = ("kind", "predicate", "hides", "transforms", "fragments", "engine_controls", "routings")
_FRAGMENT_TARGETS = ("system", "user")

#: The fragment dialect: literal text plus `{field('name')}` slots. Bare
#: braces are LITERAL in fragments (instruction prose routinely shows the
#: model JSON), so fragments speak exactly one slot spelling — the escape
#: spelling — and nothing else interpolates.
_FRAGMENT_SLOT = re.compile(r"\{field\(\s*(?:'([^']*)'|\"([^\"]*)\")\s*\)\}")


def validate_fragment_content(content: Any, *, where: str) -> None:
    """Refuse non-string content and malformed `{field(...)}` slots."""
    if not isinstance(content, str):
        raise EntryError(f"{where}: fragment content is a string, got {content!r}")
    for match in re.finditer(r"\{field\([^)]*\)\}", content):
        if not _FRAGMENT_SLOT.fullmatch(match.group(0)):
            raise EntryError(
                f"{where}: malformed field slot {match.group(0)!r} — the fragment dialect is literal "
                "text plus {field('name')} slots"
            )


def render_fragment(content: str, signature, values, input_codec) -> str:
    """Render one fragment: substitute `{field('name')}` slots, keep the
    rest literal. Unknown names refuse; absent values render empty (the
    value-slot rule)."""

    def substitute(match: re.Match) -> str:
        name = match.group(1) if match.group(1) is not None else match.group(2)
        fields = signature.fields
        if name not in fields:
            raise AdapterError(
                f"fragment references field {name!r}, but this signature declares no such field "
                f"(fields: {', '.join(fields) or 'none'})"
            )
        if not values or name not in values:
            return ""
        return input_codec.render_value(values[name], fields[name])

    return _FRAGMENT_SLOT.sub(substitute, content)


def validate_rule(rule: Any, *, where: str) -> None:
    """Refuse a malformed rule object eagerly, face by face."""
    if not isinstance(rule, dict) or rule.get("kind") != "rule":
        raise EntryError(f"{where}: a strategy rule is a dict with kind='rule', got {rule!r}")
    unknown = set(rule) - set(_RULE_FACES)
    if unknown:
        raise EntryError(f"{where}: unknown rule faces {sorted(unknown)} — valid faces: {', '.join(_RULE_FACES)}")
    missing = [key for key in ("predicate", "hides", "fragments", "engine_controls", "routings") if key not in rule]
    if missing:
        raise EntryError(f"{where}: rule is missing faces {missing} (empty faces are spelled empty, not absent)")

    validate_predicate(rule["predicate"], where=f"{where}.predicate")

    hides = rule["hides"]
    if not isinstance(hides, list) or not all(isinstance(name, str) for name in hides):
        raise EntryError(f"{where}: 'hides' is a list of field names, got {hides!r}")

    for index, transform in enumerate(rule.get("transforms", [])):
        t_where = f"{where}.transforms[{index}]"
        if not isinstance(transform, dict) or set(transform) != {"rename"}:
            raise EntryError(f"{t_where}: a transform is {{'rename': {{'from': ..., 'to': ...}}}}")
        spec = transform["rename"]
        if not isinstance(spec, dict) or set(spec) != {"from", "to"}:
            raise EntryError(f"{t_where}: rename takes exactly 'from' and 'to'")

    for index, frag in enumerate(rule["fragments"]):
        f_where = f"{where}.fragments[{index}]"
        if not isinstance(frag, dict) or set(frag) != {"target", "content"}:
            raise EntryError(f"{f_where}: a fragment is {{'target': ..., 'content': ...}}")
        if frag["target"] not in _FRAGMENT_TARGETS:
            raise EntryError(
                f"{f_where}: unknown fragment target {frag['target']!r} — valid targets: {', '.join(_FRAGMENT_TARGETS)}"
            )
        validate_fragment_content(frag["content"], where=f_where)

    controls = rule["engine_controls"]
    if not isinstance(controls, dict):
        raise EntryError(f"{where}: 'engine_controls' is a dict, got {controls!r}")
    if "request_patch" in controls and not isinstance(controls["request_patch"], dict):
        raise EntryError(f"{where}: engine_controls.request_patch is a dict")

    routings = rule["routings"]
    if not isinstance(routings, list):
        raise EntryError(f"{where}: 'routings' is a list, got {routings!r}")
    for index, routing in enumerate(routings):
        _validate_routing(routing, where=f"{where}.routings[{index}]")


def _validate_routing(routing: Any, *, where: str) -> None:
    if not isinstance(routing, dict) or "field" not in routing:
        raise EntryError(f"{where}: a routing is a dict with a 'field' key, got {routing!r}")
    if "channel" in routing:
        unknown = set(routing) - {"channel", "field", "coerce"}
        if unknown:
            raise EntryError(f"{where}: unknown channel-routing keys {sorted(unknown)}")
        coerce = routing.get("coerce", "str")
        if coerce not in COERCE_SHAPES:
            raise EntryError(
                f"{where}: unknown coerce shape {coerce!r} — shapes vocabulary: {', '.join(COERCE_SHAPES)}"
            )
        return
    if "text" in routing:
        unknown = set(routing) - {"text", "field", "consume", "coerce", "materialize"}
        if unknown:
            raise EntryError(f"{where}: unknown text-routing keys {sorted(unknown)}")
        validate_pipeline(routing["text"], where=f"{where}.text")
        if not isinstance(routing.get("consume", False), bool):
            raise EntryError(f"{where}: 'consume' is a bool")
        if "materialize" in routing:
            leaf = routing["materialize"].get("interpreter", {}).get("leaf", "?")
            raise EntryError(
                f"{where}: requires interpreter leaf {leaf!r} for value materialization — unbound in this "
                "engine; refuse or bind one (materialize is a declared-leaf door this stage does not ship)"
            )
        return
    raise EntryError(f"{where}: a routing carries either 'channel' or 'text'")


# ---------------------------------------------------------------------------
# The builtin rules (the adapter-ir-stage trios, verbatim)
# ---------------------------------------------------------------------------

BUILTIN_RULES: dict[str, dict[str, dict]] = {
    "reasoning": {
        "native": {
            "kind": "rule",
            "predicate": {"capability": "native_reasoning"},
            "hides": ["reasoning"],
            "fragments": [],
            "engine_controls": {"request_patch": {"reasoning": {"effort": "medium"}}},
            "routings": [{"channel": "reasoning", "field": "reasoning", "coerce": "str"}],
        },
        "prefix_cot": {
            "kind": "rule",
            "predicate": {"capability": "instruct"},
            "hides": [],
            "fragments": [
                {
                    "target": "user",
                    "content": "Reason step by step in the `reasoning` field before giving the final answer.",
                }
            ],
            "engine_controls": {},
            "routings": [],
        },
        "interleaved": {
            "kind": "rule",
            "predicate": {"capability": "instruct"},
            "hides": ["reasoning"],
            "fragments": [
                {
                    "target": "user",
                    "content": "After every sentence of your answer, add a brief thought inside <think>...</think> tags.",
                }
            ],
            "engine_controls": {},
            "routings": [
                {
                    "text": {
                        "kind": "pipeline",
                        "steps": [
                            {
                                "op": "regex",
                                "dialect": "re2",
                                "pattern": "<think>(?P<t>[^<]*)</think>",
                                "mode": "findall",
                                "group": "t",
                                "join": "\n",
                            }
                        ],
                    },
                    "field": "reasoning",
                    "consume": True,
                }
            ],
        },
    },
    "tools": {
        "native_fc": {
            "kind": "rule",
            "predicate": {"capability": "native_function_calling"},
            "hides": ["tools", "tool_calls"],
            "fragments": [],
            "engine_controls": {"request_patch": {"tools": {"$from": "field:tools"}, "tool_choice": "auto"}},
            "routings": [{"channel": "tool_calls", "field": "tool_calls", "coerce": "ToolCalls"}],
        },
        "cli_text": {
            "kind": "rule",
            "predicate": {"capability": "instruct"},
            "hides": ["tool_calls"],
            "fragments": [
                {
                    "target": "user",
                    "content": "To call a tool, write a line: !call <tool_name> <json arguments>. "
                    "Available tools:\n{field('tools')}",
                }
            ],
            "engine_controls": {},
            "routings": [
                {
                    "text": {
                        "kind": "pipeline",
                        "steps": [
                            {
                                "op": "regex",
                                "dialect": "re2",
                                "pattern": "(?m)^!call (?P<name>[a-zA-Z0-9_]+) (?P<args>\\{.*\\})$",
                                "mode": "findall",
                            },
                            {"op": "tool_calls", "name_group": "name", "args_group": "args", "args": "json"},
                        ],
                    },
                    "field": "tool_calls",
                    "consume": True,
                }
            ],
        },
        "xml_blocks": {
            "kind": "rule",
            "predicate": {"capability": "instruct"},
            "hides": ["tool_calls"],
            "fragments": [
                {
                    "target": "user",
                    "content": 'To call a tool, emit <tool_call name="NAME">{"arg": ...}</tool_call>. '
                    "Available tools:\n{field('tools')}",
                }
            ],
            "engine_controls": {},
            "routings": [
                {
                    "text": {
                        "kind": "pipeline",
                        "steps": [
                            {
                                "op": "regex",
                                "dialect": "re2",
                                "pattern": '<tool_call name="(?P<name>[a-zA-Z0-9_]+)">(?P<args>[^<]*)</tool_call>',
                                "mode": "findall",
                            },
                            {"op": "tool_calls", "name_group": "name", "args_group": "args", "args": "json"},
                        ],
                    },
                    "field": "tool_calls",
                    "consume": True,
                }
            ],
        },
    },
    "citations": {
        "native": {
            "kind": "rule",
            "predicate": {"capability": "native_citations"},
            "hides": ["citations"],
            "transforms": [{"rename": {"from": "answer", "to": "answer_text"}}],
            "fragments": [],
            "engine_controls": {"request_patch": {"citations": {"enabled": True}}},
            "routings": [{"channel": "citations", "field": "citations", "coerce": "Citations"}],
        },
        "inline": {
            "kind": "rule",
            "predicate": {"capability": "instruct"},
            "hides": ["citations"],
            "fragments": [
                {
                    "target": "user",
                    "content": "After each claim, cite the supporting document as [n], where n is the "
                    "1-based document index.",
                }
            ],
            "engine_controls": {},
            "routings": [
                {
                    "text": {
                        "kind": "pipeline",
                        "steps": [
                            {
                                "op": "regex",
                                "dialect": "re2",
                                "pattern": "(?P<span>[^.!?]*[.!?])\\s*\\[(?P<doc>[0-9]+)\\]",
                                "mode": "findall",
                            },
                            {"op": "citations", "span_group": "span", "doc_group": "doc"},
                        ],
                    },
                    "field": "citations",
                    "consume": False,
                }
            ],
        },
    },
}

#: role -> {name -> rule} registered through the public door.
_REGISTERED: dict[str, dict[str, dict]] = {}


def register_strategy(role: str, name: str, rule: dict) -> None:
    """Register a strategy rule under a role, bindable by name.

    Args:
        role: One of the bindable roles.
        name: The binding name; may not shadow `auto`, a builtin rule
            name, or a structural binding name.
        rule: The rule object; validated eagerly.

    Raises:
        AdapterError: On an unknown role, a shadowing name, or a
            malformed rule.
    """
    if role not in BINDABLE_ROLES:
        raise AdapterError(f"unknown strategy role {role!r} — bindable roles: {', '.join(BINDABLE_ROLES)}")
    reserved = ("auto", *BUILTIN_RULES.get(role, {}), *STRUCTURAL_BINDINGS.get(role, ()))
    if name in reserved:
        raise AdapterError(
            f"strategy name {name!r} collides with the {role} binding vocabulary ({', '.join(reserved)})"
        )
    entries = _REGISTERED.setdefault(role, {})
    if name in entries:
        raise AdapterError(
            f"a strategy named {name!r} is already registered under role {role!r} — "
            f"unregister_strategy({role!r}, {name!r}) first"
        )
    try:
        validate_rule(rule, where=f"register_strategy({role!r}, {name!r})")
    except EntryError as error:
        raise AdapterError(str(error)) from None
    entries[name] = rule


def unregister_strategy(role: str, name: str) -> None:
    """Remove a registered rule; unknown registrations refuse."""
    entries = _REGISTERED.get(role, {})
    if name not in entries:
        registered = ", ".join(entries) or "(none)"
        raise AdapterError(f"no strategy named {name!r} registered under role {role!r} — registered: {registered}")
    del entries[name]


def bindable_names(role: str) -> list[str]:
    """What a binding may name for `role`: auto, builtins, structural,
    registered."""
    return [
        "auto",
        *BUILTIN_RULES.get(role, {}),
        *STRUCTURAL_BINDINGS.get(role, ()),
        *_REGISTERED.get(role, {}),
    ]


def check_binding(role: str, value: Any, *, where: str = "strategies") -> None:
    """Refuse an unknown role, unknown name, or malformed rule object."""
    if role not in BINDABLE_ROLES:
        raise EntryError(f"{where}: unknown strategy role {role!r} — bindable roles: {', '.join(BINDABLE_ROLES)}")
    if isinstance(value, dict):
        validate_rule(value, where=f"{where}.{role}")
        return
    if not isinstance(value, str):
        raise EntryError(f"{where}.{role}: a strategy binding is a name or a rule object, got {value!r}")
    if value == "auto":
        return
    if value not in bindable_names(role):
        raise EntryError(f"{where}.{role}: unknown strategy {value!r} — bindable: {', '.join(bindable_names(role))}")


def resolve_binding(role: str, value: Any, capabilities) -> tuple[dict | None, str]:
    """Resolve one strategy binding against declared capability facts.

    Returns `(rule, note)`: the active rule (None for plain/structural
    conduct) and a resolution note for explain.

    Raises:
        AdapterError: When an explicitly named (or inline) rule's
            predicate fails — refused loudly naming the capability.
    """
    if isinstance(value, dict):
        if not evaluate_predicate(value["predicate"], capabilities):
            raise AdapterError(
                f"the {role} strategy requires {describe_predicate(value['predicate'])}, and the bound LM "
                "does not declare it — bind an LM with the capability, or choose another strategy"
            )
        return value, "rule"
    if value == "auto":
        candidates = {**BUILTIN_RULES.get(role, {}), **_REGISTERED.get(role, {})}
        for name, rule in candidates.items():
            if evaluate_predicate(rule["predicate"], capabilities):
                return rule, f"auto->{name}"
        return None, "auto->plain"
    if value in STRUCTURAL_BINDINGS.get(role, ()):
        return None, value
    rule = BUILTIN_RULES.get(role, {}).get(value) or _REGISTERED.get(role, {}).get(value)
    if rule is None:
        raise AdapterError(f"unknown {role} strategy {value!r} — bindable: {', '.join(bindable_names(role))}")
    if not evaluate_predicate(rule["predicate"], capabilities):
        raise AdapterError(
            f"the {role} strategy {value!r} requires {describe_predicate(rule['predicate'])}, and the bound "
            "LM does not declare it — bind an LM with the capability, or choose another strategy"
        )
    return rule, value


# ---------------------------------------------------------------------------
# Active effects: what the bound strategies do to one call
# ---------------------------------------------------------------------------


@dataclass
class StrategyEffects:
    """Everything the active rules contribute to one exchange."""

    hides: list[str] = field(default_factory=list)
    renames: list[tuple[str, str]] = field(default_factory=list)
    fragments: dict[str, list[str]] = field(default_factory=dict)
    request_patch: dict = field(default_factory=dict)
    engine_controls: dict = field(default_factory=dict)
    channel_routings: list[dict] = field(default_factory=list)
    text_routings: list[dict] = field(default_factory=list)
    resolutions: dict[str, str] = field(default_factory=dict)


def fields_with_role(signature, role: str) -> list[str]:
    """The signature fields carrying `role`, in declaration order."""
    from dspy.signatures.roles import resolve_semantic_role

    return [name for name, info in signature.fields.items() if resolve_semantic_role(info, field_name=name) == role]


def active_effects(strategies: dict, signature, capabilities) -> StrategyEffects:
    """Resolve every bound strategy against one signature and one LM.

    Roles with no participating field are skipped entirely — an adapter's
    strategy block never touches a signature that lacks the role.
    """
    effects = StrategyEffects()
    for role, binding in strategies.items():
        if not fields_with_role(signature, role):
            continue
        rule, note = resolve_binding(role, binding, capabilities)
        effects.resolutions[role] = note
        if rule is None:
            continue
        _check_rule_fields(rule, signature, role)
        effects.hides.extend(name for name in rule["hides"] if name not in effects.hides)
        for transform in rule.get("transforms", []):
            spec = transform["rename"]
            effects.renames.append((spec["from"], spec["to"]))
        for frag in rule["fragments"]:
            effects.fragments.setdefault(frag["target"], []).append(frag["content"])
        controls = rule["engine_controls"]
        for key, value in controls.items():
            if key == "request_patch":
                effects.request_patch.update(value)
            else:
                effects.engine_controls[key] = value
        for routing in rule["routings"]:
            if "channel" in routing:
                effects.channel_routings.append(routing)
            else:
                effects.text_routings.append(routing)
    return effects


def _check_rule_fields(rule: dict, signature, role: str) -> None:
    """A rule that names fields the signature lacks refuses, loudly."""
    declared = set(signature.fields)
    for name in rule["hides"]:
        if name not in declared:
            raise AdapterError(
                f"the {role} strategy hides field {name!r}, but this signature declares no such field "
                f"(fields: {', '.join(declared) or 'none'}) — rules name fields literally"
            )
    for routing in rule["routings"]:
        if routing["field"] not in declared:
            raise AdapterError(
                f"the {role} strategy routes into field {routing['field']!r}, but this signature declares "
                f"no such field (fields: {', '.join(declared) or 'none'})"
            )
    for transform in rule.get("transforms", []):
        source = transform["rename"]["from"]
        if source not in declared:
            raise AdapterError(
                f"the {role} strategy renames field {source!r}, but this signature declares no such field"
            )
