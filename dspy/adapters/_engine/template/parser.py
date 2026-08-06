"""Eager parser for the constrained template language (spec section 3).

Templates parse at construction, never at render: every unknown construct
refuses immediately, naming itself AND the valid set in its category, read
from ``vocabulary.VOCABULARY``. Field-name slots are the one deferred
check — a template cannot know a signature's fields, so ``{name}`` parses
as a value slot and unknown names refuse at render instead.

The grammar is deliberately closed: value slots, aggregate slots with
declared styles, positional fragment slots, single-level loop blocks over
``inputs``/``outputs`` with the fixed ``f.*`` variable set, brace escapes,
and the ``demos``/``history`` directive messages. No general expressions,
no user control flow.
"""

import re
from dataclasses import dataclass

from dspy.adapters._engine.template.vocabulary import VOCABULARY, spell_out


class TemplateError(ValueError):
    """A template construct outside the closed language, refused eagerly."""


# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Text:
    text: str


@dataclass(frozen=True)
class ValueSlot:
    """``{field_name}`` — validated against the signature at render."""

    name: str


@dataclass(frozen=True)
class InstructionSlot:
    style: str = "raw"


@dataclass(frozen=True)
class AggregateSlot:
    kind: str
    style: str = "default"
    wrap: str | None = None


@dataclass(frozen=True)
class FragmentsSlot:
    """``{fragments('system'|'user')}`` — a positional strategy-fragment slot.

    ``owns_line`` records that the slot is alone on its line in the raw
    template text; an empty render then removes the whole line, so an
    unused slot costs zero bytes (the byte-parity guarantee).
    """

    target: str
    owns_line: bool


@dataclass(frozen=True)
class LoopVarSlot:
    """``{f.attr}`` inside a loop body."""

    attr: str


@dataclass(frozen=True)
class LoopVarLiteral:
    """``{{{f.attr}}}`` — the attribute value wrapped in literal braces."""

    attr: str


@dataclass(frozen=True)
class Loop:
    var: str
    collection: str
    separator: str
    strip: bool
    body: tuple


# ---------------------------------------------------------------------------
# Parsed message-template forms
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentMessage:
    role: str
    nodes: tuple
    raw: str


@dataclass(frozen=True)
class DemosDirective:
    """``{"role": "demos"}`` — expands to user/assistant pairs per demo."""

    user: tuple | None
    assistant: tuple | None
    raw_user: str | None
    raw_assistant: str | None


@dataclass(frozen=True)
class HistoryDirective:
    """``{"role": "history"}`` — expands prior turns."""

    user: tuple | None
    assistant: tuple | None
    raw_user: str | None
    raw_assistant: str | None


@dataclass(frozen=True)
class ParsedTemplate:
    messages: tuple
    raw: tuple

    @property
    def demos_directive(self) -> DemosDirective | None:
        return next((m for m in self.messages if isinstance(m, DemosDirective)), None)

    @property
    def history_directive(self) -> HistoryDirective | None:
        return next((m for m in self.messages if isinstance(m, HistoryDirective)), None)


# ---------------------------------------------------------------------------
# Content parsing
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_]\w*")
_SLOT_OPEN = re.compile(r"\{([A-Za-z_]\w*)")
_BLOCK_TAG = re.compile(r"\{%\s*(\w+)")
_FOR_TAG = re.compile(r"\{%\s*for\s+([A-Za-z_]\w*)\s+in\s+([A-Za-z_]\w*)((?:\s[^%]*?)?)\s*%\}")
_ENDFOR_TAG = re.compile(r"\{%\s*endfor\s*%\}")
_NESTED_FOR = re.compile(r"\{%\s*for\b")
_LOOP_OPTION = re.compile(r"\s*(?:(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")|(\w+))")
_CALL_KWARG = re.compile(r"\s*(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")\s*(?:,|$)")
_POSITIONAL = re.compile(r"^\s*(?:'([^']*)'|\"([^\"]*)\")\s*$")

_STRING_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\", "'": "'", '"': '"'}


def _decode_escapes(value: str, where: str) -> str:
    out = []
    i = 0
    while i < len(value):
        ch = value[i]
        if ch == "\\":
            if i + 1 >= len(value) or value[i + 1] not in _STRING_ESCAPES:
                found = value[i : i + 2]
                raise TemplateError(
                    f"unknown escape sequence {found!r} in {where} — supported: \\n, \\t, \\r, \\\\, \\', \\\""
                )
            out.append(_STRING_ESCAPES[value[i + 1]])
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def parse_content(text: str, *, loop_var: str | None = None, allow_fragments: bool = True) -> tuple:
    """Parse one content string into AST nodes, refusing unknown constructs."""
    nodes: list = []
    buffer: list[str] = []
    fragment_targets_seen: set[str] = set()

    def flush() -> None:
        if buffer:
            nodes.append(Text("".join(buffer)))
            buffer.clear()

    i = 0
    n = len(text)
    triple_pattern = (
        re.compile(r"\{\{\{" + re.escape(loop_var) + r"\.(\w+)\}\}\}") if loop_var is not None else None
    )

    while i < n:
        if text.startswith("{%", i):
            flush()
            node, i = _parse_loop(text, i, allow_fragments)
            nodes.append(node)
        elif triple_pattern is not None and (match := triple_pattern.match(text, i)):
            attr = match.group(1)
            _check_loop_attr(attr)
            flush()
            nodes.append(LoopVarLiteral(attr))
            i = match.end()
        elif text.startswith("{{", i):
            buffer.append("{")
            i += 2
        elif text.startswith("}}", i):
            buffer.append("}")
            i += 2
        elif text[i] == "{":
            flush()
            node, i = _parse_slot(text, i, loop_var, allow_fragments, fragment_targets_seen)
            nodes.append(node)
        elif text[i] == "}":
            raise TemplateError(
                f"single '}}' at position {i} — write '}}}}' for a literal '}}' (near {text[max(0, i - 20) : i + 1]!r})"
            )
        else:
            buffer.append(text[i])
            i += 1

    flush()
    return tuple(nodes)


def _check_loop_attr(attr: str) -> None:
    variables = VOCABULARY["loop_variables"]
    if attr not in variables:
        raise TemplateError(f"unknown loop variable attribute {attr!r} — valid attributes: {spell_out(variables)}")


def _parse_slot(text: str, i: int, loop_var, allow_fragments: bool, fragment_targets_seen: set):
    n = len(text)
    match = _SLOT_OPEN.match(text, i)
    if not match:
        raise TemplateError(
            f"'{{' at position {i} does not open a slot — write '{{{{' for a literal brace "
            f"(near {text[i : i + 20]!r})"
        )
    name = match.group(1)
    j = match.end()

    if j < n and text[j] == ".":
        attr_match = re.compile(r"\.(\w+)\}").match(text, j)
        if not attr_match:
            raise TemplateError(f"malformed loop-variable reference near {text[i : i + 30]!r}")
        attr = attr_match.group(1)
        if loop_var is None:
            raise TemplateError(
                f"loop-variable reference {{{name}.{attr}}} outside a loop block — "
                f"only valid between {{% for ... %}} and {{% endfor %}}"
            )
        if name != loop_var:
            raise TemplateError(f"unknown loop variable {name!r} — this block iterates as {loop_var!r}")
        _check_loop_attr(attr)
        return LoopVarSlot(attr), attr_match.end()

    if j < n and text[j] == "(":
        args_raw, after = _read_call_args(text, j, name)
        if after >= n or text[after] != "}":
            raise TemplateError(f"slot call {{{name}(...)}} must close with '}}' (near {text[i : i + 40]!r})")
        node = _build_call_slot(name, args_raw, text, i, after + 1, allow_fragments, fragment_targets_seen)
        return node, after + 1

    if j < n and text[j] == "}":
        if name == "instruction":
            return InstructionSlot("raw"), j + 1
        if name == "fragments":
            targets = VOCABULARY["fragment_targets"]
            raise TemplateError(
                f"{{fragments}} requires a target: {{fragments('system')}} — valid targets: {spell_out(targets)}"
            )
        if name in VOCABULARY["aggregate_slots"]:
            styles = VOCABULARY["aggregate_styles"][name]
            raise TemplateError(
                f"{{{name}}} is an aggregate slot and is called: {{{name}(style='...')}} — "
                f"valid styles: {spell_out(styles)}"
            )
        return ValueSlot(name), j + 1

    raise TemplateError(f"malformed slot starting at {{{name} (near {text[i : i + 30]!r})")


def _read_call_args(text: str, open_paren: int, name: str) -> tuple[str, int]:
    """Read a call's raw argument text, honoring quotes; return (raw, index after ')')."""
    j = open_paren + 1
    start = j
    quote: str | None = None
    n = len(text)
    while j < n:
        ch = text[j]
        if quote is not None:
            if ch == quote:
                quote = None
        elif ch in {"'", '"'}:
            quote = ch
        elif ch == ")":
            return text[start:j], j + 1
        j += 1
    raise TemplateError(f"unclosed argument list in {{{name}(...)}}")


def _parse_call_kwargs(name: str, raw: str, valid_keys: tuple) -> dict[str, str]:
    kwargs: dict[str, str] = {}
    position = 0
    stripped = raw.strip()
    if not stripped:
        return kwargs
    while position < len(raw):
        match = _CALL_KWARG.match(raw, position)
        if not match:
            raise TemplateError(
                f"could not parse arguments {raw!r} for {{{name}(...)}} — expected quoted key='value' "
                f"pairs; valid keys: {spell_out(valid_keys)}"
            )
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        if key not in valid_keys:
            raise TemplateError(f"unknown argument {key!r} for {{{name}(...)}} — valid keys: {spell_out(valid_keys)}")
        if key in kwargs:
            raise TemplateError(f"duplicate argument {key!r} for {{{name}(...)}}")
        kwargs[key] = _decode_escapes(value, f"{{{name}(...)}} argument {key!r}")
        position = match.end()
        if position >= len(raw) or not raw[position:].strip():
            break
    return kwargs


def _build_call_slot(name, args_raw, text, start, end, allow_fragments, fragment_targets_seen):
    if name == "fragments":
        targets = VOCABULARY["fragment_targets"]
        if not allow_fragments:
            raise TemplateError(
                "{fragments(...)} is positional and renders once per message — "
                "not valid inside loop blocks or directive content"
            )
        match = _POSITIONAL.match(args_raw)
        if not match:
            raise TemplateError(
                f"{{fragments(...)}} takes one quoted target — {{fragments('system')}} — "
                f"valid targets: {spell_out(targets)}"
            )
        target = match.group(1) if match.group(1) is not None else match.group(2)
        if target not in targets:
            raise TemplateError(f"unknown fragments target {target!r} — valid targets: {spell_out(targets)}")
        if target in fragment_targets_seen:
            raise TemplateError(f"duplicate fragments target {target!r} — each target is placed once per template")
        fragment_targets_seen.add(target)
        owns_line = (start == 0 or text[start - 1] == "\n") and (end >= len(text) or text[end] == "\n")
        return FragmentsSlot(target, owns_line=owns_line)

    if name == "instruction":
        styles = VOCABULARY["instruction_styles"]
        kwargs = _parse_call_kwargs(name, args_raw, ("style",))
        style = kwargs.get("style", "raw")
        if style not in styles:
            raise TemplateError(f"unknown instruction style {style!r} — valid styles: {spell_out(styles)}")
        return InstructionSlot(style)

    if name in VOCABULARY["aggregate_slots"]:
        styles = VOCABULARY["aggregate_styles"][name]
        valid_keys = VOCABULARY["aggregate_kwargs"][name]
        kwargs = _parse_call_kwargs(name, args_raw, valid_keys)
        style = kwargs.get("style", "default")
        if style not in styles:
            raise TemplateError(f"unknown {name} style {style!r} — valid styles: {spell_out(styles)}")
        wrap = kwargs.get("wrap")
        if wrap is not None and style != "xml":
            raise TemplateError(f"wrap= is only meaningful with style='xml' (got style={style!r})")
        return AggregateSlot(kind=name, style=style, wrap=wrap)

    valid = ("instruction", *VOCABULARY["aggregate_slots"], "fragments")
    raise TemplateError(f"unknown slot function {{{name}(...)}} — valid slot functions: {spell_out(valid)}")


def _parse_loop(text: str, i: int, allow_fragments: bool):
    tag = _BLOCK_TAG.match(text, i)
    verb = tag.group(1) if tag else None
    if verb == "endfor":
        raise TemplateError("'{% endfor %}' without an open '{% for %}'")
    if verb != "for":
        raise TemplateError(f"unknown block tag {{% {verb} %}} — valid block tags: for, endfor")

    collections = VOCABULARY["loop_collections"]
    match = _FOR_TAG.match(text, i)
    if not match:
        raise TemplateError(
            "malformed '{% for %}' tag — expected "
            "{% for f in inputs|outputs [separator='...'] [strip] %} "
            f"(near {text[i : i + 60]!r})"
        )
    var, collection, options_raw = match.group(1), match.group(2), match.group(3) or ""
    if collection not in collections:
        raise TemplateError(f"unknown loop collection {collection!r} — valid collections: {spell_out(collections)}")

    separator, strip = _parse_loop_options(options_raw)

    end_match = _ENDFOR_TAG.search(text, match.end())
    if not end_match:
        raise TemplateError("unclosed '{% for %}' — every loop block ends with '{% endfor %}'")
    body_text = text[match.end() : end_match.start()]
    if _NESTED_FOR.search(body_text):
        raise TemplateError("nested loop blocks are not supported")
    # Authoring ergonomics: one newline trims off each end of the body so
    # multiline blocks read cleanly (upstreamed from the reference).
    if body_text.startswith("\n"):
        body_text = body_text[1:]
    if body_text.endswith("\n"):
        body_text = body_text[:-1]

    body = parse_content(body_text, loop_var=var, allow_fragments=False)
    return Loop(var=var, collection=collection, separator=separator, strip=strip, body=body), end_match.end()


def _parse_loop_options(options_raw: str) -> tuple[str, bool]:
    valid = VOCABULARY["loop_options"]
    separator = "\n"
    strip = False
    position = 0
    while position < len(options_raw):
        if not options_raw[position:].strip():
            break
        match = _LOOP_OPTION.match(options_raw, position)
        if not match:
            raise TemplateError(
                f"could not parse loop options {options_raw.strip()!r} — valid options: {spell_out(valid)}"
            )
        if match.group(4) is not None:  # bare flag
            if match.group(4) != "strip":
                raise TemplateError(
                    f"unknown loop option {match.group(4)!r} — valid options: {spell_out(valid)}"
                )
            strip = True
        else:
            key = match.group(1)
            value = match.group(2) if match.group(2) is not None else match.group(3)
            if key != "separator":
                raise TemplateError(f"unknown loop option {key!r} — valid options: {spell_out(valid)}")
            separator = _decode_escapes(value, "loop separator")
        position = match.end()
    return separator, strip


# ---------------------------------------------------------------------------
# Message-template parsing
# ---------------------------------------------------------------------------


def parse_message_template(messages) -> ParsedTemplate:
    """Parse a message-list template eagerly, refusing unknown shapes."""
    message_roles = VOCABULARY["message_roles"]
    directive_roles = VOCABULARY["directive_roles"]

    if not isinstance(messages, (list, tuple)) or not messages:
        raise TemplateError("a template is a non-empty list of message dicts")

    parsed: list = []
    fragment_targets_seen: set[str] = set()
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or "role" not in message:
            raise TemplateError(f"template message {index} must be a dict with a 'role' key (got {message!r})")
        role = message["role"]

        if role in message_roles:
            extra = set(message) - {"role", "content"}
            if extra:
                raise TemplateError(
                    f"unknown keys {sorted(extra)} on {role!r} template message — valid keys: role, content"
                )
            content = message.get("content")
            if not isinstance(content, str):
                raise TemplateError(f"{role!r} template message {index} needs a string 'content'")
            nodes = _parse_message_content(content, fragment_targets_seen)
            parsed.append(ContentMessage(role=role, nodes=nodes, raw=content))
        elif role in directive_roles:
            extra = set(message) - {"role", "user", "assistant"}
            if extra:
                raise TemplateError(
                    f"unknown keys {sorted(extra)} on {role!r} directive — valid keys: role, user, assistant"
                )
            user_raw = message.get("user")
            assistant_raw = message.get("assistant")
            if (user_raw is None) != (assistant_raw is None):
                raise TemplateError(f"{role!r} directive takes 'user' and 'assistant' together, or neither")
            user_nodes = parse_content(user_raw, allow_fragments=False) if user_raw is not None else None
            assistant_nodes = parse_content(assistant_raw, allow_fragments=False) if assistant_raw is not None else None
            cls = DemosDirective if role == "demos" else HistoryDirective
            if any(isinstance(m, cls) for m in parsed):
                raise TemplateError(f"at most one {role!r} directive per template")
            parsed.append(cls(user_nodes, assistant_nodes, user_raw, assistant_raw))
        else:
            raise TemplateError(
                f"unknown message role {role!r} — valid message roles: {spell_out(message_roles)}; "
                f"valid directives: {spell_out(directive_roles)}"
            )

    return ParsedTemplate(messages=tuple(parsed), raw=tuple(messages))


def _parse_message_content(content: str, fragment_targets_seen: set) -> tuple:
    """Parse one message's content, sharing fragment-target uniqueness across the template."""
    nodes = parse_content(content)
    for node in nodes:
        if isinstance(node, FragmentsSlot):
            if node.target in fragment_targets_seen:
                raise TemplateError(
                    f"duplicate fragments target {node.target!r} — each target is placed once per template"
                )
            fragment_targets_seen.add(node.target)
    return nodes
