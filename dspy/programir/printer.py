"""ProgramIR node JSON -> readable Python source: the compiler's inverse.

THE ROUND-TRIP LAW: for every tree this printer accepts,

    compile_forward(to_function(tree), leaves) == tree

— exact JSON equality. The printer is deterministic (same tree, same
bytes) and minimal: it prints the plain spelling of each node, in the
style of the module library's forwards, and it never prints a construct
whose reparse would desugar into a DIFFERENT tree. The few node shapes
no Python source can reproduce (a Log outside print's separator
convention, a neg on a literal number the parser would fold, a Slice
carrying an explicit step of 1 the parser would normalize away) refuse
with a teaching error instead of printing a lie.

`to_function` registers the printed source in `linecache` and execs it,
so the result is a genuine Python function: `inspect.getsource` returns
exactly the printed bytes, and the typed error names a forward may catch
resolve to the canonical classes.
"""

from __future__ import annotations

import math
import types
from typing import Any

from dspy.programir.errors import ProgramIRRefusal

__all__ = ["bind_forward", "render_forward", "to_function"]

#: Expression precedence levels (Python's table, the slice we print).
_IFEXP, _OR, _AND, _NOT, _CMP, _ADD, _MUL, _UNARY, _ATOM = 1, 2, 3, 4, 5, 6, 7, 8, 10

_CMP_OPS = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=", "in": "in", "not_in": "not in"}
_BIN_OPS = {"add": ("+", _ADD), "sub": ("-", _ADD), "mult": ("*", _MUL), "div": ("/", _MUL)}
_BUILTIN_SPELLING = {"json_dumps": "json.dumps", "json_parse": "json.loads"}

_SPACE = {"node": "Const", "value": " "}


def render_forward(tree: dict[str, Any], *, name: str = "forward") -> str:
    """Print one forward tree as Python source (a def taking self + args).

    Args:
        tree: The forward record (language/args/body envelope).
        name: The printed function's name.

    Returns:
        The source text; `compile_forward` on it reproduces `tree`.
    """
    args = "".join(f", {_argument(argument)}" for argument in tree["args"])
    lines = [f"def {name}(self{args}):"]
    lines.extend(_block(tree["body"], 1))
    return "\n".join(lines) + "\n"


def _argument(argument: Any) -> str:
    """Print one forward parameter: a plain name, or the v0.4 record bag.

    The record envelope (D-041) is `args: [{"name": <bag>, "record":
    "self"}]`; it prints as the `**<bag>` desugar door (door b), which
    `compile_forward` recognizes and lowers back to the SAME record
    binding. The bag name reparses under the `**kwargs` spelling, so the
    round trip is exact.
    """
    if isinstance(argument, dict):
        return f"**{argument['name']}"
    return argument


def to_function(tree: dict[str, Any], *, name: str = "forward", tag: str = "printed"):
    """Render + linecache-register + exec: the printed tree as a real function."""
    from dspy.modules._generate import load_generated

    return load_generated(render_forward(tree, name=name), tag=tag, name=name)


def bind_forward(module: Any, tree: dict[str, Any], *, tag: str) -> None:
    """Bind the printed tree on a module instance as its native `forward`."""
    module.forward = types.MethodType(to_function(tree, tag=tag), module)


def _unprintable(node: str, reason: str) -> None:
    raise ProgramIRRefusal(
        "PIR-E-NODE-001",
        "5_forward",
        {"node": node},
        f"printer: {node} {reason}",
    )


# ─── statements ──────────────────────────────────────────────────────


def _block(statements: list[dict[str, Any]], depth: int) -> list[str]:
    lines: list[str] = []
    for statement in statements:
        lines.extend(_stmt(statement, depth))
    if not lines:
        _unprintable("body", "blocks cannot be empty (v0.3 minItems 1); use a Pass node")
    return lines


def _stmt(node: dict[str, Any], depth: int) -> list[str]:
    pad = "    " * depth
    kind = node["node"]
    if kind == "Assign":
        value = _at(node["value"], _IFEXP)
        if node["target"] == "_" and node["value"].get("node") == "Call":
            return [pad + value]  # a statement-position call, printed bare
        return [f"{pad}{node['target']} = {value}"]
    if kind == "AssignTuple":
        return [f"{pad}{', '.join(node['targets'])} = {_at(node['value'], _IFEXP)}"]
    if kind == "SetIndex":
        return [f"{pad}{node['target']}[{_at(node['key'], _IFEXP)}] = {_at(node['value'], _IFEXP)}"]
    if kind == "Return":
        return [f"{pad}return {_at(node['value'], _IFEXP)}"]
    if kind == "If":
        lines = [f"{pad}if {_at(node['test'], _IFEXP)}:"]
        lines.extend(_block(node["body"], depth + 1))
        if node.get("orelse"):
            lines.append(f"{pad}else:")
            lines.extend(_block(node["orelse"], depth + 1))
        return lines
    if kind == "For":
        if "range" in node:
            head = f"{pad}for {node['target']} in range({node['range']}):"
        else:
            head = f"{pad}for {node['target']} in {_at(node['iter'], _IFEXP)}:"
        return [head, *_block(node["body"], depth + 1)]
    if kind == "While":
        return [f"{pad}while {_at(node['test'], _IFEXP)}:", *_block(node["body"], depth + 1)]
    if kind == "Break":
        return [pad + "break"]
    if kind == "Continue":
        return [pad + "continue"]
    if kind == "Pass":
        return [pad + "pass"]
    if kind == "Try":
        lines = [pad + "try:"]
        lines.extend(_block(node["body"], depth + 1))
        for handler in node["handlers"]:
            lines.append(f"{pad}except {handler['type']}:")
            lines.extend(_block(handler["body"], depth + 1))
        return lines
    if kind == "Raise":
        return [f"{pad}raise {node['exc']}({_scalar(node['message'])})"]
    if kind == "Log":
        return [pad + _print_call(node["parts"])]
    _unprintable(str(kind), "is not a statement the printer knows")


def _print_call(parts: list[dict[str, Any]]) -> str:
    """Invert the print -> Log desugar (arguments joined by ' ' Consts)."""
    if parts == [{"node": "Const", "value": ""}]:
        return "print()"
    arguments = []
    printable = len(parts) % 2 == 1
    for index, part in enumerate(parts):
        if index % 2:
            printable = printable and part == _SPACE
        else:
            arguments.append(part)
    if not printable:
        _unprintable(
            "Log",
            "parts do not follow print's separator convention (argument, ' ', argument, ...); "
            "no Python source lowers to this shape",
        )
    return f"print({', '.join(_at(argument, _IFEXP) for argument in arguments)})"


# ─── expressions ─────────────────────────────────────────────────────


def _at(node: dict[str, Any], minimum: int) -> str:
    """Render an expression, parenthesized when its level binds looser."""
    text, level = _expr(node)
    if level < minimum:
        return f"({text})"
    return text


def _receiver(node: dict[str, Any], *, dot: bool = False) -> str:
    """Render a subscript/method receiver (needs atom-level binding).

    `dot=True` also parenthesizes numeric literals: `5.strip()` is a
    tokenizer error, `(5).strip()` is the printable spelling.
    """
    text, level = _expr(node)
    needs_parens = level < _ATOM
    if dot and node.get("node") == "Const" and _is_number(node.get("value")):
        needs_parens = True
    return f"({text})" if needs_parens else text


def _expr(node: dict[str, Any]) -> tuple[str, int]:
    kind = node.get("node")
    if kind == "Var":
        return node["name"], _ATOM
    if kind == "Const":
        value = node["value"]
        level = _UNARY if _is_number(value) and value < 0 else _ATOM
        return _scalar(value), level
    if kind == "Attr":
        return f"{node['obj']}.{node['attr']}", _ATOM
    if kind == "Index":
        return f"{_receiver(node['obj'])}[{_at(node['key'], _IFEXP)}]", _ATOM
    if kind == "Slice":
        if node.get("step") == 1:
            _unprintable(
                "Slice",
                "with an explicit step of 1 has no printed spelling — the parser normalizes "
                "`[start:stop:1]` to a step-less Slice, so printing would silently drop the key; "
                "omit step (absent means 1)",
            )
        start = _at(node["start"], _IFEXP) if "start" in node else ""
        stop = _at(node["stop"], _IFEXP) if "stop" in node else ""
        step = ":-1" if node.get("step") == -1 else ""
        return f"{_receiver(node['obj'])}[{start}:{stop}{step}]", _ATOM
    if kind == "Format":
        return _format(node["parts"]), _ATOM
    if kind == "List":
        return f"[{', '.join(_at(item, _IFEXP) for item in node['items'])}]", _ATOM
    if kind == "Dict":
        entries = ", ".join(f"{_at(entry['key'], _IFEXP)}: {_at(entry['value'], _IFEXP)}" for entry in node["entries"])
        return f"{{{entries}}}", _ATOM
    if kind == "Compare":
        op = _CMP_OPS[node["op"]]
        # Comparison is non-associative here: a Compare operand inside a
        # Compare must parenthesize or the reparse would CHAIN it.
        return f"{_at(node['left'], _CMP + 1)} {op} {_at(node['right'], _CMP + 1)}", _CMP
    if kind == "BinOp":
        op, level = _BIN_OPS[node["op"]]
        return f"{_at(node['left'], level)} {op} {_at(node['right'], level + 1)}", level
    if kind == "BoolOp":
        op = node["op"]
        level = _AND if op == "and" else _OR
        # Nested BoolOps always parenthesize: unparenthesized same-op
        # nesting would FLATTEN on reparse.
        rendered = []
        for value in node["values"]:
            text, sub_level = _expr(value)
            rendered.append(f"({text})" if value.get("node") == "BoolOp" or sub_level < _NOT else text)
        return f" {op} ".join(rendered), level
    if kind == "UnaryOp":
        if node["op"] == "not":
            return f"not {_at(node['value'], _NOT)}", _NOT
        operand = node["value"]
        if operand.get("node") == "Const" and _is_number(operand.get("value")):
            _unprintable("UnaryOp", "neg on a literal number reparses as a folded Const; build the folded Const")
        return f"-{_at(operand, _UNARY)}", _UNARY
    if kind == "IfExp":
        body = _at(node["body"], _OR)
        test = _at(node["test"], _OR)
        orelse = _at(node["orelse"], _IFEXP)
        return f"{body} if {test} else {orelse}", _IFEXP
    if kind == "Call":
        return _call(node)
    _unprintable(str(kind), "is not an expression the printer knows")


def _call(node: dict[str, Any]) -> tuple[str, int]:
    if "builtin" in node:
        spelled = _BUILTIN_SPELLING.get(node["builtin"], node["builtin"])
        return f"{spelled}({', '.join(_at(argument, _IFEXP) for argument in node['args'])})", _ATOM
    if "method" in node:
        receiver = _receiver(node["obj"], dot=True)
        arguments = ", ".join(_at(argument, _IFEXP) for argument in node["args"])
        return f"{receiver}.{node['method']}({arguments})", _ATOM
    leaf = node["leaf"]
    # The v0.4 record-splat (D-041): `**<record>` renders FIRST, before the
    # explicit kwargs — the merge order the contract pins (expanded keys in
    # declared order, then authored kwargs). Reparse reproduces the splat.
    pieces = []
    if "splat" in node:
        pieces.append(f"**{node['splat']}")
    pieces.extend(f"{key}={_at(value, _IFEXP)}" for key, value in node["kwargs"].items())
    kwargs = ", ".join(pieces)
    if "name" in node:
        # Dynamic tool dispatch prints against the conventional table
        # attribute: `self.tools[<name expr>](...)`.
        return f"self.tools[{_at(node['name'], _IFEXP)}]({kwargs})", _ATOM
    ref = leaf.get("ref")
    if ref is None:
        _unprintable("Call", "a static leaf call without a ref names no attribute to print")
    return f"self.{ref}({kwargs})", _ATOM


def _format(parts: list[dict[str, Any]]) -> str:
    """Print Format as ``"template with {} slots".format(args...)``.

    Const string parts become template literals; everything else becomes
    a ``{}`` slot. A Const string that is empty, or that directly follows
    another literal (adjacent literals would MERGE on reparse), rides as
    an argument instead — the reparse then reproduces the exact part
    list.
    """
    template = ""
    arguments: list[str] = []
    previous_was_literal = False
    for part in parts:
        value = part.get("value")
        if part.get("node") == "Const" and isinstance(value, str) and value and not previous_was_literal:
            template += value.replace("{", "{{").replace("}", "}}")
            previous_was_literal = True
        else:
            template += "{}"
            arguments.append(_at(part, _IFEXP))
            previous_was_literal = False
    return f"{_scalar(template)}.format({', '.join(arguments)})"


def _scalar(value: Any) -> str:
    if isinstance(value, float) and not math.isfinite(value):
        _unprintable("Const", f"non-finite float {value!r} is outside the JSON value model")
    return repr(value)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
