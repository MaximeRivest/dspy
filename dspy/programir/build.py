"""Typed node builders for the ProgramIR node set v0.3 — quasi-quotation.

One constructor per node kind. Each constructor emits EXACTLY the JSON
record the frontend compiler (`forward.py`) emits — field names, field
order, shapes — so a built tree and its parsed twin are byte-compatible.

The hygiene win over string templates: names are validated at
CONSTRUCTION time. A tool named ``"foo (x)"`` or a field named
``"class"`` refuses HERE, with a teaching error naming the problem —
never later as a SyntaxError inside generated source.

Ergonomics: every expression position accepts plain Python literals
where a node is obvious — scalars become `Const`, lists become `List`,
plain dicts become `Dict` with string keys. `Forward(args, body)` builds
the envelope and runs the SAME structural admission the compiler runs
(`forward.admit_forward`), so built trees and parsed trees cannot drift.
"""

from __future__ import annotations

import keyword
import math
from typing import Any, Iterable, Mapping

from dspy.programir.errors import ProgramIRRefusal

__all__ = [
    "Assign",
    "AssignTuple",
    "Attr",
    "BinOp",
    "BoolOp",
    "Break",
    "Builtin",
    "CallInterpreter",
    "CallModule",
    "CallPredict",
    "CallTool",
    "CallToolDynamic",
    "Compare",
    "Const",
    "Continue",
    "Dict",
    "Do",
    "Except",
    "For",
    "Format",
    "Forward",
    "If",
    "IfExp",
    "Index",
    "List",
    "Log",
    "Method",
    "Pass",
    "Raise",
    "Return",
    "SetIndex",
    "Slice",
    "Try",
    "UnaryOp",
    "Var",
    "While",
]

#: The closed node vocabulary (node-set v0.3, D-037).
STATEMENT_KINDS = frozenset(
    {
        "Assign",
        "AssignTuple",
        "SetIndex",
        "Return",
        "If",
        "For",
        "While",
        "Break",
        "Continue",
        "Try",
        "Raise",
        "Pass",
        "Log",
    }
)
EXPRESSION_KINDS = frozenset(
    {
        "Call",
        "Attr",
        "Var",
        "Const",
        "Compare",
        "BinOp",
        "IfExp",
        "BoolOp",
        "UnaryOp",
        "Format",
        "List",
        "Dict",
        "Index",
        "Slice",
    }
)

#: The closed v0.3 call tables (mirrors of the schema enums; the outer
#: schema check in `Forward` keeps them honest).
BUILTINS = frozenset(
    {
        "len",
        "str",
        "int",
        "float",
        "max",
        "min",
        "sum",
        "sorted",
        "any",
        "all",
        "json_dumps",
        "json_parse",
    }
)
METHODS = frozenset(
    {
        "strip",
        "lstrip",
        "rstrip",
        "lower",
        "upper",
        "split",
        "join",
        "replace",
        "startswith",
        "endswith",
        "append",
        "extend",
        "pop",
        "get",
        "keys",
        "values",
        "items",
        "update",
    }
)

_CMP_OPS = frozenset({"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in"})
_BIN_OPS = frozenset({"add", "sub", "mult", "div"})
_BOOL_OPS = frozenset({"and", "or"})
_UNARY_OPS = frozenset({"not", "neg"})
_RAISE_TYPES = frozenset({"ToolError", "InterpreterError", "AdapterParseError", "LMError"})
_HANDLER_TYPES = _RAISE_TYPES | frozenset(
    {
        "Exception",
        "InterpreterTypeError",
        "InterpreterKeyError",
        "InterpreterArithmeticError",
    }
)

Node = dict


def _refuse(code: str, detail: dict[str, Any], message: str) -> None:
    raise ProgramIRRefusal(code, "5_forward", detail, message)


def _name(value: Any, role: str) -> str:
    """Admit one identifier, or refuse with a teaching error naming it."""
    if not isinstance(value, str):
        _refuse(
            "PIR-E-NODE-001",
            {"role": role, "got": type(value).__name__},
            f"{role} must be a name string, got {type(value).__name__}: {value!r}",
        )
    if not (value.isidentifier() and value.isascii()):
        _refuse(
            "PIR-E-NODE-001",
            {"role": role, "name": value},
            f"{role} {value!r} is not a Python identifier; IR names must print back "
            "as source, so a name the parser would refuse refuses here, at build time",
        )
    if keyword.iskeyword(value):
        _refuse(
            "PIR-E-NODE-001",
            {"role": role, "name": value},
            f"{role} {value!r} is a Python keyword; printed source could never bind it — pick another name",
        )
    if value == "self":
        _refuse(
            "PIR-E-NODE-001",
            {"role": role, "name": value},
            f"{role} 'self' is the module receiver, never a value name",
        )
    if len(value) >= 4 and value.startswith("__") and value.endswith("__"):
        _refuse(
            "PIR-E-NODE-001",
            {"role": role, "name": value},
            f"{role} {value!r} is a dunder — Python's protocol namespace, not a program name",
        )
    return value


def _scalar(value: Any, role: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _refuse(
                "PIR-E-NODE-001",
                {"role": role, "value": repr(value)},
                f"{role} {value!r} is non-finite; the JSON value model carries finite numbers only",
            )
        return value
    _refuse(
        "PIR-E-NODE-001",
        {"role": role, "got": type(value).__name__},
        f"{role} must be a JSON scalar (str/int/float/bool/None), got {type(value).__name__}; "
        "container literals are the List(...)/Dict(...) nodes",
    )


def _expr(value: Any, role: str = "expression") -> Node:
    """Coerce one expression position: node dicts pass, literals lower."""
    if isinstance(value, dict) and isinstance(value.get("node"), str):
        kind = value["node"]
        if kind in STATEMENT_KINDS:
            _refuse(
                "PIR-E-NODE-001",
                {"role": role, "node": kind},
                f"{role} got the statement node {kind!r}; statements live in bodies, not expression slots",
            )
        if kind not in EXPRESSION_KINDS:
            _refuse(
                "PIR-E-NODE-001",
                {"role": role, "node": kind},
                f"{role} got unknown node kind {kind!r}; the v0.3 vocabulary is closed",
            )
        return value
    if value is None or isinstance(value, (str, bool, int, float)):
        return Const(value)
    if isinstance(value, (list, tuple)):
        return List(*value)
    if isinstance(value, dict):
        return Dict(value)
    _refuse(
        "PIR-E-NODE-001",
        {"role": role, "got": type(value).__name__},
        f"{role} must be an expression node or a plain JSON literal, got {type(value).__name__}",
    )


def _stmt(value: Any, role: str) -> Node:
    if isinstance(value, dict) and value.get("node") in STATEMENT_KINDS:
        return value
    kind = value.get("node") if isinstance(value, dict) else type(value).__name__
    _refuse(
        "PIR-E-NODE-001",
        {"role": role, "got": kind},
        f"{role} must be a statement node, got {kind!r}; bind an expression with "
        "Assign(name, ...) or discard its value with Do(...)",
    )


def _body(statements: Any, role: str, *, allow_empty: bool = False) -> list[Node]:
    if isinstance(statements, dict):
        statements = [statements]
    items = [_stmt(statement, role) for statement in statements]
    if not items and not allow_empty:
        _refuse("PIR-E-NODE-001", {"role": role}, f"{role} cannot be empty (v0.3 minItems 1); use Pass()")
    return items


# ─── statements ──────────────────────────────────────────────────────


def Assign(target: str, value: Any) -> Node:
    """``target = value``."""
    return {"node": "Assign", "target": _name(target, "Assign target"), "value": _expr(value, "Assign value")}


def Do(value: Any) -> Node:
    """A statement-position call; the discarded result binds ``_``."""
    return {"node": "Assign", "target": "_", "value": _expr(value, "Do value")}


def AssignTuple(targets: Iterable[str], value: Any) -> Node:
    """``a, b = value`` — a fixed-shape destructuring bind."""
    names = [_name(target, "AssignTuple target") for target in targets]
    if len(names) < 2:
        _refuse(
            "PIR-E-NODE-001",
            {"targets": names},
            "AssignTuple binds two or more names; a one-target destructuring is a plain Assign",
        )
    if len(set(names)) != len(names):
        _refuse("PIR-E-NODE-001", {"targets": names}, "AssignTuple targets must be distinct names")
    return {"node": "AssignTuple", "targets": names, "value": _expr(value, "AssignTuple value")}


def SetIndex(target: str, key: Any, value: Any) -> Node:
    """``target[key] = value`` — the subscript write statement."""
    return {
        "node": "SetIndex",
        "target": _name(target, "SetIndex target"),
        "key": _expr(key, "SetIndex key"),
        "value": _expr(value, "SetIndex value"),
    }


def Return(value: Any) -> Node:
    """``return value``."""
    return {"node": "Return", "value": _expr(value, "Return value")}


def If(test: Any, body: Any, orelse: Any = ()) -> Node:
    """``if test: body [else: orelse]``."""
    return {
        "node": "If",
        "test": _expr(test, "If test"),
        "body": _body(body, "If body"),
        "orelse": _body(orelse, "If orelse", allow_empty=True),
    }


def For(target: str, body: Any, *, range: Any = None, iter: Any = None) -> Node:
    """A bounded loop: exactly one of ``range=`` (literal) or ``iter=`` (list)."""
    record: Node = {"node": "For", "target": _name(target, "For target")}
    if (range is None) == (iter is None):
        _refuse(
            "PIR-E-NODE-001",
            {"node": "For"},
            "For takes exactly one of range= (a non-negative int literal) or iter= (a list expression)",
        )
    if range is not None:
        if isinstance(range, bool) or not isinstance(range, int) or range < 0:
            _refuse(
                "PIR-E-NODE-001",
                {"node": "For", "range": repr(range)},
                f"For range must be a non-negative int literal, got {range!r}; a computed bound "
                "needs the iter= list form or a While",
            )
        record["range"] = range
    else:
        record["iter"] = _expr(iter, "For iter")
    record["body"] = _body(body, "For body")
    return record


def While(test: Any, body: Any) -> Node:
    """``while test: body`` (engine-capped at 1000 iterations, SEM-6)."""
    return {"node": "While", "test": _expr(test, "While test"), "body": _body(body, "While body")}


def Break() -> Node:
    """``break``."""
    return {"node": "Break"}


def Continue() -> Node:
    """``continue``."""
    return {"node": "Continue"}


def Pass() -> Node:
    """``pass``."""
    return {"node": "Pass"}


def Try(body: Any, handlers: Any) -> Node:
    """``try: body`` with one or more `Except` handlers."""
    if isinstance(handlers, dict):
        handlers = [handlers]
    checked = []
    for handler in handlers:
        if not (isinstance(handler, dict) and set(handler) == {"type", "body"}):
            _refuse(
                "PIR-E-NODE-001",
                {"node": "Try"},
                "Try handlers must be Except(type, body) records",
            )
        checked.append(handler)
    if not checked:
        _refuse("PIR-E-NODE-001", {"node": "Try"}, "Try requires at least one Except handler")
    return {"node": "Try", "body": _body(body, "Try body"), "handlers": checked}


def Except(type: str, body: Any) -> Node:
    """One ``except <TypedError>:`` handler."""
    if type not in _HANDLER_TYPES:
        _refuse(
            "PIR-E-NODE-001",
            {"type": type},
            f"except type {type!r} is outside the typed error table ({', '.join(sorted(_HANDLER_TYPES))})",
        )
    return {"type": type, "body": _body(body, "except body")}


def Raise(exc: str, message: Any) -> Node:
    """``raise Exc(message)`` — a typed-table error with a scalar payload."""
    if exc not in _RAISE_TYPES:
        _refuse(
            "PIR-E-NODE-001",
            {"exc": exc},
            f"Raise type {exc!r} is outside the typed error table ({', '.join(sorted(_RAISE_TYPES))})",
        )
    return {"node": "Raise", "exc": exc, "message": _scalar(message, "Raise message")}


def Log(*values: Any) -> Node:
    """``print(...)``'s lawful home — parts follow the print desugar."""
    parts: list[Node] = []
    for index, value in enumerate(values):
        if index:
            parts.append({"node": "Const", "value": " "})
        parts.append(_expr(value, "Log part"))
    if not parts:
        parts = [{"node": "Const", "value": ""}]
    return {"node": "Log", "parts": parts}


# ─── expressions ─────────────────────────────────────────────────────


def Var(name: str) -> Node:
    """Read a bound name."""
    return {"node": "Var", "name": _name(name, "Var name")}


def Const(value: Any) -> Node:
    """A JSON-scalar literal."""
    return {"node": "Const", "value": _scalar(value, "Const value")}


def Attr(obj: str, attr: str) -> Node:
    """``obj.attr`` — field access on a record held in a VARIABLE.

    `obj` is the variable's name (a bare string, the v0 encoding), not an
    expression; bind intermediate values to a name first.
    """
    if isinstance(obj, dict):
        _refuse(
            "PIR-E-NODE-001",
            {"node": "Attr"},
            "Attr reads one variable.field: obj is the variable NAME (a string), not an "
            "expression — bind the intermediate value to a name first",
        )
    return {"node": "Attr", "obj": _name(obj, "Attr obj"), "attr": _name(attr, "Attr field")}


def Index(obj: Any, key: Any) -> Node:
    """``obj[key]`` — subscript read."""
    return {"node": "Index", "obj": _expr(obj, "Index obj"), "key": _expr(key, "Index key")}


def Slice(obj: Any, start: Any = None, stop: Any = None, step: int | None = None) -> Node:
    """``obj[start:stop:step]`` — read-only; step is literal 1 or -1 only."""
    record: Node = {"node": "Slice", "obj": _expr(obj, "Slice obj")}
    if start is not None:
        record["start"] = _expr(start, "Slice start")
    if stop is not None:
        record["stop"] = _expr(stop, "Slice stop")
    if step not in (None, 1, -1):
        _refuse(
            "PIR-E-NODE-001",
            {"node": "Slice", "step": repr(step)},
            f"Slice step is admitted as literal 1 or -1 only, got {step!r}",
        )
    if step == -1:
        record["step"] = -1
    return record


def Format(*parts: Any) -> Node:
    """String building — the f-string / .format compile target."""
    if not parts:
        _refuse("PIR-E-NODE-001", {"node": "Format"}, "Format needs at least one part")
    return {"node": "Format", "parts": [_expr(part, "Format part") for part in parts]}


def List(*items: Any) -> Node:
    """A list literal; may be empty."""
    return {"node": "List", "items": [_expr(item, "List item") for item in items]}


def Dict(entries: Mapping[str, Any] | Iterable[tuple[Any, Any]] = ()) -> Node:
    """An object literal: a plain mapping (string keys) or (key, value) pairs."""
    if isinstance(entries, Mapping):
        pairs = [(Const(_string_key(key)), value) for key, value in entries.items()]
    else:
        pairs = [(_expr(key, "Dict key"), value) for key, value in entries]
    return {
        "node": "Dict",
        "entries": [{"key": key, "value": _expr(value, "Dict value")} for key, value in pairs],
    }


def _string_key(key: Any) -> str:
    if not isinstance(key, str):
        _refuse(
            "PIR-E-NODE-001",
            {"node": "Dict", "got": type(key).__name__},
            f"Dict mapping keys must be strings (runtime dict keys are strings), got {type(key).__name__}",
        )
    return key


def Compare(op: str, left: Any, right: Any) -> Node:
    """One comparison from the closed operator set."""
    if op not in _CMP_OPS:
        _refuse(
            "PIR-E-NODE-001",
            {"op": op},
            f"Compare operator {op!r} is outside {sorted(_CMP_OPS)} (SEM-4)",
        )
    return {
        "node": "Compare",
        "op": op,
        "left": _expr(left, "Compare left"),
        "right": _expr(right, "Compare right"),
    }


def BinOp(op: str, left: Any, right: Any) -> Node:
    """One binary operation: add/sub/mult/div."""
    if op not in _BIN_OPS:
        _refuse(
            "PIR-E-NODE-001",
            {"op": op},
            f"BinOp operator {op!r} is outside {sorted(_BIN_OPS)} (SEM-4)",
        )
    return {
        "node": "BinOp",
        "op": op,
        "left": _expr(left, "BinOp left"),
        "right": _expr(right, "BinOp right"),
    }


def BoolOp(op: str, *values: Any) -> Node:
    """``and``/``or`` over two or more operands, short-circuit."""
    if op not in _BOOL_OPS:
        _refuse("PIR-E-NODE-001", {"op": op}, f"BoolOp operator {op!r} is outside and/or")
    if len(values) < 2:
        _refuse("PIR-E-NODE-001", {"op": op}, "BoolOp needs two or more operands")
    return {"node": "BoolOp", "op": op, "values": [_expr(value, "BoolOp operand") for value in values]}


def UnaryOp(op: str, value: Any) -> Node:
    """``not``/``neg``. Neg of a literal number folds to Const (the parser's rule)."""
    if op not in _UNARY_OPS:
        _refuse("PIR-E-NODE-001", {"op": op}, f"unary operator {op!r} is outside not/neg")
    operand = _expr(value, "UnaryOp operand")
    if op == "neg" and operand["node"] == "Const" and _is_number(operand["value"]):
        folded = -operand["value"]
        if isinstance(folded, float) and folded == 0.0:
            folded = 0.0  # negative zero is kept out of the value model
        return {"node": "Const", "value": folded}
    return {"node": "UnaryOp", "op": op, "value": operand}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def IfExp(test: Any, body: Any, orelse: Any) -> Node:
    """``body if test else orelse``."""
    return {
        "node": "IfExp",
        "test": _expr(test, "IfExp test"),
        "body": _expr(body, "IfExp body"),
        "orelse": _expr(orelse, "IfExp orelse"),
    }


# ─── calls ───────────────────────────────────────────────────────────


def Builtin(name: str, *args: Any) -> Node:
    """A closed-table builtin call (json.dumps/json.loads spell json_dumps/json_parse)."""
    if name not in BUILTINS:
        _refuse(
            "PIR-E-NODE-002",
            {"target": name},
            f"unknown builtin {name!r} — the table is closed (v0.3, D-037); "
            "an unlisted name is an unresolved call — use a tool leaf for anything impure",
        )
    if not args:
        _refuse("PIR-E-NODE-001", {"builtin": name}, f"builtin {name} needs at least one argument")
    if name in ("json_dumps", "json_parse") and len(args) != 1:
        _refuse(
            "PIR-E-NODE-001",
            {"builtin": name},
            f"{name} takes exactly one positional argument (the canonical serde profile, D-037)",
        )
    return {"node": "Call", "builtin": name, "args": [_expr(arg, f"{name} argument") for arg in args]}


def Method(obj: Any, name: str, *args: Any) -> Node:
    """A closed-table value-method call: ``obj.name(args)``."""
    if name not in METHODS:
        _refuse(
            "PIR-E-NODE-002",
            {"target": name},
            f"method .{name}() is outside the closed 18-name str/list/dict table (v0.3, D-037); "
            "unknown names refuse at build — use a tool leaf for anything impure",
        )
    return {
        "node": "Call",
        "method": name,
        "obj": _expr(obj, "method receiver"),
        "args": [_expr(arg, f".{name}() argument") for arg in args],
    }


def _leaf_call(kind: str, ref: str | None, kwargs: dict[str, Any], name: Any = None) -> Node:
    checked_kwargs = {}
    for key, value in kwargs.items():
        _name(key, f"{kind} call keyword")
        checked_kwargs[key] = _expr(value, f"{kind} call argument {key!r}")
    leaf: Node = {"kind": kind}
    if ref is not None:
        leaf["ref"] = _name(ref, f"{kind} leaf name")
    call: Node = {"node": "Call", "leaf": leaf, "kwargs": checked_kwargs}
    if name is not None:
        call["name"] = _expr(name, "dynamic tool name")
    return call


def CallPredict(ref: str, **kwargs: Any) -> Node:
    """Invoke a declared Predict leaf: ``self.<ref>(**kwargs)``."""
    return _leaf_call("predict", ref, kwargs)


def CallModule(ref: str, **kwargs: Any) -> Node:
    """Invoke a declared sub-module leaf: ``self.<ref>(**kwargs)``."""
    return _leaf_call("module", ref, kwargs)


def CallTool(ref: str, **kwargs: Any) -> Node:
    """Invoke a declared tool leaf statically: ``self.<ref>(**kwargs)``."""
    return _leaf_call("tool", ref, kwargs)


def CallToolDynamic(name: Any, **kwargs: Any) -> Node:
    """Model-dispatched tool call: ``self.tools[<name expr>](**kwargs)``."""
    return _leaf_call("tool", None, kwargs, name=_expr(name, "dynamic tool name"))


def CallInterpreter(ref: str, **kwargs: Any) -> Node:
    """Invoke a declared interpreter leaf: ``self.<ref>(**kwargs)``."""
    return _leaf_call("interpreter", ref, kwargs)


# ─── the envelope ────────────────────────────────────────────────────


def Forward(args: Iterable[str], body: Any, *, leaves: Mapping[str, Any] | None = None) -> Node:
    """The forward root: argument names + statement body, admitted.

    Finalize runs the SAME structural admission the parse compiler runs
    (`admit_forward`: schema, closed tables, arity, leaf resolution when
    `leaves` is given) — built and parsed trees cannot drift.
    """
    from dspy.programir.forward import admit_forward

    names = [_name(argument, "forward argument") for argument in args]
    if len(set(names)) != len(names):
        _refuse("PIR-E-NODE-001", {"args": names}, "forward argument names must be distinct")
    tree = {
        "language": "restricted-python-ast",
        "args": names,
        "body": _body(body, "forward body"),
    }
    return admit_forward(tree, leaves)
