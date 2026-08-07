"""Compile Python forward methods into the ProgramIR node set v0.1."""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dspy.programir.errors import ProgramIRRefusal

_ERROR_TYPES = {"ToolError", "InterpreterError", "AdapterParseError", "LMError"}
_PROPOSED_V02 = {
    "JoinedStr",
    "FormattedValue",
    "Dict",
    "List",
    "Tuple",
    "Subscript",
    "BoolOp",
    "IfExp",
    "UnaryOp",
    "AugAssign",
    "AnnAssign",
    "SetComp",
    "ListComp",
    "DictComp",
    "GeneratorExp",
}


@dataclass(frozen=True)
class LeafRef:
    """Name one child callable admitted by a module forward."""

    kind: str
    ref: str | None = None


def compile_forward(function: Any, leaves: Mapping[str, LeafRef]) -> dict[str, Any]:
    """Compile one Python method into a node-set v0.1 forward record.

    Args:
        function: The authored `forward` function.
        leaves: Immediate child names mapped to their typed leaf references.

    Returns:
        The canonical node-set JSON record.
    """
    try:
        lines, start_line = inspect.getsourcelines(function)
        filename = inspect.getsourcefile(function) or "<unknown>"
    except (OSError, TypeError) as error:
        raise ProgramIRRefusal(
            "PIR-E-NODE-001",
            "5_forward",
            {"node": "Source", "line": None},
            f"forward source is not introspectable for {function!r}",
        ) from error

    tree = ast.parse(textwrap.dedent("".join(lines)))
    definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(definitions) != 1 or not isinstance(definitions[0], ast.FunctionDef):
        node = definitions[0] if definitions else tree
        _refuse_node(node, filename, start_line, "forward must be one synchronous function")
    compiler = _Compiler(leaves=leaves, filename=filename, start_line=start_line)
    return compiler.function(definitions[0])


class _Compiler:
    def __init__(self, *, leaves: Mapping[str, LeafRef], filename: str, start_line: int):
        self.leaves = leaves
        self.filename = filename
        self.start_line = start_line

    def function(self, node: ast.FunctionDef) -> dict[str, Any]:
        args = node.args
        if args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults or args.kw_defaults:
            self.refuse(node, "forward parameters must be required named arguments; defaults and variadics are not in v0.1")
        names = [argument.arg for argument in args.args]
        if names and names[0] == "self":
            names = names[1:]
        if not node.body:
            self.refuse(node, "forward body cannot be empty")
        return {
            "language": "restricted-python-ast",
            "args": names,
            "body": [self.statement(statement) for statement in node.body],
        }

    def statement(self, node: ast.stmt) -> dict[str, Any]:
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                self.refuse(node, "Assign targets must be one simple name")
            return {"node": "Assign", "target": node.targets[0].id, "value": self.expression(node.value)}
        if isinstance(node, ast.Return):
            if node.value is None:
                self.refuse(node, "Return must carry the module output")
            return {"node": "Return", "value": self.expression(node.value)}
        if isinstance(node, ast.If):
            if not node.body:
                self.refuse(node, "If body cannot be empty")
            return {
                "node": "If",
                "test": self.expression(node.test),
                "body": [self.statement(item) for item in node.body],
                "orelse": [self.statement(item) for item in node.orelse],
            }
        if isinstance(node, ast.For):
            if node.orelse:
                self.refuse(node, "for/else is not in node-set v0.1")
            if not isinstance(node.target, ast.Name):
                self.refuse(node, "For target must be one simple name")
            count = self._range_literal(node.iter)
            return {
                "node": "For",
                "target": node.target.id,
                "range": count,
                "body": [self.statement(item) for item in node.body],
            }
        if isinstance(node, ast.While):
            if node.orelse:
                self.refuse(node, "while/else is not in node-set v0.1")
            return {
                "node": "While",
                "test": self.expression(node.test),
                "body": [self.statement(item) for item in node.body],
            }
        if isinstance(node, ast.Break):
            return {"node": "Break"}
        if isinstance(node, ast.Continue):
            return {"node": "Continue"}
        if isinstance(node, ast.Try):
            if node.orelse or node.finalbody or not node.handlers:
                self.refuse(node, "Try in v0.1 requires handlers and permits no else or finally")
            return {
                "node": "Try",
                "body": [self.statement(item) for item in node.body],
                "handlers": [self.handler(handler) for handler in node.handlers],
            }
        if isinstance(node, ast.Raise):
            return self.raise_statement(node)
        self.refuse(node)

    def expression(self, node: ast.expr) -> dict[str, Any]:
        if isinstance(node, ast.Name):
            return {"node": "Var", "name": node.id}
        if isinstance(node, ast.Constant):
            if node.value is not None and not isinstance(node.value, (str, int, float, bool)):
                self.refuse(node, "Const must be a JSON scalar")
            return {"node": "Const", "value": node.value}
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                self.refuse(node, "Attr supports one variable.field lookup in v0.1")
            return {"node": "Attr", "obj": node.value.id, "attr": node.attr}
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.Compare):
            if len(node.ops) != 1 or len(node.comparators) != 1:
                self.refuse(node, "chained comparisons are not in node-set v0.1")
            operators = {ast.Eq: "eq", ast.NotEq: "ne"}
            op = operators.get(type(node.ops[0]))
            if op is None:
                self.refuse(node, f"Compare operator {type(node.ops[0]).__name__} is not in node-set v0.1")
            return {
                "node": "Compare",
                "op": op,
                "left": self.expression(node.left),
                "right": self.expression(node.comparators[0]),
            }
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, ast.Add):
                self.refuse(node, f"BinOp operator {type(node.op).__name__} is not in node-set v0.1")
            return {
                "node": "BinOp",
                "op": "add",
                "left": self.expression(node.left),
                "right": self.expression(node.right),
            }
        self.refuse(node)

    def call(self, node: ast.Call) -> dict[str, Any]:
        if node.args:
            self.refuse(node, "leaf calls accept keyword arguments only")
        dynamic_name = None
        if (
            isinstance(node.func, ast.Subscript)
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "self"
        ):
            target = node.func.value.attr
            leaf = self.leaves.get(target)
            if leaf is None or leaf.kind != "tool" or leaf.ref is not None:
                self._refuse_call(node, target=f"self.{target}[...]")
            dynamic_name = self.expression(node.func.slice)
        else:
            if not isinstance(node.func, ast.Attribute) or not isinstance(node.func.value, ast.Name):
                self._refuse_call(node)
            if node.func.value.id != "self":
                self._refuse_call(node)
            target = node.func.attr
            leaf = self.leaves.get(target)
            if leaf is None:
                self._refuse_call(node, target=f"self.{target}")
        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                self.refuse(keyword.value, "keyword splats are not in node-set v0.1")
            if keyword.arg in kwargs:
                self.refuse(keyword.value, f"duplicate keyword {keyword.arg!r}")
            kwargs[keyword.arg] = self.expression(keyword.value)
        leaf_record = {"kind": leaf.kind}
        if leaf.ref is not None:
            leaf_record["ref"] = leaf.ref
        call = {"node": "Call", "leaf": leaf_record, "kwargs": kwargs}
        if dynamic_name is not None:
            call["name"] = dynamic_name
        return call

    def handler(self, node: ast.ExceptHandler) -> dict[str, Any]:
        if node.name is not None or not isinstance(node.type, ast.Name):
            self.refuse(node, "except handlers name one typed error and do not bind it in v0.1")
        allowed = _ERROR_TYPES | {"Exception"}
        if node.type.id not in allowed:
            self.refuse(node, f"except type {node.type.id!r} is outside the typed error table")
        return {"type": node.type.id, "body": [self.statement(item) for item in node.body]}

    def raise_statement(self, node: ast.Raise) -> dict[str, Any]:
        call = node.exc
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id not in _ERROR_TYPES
            or len(call.args) != 1
            or call.keywords
            or not isinstance(call.args[0], ast.Constant)
        ):
            self.refuse(node, "Raise must be SomeTypedError(<JSON scalar>) in v0.1")
        value = call.args[0].value
        if value is not None and not isinstance(value, (str, int, float, bool)):
            self.refuse(node, "Raise message must be a JSON scalar")
        return {"node": "Raise", "exc": call.func.id, "message": value}

    def _range_literal(self, node: ast.expr) -> int:
        if (
            not isinstance(node, ast.Call)
            or not isinstance(node.func, ast.Name)
            or node.func.id != "range"
            or len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.Constant)
            or isinstance(node.args[0].value, bool)
            or not isinstance(node.args[0].value, int)
            or node.args[0].value < 0
        ):
            self.refuse(node, "For iterates range(<non-negative int literal>) only in v0.1")
        return node.args[0].value

    def _refuse_call(self, node: ast.Call, *, target: str | None = None) -> None:
        rendered = target or ast.unparse(node.func)
        line = self.absolute_line(node)
        raise ProgramIRRefusal(
            "PIR-E-NODE-002",
            "5_forward",
            {"leaf": rendered, "line": line},
            f"forward call to undeclared leaf {rendered!r} at {self.location(node)}; "
            "every call must resolve to a Predict, sub-module, tool, or interpreter",
        )

    def refuse(self, node: ast.AST, reason: str | None = None) -> None:
        name = type(node).__name__
        detail: dict[str, Any] = {"node": name, "line": self.absolute_line(node)}
        hint = ""
        if name in _PROPOSED_V02:
            detail["proposed"] = "node-set v0.2"
            hint = "; proposed in node-set v0.2, not ratified"
        explanation = f": {reason}" if reason else ""
        raise ProgramIRRefusal(
            "PIR-E-NODE-001",
            "5_forward",
            detail,
            f"forward: <{name}> at {self.location(node)} is not in the representable node set{explanation}{hint}",
        )

    def absolute_line(self, node: ast.AST) -> int:
        return self.start_line + getattr(node, "lineno", 1) - 1

    def location(self, node: ast.AST) -> str:
        return f"{Path(self.filename).name}:{self.absolute_line(node)}"


def _refuse_node(node: ast.AST, filename: str, start_line: int, reason: str) -> None:
    compiler = _Compiler(leaves={}, filename=filename, start_line=start_line)
    compiler.refuse(node, reason)
