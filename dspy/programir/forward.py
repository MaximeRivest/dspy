"""Compile Python forward methods into the ProgramIR node set v0.4.

The frontend contract (D-034/D-035/D-037/D-041, spec/node-set.md): admit
semantics, desugar syntax, refuse the rest loudly by name. This compiler
lowers a restricted Python `forward` (or its `aforward` twin — the
aforward rule compiles both to the IDENTICAL graph) into the ratified
node-set 0.4 JSON encoding.

The v0.4 inputs-bag admission (D-041) adds a second lawful envelope — the
RECORD ENVELOPE — and one lawful `**splat`. A signature-polymorphic
forward takes the module's OWN input record as one parameter and threads
it into an inner leaf. Two authoring doors compile to that envelope; both
require the module's declared input signature (component 2, D-036), which
the caller supplies as `signature` (the declared input field names in
order). Without a signature the compiler stays byte-identical to v0.3.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dspy.programir.errors import ProgramIRRefusal

_ERROR_TYPES = {"ToolError", "InterpreterError", "AdapterParseError", "LMError"}

#: v0.3 closed builtin table (D-037). `json_dumps`/`json_parse` arrive via
#: the `json.dumps`/`json.loads` surface spelling, mapped below.
_BUILTINS = {
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
}
_JSON_BUILTINS = {"dumps": "json_dumps", "loads": "json_parse"}

#: v0.3 closed value-method table (D-037): str, list, and dict families.
_METHODS = {
    # str — fresh values, code-point semantics
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
    # list — mutators under the v0.2 aliasing ruling
    "append",
    "extend",
    "pop",
    # dict — ordered per the v0.2 Dict ruling
    "get",
    "keys",
    "values",
    "items",
    "update",
}

_BINOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mult", ast.Div: "div"}
_CMPOPS = {
    ast.Eq: "eq",
    ast.NotEq: "ne",
    ast.Lt: "lt",
    ast.LtE: "le",
    ast.Gt: "gt",
    ast.GtE: "ge",
    ast.In: "in",
    ast.NotIn: "not_in",
}

#: Explicit concurrency refuses by name (D-037): scheduling is engine
#: policy, never forward semantics.
_CONCURRENCY_NAMES = {
    "gather",
    "create_task",
    "ensure_future",
    "wait",
    "wait_for",
    "as_completed",
    "run",
    "sleep",
    "TaskGroup",
    "Lock",
    "Semaphore",
    "Event",
    "Condition",
    "Queue",
    "to_thread",
    "shield",
}

#: Named statement refusals: construct -> (reason, supported alternative).
_NAMED_STMT_REFUSALS: dict[type, tuple[str, str]] = {
    ast.With: (
        "`with` is dspy.context — ambient dynamic scoping, the zero-reach-back rule's target",
        "hoist the configuration out of forward; context-scoped forwards are non-exportable (D-035)",
    ),
    ast.AsyncWith: (
        "async `with` is explicit concurrency plumbing",
        "the aforward rule admits only the plain `await leaf.acall(...)` mirror (D-037)",
    ),
    ast.AsyncFor: (
        "async `for` is explicit concurrency plumbing",
        "iterate a plain list; execution mode belongs to the engine (D-037)",
    ),
    ast.ClassDef: (
        "runtime class definition is the refused module family (Refine/BestOfN)",
        "declare modules in __init__ and call them as leaves (D-035)",
    ),
    ast.Import: (
        "imports inside forward are undeclared leaves and hidden state — the leaf rule's whole point",
        "move the dependency into a tool leaf with a `# deps:` line (D-037)",
    ),
    ast.ImportFrom: (
        "imports inside forward are undeclared leaves and hidden state — the leaf rule's whole point",
        "move the dependency into a tool leaf with a `# deps:` line (D-037)",
    ),
    ast.FunctionDef: (
        "nested def is an undeclared leaf and a closure over hidden state",
        "declare the helper as a tool leaf in __init__",
    ),
    ast.AsyncFunctionDef: (
        "nested async def is an undeclared leaf and a closure over hidden state",
        "declare the helper as a tool leaf in __init__",
    ),
    ast.Global: ("global mutation is hidden state", "thread values through Assign"),
    ast.Nonlocal: ("nonlocal mutation is hidden state", "thread values through Assign"),
    ast.Delete: ("del is not in the node set", "rebind the name instead"),
}


@dataclass(frozen=True)
class LeafRef:
    """Name one child callable admitted by a module forward."""

    kind: str
    ref: str | None = None


def compile_forward(
    function: Any,
    leaves: Mapping[str, LeafRef],
    signature: list[str] | None = None,
) -> dict[str, Any]:
    """Compile one Python method into a node-set v0.4 forward record.

    Args:
        function: The authored `forward` (or `aforward`) function.
        leaves: Immediate child names mapped to their typed leaf references.
        signature: The module's declared input field names, in declared
            order (component 2, D-036), or None when the module has no
            declared signature. Supplying it enables the v0.4 record
            envelope: a single non-self parameter used as a record, or a
            `**kwargs` desugar, lowers to `args: [{name, record: "self"}]`.
            When None, the compiler is byte-identical to v0.3 — plain input
            names only, and `**kwargs`/`*args` refuse.

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
    if len(definitions) != 1:
        node = definitions[0] if definitions else tree
        _refuse_node(node, filename, start_line, "forward must be exactly one function definition")
    compiler = _Compiler(leaves=leaves, filename=filename, start_line=start_line, signature=signature)
    return compiler.function(definitions[0])


class _Compiler:
    def __init__(
        self,
        *,
        leaves: Mapping[str, LeafRef],
        filename: str,
        start_line: int,
        signature: list[str] | None = None,
    ):
        self.leaves = leaves
        self.filename = filename
        self.start_line = start_line
        #: The module's declared input field set (D-036 component 2), or
        #: None. When present, the record envelope (D-041) is available.
        self.signature = list(signature) if signature is not None else None
        #: The record parameter's name when this forward is the record
        #: envelope, else None. Set by `function` before the body compiles;
        #: read by the splat/read-only-record checks (D-041).
        self._record: str | None = None
        self._pending: list[list[dict[str, Any]]] = []
        self._hoist_blocked = 0
        self._temp = 0

    # -- envelope ---------------------------------------------------------

    def function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
        # The aforward rule (D-037): `async def aforward` compiles to the
        # IDENTICAL graph as the sync twin — async-ness is stripped here.
        args = node.args
        if args.posonlyargs or args.kwonlyargs or args.defaults or args.kw_defaults:
            self.refuse(
                node,
                "forward parameters must be required named arguments; "
                "positional-only, keyword-only, and default parameters are not in the forward envelope",
            )
        names = [argument.arg for argument in args.args]
        if names and names[0] == "self":
            names = names[1:]

        record_arg = self._record_envelope(node, names)
        if record_arg is not None:
            # The v0.4 record envelope (D-041): args carry exactly one
            # record binding, bound to the module's OWN input signature.
            self._record = record_arg
            emitted_args: list[Any] = [{"name": record_arg, "record": "self"}]
        else:
            if args.vararg or args.kwarg:
                self.refuse(
                    node,
                    "forward parameters must be required named arguments; "
                    "variadics (*args/**kwargs) are the v0.4 record envelope, which needs a declared "
                    "module signature (D-041) — this module declares none",
                )
            emitted_args = names

        if not node.body:
            self.refuse(node, "forward body cannot be empty")
        body = self._body(node.body, allow_docstring=True)
        if not body:
            self.refuse(node, "forward body cannot be empty")
        return {
            "language": "restricted-python-ast",
            "args": emitted_args,
            "body": body,
        }

    def _record_envelope(self, node: ast.FunctionDef | ast.AsyncFunctionDef, names: list[str]) -> str | None:
        """Return the record parameter's name, or None for the plain form.

        The v0.4 inputs-bag admission (D-041) has TWO authoring doors,
        both requiring a declared module signature (component 2, D-036 —
        `self.signature`). The record binds under the parameter's own name.

        Door (a) — explicit `def forward(self, inputs)`: exactly one
        non-self positional parameter whose name is NOT a declared input
        field (a bag name like `inputs`, never a plain field) AND which the
        body USES as a record — it appears as a `**<name>` splat on a leaf
        call. A single parameter that names a declared field stays the
        plain v0.1 form (byte-identical); a bag-named parameter never
        splatted is not a record envelope and refuses downstream as an
        unbound read.

        Door (b) — THE DESUGAR `def forward(self, **kwargs)`: a bare
        `**kwargs` parameter (no positional inputs, no `*args`). The classic
        signature-polymorphic dspy authoring (ReAct, ChainOfThought) lowers
        clean here; the record binds under the `**kwargs` name.

        Without a declared signature both doors are unavailable: `**kwargs`
        refuses (v0.3 behavior) and a lone bag parameter is a plain input.
        """
        args = node.args
        if not self.signature:
            # No declared inputs (None, or an empty field set): neither door
            # opens. `**kwargs` keeps its v0.3 refusal, and a lone bag
            # parameter stays a plain input — an empty record would bind a
            # zero-key envelope where the plain form is honest.
            return None
        # Door (b): the `**kwargs` desugar. A bare double-star, nothing else.
        if args.kwarg is not None and args.vararg is None and not names:
            return args.kwarg.arg
        # Door (a): a single explicit bag parameter used as a record.
        if args.kwarg is None and args.vararg is None and len(names) == 1:
            candidate = names[0]
            if candidate not in self.signature and self._is_splatted(node, candidate):
                return candidate
        return None

    def _is_splatted(self, node: ast.AST, name: str) -> bool:
        """True when `**name` appears as a call keyword-splat in the body.

        The record's one observable use is the `**splat` (D-041). Reading a
        bag parameter's fields with `Attr`/`Index` is possible too, but the
        splat is the unambiguous record signal; a bag parameter never
        splatted is not the record envelope.
        """
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                for keyword in item.keywords:
                    if keyword.arg is None and isinstance(keyword.value, ast.Name) and keyword.value.id == name:
                        return True
        return False

    # -- statements -------------------------------------------------------

    def _body(self, statements: list[ast.stmt], *, allow_docstring: bool = False) -> list[dict[str, Any]]:
        compiled: list[dict[str, Any]] = []
        for index, statement in enumerate(statements):
            if (
                allow_docstring
                and index == 0
                and isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue  # the docstring carries no semantics
            compiled.extend(self.statement(statement))
        return compiled

    def statement(self, node: ast.stmt) -> list[dict[str, Any]]:
        """Compile one statement; desugars may emit several nodes."""
        self._pending.append([])
        try:
            produced = self._statement(node)
        finally:
            hoisted = self._pending.pop()
        return hoisted + produced

    def _statement(self, node: ast.stmt) -> list[dict[str, Any]]:
        named = _NAMED_STMT_REFUSALS.get(type(node))
        if named is not None:
            reason, alternative = named
            self.refuse(node, f"{reason}; instead: {alternative}")
        if isinstance(node, ast.Assign):
            return self.assign(node)
        if isinstance(node, ast.AnnAssign):
            # D-035 desugar: AnnAssign -> Assign, annotation dropped.
            if not isinstance(node.target, ast.Name) or node.value is None:
                self.refuse(node, "annotated assignment must bind one simple name to a value")
            return [{"node": "Assign", "target": node.target.id, "value": self.expression(node.value)}]
        if isinstance(node, ast.AugAssign):
            return self.aug_assign(node)
        if isinstance(node, ast.Expr):
            return self.expr_statement(node)
        if isinstance(node, ast.Assert):
            return self.assert_statement(node)
        if isinstance(node, ast.Pass):
            return [{"node": "Pass"}]
        if isinstance(node, ast.Return):
            if node.value is None:
                self.refuse(node, "Return must carry the module output")
            return [{"node": "Return", "value": self.expression(node.value)}]
        if isinstance(node, ast.If):
            if not node.body:
                self.refuse(node, "If body cannot be empty")
            test = self.expression(node.test)
            return [
                {
                    "node": "If",
                    "test": test,
                    "body": self._block(node.body),
                    "orelse": self._block(node.orelse) if node.orelse else [],
                }
            ]
        if isinstance(node, ast.For):
            return self.for_statement(node)
        if isinstance(node, ast.While):
            if node.orelse:
                self.refuse(node, "while/else is not in the node set; move the tail after the loop")
            self._hoist_blocked += 1
            try:
                test = self.expression(node.test)
            finally:
                self._hoist_blocked -= 1
            return [
                {
                    "node": "While",
                    "test": test,
                    "body": self._block(node.body),
                }
            ]
        if isinstance(node, ast.Break):
            return [{"node": "Break"}]
        if isinstance(node, ast.Continue):
            return [{"node": "Continue"}]
        if isinstance(node, ast.Try):
            if node.orelse or node.finalbody or not node.handlers:
                self.refuse(node, "Try requires handlers and permits no else or finally")
            return [
                {
                    "node": "Try",
                    "body": self._block(node.body),
                    "handlers": [self.handler(handler) for handler in node.handlers],
                }
            ]
        if isinstance(node, ast.Raise):
            return [self.raise_statement(node)]
        self.refuse(node)

    def _block(self, statements: list[ast.stmt]) -> list[dict[str, Any]]:
        body = self._body(statements)
        return body or [{"node": "Pass"}]

    def _guard_record_write(self, name: str, node: ast.AST) -> None:
        """Refuse a write to the read-only input record (D-041, PIR-E-NODE-004).

        The record parameter cannot be an Assign / AssignTuple / For /
        SetIndex target (SEM-5 as amended v0.4). This is the standalone,
        compile-visible half; an alias write (`x = inputs; x[k] = v`) is
        undecidable here and is the interpreter's runtime InterpreterTypeError.
        """
        if self._record is not None and name == self._record:
            self._refuse_record_write(node, name)

    def assign(self, node: ast.Assign) -> list[dict[str, Any]]:
        if len(node.targets) != 1:
            self.refuse(node, "chained assignment (a = b = ...) is not in the node set; assign each name")
        target = node.targets[0]
        if isinstance(target, ast.Name):
            self._guard_record_write(target.id, node)
            return [{"node": "Assign", "target": target.id, "value": self.expression(node.value)}]
        if isinstance(target, ast.Tuple):
            # D-035 desugar: tuple destructuring -> AssignTuple.
            names = self._tuple_target_names(target)
            return [{"node": "AssignTuple", "targets": names, "value": self.expression(node.value)}]
        if isinstance(target, ast.Subscript):
            # SEM-5 as amended: subscript writes are the SetIndex statement.
            if isinstance(target.slice, ast.Slice):
                self.refuse(node, "slice assignment is not admitted (v0.3 Slice is read-only)")
            if not isinstance(target.value, ast.Name):
                self.refuse(node, "SetIndex target must be a variable name; bind the container first")
            self._guard_record_write(target.value.id, node)
            return [
                {
                    "node": "SetIndex",
                    "target": target.value.id,
                    "key": self.expression(target.slice),
                    "value": self.expression(node.value),
                }
            ]
        if isinstance(target, ast.Attribute):
            self.refuse(node, "attribute assignment mutates module state; forwards thread values through names (SEM-5)")
        self.refuse(node, "Assign target must be a simple name, a tuple of names, or one subscript")

    def _tuple_target_names(self, target: ast.Tuple) -> list[str]:
        names: list[str] = []
        for element in target.elts:
            if isinstance(element, ast.Starred):
                self.refuse(
                    element, "starred unpacking is the splat convention; AssignTuple binds a fixed shape (D-037)"
                )
            if not isinstance(element, ast.Name):
                self.refuse(element, "tuple destructuring binds simple names only (SEM-5)")
            self._guard_record_write(element.id, element)
            names.append(element.id)
        if len(names) < 2:
            self.refuse(target, "one-target destructuring is a plain Assign; write it as one")
        if len(set(names)) != len(names):
            self.refuse(target, "AssignTuple targets must be distinct names")
        return names

    def aug_assign(self, node: ast.AugAssign) -> list[dict[str, Any]]:
        # D-035 desugar: AugAssign -> Assign + BinOp.
        op = _BINOPS.get(type(node.op))
        if op is None:
            self.refuse(node, f"augmented operator {type(node.op).__name__} is outside add/sub/mult/div (SEM-4)")
        value = self.expression(node.value)
        if isinstance(node.target, ast.Name):
            self._guard_record_write(node.target.id, node)
            read: dict[str, Any] = {"node": "Var", "name": node.target.id}
            return [
                {
                    "node": "Assign",
                    "target": node.target.id,
                    "value": {"node": "BinOp", "op": op, "left": read, "right": value},
                }
            ]
        if isinstance(node.target, ast.Subscript) and isinstance(node.target.value, ast.Name):
            if not isinstance(node.target.slice, (ast.Name, ast.Constant)):
                self.refuse(
                    node,
                    "augmented subscript assignment needs a Var or Const key so the key evaluates once "
                    "(the single-evaluation rule, D-035)",
                )
            key = self.expression(node.target.slice)
            container = node.target.value.id
            self._guard_record_write(container, node)
            return [
                {
                    "node": "SetIndex",
                    "target": container,
                    "key": key,
                    "value": {
                        "node": "BinOp",
                        "op": op,
                        "left": {"node": "Index", "obj": {"node": "Var", "name": container}, "key": key},
                        "right": value,
                    },
                }
            ]
        self.refuse(node, "augmented assignment binds one simple name or one subscript on a named container")

    def expr_statement(self, node: ast.Expr) -> list[dict[str, Any]]:
        value = node.value
        if isinstance(value, ast.Constant):
            return [{"node": "Pass"}]  # a bare constant carries no semantics
        if isinstance(value, ast.Await):
            value = self._unwrap_await(value)
        if isinstance(value, ast.Call):
            # D-037 desugar: print(...) -> the Log statement.
            if isinstance(value.func, ast.Name) and value.func.id == "print":
                return [self.log_statement(value)]
            call = self.call(value)
            # Statement-position calls (list.append, dict.update, bare leaf
            # calls) keep their effects; the discarded result binds `_`.
            return [{"node": "Assign", "target": "_", "value": call}]
        self.refuse(node, "an expression statement with no effect is not in the node set; bind or remove it")

    def assert_statement(self, node: ast.Assert) -> list[dict[str, Any]]:
        # D-035 desugar: Assert -> If + UnaryOp not + Raise InterpreterError.
        message: Any = "assertion failed"
        if node.msg is not None:
            if not isinstance(node.msg, ast.Constant) or not _is_scalar(node.msg.value):
                self.refuse(node, "assert message must be a constant JSON scalar (Raise carries constants only)")
            message = node.msg.value
        return [
            {
                "node": "If",
                "test": {"node": "UnaryOp", "op": "not", "value": self.expression(node.test)},
                "body": [{"node": "Raise", "exc": "InterpreterError", "message": message}],
                "orelse": [],
            }
        ]

    def log_statement(self, call: ast.Call) -> dict[str, Any]:
        if call.keywords:
            self.refuse(call, "print keywords (sep/end/file) are not in the Log desugar; join the parts yourself")
        parts: list[dict[str, Any]] = []
        for index, argument in enumerate(call.args):
            if index:
                parts.append({"node": "Const", "value": " "})
            parts.append(self.expression(argument))
        if not parts:
            parts = [{"node": "Const", "value": ""}]
        return {"node": "Log", "parts": parts}

    def for_statement(self, node: ast.For) -> list[dict[str, Any]]:
        if node.orelse:
            self.refuse(node, "for/else is not in the node set; move the tail after the loop")
        # D-037 desugar: enumerate-For (For-position only).
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
        ):
            return self.enumerate_for(node)
        head: list[dict[str, Any]] = []
        if isinstance(node.target, ast.Name):
            self._guard_record_write(node.target.id, node)
            target = node.target.id
        elif isinstance(node.target, ast.Tuple):
            target = self._gensym()
            head = [
                {
                    "node": "AssignTuple",
                    "targets": self._tuple_target_names(node.target),
                    "value": {"node": "Var", "name": target},
                }
            ]
        else:
            self.refuse(node, "For target must be a simple name or a tuple of names")
        count = self._range_literal(node.iter)
        record: dict[str, Any] = {"node": "For", "target": target}
        if count is not None:
            record["range"] = count
        else:
            record["iter"] = self.expression(node.iter)
        record["body"] = head + self._block(node.body)
        return [record]

    def enumerate_for(self, node: ast.For) -> list[dict[str, Any]]:
        call = node.iter
        assert isinstance(call, ast.Call)
        if len(call.args) != 1 or call.keywords:
            self.refuse(node, "enumerate takes one iterable and no start in the For desugar (D-037)")
        if not isinstance(node.target, ast.Tuple) or len(node.target.elts) != 2:
            self.refuse(node, "enumerate-For binds exactly (index, element)")
        index_target, element_target = node.target.elts
        if not isinstance(index_target, ast.Name) or not isinstance(element_target, ast.Name):
            self.refuse(node, "enumerate-For binds simple names only")
        self._guard_record_write(index_target.id, node)
        self._guard_record_write(element_target.id, node)
        iterated = self.expression(call.args[0])
        index_name = index_target.id
        increment: dict[str, Any] = {
            "node": "Assign",
            "target": index_name,
            "value": {
                "node": "BinOp",
                "op": "add",
                "left": {"node": "Var", "name": index_name},
                "right": {"node": "Const", "value": 1},
            },
        }
        return [
            {"node": "Assign", "target": index_name, "value": {"node": "Const", "value": -1}},
            {
                "node": "For",
                "target": element_target.id,
                "iter": iterated,
                "body": [increment] + self._block(node.body),
            },
        ]

    # -- expressions ------------------------------------------------------

    def expression(self, node: ast.expr) -> dict[str, Any]:
        if isinstance(node, ast.Name):
            return {"node": "Var", "name": node.id}
        if isinstance(node, ast.Constant):
            if not _is_scalar(node.value):
                self.refuse(node, "Const must be a JSON scalar")
            return {"node": "Const", "value": node.value}
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                self.refuse(node, "Attr reads one variable.field; bind the intermediate value to a name first")
            if node.value.id == "self":
                self.refuse(
                    node,
                    f"self.{node.attr} reads module state — the zero-reach-back rule; "
                    "pass the value in as a forward argument or a leaf result",
                )
            return {"node": "Attr", "obj": node.value.id, "attr": node.attr}
        if isinstance(node, ast.Await):
            return self.call(self._unwrap_await(node))
        if isinstance(node, ast.Call):
            return self.call(node)
        if isinstance(node, ast.NamedExpr):
            return self.walrus(node)
        if isinstance(node, ast.Compare):
            return self.compare(node)
        if isinstance(node, ast.BinOp):
            return self.binop(node)
        if isinstance(node, ast.BoolOp):
            op = "and" if isinstance(node.op, ast.And) else "or"
            return {"node": "BoolOp", "op": op, "values": [self.expression(value) for value in node.values]}
        if isinstance(node, ast.UnaryOp):
            return self.unaryop(node)
        if isinstance(node, ast.IfExp):
            return {
                "node": "IfExp",
                "test": self.expression(node.test),
                "body": self.expression(node.body),
                "orelse": self.expression(node.orelse),
            }
        if isinstance(node, ast.JoinedStr):
            return self.fstring(node)
        if isinstance(node, (ast.List, ast.Tuple)):
            # Tuple values do not exist — tuple literals lower to List (D-035).
            for element in node.elts:
                if isinstance(element, ast.Starred):
                    self.refuse(element, "starred elements are the splat convention; write the elements out (D-037)")
            return {"node": "List", "items": [self.expression(element) for element in node.elts]}
        if isinstance(node, ast.Dict):
            entries = []
            for key, value in zip(node.keys, node.values, strict=False):
                if key is None:
                    self.refuse(node, "dict **splat is the splat convention; merge with `|` or update instead (D-037)")
                entries.append({"key": self.expression(key), "value": self.expression(value)})
            return {"node": "Dict", "entries": entries}
        if isinstance(node, ast.Set):
            self.refuse(
                node,
                "no set value exists in the value model; use a list (a faithful set admission is impossible, D-037)",
            )
        if isinstance(node, ast.Subscript):
            return self.subscript(node)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            return self.list_comprehension(node)
        if isinstance(node, ast.DictComp):
            return self.dict_comprehension(node)
        if isinstance(node, ast.SetComp):
            self.refuse(node, "no set value exists in the value model; use a list comprehension (D-037)")
        if isinstance(node, ast.Lambda):
            self.refuse(
                node, "lambda reopens hidden state via closures; use the no-key builtin forms or a tool leaf (D-037)"
            )
        if isinstance(node, ast.Starred):
            self.refuse(node, "starred expressions are the splat convention, outside the kwargs-only call form (D-037)")
        self.refuse(node)

    def walrus(self, node: ast.NamedExpr) -> dict[str, Any]:
        # D-037 desugar: walrus -> hoisted Assign + Var, only where the
        # enclosing statement evaluates the expression exactly once.
        if not isinstance(node.target, ast.Name):
            self.refuse(node, "walrus target must be a simple name")
        self._guard_record_write(node.target.id, node)
        self._hoist(node, {"node": "Assign", "target": node.target.id, "value": self.expression(node.value)})
        return {"node": "Var", "name": node.target.id}

    def compare(self, node: ast.Compare) -> dict[str, Any]:
        operands = [node.left, *node.comparators]
        if len(node.ops) > 1:
            # D-035: chained comparisons lower to BoolOp-and of pairwise
            # Compares only when every interior operand is Var or Const
            # (single evaluation trivially preserved); otherwise refuse.
            for interior in operands[1:-1]:
                if not isinstance(interior, (ast.Name, ast.Constant)):
                    self.refuse(
                        node,
                        "chained comparison with a computed interior operand would evaluate it twice; "
                        "bind it to a name first (D-035)",
                    )
            pairs = [
                self._compare_pair(op, left, right, node)
                for op, left, right in zip(node.ops, operands[:-1], operands[1:], strict=False)
            ]
            return {"node": "BoolOp", "op": "and", "values": pairs}
        return self._compare_pair(node.ops[0], node.left, node.comparators[0], node)

    def _compare_pair(self, op: ast.cmpop, left: ast.expr, right: ast.expr, origin: ast.AST) -> dict[str, Any]:
        if isinstance(op, (ast.Is, ast.IsNot)):
            # D-035 desugar: `x is None` -> eq/ne against Const null.
            if isinstance(right, ast.Constant) and right.value is None:
                other = left
            elif isinstance(left, ast.Constant) and left.value is None:
                other = right
            else:
                self.refuse(
                    origin, "`is` compares identity; only `is None` / `is not None` lower (to eq/ne null, D-035)"
                )
            return {
                "node": "Compare",
                "op": "eq" if isinstance(op, ast.Is) else "ne",
                "left": self.expression(other),
                "right": {"node": "Const", "value": None},
            }
        name = _CMPOPS.get(type(op))
        if name is None:
            self.refuse(origin, f"Compare operator {type(op).__name__} is outside eq/ne/orderings/membership (SEM-4)")
        return {
            "node": "Compare",
            "op": name,
            "left": self.expression(left),
            "right": self.expression(right),
        }

    def binop(self, node: ast.BinOp) -> dict[str, Any]:
        op = _BINOPS.get(type(node.op))
        if op is not None:
            return {
                "node": "BinOp",
                "op": op,
                "left": self.expression(node.left),
                "right": self.expression(node.right),
            }
        if isinstance(node.op, ast.Mod):
            if isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                return self.percent_format(node)
            self.refuse(node, "numeric % is refused (sign semantics differ across languages, D-037); use sub/mult/div")
        if isinstance(node.op, ast.BitOr):
            return self.dict_merge(node)
        if isinstance(node.op, ast.FloorDiv):
            self.refuse(node, "// is not admitted; use div and int() (D-037)")
        if isinstance(node.op, ast.BitAnd):
            self.refuse(node, "& is not admitted — no set value and no evidence (D-037)")
        self.refuse(node, f"BinOp operator {type(node.op).__name__} is outside add/sub/mult/div (SEM-4)")

    def dict_merge(self, node: ast.BinOp) -> dict[str, Any]:
        # D-037 desugar: `d1 | d2` -> fresh Dict + two update calls,
        # hoisted before the statement (update's ruling supplies the
        # merge semantics; un-defers the v0.2 BitOr refusal).
        temp = self._gensym()
        self._hoist(node, {"node": "Assign", "target": temp, "value": {"node": "Dict", "entries": []}})
        for operand in (node.left, node.right):
            self._hoist(
                node,
                {
                    "node": "Assign",
                    "target": "_",
                    "value": {
                        "node": "Call",
                        "method": "update",
                        "obj": {"node": "Var", "name": temp},
                        "args": [self.expression(operand)],
                    },
                },
            )
        return {"node": "Var", "name": temp}

    def unaryop(self, node: ast.UnaryOp) -> dict[str, Any]:
        if isinstance(node.op, ast.Not):
            return {"node": "UnaryOp", "op": "not", "value": self.expression(node.operand)}
        if isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and _is_number(node.operand.value):
                value = -node.operand.value
                if isinstance(value, float) and value == 0.0:
                    value = 0.0  # −0.0 is kept out of the value model
                return {"node": "Const", "value": value}
            return {"node": "UnaryOp", "op": "neg", "value": self.expression(node.operand)}
        self.refuse(node, f"unary operator {type(node.op).__name__} is outside not/neg (v0.2)")

    def fstring(self, node: ast.JoinedStr) -> dict[str, Any]:
        # D-035 desugar: f-string -> Format.
        parts: list[dict[str, Any]] = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                parts.append({"node": "Const", "value": value.value})
            elif isinstance(value, ast.FormattedValue):
                if value.conversion != -1 or value.format_spec is not None:
                    self.refuse(
                        value,
                        "f-string conversions and format specs are not in Format; pre-render with str() (D-037)",
                    )
                parts.append(self.expression(value.value))
            else:
                self.refuse(value, "unrecognized f-string part")
        if not parts:
            return {"node": "Const", "value": ""}
        return {"node": "Format", "parts": parts}

    def percent_format(self, node: ast.BinOp) -> dict[str, Any]:
        # D-037 desugar: "%s..." % (...) -> Format. %s and %d slots only.
        template = node.left.value  # type: ignore[union-attr]
        values = node.right.elts if isinstance(node.right, ast.Tuple) else [node.right]
        parts: list[dict[str, Any]] = []
        literal = ""
        slot = 0
        index = 0
        while index < len(template):
            character = template[index]
            if character == "%":
                if index + 1 >= len(template):
                    self.refuse(node, "%-format template ends with a bare %")
                marker = template[index + 1]
                if marker == "%":
                    literal += "%"
                elif marker in ("s", "d"):
                    if slot >= len(values):
                        self.refuse(node, "%-format has more slots than values")
                    if literal:
                        parts.append({"node": "Const", "value": literal})
                        literal = ""
                    parts.append(self.expression(values[slot]))
                    slot += 1
                else:
                    self.refuse(node, f"%-format marker %{marker} is outside %s/%d; pre-render the value (D-037)")
                index += 2
                continue
            literal += character
            index += 1
        if slot != len(values):
            self.refuse(node, "%-format has more values than slots")
        if literal:
            parts.append({"node": "Const", "value": literal})
        if not parts:
            return {"node": "Const", "value": ""}
        return {"node": "Format", "parts": parts}

    def subscript(self, node: ast.Subscript) -> dict[str, Any]:
        if isinstance(node.slice, ast.Slice):
            if node.slice.step is not None:
                step = node.slice.step
                is_literal = (isinstance(step, ast.Constant) and step.value in (1, -1)) or (
                    isinstance(step, ast.UnaryOp)
                    and isinstance(step.op, ast.USub)
                    and isinstance(step.operand, ast.Constant)
                    and step.operand.value == 1
                )
                if not is_literal:
                    self.refuse(node, "Slice step is admitted as literal 1 or -1 only (v0.3, D-037)")
            record: dict[str, Any] = {"node": "Slice", "obj": self.expression(node.value)}
            if node.slice.lower is not None:
                record["start"] = self.expression(node.slice.lower)
            if node.slice.upper is not None:
                record["stop"] = self.expression(node.slice.upper)
            if node.slice.step is not None:
                literal = node.slice.step
                value = literal.value if isinstance(literal, ast.Constant) else -1
                if value == -1:
                    record["step"] = -1
            return record
        if isinstance(node.slice, ast.Tuple):
            self.refuse(node, "multi-dimensional subscripts are not in the node set")
        return {
            "node": "Index",
            "obj": self.expression(node.value),
            "key": self.expression(node.slice),
        }

    # -- comprehensions (D-035 desugar: For-accumulation) -----------------

    def _comprehension_frame(
        self, node: ast.ListComp | ast.GeneratorExp | ast.DictComp
    ) -> tuple[str, ast.comprehension, list[dict[str, Any]]]:
        if len(node.generators) != 1:
            self.refuse(node, "nested comprehension generators are not in the desugar table; write nested loops")
        generator = node.generators[0]
        if generator.is_async:
            self.refuse(node, "async comprehensions are explicit concurrency; the engine owns scheduling (D-037)")
        head: list[dict[str, Any]] = []
        if isinstance(generator.target, ast.Name):
            self._guard_record_write(generator.target.id, generator.target)
            loop_target = generator.target.id
        elif isinstance(generator.target, ast.Tuple):
            loop_target = self._gensym()
            head = [
                {
                    "node": "AssignTuple",
                    "targets": self._tuple_target_names(generator.target),
                    "value": {"node": "Var", "name": loop_target},
                }
            ]
        else:
            self.refuse(node, "comprehension target must be a simple name or a tuple of names")
        return loop_target, generator, head

    def list_comprehension(self, node: ast.ListComp | ast.GeneratorExp) -> dict[str, Any]:
        loop_target, generator, head = self._comprehension_frame(node)
        accumulator = self._gensym()
        self._hoist(node, {"node": "Assign", "target": accumulator, "value": {"node": "List", "items": []}})
        inner, element = self._capture(node.elt)
        accumulate: list[dict[str, Any]] = inner + [
            {
                "node": "Assign",
                "target": accumulator,
                "value": {
                    "node": "BinOp",
                    "op": "add",
                    "left": {"node": "Var", "name": accumulator},
                    "right": {"node": "List", "items": [element]},
                },
            }
        ]
        self._hoist(node, self._comprehension_loop(generator, loop_target, head, accumulate))
        return {"node": "Var", "name": accumulator}

    def dict_comprehension(self, node: ast.DictComp) -> dict[str, Any]:
        loop_target, generator, head = self._comprehension_frame(node)
        accumulator = self._gensym()
        self._hoist(node, {"node": "Assign", "target": accumulator, "value": {"node": "Dict", "entries": []}})
        key_inner, key = self._capture(node.key)
        value_inner, value = self._capture(node.value)
        accumulate = (
            key_inner
            + value_inner
            + [
                {
                    "node": "SetIndex",
                    "target": accumulator,
                    "key": key,
                    "value": value,
                }
            ]
        )
        self._hoist(node, self._comprehension_loop(generator, loop_target, head, accumulate))
        return {"node": "Var", "name": accumulator}

    def _comprehension_loop(
        self,
        generator: ast.comprehension,
        loop_target: str,
        head: list[dict[str, Any]],
        accumulate: list[dict[str, Any]],
    ) -> dict[str, Any]:
        body = accumulate
        for condition in reversed(generator.ifs):
            inner, test = self._capture(condition)
            body = inner + [{"node": "If", "test": test, "body": body, "orelse": []}]
        count = self._range_literal(generator.iter)
        loop: dict[str, Any] = {"node": "For", "target": loop_target}
        if count is not None:
            loop["range"] = count
        else:
            loop["iter"] = self.expression(generator.iter)
        loop["body"] = head + body
        return loop

    def _capture(self, node: ast.expr) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Compile an expression, collecting its hoists into a local list."""
        self._pending.append([])
        try:
            expression = self.expression(node)
        finally:
            hoisted = self._pending.pop()
        return hoisted, expression

    # -- calls ------------------------------------------------------------

    def _unwrap_await(self, node: ast.Await) -> ast.Call:
        # The aforward rule (D-037): `await leaf.acall(...)` is the sync
        # call; anything else under await refuses.
        if isinstance(node.value, ast.Call):
            return node.value
        self.refuse(
            node,
            "await wraps only a declared leaf's .acall(...); "
            "explicit concurrency is engine policy, never forward semantics (D-037)",
        )

    def call(self, node: ast.Call) -> dict[str, Any]:
        func = node.func
        # Strip the .acall async spelling — the aforward same-graph rule.
        if isinstance(func, ast.Attribute) and func.attr == "acall" and self._is_leaf_callee(func.value):
            func = func.value
            node = ast.Call(func=func, args=node.args, keywords=node.keywords)
            ast.copy_location(node, func)
        # Explicit concurrency refuses by name (D-037).
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "asyncio":
            self.refuse(
                node,
                f"asyncio.{func.attr} is explicit concurrency; scheduling is the engine's, never the artifact's (D-037)",
            )
        if isinstance(func, ast.Name) and func.id in _CONCURRENCY_NAMES:
            self.refuse(
                node, f"{func.id} is explicit concurrency; scheduling is the engine's, never the artifact's (D-037)"
            )
        # Leaf calls: self.<name>(...) and self.<tools>[<expr>](...).
        if self._is_leaf_callee(func):
            return self.leaf_call(node, func)
        # Builtin calls: the closed v0.3 table.
        if isinstance(func, ast.Name):
            return self.builtin_call(node, func.id)
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "json" and func.attr in _JSON_BUILTINS:
                return self.json_call(node, _JSON_BUILTINS[func.attr])
            if func.attr == "format" and isinstance(func.value, ast.Constant) and isinstance(func.value.value, str):
                return self.str_format(node, func.value.value)
            return self.method_call(node, func)
        self._refuse_call(node)

    def _is_leaf_callee(self, func: ast.expr) -> bool:
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            return True
        return (
            isinstance(func, ast.Subscript)
            and isinstance(func.value, ast.Attribute)
            and isinstance(func.value.value, ast.Name)
            and func.value.value.id == "self"
        )

    def leaf_call(self, node: ast.Call, func: ast.expr) -> dict[str, Any]:
        if node.args:
            self.refuse(node, "leaf calls accept keyword arguments only (the kwargs-only call convention)")
        dynamic_name = None
        if isinstance(func, ast.Subscript):
            target = func.value.attr  # type: ignore[union-attr]
            leaf = self.leaves.get(target)
            if leaf is None or leaf.kind != "tool" or leaf.ref is not None:
                self._refuse_call(node, target=f"self.{target}[...]")
            dynamic_name = self.expression(func.slice)
        else:
            target = func.attr  # type: ignore[union-attr]
            leaf = self.leaves.get(target)
            if leaf is None:
                self._refuse_call(node, target=f"self.{target}")
        # The v0.4 record-splat (D-041): a `**<name>` splat is lawful ONLY
        # when it names the forward's declared record parameter; anything
        # else is the LM-decided-dict form — the line that stays.
        splat = self._leaf_splat(node)
        kwargs: dict[str, Any] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue  # the record splat, handled above
            if keyword.arg in kwargs:
                self.refuse(keyword.value, f"duplicate keyword {keyword.arg!r}")
            # Splat/explicit-kwarg collision: the splat expands the whole
            # declared signature, so an explicit kwarg naming a declared
            # input field collides — never last-writer-wins (PIR-E-NODE-003).
            if splat is not None and self.signature is not None and keyword.arg in self.signature:
                self._refuse_splat_collision(keyword.value, keyword.arg)
            kwargs[keyword.arg] = self.expression(keyword.value)
        leaf_record = {"kind": leaf.kind}
        if leaf.ref is not None:
            leaf_record["ref"] = leaf.ref
        call = {"node": "Call", "leaf": leaf_record, "kwargs": kwargs}
        if dynamic_name is not None:
            call["name"] = dynamic_name
        if splat is not None:
            call["splat"] = splat
        return call

    def _leaf_splat(self, node: ast.Call) -> str | None:
        """Resolve a leaf call's `**splat`, or None when it has no splat.

        The v0.4 record-splat convention (D-041): a `**<expr>` on a leaf
        call is lawful ONLY when `<expr>` is a bare `Name` naming the
        forward's declared record parameter (`self._record`). Every other
        `**` — a splat of a plain variable, an LM-decided dict
        (`**pred.next_tool_args`), or any expression — refuses
        PIR-E-NODE-002 naming the target: splat what the signature
        declares, never what the model decides. At most one splat per call
        (Python allows several `**`; the record envelope creates one
        record, so a second splat cannot resolve).
        """
        splat: str | None = None
        for keyword in node.keywords:
            if keyword.arg is not None:
                continue
            target = self._splat_target_name(keyword.value)
            if self._record is None or target != self._record:
                self._refuse_splat(keyword.value, target)
            if splat is not None:
                self._refuse_splat(keyword.value, target)
            splat = target
        return splat

    def _splat_target_name(self, value: ast.expr) -> str:
        """A readable name for a `**<value>` splat target (for the refusal)."""
        if isinstance(value, ast.Name):
            return value.id
        try:
            return ast.unparse(value)
        except Exception:  # pragma: no cover - unparse is total on valid AST
            return type(value).__name__

    def _refuse_splat(self, node: ast.AST, target: str) -> None:
        line = self.absolute_line(node)
        raise ProgramIRRefusal(
            "PIR-E-NODE-002",
            "5_forward",
            {"target": target, "line": line, "reason": "splat"},
            f"forward `**{target}` splat at {self.location(node)} resolves to no declared input "
            "record — splat what the signature declares, never what the model decides (D-041; "
            "the LM-decided-dict form stays refused). Only `**<record>` on the module's own "
            "signature record is lawful.",
        )

    def _refuse_splat_collision(self, node: ast.AST, key: str) -> None:
        line = self.absolute_line(node)
        raise ProgramIRRefusal(
            "PIR-E-NODE-003",
            "5_forward",
            {"key": key, "line": line, "reason": "collision"},
            f"forward record-splat and explicit kwarg both name {key!r} at {self.location(node)} "
            "(D-041) — never last-writer-wins; drop one.",
        )

    def _refuse_record_write(self, node: ast.AST, record: str) -> None:
        line = self.absolute_line(node)
        raise ProgramIRRefusal(
            "PIR-E-NODE-004",
            "5_forward",
            {"record": record, "line": line},
            f"forward writes to the read-only input record {record!r} at {self.location(node)} "
            "(D-041; SEM-5 as amended v0.4) — the record parameter cannot be an Assign / "
            "AssignTuple / For / SetIndex target.",
        )

    def builtin_call(self, node: ast.Call, name: str) -> dict[str, Any]:
        if name == "print":
            self.refuse(node, "print is the Log statement (D-037) and has no value; use it in statement position")
        if name in ("range", "enumerate"):
            self.refuse(node, f"{name} exists only in For position (SEM-2 / the enumerate desugar, D-037)")
        if name in ("isinstance", "type", "getattr", "hasattr", "setattr", "issubclass"):
            self.refuse(
                node, f"{name} is host introspection; typed values make runtime type-dispatch unnecessary (D-037)"
            )
        if name == "set":
            self.refuse(node, "no set value exists in the value model (D-037); use a list")
        if name in ("open", "input", "eval", "exec"):
            self.refuse(node, f"{name} is I/O or host reach-back; the lawful home is a tool leaf (D-037)")
        if name in _ERROR_TYPES or name == "Exception":
            self.refuse(node, "error constructors exist only under Raise (SEM-3)")
        if name not in _BUILTINS:
            self._refuse_call(node, target=name)
        if node.keywords:
            self.refuse(
                node, f"builtin {name} takes positional arguments only (lambda keys are the refused construct, D-037)"
            )
        if not node.args:
            self.refuse(node, f"builtin {name} needs at least one argument")
        return {
            "node": "Call",
            "builtin": name,
            "args": [self.expression(argument) for argument in node.args],
        }

    def json_call(self, node: ast.Call, builtin: str) -> dict[str, Any]:
        if node.keywords or len(node.args) != 1:
            self.refuse(node, f"{builtin} takes exactly one positional argument (the canonical serde profile, D-037)")
        return {"node": "Call", "builtin": builtin, "args": [self.expression(node.args[0])]}

    def method_call(self, node: ast.Call, func: ast.Attribute) -> dict[str, Any]:
        name = func.attr
        if name not in _METHODS:
            self.refuse(
                node,
                f"method .{name}() is outside the closed 18-name str/list/dict table (v0.3, D-037); "
                "unknown names refuse at compile — use a tool leaf for anything impure",
            )
        if node.keywords:
            self.refuse(node, f"method .{name}() takes positional arguments only")
        return {
            "node": "Call",
            "method": name,
            "obj": self.expression(func.value),
            "args": [self.expression(argument) for argument in node.args],
        }

    def str_format(self, node: ast.Call, template: str) -> dict[str, Any]:
        # D-037 desugar: "…{}…".format(a, b) -> Format ({} slots only).
        if node.keywords:
            self.refuse(node, ".format() named fields are not in the Format desugar; use positional {} slots")
        parts: list[dict[str, Any]] = []
        literal = ""
        slot = 0
        index = 0
        while index < len(template):
            pair = template[index : index + 2]
            if pair in ("{{", "}}"):
                literal += pair[0]
                index += 2
                continue
            if template[index] == "{":
                if pair != "{}":
                    self.refuse(node, ".format() indexed or named slots are not in the Format desugar; use bare {}")
                if slot >= len(node.args):
                    self.refuse(node, ".format() has more slots than values")
                if literal:
                    parts.append({"node": "Const", "value": literal})
                    literal = ""
                parts.append(self.expression(node.args[slot]))
                slot += 1
                index += 2
                continue
            if template[index] == "}":
                self.refuse(node, ".format() template has an unmatched }")
            literal += template[index]
            index += 1
        if slot != len(node.args):
            self.refuse(node, ".format() has more values than slots")
        if literal:
            parts.append({"node": "Const", "value": literal})
        if not parts:
            return {"node": "Const", "value": ""}
        return {"node": "Format", "parts": parts}

    # -- handlers and raise ----------------------------------------------

    def handler(self, node: ast.ExceptHandler) -> dict[str, Any]:
        if node.name is not None or not isinstance(node.type, ast.Name):
            self.refuse(node, "except handlers name one typed error and do not bind it")
        allowed = _ERROR_TYPES | {
            "Exception",
            "InterpreterTypeError",
            "InterpreterKeyError",
            "InterpreterArithmeticError",
        }
        if node.type.id not in allowed:
            self.refuse(
                node,
                f"except type {node.type.id!r} is outside the typed error table "
                f"({', '.join(sorted(allowed))}) (SEM-3 as amended v0.2)",
            )
        return {"type": node.type.id, "body": self._block(node.body)}

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
            self.refuse(
                node,
                "Raise must be SomeTypedError(<JSON scalar>) with a name from the typed error table "
                f"({', '.join(sorted(_ERROR_TYPES))}) (SEM-3)",
            )
        value = call.args[0].value
        if not _is_scalar(value):
            self.refuse(node, "Raise message must be a JSON scalar")
        return {"node": "Raise", "exc": call.func.id, "message": value}

    # -- helpers ----------------------------------------------------------

    def _range_literal(self, node: ast.expr) -> int | None:
        """Return the literal count for `range(<literal>)`, else None."""
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range"):
            return None
        if (
            len(node.args) != 1
            or node.keywords
            or not isinstance(node.args[0], ast.Constant)
            or isinstance(node.args[0].value, bool)
            or not isinstance(node.args[0].value, int)
            or node.args[0].value < 0
        ):
            self.refuse(
                node,
                "For iterates range(<non-negative int literal>) or a list expression (SEM-2 as amended v0.2); "
                "a computed bound needs the list form or a While",
            )
        return node.args[0].value

    def _hoist(self, node: ast.AST, statement: dict[str, Any]) -> None:
        if self._hoist_blocked or not self._pending:
            self.refuse(
                node,
                "this desugar hoists statements, which a While test re-evaluates every iteration — "
                "single evaluation cannot be preserved; restructure the loop (D-035/D-037)",
            )
        self._pending[-1].append(statement)

    def _gensym(self) -> str:
        self._temp += 1
        return f"_pir{self._temp}"

    def _refuse_call(self, node: ast.Call, *, target: str | None = None) -> None:
        rendered = target or ast.unparse(node.func)
        line = self.absolute_line(node)
        raise ProgramIRRefusal(
            "PIR-E-NODE-002",
            "5_forward",
            {"leaf": rendered, "line": line},
            f"forward call to undeclared leaf {rendered!r} at {self.location(node)}; "
            "every call must resolve to a Predict, sub-module, tool, interpreter, "
            "or a name in the closed v0.3 builtin/method tables",
        )

    def refuse(self, node: ast.AST, reason: str | None = None) -> None:
        name = type(node).__name__
        detail: dict[str, Any] = {"node": name, "line": self.absolute_line(node)}
        explanation = f": {reason}" if reason else ""
        raise ProgramIRRefusal(
            "PIR-E-NODE-001",
            "5_forward",
            detail,
            f"forward: <{name}> at {self.location(node)} is not in the representable node set{explanation}",
        )

    def absolute_line(self, node: ast.AST) -> int:
        return self.start_line + getattr(node, "lineno", 1) - 1

    def location(self, node: ast.AST) -> str:
        return f"{Path(self.filename).name}:{self.absolute_line(node)}"


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _refuse_node(node: ast.AST, filename: str, start_line: int, reason: str) -> None:
    compiler = _Compiler(leaves={}, filename=filename, start_line=start_line)
    compiler.refuse(node, reason)
