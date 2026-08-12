"""Closed-grammar interpreter for the ProgramIR forward layer.

Executes one compiled forward map — a manifest's ``5_forward`` component —
against caller-supplied leaves. The semantics mirror the programir-contract
reference interpreter (`reference/interp.py` at the pinned SHA): the eight
SEM rulings, the v0.2 value model (exact int64, no NaN/Inf, strict JSON
equality), the SEM-6 loop and recursion caps, and the v0.3 builtin and
value-method tables (D-037). dspy's compiler currently emits node-set v0.1
trees; this interpreter runs those and is structured for v0.3.

v0.4 (D-041, the inputs-bag admission): the record envelope binds ONE
shallowly immutable record (`_Record`) instead of per-name inputs; a
leaf call's ``splat`` expands the record's static key list into the
kwargs ahead of the explicit ones. Direct writes to the record and
splat/kwarg collisions are compile refusals and surface here as
`MalformedNodeError`; alias-path mutation is the runtime
`InterpreterTypeError`.

The interpreter performs no I/O of its own: every predict / tool /
interpreter call goes through the `LeafRunner` the caller supplies, so a
scripted LM can drive execution without a live model.
"""

from __future__ import annotations

import json as _json
from typing import Any, Protocol

from dspy.programir.engine.errors import (
    HANDLER_NAMES,
    RAISEABLE,
    CatchableError,
    InterpreterArithmeticError,
    InterpreterKeyError,
    InterpreterTypeError,
    LoopCapError,
    MalformedNodeError,
    handler_matches,
)

__all__ = [
    "LeafRunner",
    "run_forward",
    "json_equal",
    "render_float",
    "WHILE_CAP",
    "RECURSION_CAP",
    "INT_MAX",
    "SAFE_INT",
]

#: v0.2 value model: ints are a distinct exact type covering ±(2^63-1).
INT_MAX = 2**63 - 1

#: v0.2 arithmetic: div refuses when an int operand's magnitude exceeds 2^53.
SAFE_INT = 2**53

#: SEM-6: the ratified default ``While`` iteration cap.
WHILE_CAP = 1000

#: SEM-7: module recursion depth cap (frames of `run_forward`).
RECURSION_CAP = 32


class LeafRunner(Protocol):
    """What the caller supplies: the typed leaves, scriptable or live.

    The interpreter resolves every ``Call`` to one of these three entry
    points (module leaves recurse into `run_forward` instead). Leaf
    failures must be raised as the typed table errors from ``errors.py``.
    """

    def predict(self, path: str, kwargs: dict) -> dict:
        """Run the predictor at dotted `path`; return a Prediction record."""
        ...

    def tool(self, name: str, kwargs: dict) -> Any:
        """Run tool `name`; return a JSON value or raise ``ToolError``."""
        ...

    def interpreter(self, ref: str, code: str) -> Any:
        """Run `code` on interpreter pool entry `ref` (D-029)."""
        ...


class _Record(dict):
    """The v0.4 input record (D-041): a shallowly immutable mapping.

    Subclasses dict so every READ path (Attr, Index, membership, the
    non-mutating dict methods, splat expansion, trace serialization)
    works unchanged; the mutation sites (SetIndex, the mutating dict
    methods) check for this marker and raise `InterpreterTypeError` —
    the runtime half of the read-only rule, reachable only through an
    alias (the direct spelling is a compile refusal, PIR-E-NODE-004).
    Values it holds follow the ordinary v0.2 aliasing rules (shallow
    immutability, spec/node-set.md v0.4).
    """


class _BreakSignal(Exception):  # noqa: N818 - control-flow signal, not an error
    """Internal loop signal; never crosses a statement-list boundary."""


class _ContinueSignal(Exception):  # noqa: N818 - control-flow signal, not an error
    """Internal loop signal; never crosses a statement-list boundary."""


class _ReturnSignal(Exception):  # noqa: N818 - control-flow signal, not an error
    """Internal signal carrying a forward's return value."""

    def __init__(self, value: Any):
        self.value = value


def _is_number(v: Any) -> bool:
    """True for int/float, excluding bool (a distinct JSON type here)."""
    return isinstance(v, (int, float)) and type(v) is not bool


def json_equal(a: Any, b: Any) -> bool:
    """Compare two values with strict JSON-value equality (SEM-4).

    Different JSON types are unequal, never an error: ``true != 1``,
    ``1 != 1.0``, ``"1" != 1``. Arrays and objects compare deeply under
    the same rule.

    Args:
        a: Left value.
        b: Right value.

    Returns:
        True when the two values are the same JSON value.
    """
    if type(a) is bool or type(b) is bool:
        return type(a) is bool and type(b) is bool and a == b
    if _is_number(a) or _is_number(b):
        return _is_number(a) and _is_number(b) and type(a) is type(b) and a == b
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, list) or isinstance(b, list):
        return (
            isinstance(a, list)
            and isinstance(b, list)
            and len(a) == len(b)
            and all(json_equal(x, y) for x, y in zip(a, b, strict=False))
        )
    if isinstance(a, dict) or isinstance(b, dict):
        return (
            isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and all(json_equal(a[k], b[k]) for k in a)
        )
    return False


def render_float(x: float) -> str:
    """Render one float under the v0.3 float-repr pin (D-037).

    Shortest round-trip decimal; fixed notation for 1e-6 <= |x| < 1e21,
    exponent form outside; ``-0.0`` renders ``"0"``.
    """
    if x == 0.0:
        return "0"
    s = repr(x)
    neg = s.startswith("-")
    if neg:
        s = s[1:]
    if "e" in s:
        mant, _, exp_s = s.partition("e")
        exp = int(exp_s)
    else:
        mant, exp = s, 0
    ip, _, fp = mant.partition(".")
    all_digits = ip + fp
    digits = all_digits.lstrip("0")
    lead = len(all_digits) - len(digits)
    e10 = len(ip) + exp - lead - 1
    digits = digits.rstrip("0") or "0"
    if 1e-6 <= abs(x) < 1e21:
        point = e10 + 1
        if point <= 0:
            body = "0." + "0" * (-point) + digits
        elif point >= len(digits):
            body = digits + "0" * (point - len(digits))
        else:
            body = digits[:point] + "." + digits[point:]
    else:
        head, rest = digits[0], digits[1:]
        body = head + ("." + rest if rest else "") + "e" + ("+" if e10 >= 0 else "-") + str(abs(e10))
    return ("-" if neg else "") + body


def _render_part(v: Any, where: str) -> str:
    """Render one value under the Format/Log/str rules (v0.2 + v0.3)."""
    if isinstance(v, str):
        return v
    if type(v) is bool:
        raise InterpreterTypeError(f"{where} cannot render a bool")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return render_float(v)
    raise InterpreterTypeError(f"{where} cannot render {type(v).__name__} (strings, ints, and floats only)")


def _int_guard(value: int, op: str) -> int:
    """Keep exact int results within ±(2^63-1) (v0.2 arithmetic rule 1)."""
    if -INT_MAX <= value <= INT_MAX:
        return value
    raise InterpreterArithmeticError(
        f"{op}: exact int result {value} outside ±2^63-1 — never wraps, never silently promotes (v0.2, D-034)"
    )


def _float_guard(value: float, op: str) -> float:
    """Refuse non-finite float results (v0.2 arithmetic rule 2)."""
    if value != value or value in (float("inf"), float("-inf")):
        raise InterpreterArithmeticError(
            f"{op}: non-finite float result — Inf and NaN do not exist in the value model (v0.2, D-034)"
        )
    return value


def _node(obj: Any, what: str) -> dict:
    """Require a node object; refuse anything else as malformed."""
    if not isinstance(obj, dict) or not isinstance(obj.get("node"), str):
        raise MalformedNodeError(f"{what} is not a node object")
    return obj


def _field(node: dict, name: str) -> Any:
    """Require field `name` on `node` — absent is malformed, not None."""
    if name not in node:
        raise MalformedNodeError(f"{node['node']} node is missing {name!r}")
    return node[name]


def _index_read(container: Any, key: Any) -> Any:
    """Read one v0.2 Index: dict+str member or list+int element."""
    if isinstance(container, dict):
        if not isinstance(key, str):
            raise InterpreterTypeError(f"Index into a dict needs a string key, got {type(key).__name__}")
        if key not in container:
            raise InterpreterKeyError(f"no key {key!r}")
        return container[key]
    if isinstance(container, list):
        if type(key) is bool or not isinstance(key, int):
            raise InterpreterTypeError(f"Index into a list needs an int key, got {type(key).__name__}")
        if not 0 <= key < len(container):
            raise InterpreterKeyError(f"index {key} out of range (negative indices are Slice territory — v0.2)")
        return container[key]
    raise InterpreterTypeError(f"Index into {type(container).__name__} (v0.2)")


def _all_numbers(xs: list) -> bool:
    return all(_is_number(x) for x in xs)


def _all_strings(xs: list) -> bool:
    return all(isinstance(x, str) for x in xs)


def _json_dumps(value: Any) -> str:
    """Serialize one value under the canonical-JSON profile (PIR-004)."""

    def scan(v: Any) -> None:
        if isinstance(v, float):
            _float_guard(v, "json_dumps")
            if v == 0.0 and str(v)[0] == "-":
                raise InterpreterArithmeticError("json_dumps: -0.0 refuses (canonical profile)")
        elif isinstance(v, dict):
            for k, sub in v.items():
                if not isinstance(k, str):
                    raise InterpreterTypeError("json_dumps: object keys must be strings")
                scan(sub)
        elif isinstance(v, list):
            for sub in v:
                scan(sub)
        elif v is None or type(v) is bool or isinstance(v, (int, str)):
            pass
        else:
            raise InterpreterTypeError(f"json_dumps: {type(v).__name__} is not a JSON value")

    scan(value)
    return _json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _json_parse(text: str) -> Any:
    def refuse_const(name: str) -> Any:
        raise InterpreterTypeError(f"json_parse: {name} is not a JSON value")

    try:
        return _json.loads(text, parse_constant=refuse_const)
    except InterpreterTypeError:
        raise
    except ValueError as exc:
        raise InterpreterTypeError(f"json_parse: {exc}") from None


def _arity(name: str, args: list, low: int, high: int) -> None:
    if not (low <= len(args) <= high):
        raise InterpreterTypeError(
            f"{name}: takes {low}" + (f"..{high}" if high != low else "") + f" argument(s), got {len(args)}"
        )


def _run_builtin(name: str, args: list) -> Any:
    """Run one builtin from the closed v0.3 table (D-037)."""
    if name == "len":
        _arity("len", args, 1, 1)
        v = args[0]
        if isinstance(v, (str, list, dict)):
            return len(v)
        raise InterpreterTypeError(f"len of {type(v).__name__}")
    if name == "str":
        _arity("str", args, 1, 1)
        v = args[0]
        if type(v) is bool:
            return "true" if v else "false"
        if v is None:
            return "null"
        return _render_part(v, "str")
    if name == "int":
        _arity("int", args, 1, 1)
        v = args[0]
        if type(v) is bool:
            raise InterpreterTypeError("int of bool refuses")
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return _int_guard(int(v), "int")
        if isinstance(v, str):
            import re as _re

            if _re.fullmatch(r"-?(0|[1-9][0-9]*)", v):
                return _int_guard(int(v), "int")
            raise InterpreterTypeError(f"int: {v!r} is not canonical decimal")
        raise InterpreterTypeError(f"int of {type(v).__name__}")
    if name == "float":
        _arity("float", args, 1, 1)
        v = args[0]
        if type(v) is bool:
            raise InterpreterTypeError("float of bool refuses")
        if isinstance(v, float):
            return v
        if isinstance(v, int):
            if abs(v) > SAFE_INT:
                raise InterpreterArithmeticError("float: int beyond 2^53 is not exactly representable")
            return float(v)
        if isinstance(v, str):
            try:
                out = float(v)
            except ValueError:
                raise InterpreterTypeError(f"float: cannot parse {v!r}") from None
            return _float_guard(out, "float")
        raise InterpreterTypeError(f"float of {type(v).__name__}")
    if name in ("max", "min", "sorted"):
        _arity(name, args, 1, 1)
        xs = args[0]
        if not isinstance(xs, list):
            raise InterpreterTypeError(f"{name} of {type(xs).__name__}")
        if name != "sorted" and not xs:
            raise InterpreterTypeError(f"{name} of an empty list")
        if not (_all_numbers(xs) or _all_strings(xs)):
            raise InterpreterTypeError(f"{name}: list must be all numbers or all strings")
        if name == "max":
            return max(xs)
        if name == "min":
            return min(xs)
        return sorted(xs)
    if name == "sum":
        _arity("sum", args, 1, 1)
        xs = args[0]
        if not isinstance(xs, list) or not _all_numbers(xs):
            raise InterpreterTypeError("sum: takes a list of numbers")
        total: Any = 0
        for x in xs:
            total = total + x
            if isinstance(total, int):
                _int_guard(total, "sum")
            else:
                _float_guard(total, "sum")
        return total
    if name in ("any", "all"):
        _arity(name, args, 1, 1)
        xs = args[0]
        if not isinstance(xs, list) or not all(type(x) is bool for x in xs):
            raise InterpreterTypeError(f"{name}: takes a list of booleans (SEM-1 discipline)")
        return any(xs) if name == "any" else all(xs)
    if name == "json_dumps":
        _arity("json_dumps", args, 1, 1)
        return _json_dumps(args[0])
    if name == "json_parse":
        _arity("json_parse", args, 1, 1)
        if not isinstance(args[0], str):
            raise InterpreterTypeError("json_parse: takes a string")
        return _json_parse(args[0])
    raise MalformedNodeError(
        f"unknown builtin {name!r} — the table is closed (v0.3, D-037); a compile-side refusal, never a runtime fallback"
    )


_STR_METHODS = frozenset(
    {"strip", "lstrip", "rstrip", "lower", "upper", "split", "join", "replace", "startswith", "endswith"}
)
_LIST_METHODS = frozenset({"append", "extend", "pop"})
_DICT_METHODS = frozenset({"get", "keys", "values", "items", "pop", "update"})
_METHOD_NAMES = _STR_METHODS | _LIST_METHODS | _DICT_METHODS


def _run_method(name: str, obj: Any, args: list) -> Any:
    """Run one value method from the closed v0.3 table (D-037)."""
    if name not in _METHOD_NAMES:
        raise MalformedNodeError(
            f"unknown method {name!r} — the table is closed (v0.3, D-037); a compile-side refusal, never a runtime fallback"
        )
    if isinstance(obj, str) and name in _STR_METHODS:
        if name in ("strip", "lstrip", "rstrip"):
            _arity(name, args, 0, 0)
            return getattr(obj, name)()
        if name in ("lower", "upper"):
            _arity(name, args, 0, 0)
            return obj.lower() if name == "lower" else obj.upper()
        if name == "split":
            _arity("split", args, 0, 1)
            if not args:
                return obj.split()
            sep = args[0]
            if not isinstance(sep, str) or sep == "":
                raise InterpreterTypeError("split: separator must be a non-empty string")
            return obj.split(sep)
        if name == "join":
            _arity("join", args, 1, 1)
            xs = args[0]
            if not isinstance(xs, list) or not _all_strings(xs):
                raise InterpreterTypeError("join: takes a list of strings")
            return obj.join(xs)
        if name == "replace":
            _arity("replace", args, 2, 2)
            old, new = args
            if not isinstance(old, str) or not isinstance(new, str):
                raise InterpreterTypeError("replace: takes two strings")
            return obj.replace(old, new)
        if name in ("startswith", "endswith"):
            _arity(name, args, 1, 1)
            needle = args[0]
            if not isinstance(needle, str):
                raise InterpreterTypeError(f"{name}: takes a string")
            return getattr(obj, name)(needle)
    if isinstance(obj, list) and name in _LIST_METHODS:
        if name == "append":
            _arity("append", args, 1, 1)
            obj.append(args[0])
            return None
        if name == "extend":
            _arity("extend", args, 1, 1)
            xs = args[0]
            if not isinstance(xs, list):
                raise InterpreterTypeError("extend: takes a list")
            obj.extend(list(xs))
            return None
        if name == "pop":
            _arity("pop", args, 0, 1)
            if not obj:
                raise InterpreterKeyError("pop from an empty list")
            if not args:
                return obj.pop()
            idx = args[0]
            if type(idx) is bool or not isinstance(idx, int):
                raise InterpreterTypeError("pop: index must be an int")
            if not -len(obj) <= idx < len(obj):
                raise InterpreterKeyError(f"pop: index {idx} out of range")
            return obj.pop(idx)
    if isinstance(obj, dict) and name in _DICT_METHODS:
        if isinstance(obj, _Record) and name in ("pop", "update"):
            # The v0.4 input record is immutable at execution (D-041);
            # the non-mutating reads (get/keys/values/items) apply.
            raise InterpreterTypeError(
                f"{name} on the input record — the record is read-only "
                "(D-041), through every alias"
            )
        if name == "get":
            _arity("get", args, 1, 2)
            key = args[0]
            if not isinstance(key, str):
                raise InterpreterTypeError("get: key must be a string")
            default = args[1] if len(args) == 2 else None
            return obj.get(key, default)
        if name in ("keys", "values"):
            _arity(name, args, 0, 0)
            return list(obj.keys() if name == "keys" else obj.values())
        if name == "items":
            _arity("items", args, 0, 0)
            return [[k, v] for k, v in obj.items()]
        if name == "pop":
            _arity("pop", args, 1, 2)
            key = args[0]
            if not isinstance(key, str):
                raise InterpreterTypeError("pop: key must be a string")
            if key in obj:
                return obj.pop(key)
            if len(args) == 2:
                return args[1]
            raise InterpreterKeyError(f"pop: no key {key!r}")
        if name == "update":
            _arity("update", args, 1, 1)
            other = args[0]
            if not isinstance(other, dict):
                raise InterpreterTypeError("update: takes a dict")
            obj.update(other)
            return None
    raise InterpreterTypeError(f"method {name!r} does not apply to {type(obj).__name__}")


def _stmt_list(node: dict, name: str) -> list:
    """Require field `name` to be a statement list."""
    body = _field(node, name)
    if not isinstance(body, list):
        raise MalformedNodeError(f"{node['node']}.{name} must be a statement list")
    return body


def run_forward(
    forwards: dict,
    path: str,
    inputs: dict,
    leaves: LeafRunner,
    *,
    normalize_legacy: bool = True,
    log_sink: list | None = None,
    _depth: int = 1,
) -> dict:
    """Execute module `path`'s compiled forward and return its Prediction.

    Args:
        forwards: The ``5_forward`` map, dotted module path -> forward
            spec; the root module's key is ``"self"`` (pass `path` = "").
        path: Dotted path of the module to run ("" for the root).
        inputs: Input bindings for the forward's environment.
        leaves: The `LeafRunner` supplying predict / tool / interpreter.
        normalize_legacy: Accept the pre-D-029 interpreter leaf without a
            ``ref`` by normalizing it to ``ref: "main"``.
        log_sink: Where ``Log`` statements append rendered strings (v0.3,
            D-037). None discards logs.

    Returns:
        The Prediction record (a dict) the forward returned.

    Raises:
        CatchableError: A table error the program did not handle.
        LoopCapError: SEM-6 cap breach (never program-catchable).
        MalformedNodeError: Input the compiler should have refused.
    """
    if _depth > RECURSION_CAP:
        raise LoopCapError(f"module recursion exceeded the depth cap {RECURSION_CAP} (SEM-6)")
    key = path or "self"
    spec = forwards.get(key)
    if not isinstance(spec, dict):
        raise MalformedNodeError(f"no forward entry for module path {key!r}")
    top = spec.get("body")
    if not isinstance(top, list):
        raise MalformedNodeError(f"forward[{key}] has no statement list")
    # The v0.4 envelope split (D-041): plain args bind each input name;
    # the record envelope binds ONE shallowly immutable record. A
    # missing key at runtime is impossible by construction (validation
    # admits only inputs matching the signature) — no check here.
    record_name: str | None = None
    args_spec = spec.get("args")
    if (isinstance(args_spec, list) and len(args_spec) == 1
            and isinstance(args_spec[0], dict)):
        binding = args_spec[0]
        name = binding.get("name")
        if not isinstance(name, str) or not name:
            raise MalformedNodeError(
                "record binding needs a name (D-041; a compile-side "
                "refusal, never a runtime fallback)"
            )
        if binding.get("record") != "self":
            raise MalformedNodeError(
                "record binding target must be 'self' — the module's own "
                "input signature is the only lawful target (D-041)"
            )
        record_name = name
        env: dict = {record_name: _Record(inputs)}
    else:
        env = dict(inputs)

    def require_bool(v: Any, where: str) -> bool:
        if type(v) is not bool:
            raise InterpreterTypeError(f"{where} must be a boolean, got {type(v).__name__} (SEM-1)")
        return v

    def ev(e: Any) -> Any:
        e = _node(e, "expression")
        n = e["node"]
        if n == "Var":
            name = _field(e, "name")
            if not isinstance(name, str):
                raise MalformedNodeError("Var name must be a string")
            if name not in env:
                raise InterpreterKeyError(f"unbound name {name!r} (SEM-7)")
            return env[name]
        if n == "Const":
            return _field(e, "value")
        if n == "Attr":
            obj_name, attr = _field(e, "obj"), _field(e, "attr")
            if not isinstance(obj_name, str) or not isinstance(attr, str):
                raise MalformedNodeError("Attr obj and attr must be strings")
            if obj_name not in env:
                raise InterpreterKeyError(f"unbound name {obj_name!r} (SEM-7)")
            record = env[obj_name]
            if not isinstance(record, dict):
                raise InterpreterTypeError(
                    f"Attr on {type(record).__name__}; field access is key lookup on a Prediction/dict only (SEM-7)"
                )
            if attr not in record:
                raise InterpreterKeyError(f"{obj_name!r} has no field {attr!r} (SEM-7)")
            return record[attr]
        if n == "Compare":
            op = _field(e, "op")
            left, right = ev(_field(e, "left")), ev(_field(e, "right"))
            if op in ("eq", "ne"):
                equal = json_equal(left, right)
                return equal if op == "eq" else not equal
            if op in ("lt", "le", "gt", "ge"):
                if (_is_number(left) and _is_number(right)) or (isinstance(left, str) and isinstance(right, str)):
                    if op == "lt":
                        return left < right
                    if op == "le":
                        return left <= right
                    if op == "gt":
                        return left > right
                    return left >= right
                raise InterpreterTypeError(
                    f"{op} of {type(left).__name__} and {type(right).__name__} (SEM-4 as amended v0.2)"
                )
            if op in ("in", "not_in"):
                if isinstance(right, list):
                    found = any(json_equal(left, x) for x in right)
                elif isinstance(right, str) and isinstance(left, str):
                    found = left in right
                elif isinstance(right, dict) and isinstance(left, str):
                    found = left in right
                else:
                    raise InterpreterTypeError(
                        f"membership of {type(left).__name__} in {type(right).__name__} (SEM-4 as amended v0.2)"
                    )
                return found if op == "in" else not found
            raise MalformedNodeError(f"Compare op {op!r} outside the SEM-4 set")
        if n == "BinOp":
            op = _field(e, "op")
            if op not in ("add", "sub", "mult", "div"):
                raise MalformedNodeError(f"BinOp op {op!r} outside add/sub/mult/div (SEM-4)")
            left, right = ev(_field(e, "left")), ev(_field(e, "right"))
            if op == "add":
                if isinstance(left, str) and isinstance(right, str):
                    return left + right
                if isinstance(left, list) and isinstance(right, list):
                    return list(left) + list(right)
            if _is_number(left) and _is_number(right):
                if op == "div":
                    if right == 0:
                        raise InterpreterArithmeticError("div by zero (v0.2, D-034)")
                    for operand in (left, right):
                        if isinstance(operand, int) and abs(operand) > SAFE_INT:
                            raise InterpreterArithmeticError(
                                "div with an int operand beyond 2^53 (the double-rounding trap; v0.2, D-034)"
                            )
                    return _float_guard(left / right, "div")
                result = left + right if op == "add" else left - right if op == "sub" else left * right
                if isinstance(result, int):
                    return _int_guard(result, op)
                return _float_guard(result, op)
            raise InterpreterTypeError(f"{op} of {type(left).__name__} and {type(right).__name__} (SEM-4)")
        if n == "UnaryOp":
            op = _field(e, "op")
            v = ev(_field(e, "value"))
            if op == "not":
                return not require_bool(v, "UnaryOp not operand")
            if op == "neg":
                if not _is_number(v):
                    raise InterpreterTypeError(f"neg of {type(v).__name__} (v0.2)")
                out = -v
                if isinstance(out, float) and out == 0.0:
                    return 0.0
                if isinstance(out, int):
                    return _int_guard(out, "neg")
                return out
            raise MalformedNodeError(f"UnaryOp op {op!r} outside not/neg")
        if n == "Format":
            parts = _field(e, "parts")
            if not isinstance(parts, list) or not parts:
                raise MalformedNodeError("Format needs a non-empty parts list")
            return "".join(_render_part(ev(p), "Format") for p in parts)
        if n == "List":
            items = _field(e, "items")
            if not isinstance(items, list):
                raise MalformedNodeError("List items must be a list")
            return [ev(x) for x in items]
        if n == "Dict":
            entries = _field(e, "entries")
            if not isinstance(entries, list):
                raise MalformedNodeError("Dict entries must be a list")
            out: dict = {}
            for entry in entries:
                if not isinstance(entry, dict) or "key" not in entry or "value" not in entry:
                    raise MalformedNodeError("Dict entry must carry key and value")
                key_value = ev(entry["key"])
                if not isinstance(key_value, str):
                    raise InterpreterTypeError(f"Dict key must be a string, got {type(key_value).__name__} (v0.2)")
                out[key_value] = ev(entry["value"])
            return out
        if n == "Index":
            container = ev(_field(e, "obj"))
            index_key = ev(_field(e, "key"))
            return _index_read(container, index_key)
        if n == "Slice":
            container = ev(_field(e, "obj"))
            if not isinstance(container, (list, str)):
                raise InterpreterTypeError(f"Slice of {type(container).__name__} (v0.3)")
            step = e.get("step", 1)
            if step not in (1, -1):
                raise MalformedNodeError("Slice step is literal 1 or -1 (v0.3)")
            bounds = []
            for field_name in ("start", "stop"):
                if field_name in e:
                    b = ev(e[field_name])
                    if type(b) is bool or not isinstance(b, int):
                        raise InterpreterTypeError(f"Slice {field_name} must be an int")
                    bounds.append(b)
                else:
                    bounds.append(None)
            sliced = container[bounds[0] : bounds[1] : (None if step == 1 else -1)]
            return list(sliced) if isinstance(container, list) else sliced
        if n == "BoolOp":
            op = _field(e, "op")
            if op not in ("and", "or"):
                raise MalformedNodeError(f"BoolOp op {op!r} outside and/or")
            values = _field(e, "values")
            if not isinstance(values, list) or len(values) < 2:
                raise MalformedNodeError("BoolOp needs a list of 2+ operands")
            result = False
            for v in values:
                result = require_bool(ev(v), "BoolOp operand")
                if op == "and" and not result:
                    return False
                if op == "or" and result:
                    return True
            return result
        if n == "IfExp":
            take = require_bool(ev(_field(e, "test")), "IfExp test")
            return ev(_field(e, "body") if take else _field(e, "orelse"))
        if n == "Call":
            return call(e)
        raise MalformedNodeError(f"<{n}> is not in the expression node set")

    def call(e: dict) -> Any:
        if "builtin" in e or "method" in e:
            if "splat" in e:
                raise MalformedNodeError(
                    "splat is a leaf-call form (D-041) — the builtin/"
                    "method call forms carry no kwargs"
                )
        if "builtin" in e:
            name = e["builtin"]
            args_spec = _field(e, "args")
            if not isinstance(name, str) or not isinstance(args_spec, list):
                raise MalformedNodeError("builtin call needs name + args")
            return _run_builtin(name, [ev(a) for a in args_spec])
        if "method" in e:
            name = e["method"]
            args_spec = _field(e, "args")
            if not isinstance(name, str) or not isinstance(args_spec, list):
                raise MalformedNodeError("method call needs name + args")
            receiver = ev(_field(e, "obj"))
            return _run_method(name, receiver, [ev(a) for a in args_spec])
        leaf = _field(e, "leaf")
        if not isinstance(leaf, dict) or "kind" not in leaf:
            raise MalformedNodeError(
                "Call without a typed leaf — every call resolves to a declared leaf (the leaf rule)"
            )
        kind = leaf["kind"]
        kwargs_spec = e.get("kwargs", {})
        if not isinstance(kwargs_spec, dict):
            raise MalformedNodeError("Call kwargs must be an object")
        # v0.4 record-splat (D-041): expanded keys first (the record's
        # order IS the declared field order by construction), then the
        # explicit kwargs in authored order. Collisions and non-record
        # splats are compile refusals; reaching them here is malformed
        # input, never a runtime fallback.
        splat = e.get("splat")
        kw: dict = {}
        if splat is not None:
            if not isinstance(splat, str) or not splat:
                raise MalformedNodeError("splat must be a simple name")
            if record_name is None or splat != record_name:
                raise MalformedNodeError(
                    f"splat {splat!r} resolves to no declared input record "
                    "(D-041; PIR-E-NODE-002 at compile — splat what the "
                    "signature declares, never what the model decides)"
                )
            kw.update(env[splat])
        for k, v in kwargs_spec.items():
            if splat is not None and k in kw:
                raise MalformedNodeError(
                    f"splat and explicit kwarg both name {k!r} (D-041; "
                    "PIR-E-NODE-003 at compile)"
                )
            kw[k] = ev(v)
        if kind in ("predict", "module"):
            ref = leaf.get("ref")
            if not isinstance(ref, str) or not ref:
                raise MalformedNodeError(f"{kind} leaf without a ref")
            child = f"{path}.{ref}" if path else ref
            if kind == "predict":
                return leaves.predict(child, kw)
            return run_forward(
                forwards,
                child,
                kw,
                leaves,
                normalize_legacy=normalize_legacy,
                log_sink=log_sink,
                _depth=_depth + 1,
            )
        if kind == "tool":
            has_name, has_ref = "name" in e, "ref" in leaf
            if has_name == has_ref:
                raise MalformedNodeError(
                    "tool leaf needs exactly one of a static ref or a dynamic name expression (SEM-8)"
                )
            if has_name:
                name = ev(e["name"])
                if not isinstance(name, str):
                    raise InterpreterTypeError(f"dynamic tool name must be a string, got {type(name).__name__}")
            else:
                name = leaf["ref"]
                if not isinstance(name, str) or not name:
                    raise MalformedNodeError("tool leaf ref must be a string")
            return leaves.tool(name, kw)
        if kind == "interpreter":
            if "ref" in leaf:
                ref = leaf["ref"]
                if not isinstance(ref, str) or not ref:
                    raise MalformedNodeError("interpreter leaf ref must be a string")
            elif normalize_legacy:
                ref = "main"
            else:
                raise MalformedNodeError(
                    "interpreter leaf without ref (pre-D-029 legacy form; pass normalize_legacy=True to accept it)"
                )
            if set(kw) != {"code"}:
                raise MalformedNodeError("interpreter call takes exactly one kwarg: code")
            code = kw["code"]
            if not isinstance(code, str):
                raise InterpreterTypeError(f"interpreter code must be a string, got {type(code).__name__}")
            return leaves.interpreter(ref, code)
        raise MalformedNodeError(f"unknown leaf kind {kind!r}")

    def guard_record_target(target: str, what: str) -> None:
        """The read-only record rule's runtime backstop (D-041): a direct
        bind/mutate of the record name is compile-refused (NODE-004), so
        seeing one here is malformed input, never a runtime rebinding."""
        if record_name is not None and target == record_name:
            raise MalformedNodeError(
                f"{what} targets the read-only input record "
                f"{target!r} (D-041; PIR-E-NODE-004 at compile)"
            )

    def ex(stmts: list) -> None:
        for s in stmts:
            s = _node(s, "statement")
            k = s["node"]
            if k == "Assign":
                target = _field(s, "target")
                if not isinstance(target, str) or not target:
                    raise MalformedNodeError("Assign target must be a simple name (SEM-5)")
                guard_record_target(target, "Assign")
                env[target] = ev(_field(s, "value"))
            elif k == "If":
                take = require_bool(ev(_field(s, "test")), "If test")
                if take:
                    ex(_stmt_list(s, "body"))
                else:
                    orelse = s.get("orelse", [])
                    if not isinstance(orelse, list):
                        raise MalformedNodeError("If orelse must be a statement list")
                    ex(orelse)
            elif k == "For":
                target = _field(s, "target")
                if not isinstance(target, str) or not target:
                    raise MalformedNodeError("For target must be a simple name")
                guard_record_target(target, "For target")
                has_range, has_iter = "range" in s, "iter" in s
                if has_range == has_iter:
                    raise MalformedNodeError(
                        "For takes exactly one of range (literal) or iter (list expression) (SEM-2 as amended v0.2)"
                    )
                if has_range:
                    rng = s["range"]
                    if type(rng) is not int or rng < 0:
                        raise MalformedNodeError("For range must be a non-negative int literal (SEM-2)")
                    sequence: list = list(range(rng))
                else:
                    iterated = ev(s["iter"])
                    if not isinstance(iterated, list):
                        raise InterpreterTypeError(
                            f"For iter must be a list, got {type(iterated).__name__} (SEM-2 as amended v0.2)"
                        )
                    sequence = list(iterated)
                body = _stmt_list(s, "body")
                for item in sequence:
                    env[target] = item
                    try:
                        ex(body)
                    except _BreakSignal:
                        break
                    except _ContinueSignal:
                        continue
            elif k == "While":
                body = _stmt_list(s, "body")
                iters = 0
                while True:
                    if not require_bool(ev(_field(s, "test")), "While test"):
                        break
                    iters += 1
                    if iters > WHILE_CAP:
                        raise LoopCapError(f"While exceeded the {WHILE_CAP}-iteration cap (SEM-6)")
                    try:
                        ex(body)
                    except _BreakSignal:
                        break
                    except _ContinueSignal:
                        continue
            elif k == "Break":
                raise _BreakSignal()
            elif k == "Continue":
                raise _ContinueSignal()
            elif k == "Try":
                handlers = _field(s, "handlers")
                if not isinstance(handlers, list):
                    raise MalformedNodeError("Try handlers must be a list")
                try:
                    ex(_stmt_list(s, "body"))
                except CatchableError as err:
                    chosen = None
                    for h in handlers:
                        htype = h.get("type") if isinstance(h, dict) else None
                        if htype not in HANDLER_NAMES:
                            raise MalformedNodeError(
                                f"unknown handler type {htype!r} — legal names are {sorted(HANDLER_NAMES)} (SEM-3)"
                            )
                        if chosen is None and handler_matches(err, htype):
                            chosen = h
                    if chosen is None:
                        raise
                    ex(_stmt_list(chosen, "body"))
            elif k == "Raise":
                exc_name = _field(s, "exc")
                cls = RAISEABLE.get(exc_name) if isinstance(exc_name, str) else None
                if cls is None:
                    raise MalformedNodeError(
                        f"Raise of unknown error type {exc_name!r} — raise-able names are {sorted(RAISEABLE)} (SEM-3)"
                    )
                message = s.get("message", "")
                if not isinstance(message, str):
                    raise MalformedNodeError("Raise message must be a string")
                raise cls(message)
            elif k == "Return":
                raise _ReturnSignal(ev(_field(s, "value")))
            elif k == "Pass":
                pass
            elif k == "Log":
                parts = _field(s, "parts")
                if not isinstance(parts, list) or not parts:
                    raise MalformedNodeError("Log needs a non-empty parts list")
                rendered = "".join(_render_part(ev(p), "Log") for p in parts)
                if log_sink is not None:
                    log_sink.append(rendered)
            elif k == "SetIndex":
                target = _field(s, "target")
                if not isinstance(target, str) or not target:
                    raise MalformedNodeError("SetIndex target must be a simple name")
                guard_record_target(target, "SetIndex")
                if target not in env:
                    raise InterpreterKeyError(f"unbound name {target!r} (SEM-7)")
                container = env[target]
                set_key = ev(_field(s, "key"))
                value = ev(_field(s, "value"))
                if isinstance(container, _Record):
                    # Reached only through an alias — the direct spelling
                    # is compile-refused (guard above). The record is
                    # immutable at execution (D-041).
                    raise InterpreterTypeError(
                        "SetIndex into the input record — the record is "
                        "read-only (D-041), through every alias"
                    )
                if isinstance(container, dict):
                    if not isinstance(set_key, str):
                        raise InterpreterTypeError("SetIndex into a dict needs a string key")
                    container[set_key] = value
                elif isinstance(container, list):
                    if type(set_key) is bool or not isinstance(set_key, int):
                        raise InterpreterTypeError("SetIndex into a list needs an int key")
                    if not 0 <= set_key < len(container):
                        raise InterpreterKeyError(f"SetIndex index {set_key} out of range (existing index only — v0.2)")
                    container[set_key] = value
                else:
                    raise InterpreterTypeError(f"SetIndex into {type(container).__name__} (v0.2)")
            elif k == "AssignTuple":
                targets = _field(s, "targets")
                if (
                    not isinstance(targets, list)
                    or len(targets) < 2
                    or not all(isinstance(t, str) and t for t in targets)
                ):
                    raise MalformedNodeError("AssignTuple targets must be 2+ simple names")
                for t in targets:
                    guard_record_target(t, "AssignTuple target")
                value = ev(_field(s, "value"))
                if not isinstance(value, list):
                    raise InterpreterTypeError(f"AssignTuple value must be a list, got {type(value).__name__} (v0.2)")
                if len(value) != len(targets):
                    raise InterpreterTypeError(f"AssignTuple: {len(targets)} targets vs {len(value)} elements (v0.2)")
                for t, v in zip(targets, value, strict=False):
                    env[t] = v
            else:
                raise MalformedNodeError(f"<{k}> is not in the statement whitelist")

    try:
        ex(top)
    except _ReturnSignal as ret:
        result = ret.value
    except (_BreakSignal, _ContinueSignal):
        raise MalformedNodeError(f"forward[{key}]: break/continue outside a loop") from None
    else:
        raise MalformedNodeError(f"forward[{key}] fell off the end without Return")
    if not isinstance(result, dict):
        raise InterpreterTypeError(
            f"forward[{key}] must return a Prediction record (a dict), got {type(result).__name__} (SEM-7)"
        )
    return result
