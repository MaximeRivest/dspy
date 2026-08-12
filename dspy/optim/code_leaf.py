"""Admit optimizer-authored Python as a declared tool leaf.

FlexIR v2's code ops (``replace_predict_with_code``,
``replace_predict_with_code_partial``) carry ONE field of code —
``python_source`` — and nothing else. That source is DATA to this module:
the optimizer never execs, formats, or splices it. `admit_tool_source`
runs the section-3 admission chain and, only on success, defines the one
function in a clean namespace (the `to_function` linecache convention, so
the artifact carries reviewable, reparseable source). A source that fails
ANY step refuses with a teaching ValueError — the loud refusal FlexIR
records in its ledger and shows the next reflection call.

The chain, in order (a refusal at any step stops the rest):

1. `ast.parse` succeeds and the module body is EXACTLY one undecorated
   `def` (the `leaves._source` check, reused).
2. Self-containment: no closures possible (source-born), no global reads
   (`leaves._check_self_contained` on the parsed function, reused — not
   forked).
3. Import allowlist: a closed pure-stdlib set. Any network/process/
   reflection module (socket, http, urllib, requests, subprocess, os,
   sys, importlib, ctypes, ...) refuses, and `# deps:` must be empty —
   optimizer-authored code declares no third-party dependencies
   (risk 2: a code leaf may not hide an LM call over HTTP).
4. io-contract from the REPLACED predict's signature: the parameter
   names are exactly the signature's input fields (type hints required),
   and the return annotation matches the op — `dict` for the full
   replace, `dict | None` for the partial (None declines to the LM).

Injection-shaped source (an `os.system(...)`-looking string) is admitted
only as a tool leaf's carried source, and NEVER executed by the optimizer
— step 3 refuses `os` at admission, and even a source that passed would
run only through the engine at score time, inside the engine's leaf
machinery, never in the applier.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Callable

from dspy.modules._generate import load_generated
from dspy.programir.leaves import AuthoredLeaf, _check_self_contained, extract_tool, parse_deps

__all__ = ["ADMITTED_IMPORTS", "DISALLOWED_BUILTINS", "AdmittedCode", "admit_tool_source"]

#: Builtin names that reach import, code-exec, filesystem, or reflection
#: machinery WITHOUT an `import` statement — the exact doors the module
#: import allowlist (`_refuse_disallowed_imports`) cannot see, and that
#: `leaves._check_self_contained` waves through because every one lives in
#: `dir(builtins)`. `__import__` imports any module (a hidden HTTP LM call
#: over `urllib`/`http`); `eval`/`exec`/`compile` run arbitrary source;
#: `open` reads/writes files; `globals`/`vars`/`locals` hand out the module
#: namespace; `input`/`breakpoint` block or drop to a debugger. A code leaf
#: that names any of these is refused at admission (risk 2: a leaf may not
#: hide an LM call or a side effect). AST filtering of Python is not a hard
#: security boundary, but it closes the builtin-mediated escapes the
#: import-node walk structurally cannot.
DISALLOWED_BUILTINS = frozenset(
    {
        "__import__",
        "eval",
        "exec",
        "compile",
        "open",
        "globals",
        "vars",
        "locals",
        "input",
        "breakpoint",
    }
)

#: Dunder attributes that walk from an ordinary value back to import/exec
#: machinery — the classic sandbox-escape chain
#: (`().__class__.__bases__[0].__subclasses__()`, `f.__globals__`,
#: `x.__builtins__`). A dotted access to any of these is refused so
#: `getattr`-free attribute chains cannot reach what the name denylist
#: bans. `getattr`/`setattr`/`delattr` are name-denied above via
#: `_reflection_call_names` so a dynamic form cannot slip past.
DISALLOWED_ATTRIBUTES = frozenset(
    {
        "__globals__",
        "__builtins__",
        "__class__",
        "__bases__",
        "__base__",
        "__subclasses__",
        "__mro__",
        "__code__",
        "__import__",
        "__loader__",
        "__dict__",
        "__getattribute__",
    }
)

#: Builtins that dispatch on a NAME at runtime, so a code leaf could reach
#: a denied builtin or a dunder indirectly (`getattr(x, "__globals__")`,
#: `__builtins__["eval"]`). Denied at admission alongside the doors above.
_REFLECTION_CALL_NAMES = frozenset({"getattr", "setattr", "delattr"})

#: The closed stdlib import allowlist for optimizer-authored code. Pure,
#: deterministic, offline modules only — no network, process, filesystem,
#: reflection, or foreign-code doors. A code leaf that needs anything
#: outside this set is either impure (belongs behind a placement rung a
#: receiver chooses, not an optimizer edit) or is faking cheapness.
ADMITTED_IMPORTS = frozenset(
    {
        "re",
        "json",
        "math",
        "string",
        "textwrap",
        "datetime",
        "collections",
        "itertools",
        "functools",
        "unicodedata",
        "decimal",
        "fractions",
        "statistics",
        "heapq",
        "bisect",
    }
)


@dataclass(frozen=True)
class AdmittedCode:
    """One admitted code leaf: its pool entry, sidecar, and live callable.

    Attributes:
        leaf: The `AuthoredLeaf` (pool entry + source sidecar) produced by
            `extract_tool`, carrying `authored_by: "optimizer"` provenance.
        function: The live function, defined via the `to_function`
            linecache convention (the single def, nothing else executed).
    """

    leaf: AuthoredLeaf
    function: Callable[..., Any]


def admit_tool_source(
    name: str,
    python_source: str,
    signature: Any,
    *,
    partial: bool,
    extra_imports: frozenset[str] | None = None,
    code_trust: str = "isolated",
    allowed_deps: frozenset[str] | None = None,
    session: bool = False,
) -> AdmittedCode:
    """Admit optimizer-authored source as a tool leaf, or refuse loudly.

    Args:
        name: The tool leaf's pool name (a validated identifier).
        python_source: The proposal's code field — DATA, never exec'd
            here beyond the one admitted def.
        signature: The replaced predict's `dspy.Signature`; the io-contract
            derives from its input/output fields (the unified-leaf law).
            `None` admits a FREE tool (FlexIR v3 `add_tool`): the
            io-contract then derives from the source's own hints — every
            parameter type-hinted, return annotation `dict` — and
            `partial` must be False (a free tool has no LM to decline to).
        partial: `False` for `replace_predict_with_code` (return `dict`);
            `True` for the partial op (return `dict | None`; None declines
            to the LM).
        extra_imports: Additional module names admitted ALONGSIDE
            `ADMITTED_IMPORTS` for this call only. Widening the allowlist
            widens what generated code can do — the caller (the optimizing
            user, via `FlexIR(extra_imports=...)`) owns that choice. The
            denied builtins and dunder walks stay denied regardless; the
            module-level frozenset is never mutated.
        code_trust: `"isolated"` (the default) keeps the trust pairing
            rule: the leaf carries the isolation-required rung and fails
            closed at load unless the receiver grants it.
            `"in_process"` stamps the function so the leaf gets the
            in-process placement — the optimizing user's own loop and
            saved artifact run WITHOUT a grant ceremony. This is NOT a
            sandbox: in-process optimizer-authored code runs with full
            ambient authority; the caller makes that choice for their own
            loop. `authored_by: "optimizer"` survives in the entry either
            way, so a receiver can audit or re-place the leaf.
        allowed_deps: Package names a `# deps:` line may declare. The
            default (None) keeps the hard rule: any dep refuses. A dep
            outside the set refuses naming the allowed set. Admitted deps
            flow into the pool entry's `deps[]` (via `parse_deps`, as for
            any authored leaf), so the artifact's environment union picks
            them up. A dep name does NOT admit its IMPORT name — the
            import allowlist is governed by `extra_imports` alone
            (beautifulsoup4 imports as bs4; grant "bs4" there too).

    Returns:
        An `AdmittedCode` (leaf entry + sidecar + live callable).

    Raises:
        ValueError: On any admission failure — non-parseable source, more
            than one def, a decorator, a closure or global read, a
            disallowed import, a `# deps:` package outside `allowed_deps`
            (or any dep when it is None), or an io-contract mismatch
            against the signature. The message teaches the fix.
    """
    if not isinstance(python_source, str) or not python_source.strip():
        raise ValueError(f"tool {name!r}: python_source must be a non-empty string of one function definition")
    if code_trust not in ("isolated", "in_process"):
        raise ValueError(f"tool {name!r}: code_trust must be 'isolated' or 'in_process', got {code_trust!r}")
    if signature is None and partial:
        raise ValueError(f"tool {name!r}: a free tool (no replaced signature) has no LM to decline to; partial refuses")

    subject = f"tool {name!r}"
    # Step 1: exactly one undecorated function def (reuse the leaves check,
    # but source-first — the optimizer holds text, not a live function).
    # A parse failure is a REFUSAL, not a crash: `SyntaxError` is not a
    # `ValueError`, so it would otherwise escape the applier's
    # `except ValueError` and abort the whole compile() run on one
    # malformed reflection-LM proposal. Re-raise it as the teaching
    # `ValueError` this module's contract promises (Raises: ValueError).
    try:
        tree = ast.parse(python_source)
    except SyntaxError as error:
        raise ValueError(
            f"ProgramIR {subject} source does not parse as Python ({error.msg} at line {error.lineno}); "
            "author one valid, undecorated function definition"
        ) from error
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef) or len(tree.body) != 1:
        raise ValueError(
            f"ProgramIR {subject} source must be EXACTLY one undecorated function definition "
            "(no imports at module scope, no second def, no statements around it)"
        )
    definition = tree.body[0]
    if definition.decorator_list:
        raise ValueError(f"ProgramIR {subject} uses decorators; author the undecorated function instead")

    # Step 2: self-containment — no global reads, no closures. Source-born
    # code has no closure cell, so only the global-read arm can fire; reuse
    # the shared AST check so the rule never drifts from leaves.py.
    _check_self_contained(_SourceOnly(definition.name), python_source, subject=subject)

    # Step 3: the import allowlist + empty deps (risk 2), PLUS the builtin
    # denylist — `__import__`/`eval`/`exec`/`compile`/`open` reach import,
    # code-exec, and filesystem machinery WITHOUT an import node, so the
    # allowlist walk alone cannot catch them and `_check_self_contained`
    # admits them (they live in `dir(builtins)`). This is the only layer
    # that closes those builtin-mediated escapes.
    # Deps are validated FIRST so the import walk can teach the dep/import
    # pairing rule when the source both declares deps and imports outside
    # the allowlist (a dep name never admits its import name).
    deps = parse_deps(python_source)
    granted = allowed_deps or frozenset()
    if deps and not granted:
        raise ValueError(
            f"ProgramIR {subject} declares `# deps: {', '.join(deps)}` — optimizer-authored code carries no "
            "third-party dependencies; use the stdlib allowlist only"
        )
    disallowed_deps = sorted(set(deps) - granted)
    if disallowed_deps:
        raise ValueError(
            f"ProgramIR {subject} declares dep(s) {disallowed_deps} outside the allowed set "
            f"{sorted(granted)}; only the packages the user granted via allowed_deps may ride `# deps:`"
        )
    _refuse_disallowed_imports(tree, subject=subject, extra_imports=extra_imports, declares_deps=bool(deps))
    _refuse_builtin_escapes(definition, subject=subject)

    # Step 4: the io-contract from the replaced signature.
    _check_io_contract(definition, signature, subject=subject, partial=partial, session=session)

    # Only now define the function (the single def in a clean namespace via
    # linecache — defines the name, runs no body) and extract the pool
    # entry the leaves.py way. `extract_tool` re-runs steps 1-2 as a
    # backstop; the io-contract and import rules are ours.
    function = load_generated(python_source, tag=f"flex-code-{name}", name=definition.name)
    # Stamp provenance on the function itself (PIR-014) so `extract_tool`
    # carries `authored_by: "optimizer"` into the entry at EVERY recompile,
    # not only this one — the compiler re-extracts from the live function.
    function._dspy_authored_by = "optimizer"
    if code_trust == "in_process":
        # The optimizing user's recorded choice (see the Args note): the
        # leaf gets the in-process placement instead of the fail-closed
        # isolation rung. Provenance above survives regardless.
        function._dspy_placement_rung = "in_process"
    leaf = extract_tool(function, name=name)
    return AdmittedCode(leaf=leaf, function=function)


class _SourceOnly:
    """A stand-in the source-first checks accept: no closure, named def.

    `leaves._check_self_contained` reads `__closure__` (source-born code
    has none) and parses the source itself for the global-read walk. It
    never calls the object, so a lightweight stand-in carrying the right
    `__closure__` is enough to reuse the exact shared check.
    """

    __closure__ = None

    def __init__(self, name: str):
        self.__name__ = name


def _refuse_disallowed_imports(
    tree: ast.Module, *, subject: str, extra_imports: frozenset[str] | None = None, declares_deps: bool = False
) -> None:
    """Refuse any import outside the pure-stdlib allowlist (risk 2).

    `extra_imports` widens the allowlist for THIS call only — a per-
    optimizer-instance grant the user made explicitly. The module-level
    `ADMITTED_IMPORTS` frozenset is never mutated. An admitted `# deps:`
    package does NOT widen this list (`declares_deps` only sharpens the
    teaching message): a dep name and its import name differ
    (beautifulsoup4 imports as bs4), so the import name must be granted
    via extra_imports in its own right.
    """
    allowed = ADMITTED_IMPORTS | (extra_imports or frozenset())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        disallowed = sorted({module for module in names if module not in allowed})
        if disallowed:
            pairing = (
                "; a declared `# deps:` package does NOT admit its import name — grant the IMPORT name via "
                "FlexIR(extra_imports=...) as well (beautifulsoup4 imports as bs4)"
                if declares_deps
                else ""
            )
            raise ValueError(
                f"ProgramIR {subject} imports {disallowed}, which is outside the optimizer allowlist "
                f"{sorted(allowed)}; network/process/reflection modules refuse (a code leaf may not "
                f"hide an LM call or a side effect){pairing}"
            )


def _refuse_builtin_escapes(definition: ast.FunctionDef, *, subject: str) -> None:
    """Refuse code-exec / dynamic-import / reflection builtins (risk 2).

    Walks every `Name` load and `Attribute` access in the function body.
    Refuses a reference to any name in `DISALLOWED_BUILTINS` or
    `_REFLECTION_CALL_NAMES` (`__import__`, `eval`, `exec`, `compile`,
    `open`, `globals`, `getattr`, ...) and any dotted access to a dunder in
    `DISALLOWED_ATTRIBUTES` (`x.__globals__`, `().__class__.__bases__`).
    These are the builtin-mediated doors the module import allowlist
    (`_refuse_disallowed_imports`) and the `dir(builtins)` self-containment
    allowance (`leaves._check_self_contained`) structurally cannot see.

    A local rebinding (`def eval(...)`, `eval = ...`, a parameter named
    `open`) is NOT an escape — only a Load of a denied NAME that is not
    bound locally, and any denied ATTRIBUTE access, is refused.
    """
    local = _locally_bound_names(definition)
    banned_names = DISALLOWED_BUILTINS | _REFLECTION_CALL_NAMES
    for node in ast.walk(definition):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in banned_names and node.id not in local:
                raise ValueError(
                    f"ProgramIR {subject} uses the builtin {node.id!r}, which reaches import, code-exec, "
                    "filesystem, or reflection machinery — refused (a code leaf may not hide an LM call or a "
                    f"side effect; the denied builtins are {sorted(DISALLOWED_BUILTINS | _REFLECTION_CALL_NAMES)})"
                )
        elif isinstance(node, ast.Attribute) and node.attr in DISALLOWED_ATTRIBUTES:
            raise ValueError(
                f"ProgramIR {subject} accesses the reflection attribute {'.' + node.attr!r} — refused; a code "
                "leaf may not walk from a value back to import or exec machinery"
            )


def _locally_bound_names(definition: ast.FunctionDef) -> set[str]:
    """Names bound inside the def (params, assignments, nested defs, imports).

    A Load of one of these is the leaf's OWN name, not the dangerous
    builtin, so it must not trip the denylist. Mirrors the local-name
    collection `leaves._check_self_contained` does for the global-read walk.
    """
    local: set[str] = set()
    for item in ast.walk(definition):
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(item, ast.Lambda):
                local.add(item.name)
            inner = item.args
            local.update(argument.arg for argument in (*inner.posonlyargs, *inner.args, *inner.kwonlyargs))
            if inner.vararg:
                local.add(inner.vararg.arg)
            if inner.kwarg:
                local.add(inner.kwarg.arg)
        elif isinstance(item, ast.ExceptHandler) and item.name:
            local.add(item.name)
        elif isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
            local.add(item.id)
        elif isinstance(item, (ast.Import, ast.ImportFrom)):
            for alias in item.names:
                local.add(alias.asname or alias.name.split(".")[0])
    return local


def _check_io_contract(
    definition: ast.FunctionDef, signature: Any, *, subject: str, partial: bool, session: bool = False
) -> None:
    """Refuse unless params == input fields (typed) and return matches the op.

    The unified-leaf law: the code leaf's interface IS the replaced
    predict's signature. Parameters are exactly the signature's input
    fields, each type-hinted; the return annotation is `dict` (full
    replace) or `dict | None` (partial — None declines to the LM).

    With `signature=None` (a FREE tool, FlexIR v3 `add_tool`), the
    io-contract derives from the source's OWN hints: every parameter must
    carry a type hint, and the return annotation must be `dict`.

    With `session=True` (PIR-021 session leaf), the FIRST parameter is the
    grant bridge — the leaf's only capability surface. It is exempt from
    the typed-input rule (it is not a data input); the rest still must be
    typed, and the return still must be `dict`.
    """
    inputs = list(signature.input_fields) if signature is not None else None
    args = definition.args
    if args.posonlyargs or args.vararg or args.kwarg:
        wanted = f"exactly the signature's input fields {inputs}" if inputs is not None else "plain typed parameters"
        raise ValueError(
            f"ProgramIR {subject} must take {wanted} as plain parameters — no positional-only, *args, or **kwargs"
        )
    all_positional = [*args.args, *args.kwonlyargs]
    # Every parameter — the bridge included — must be typed (extract_tool
    # enforces this too; a session bridge types as `object`). The bridge
    # is exempt only from the input-FIELD equality rule below, not typing.
    untyped = [argument.arg for argument in all_positional if argument.annotation is None]
    if untyped:
        raise ValueError(f"ProgramIR {subject} parameters {untyped} need type hints (every leaf parameter is typed)")
    positional = all_positional
    if session:
        if not positional:
            raise ValueError(
                f"ProgramIR {subject} is a session leaf and must take the grant bridge as its FIRST parameter "
                "(a session leaf reaches its grants through the bridge, never ambient pools)"
            )
        positional = positional[1:]  # exempt the bridge from the input-field rule
    names = [argument.arg for argument in positional]
    if inputs is not None and names != inputs:
        raise ValueError(
            f"ProgramIR {subject} parameters {names} must be exactly the replaced signature's input "
            f"fields {inputs}, in order (the unified-leaf law: the code's interface IS the signature)"
        )
    if definition.returns is None:
        expected = "dict | None" if partial else "dict"
        record = (
            f"(one key per signature output field: {list(signature.output_fields)})"
            if signature is not None
            else "(the tool's output record)"
        )
        raise ValueError(f"ProgramIR {subject} needs a return annotation of `{expected}` — the output record {record}")
    got = ast.unparse(definition.returns).replace(" ", "")
    allowed = {"dict|None", "Optional[dict]", "None|dict"} if partial else {"dict"}
    if got not in allowed:
        expected = "dict | None" if partial else "dict"
        raise ValueError(
            f"ProgramIR {subject} returns `{ast.unparse(definition.returns)}` but this op requires "
            f"`{expected}` (the output record{'; return None to decline to the LM' if partial else ''})"
        )
