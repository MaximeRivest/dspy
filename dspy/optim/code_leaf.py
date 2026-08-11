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

__all__ = ["ADMITTED_IMPORTS", "AdmittedCode", "admit_tool_source"]

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
) -> AdmittedCode:
    """Admit optimizer-authored source as a tool leaf, or refuse loudly.

    Args:
        name: The tool leaf's pool name (a validated identifier).
        python_source: The proposal's code field — DATA, never exec'd
            here beyond the one admitted def.
        signature: The replaced predict's `dspy.Signature`; the io-contract
            derives from its input/output fields (the unified-leaf law).
        partial: `False` for `replace_predict_with_code` (return `dict`);
            `True` for the partial op (return `dict | None`; None declines
            to the LM).

    Returns:
        An `AdmittedCode` (leaf entry + sidecar + live callable).

    Raises:
        ValueError: On any admission failure — non-parseable source, more
            than one def, a decorator, a closure or global read, a
            disallowed import, a non-empty `# deps:`, or an io-contract
            mismatch against the signature. The message teaches the fix.
    """
    if not isinstance(python_source, str) or not python_source.strip():
        raise ValueError(f"tool {name!r}: python_source must be a non-empty string of one function definition")

    subject = f"tool {name!r}"
    # Step 1: exactly one undecorated function def (reuse the leaves check,
    # but source-first — the optimizer holds text, not a live function).
    tree = ast.parse(python_source)
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

    # Step 3: the import allowlist + empty deps (risk 2).
    _refuse_disallowed_imports(tree, subject=subject)
    deps = parse_deps(python_source)
    if deps:
        raise ValueError(
            f"ProgramIR {subject} declares `# deps: {', '.join(deps)}` — optimizer-authored code carries no "
            "third-party dependencies; use the stdlib allowlist only"
        )

    # Step 4: the io-contract from the replaced signature.
    _check_io_contract(definition, signature, subject=subject, partial=partial)

    # Only now define the function (the single def in a clean namespace via
    # linecache — defines the name, runs no body) and extract the pool
    # entry the leaves.py way. `extract_tool` re-runs steps 1-2 as a
    # backstop; the io-contract and import rules are ours.
    function = load_generated(python_source, tag=f"flex-code-{name}", name=definition.name)
    # Stamp provenance on the function itself (PIR-014) so `extract_tool`
    # carries `authored_by: "optimizer"` into the entry at EVERY recompile,
    # not only this one — the compiler re-extracts from the live function.
    function._dspy_authored_by = "optimizer"
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


def _refuse_disallowed_imports(tree: ast.Module, *, subject: str) -> None:
    """Refuse any import outside the pure-stdlib allowlist (risk 2)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        disallowed = sorted({module for module in names if module not in ADMITTED_IMPORTS})
        if disallowed:
            raise ValueError(
                f"ProgramIR {subject} imports {disallowed}, which is outside the optimizer allowlist "
                f"{sorted(ADMITTED_IMPORTS)}; network/process/reflection modules refuse (a code leaf may not "
                "hide an LM call or a side effect)"
            )


def _check_io_contract(definition: ast.FunctionDef, signature: Any, *, subject: str, partial: bool) -> None:
    """Refuse unless params == input fields (typed) and return matches the op.

    The unified-leaf law: the code leaf's interface IS the replaced
    predict's signature. Parameters are exactly the signature's input
    fields, each type-hinted; the return annotation is `dict` (full
    replace) or `dict | None` (partial — None declines to the LM).
    """
    inputs = list(signature.input_fields)
    args = definition.args
    if args.posonlyargs or args.vararg or args.kwarg:
        raise ValueError(
            f"ProgramIR {subject} must take exactly the signature's input fields {inputs} as plain "
            "parameters — no positional-only, *args, or **kwargs"
        )
    names = [argument.arg for argument in (*args.args, *args.kwonlyargs)]
    if names != inputs:
        raise ValueError(
            f"ProgramIR {subject} parameters {names} must be exactly the replaced signature's input "
            f"fields {inputs}, in order (the unified-leaf law: the code's interface IS the signature)"
        )
    untyped = [argument.arg for argument in (*args.args, *args.kwonlyargs) if argument.annotation is None]
    if untyped:
        raise ValueError(f"ProgramIR {subject} parameters {untyped} need type hints (every leaf parameter is typed)")
    if definition.returns is None:
        expected = "dict | None" if partial else "dict"
        raise ValueError(
            f"ProgramIR {subject} needs a return annotation of `{expected}` — the output record "
            f"(one key per signature output field: {list(signature.output_fields)})"
        )
    got = ast.unparse(definition.returns).replace(" ", "")
    allowed = {"dict|None", "Optional[dict]", "None|dict"} if partial else {"dict"}
    if got not in allowed:
        expected = "dict | None" if partial else "dict"
        raise ValueError(
            f"ProgramIR {subject} returns `{ast.unparse(definition.returns)}` but this op requires "
            f"`{expected}` (the output record{'; return None to decline to the LM' if partial else ''})"
        )
