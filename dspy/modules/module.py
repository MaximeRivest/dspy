"""Module: the program IS the IR.

A Module is authored exactly as in classic dspy — declare children in
`__init__`, write a `forward` — but calling it does not run your Python.
`program(**inputs)` compiles `forward` into the ProgramIR node set
(cached), materializes the IR against the live bindings, and executes
the engine interpreter. What runs is exactly what `program.save(dir)`
writes; there is no second semantics.

The native Python path survives only as `forward_native(**inputs)` — the
equivalence test's control arm, never the default door.

A forward that steps outside the representable node set raises the
compiler's teaching error at the first call (or at an explicit
`compile_ir()`), naming the construct, the line, and the supported
alternative.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["Module"]


def _hashable(value: Any) -> Any:
    """Coerce a declared-literal value into a hashable fingerprint part."""
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


class Module:
    """Base class for IR-native dspy programs.

    Subclass it, declare children (`Predict`, sub-modules, tools,
    interpreters) as attributes in `__init__`, and write a `forward`
    inside the representable subset. Calling the instance executes the
    compiled IR through the engine.

    Examples:
        ```python
        class QA(dspy.Module):
            def __init__(self):
                self.answer = dspy.Predict("question -> answer")

            def forward(self, question):
                result = self.answer(question=question)
                return result

        program = QA()
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
        prediction = program(question="Why is the sky blue?")
        ```
    """

    _ir_cache: tuple[Any, Any] | None = None

    def forward(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} declares no forward. A Module subclass defines "
            "forward(self, <inputs>) inside the representable node set; the engine "
            "compiles and runs it."
        )

    # ------------------------------------------------------------------
    # Execution: the engine path is the default; native is the control arm
    # ------------------------------------------------------------------

    def __call__(self, **inputs: Any):
        """Compile (cached), materialize against live bindings, interpret."""
        return self._executable()(**inputs)

    def forward_native(self, **inputs: Any):
        """Run the authored Python forward directly.

        This is the equivalence test's control arm only. It exists to
        prove the engine path faithful, not as an execution mode.
        """
        return self.forward(**inputs)

    def _executable(self):
        from dspy.programir._dspy import compile_with_live
        from dspy.programir.engine.materialize import materialize

        fingerprint = self._fingerprint()
        cached = self._ir_cache
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        ir, live = compile_with_live(self)
        executable = materialize(ir, bindings=live)
        self._ir_cache = (fingerprint, executable)
        return executable

    def _fingerprint(self) -> tuple:
        """A cheap identity of everything the compiled IR depends on.

        Covers each predictor's resolved bindings, demos, instructions,
        and config, plus every module's declared-literal values
        (`ir_literals` — edits to a baked loop cap recompile). Structural
        edits after `__init__` (rebinding a child attribute) are not
        fingerprinted — call `invalidate_ir()` after those.
        """
        parts = []
        for path, predictor in self.named_predictors():
            parts.append(
                (
                    path,
                    id(predictor.resolve_lm()),
                    id(predictor.resolve_adapter()),
                    tuple(id(demo) for demo in predictor.demos),
                    predictor.signature.instructions,
                    repr(sorted(predictor.config.items(), key=repr)),
                )
            )
        for path, module in self._named_modules():
            names = getattr(type(module), "ir_literals", ())
            if names:
                # A declared literal may be a scalar or a scalar list;
                # coerce list values to tuples so the fingerprint stays
                # hashable (edits to a baked output-name list recompile).
                parts.append((path, tuple((name, _hashable(getattr(module, name, None))) for name in names)))
        return tuple(parts)

    def _named_modules(self) -> list[tuple[str, Module]]:
        """Every module in the tree (self included) as `(path, module)`."""
        found: list[tuple[str, Module]] = [("self", self)]

        def walk(module: Module, prefix: str) -> None:
            for name, child in module.__dict__.items():
                if not (name.isidentifier() and name.isascii()):
                    continue
                if isinstance(child, Module):
                    path = name if not prefix else f"{prefix}.{name}"
                    found.append((path, child))
                    walk(child, path)

        walk(self, "")
        return found

    def invalidate_ir(self) -> None:
        """Drop the cached compiled program; the next call recompiles."""
        self._ir_cache = None

    # ------------------------------------------------------------------
    # Compilation and the one save path
    # ------------------------------------------------------------------

    def compile_ir(self):
        """Compile this program into its ProgramIR value.

        Returns:
            The detached `ProgramIR` (manifest + sidecars).

        Raises:
            ProgramIRRefusal: When `forward` steps outside the
                representable node set — the teaching error names the
                construct and the alternative.
        """
        from dspy.programir._dspy import compile_program

        return compile_program(self)

    def to_manifest(self) -> dict[str, Any]:
        """Compile and return the manifest dict (the tools' input form)."""
        return self.compile_ir().to_manifest()

    def save(self, path: str | os.PathLike[str]):
        """Write this program as a ProgramIR artifact directory.

        The one save path: `compile_ir()` + the artifact writer. Load it
        back with `dspy.load(path, bindings=...)`.

        Args:
            path: Destination directory; must not already exist.

        Returns:
            The finalized `ProgramIR`, including generated lockfiles.
        """
        from dspy.programir.export import export

        return export(self, path)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def named_predictors(self) -> list[tuple[str, Any]]:
        """Every `Predict` leaf as `(dotted_path, predictor)` pairs."""
        from dspy.modules.predict import Predict

        if isinstance(self, Predict):
            return [("self", self)]
        found: list[tuple[str, Any]] = []

        def walk(module: Module, prefix: str) -> None:
            for name, child in module.__dict__.items():
                if not (name.isidentifier() and name.isascii()):
                    continue
                path = name if not prefix else f"{prefix}.{name}"
                if isinstance(child, Predict):
                    found.append((path, child))
                elif isinstance(child, Module):
                    walk(child, path)

        walk(self, "")
        return found

    # ------------------------------------------------------------------
    # Observability: pure views over the compiled manifest
    # ------------------------------------------------------------------

    def explain(self) -> str:
        """The static explain view (View 1) of the compiled program."""
        from dspy.programir import explain_view

        return explain_view.build_text(self.to_manifest())

    def lint(self) -> list:
        """Static lint findings over the compiled program's manifest."""
        from dspy.programir.tools.lint import lint

        return lint(self.to_manifest())

    def cost(self) -> dict[str, Any]:
        """Static call/token bounds per program invocation."""
        from dspy.programir.tools.cost import estimate

        return estimate(self.to_manifest())

    def __repr__(self) -> str:
        predictors = ", ".join(path for path, _ in self.named_predictors())
        return f"{type(self).__name__}(predictors=[{predictors}])"
