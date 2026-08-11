"""Predict: the leaf where program meets model.

One exchange, four moves: resolve the LM and adapter from bindings,
format, call, parse. Bindings are explicit — a Predict holds its own
(`lm=` / `adapter=` constructor kwargs, or `set_lm()` / `set_adapter()`),
else the default binding table written by `dspy.configure(...)` answers.
A string binding is a name looked up in that table at call time. There
is no ambient state anywhere on this path.

Unlike composite modules, calling a Predict runs this exchange directly:
the leaf IS its own machinery, on the native path and under the engine
interpreter alike.
"""

from __future__ import annotations

from typing import Any

from dspy.core.prediction import Prediction
from dspy.lm.bindings import BINDINGS, BindingError, resolve
from dspy.modules.module import Module
from dspy.signatures.signature import ensure_signature

__all__ = ["Predict"]

#: One shared default adapter: stable identity keeps the compile cache and
#: the adapter pool honest when nothing is bound.
_DEFAULT_ADAPTER = None


def _default_adapter():
    global _DEFAULT_ADAPTER
    if _DEFAULT_ADAPTER is None:
        from dspy.adapters.presets import ChatAdapter

        _DEFAULT_ADAPTER = ChatAdapter()
    return _DEFAULT_ADAPTER


class Predict(Module):
    """The predictor leaf: one signature, one LM exchange per call.

    Args:
        signature: A `dspy.Signature` class or a `"inputs -> outputs"`
            string.
        lm: Per-predictor LM binding — a live LM or a binding-table name.
        adapter: Per-predictor adapter binding — a live adapter or a
            binding-table name.
        **config: Per-call LM request kwargs (`temperature=0.7`, ...),
            saved as component 3c of the IR.

    Examples:
        ```python
        answer = dspy.Predict("question -> answer")
        answer.set_lm(dspy.LM("openai/gpt-4o-mini"))
        prediction = answer(question="Why is the sky blue?")
        ```
    """

    def __init__(self, signature, *, lm=None, adapter=None, **config: Any):
        self.signature = ensure_signature(signature)
        self.demos: list = []
        self.config: dict[str, Any] = dict(config)
        self.bindings: dict[str, Any] = {}
        if lm is not None:
            self.bindings["lm"] = lm
        if adapter is not None:
            self.bindings["adapter"] = adapter

    # ------------------------------------------------------------------
    # Bindings: per-predictor overrides, else the default table
    # ------------------------------------------------------------------

    def set_lm(self, lm) -> None:
        """Bind this predictor's LM: a live LM or a binding-table name."""
        self.bindings["lm"] = lm

    def set_adapter(self, adapter) -> None:
        """Bind this predictor's adapter: a live adapter or a table name."""
        self.bindings["adapter"] = adapter

    def resolve_lm(self):
        """The live LM this predictor runs with, resolved loudly."""
        return self._resolve_binding("lm")

    def resolve_adapter(self):
        """The live adapter, defaulting to the shared ChatAdapter."""
        if self.bindings.get("adapter") is None and "adapter" not in BINDINGS:
            return _default_adapter()
        return self._resolve_binding("adapter")

    def _resolve_binding(self, name: str):
        bound = self.bindings.get(name)
        if isinstance(bound, str):
            if bound in BINDINGS:
                return BINDINGS[bound]
            raise BindingError(
                f"this predictor binds {name!r} to the name {bound!r}, but the default "
                f"binding table has no such entry. Call dspy.configure({bound}=...) or "
                "bind a live object on the predictor."
            )
        if bound is not None:
            return bound
        return resolve(name, None)

    # ------------------------------------------------------------------
    # The exchange
    # ------------------------------------------------------------------

    def forward(self, **inputs: Any) -> Prediction:
        """Run one exchange: format, call the LM, parse the completion.

        Args:
            **inputs: Values for the signature's input fields.

        Returns:
            A `Prediction` whose fields are exactly the signature's
            declared outputs; the raw completion rides in the
            `_trajectory` exhaust channel.
        """
        unknown = sorted(set(inputs) - set(self.signature.input_fields))
        if unknown:
            raise ValueError(
                f"Predict({self.signature.signature!r}) got unknown input(s) {unknown}; "
                f"declared inputs: {sorted(self.signature.input_fields)}"
            )
        lm = self.resolve_lm()
        adapter = self.resolve_adapter()
        demos = [demo.toDict() if hasattr(demo, "toDict") else dict(demo) for demo in self.demos]
        call = adapter.format(self.signature, inputs=inputs, demos=demos, lm=lm)
        outputs = lm(messages=call.messages, **{**self.config, **call.request})
        fields = adapter.parse(self.signature, outputs[0], lm=lm)
        prediction = Prediction(**fields)
        prediction._trajectory["completion"] = outputs[0]
        return prediction

    def __call__(self, **inputs: Any) -> Prediction:
        """The leaf runs its exchange directly — it IS the machinery."""
        return self.forward(**inputs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.signature.signature})"
