"""ChainOfThought: a Predict leaf with the reasoning role engaged.

There is no wrapper forward and no separate module node — ChainOfThought
IS a Predict whose signature declares a reasoning-role output field. HOW
the model reasons is the adapter's reasoning strategy: the default
`"auto"` binding resolves to native thinking when the bound LM declares
`native_reasoning`, else to `prefix_cot` for instruct models, else to
plain conduct. The program stays the same across all three; only the
strategy (data on the adapter entry) changes.
"""

from __future__ import annotations

from typing import Any

from dspy.modules.predict import Predict, _default_adapter
from dspy.signatures.field import OutputField
from dspy.signatures.signature import ensure_signature

__all__ = ["ChainOfThought"]


class ChainOfThought(Predict):
    """Predict with a reasoning-role output field prepended.

    Args:
        signature: A `dspy.Signature` class or a `"inputs -> outputs"`
            string. If it already declares a `reasoning` output, that
            field is the contract and no field is added.
        strategies: Adapter strategy bindings to apply, e.g.
            `{"reasoning": "prefix_cot"}`. Omitted = the adapter's own
            block (the default adapter binds `reasoning: "auto"`).
        lm: Per-predictor LM binding, as for `Predict`.
        adapter: Per-predictor adapter binding, as for `Predict`.
        **config: Per-call LM request kwargs, as for `Predict`.

    Examples:
        ```python
        cot = dspy.ChainOfThought("question -> answer")
        cot.set_lm(dspy.LM("openai/gpt-4o-mini"))
        prediction = cot(question="What is 17 * 4?")
        print(prediction.reasoning)
        print(prediction.answer)
        ```
    """

    def __init__(self, signature, *, strategies=None, lm=None, adapter=None, **config: Any):
        signature = ensure_signature(signature)
        if "reasoning" not in signature.output_fields:
            signature = signature.prepend("reasoning", OutputField(role="reasoning"), type_=str)
        if strategies:
            if isinstance(adapter, str):
                raise ValueError(
                    "ChainOfThought(strategies=...) needs a live adapter to derive from; a binding-table "
                    "name resolves at call time, after derivation. Bind the live adapter (or none, for "
                    "the default) instead."
                )
            base = adapter if adapter is not None else _default_adapter()
            adapter = base.with_strategies(**strategies)
        super().__init__(signature, lm=lm, adapter=adapter, **config)
