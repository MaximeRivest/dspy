from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        native: bool = False,
        reasoning_effort: str | int | None = None,
        **config: dict[str, Any],
    ):
        """
        A module that reasons step by step in order to predict the output of a task.

        When ``native=True`` and the LM supports it, reasoning is handled by the
        model's built-in thinking capability (e.g. extended thinking on Claude,
        reasoning tokens on o3/o4-mini) instead of being produced as a regular
        output field in the prompt. On models without native reasoning the field
        is still included in the prompt as a graceful fallback.

        Args:
            signature: The signature of the module.
            rationale_field: The field that will contain the reasoning.
            rationale_field_type: The type of the rationale field.
            native: If True, use ``dspy.Reasoning`` as the rationale field type
                so that models with native reasoning emit thinking tokens rather
                than generating reasoning through the prompt.
            reasoning_effort: Control how much the model reasons. Accepts
                ``"low"``, ``"medium"``, or ``"high"`` (mapped to provider
                defaults, e.g. 1024/2048/4096 budget tokens on Anthropic), or
                an ``int`` for an exact token budget. Implies ``native=True``.
            **config: The configuration for the module.
        """
        super().__init__()
        signature = ensure_signature(signature)
        desc = "${reasoning}"
        if reasoning_effort is not None:
            native = True
            config["reasoning_effort"] = reasoning_effort
        if native:
            from dspy.adapters.types import Reasoning
            rationale_field_type = Reasoning
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
