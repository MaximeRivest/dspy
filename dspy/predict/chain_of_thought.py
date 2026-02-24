from typing import Any

from pydantic.fields import FieldInfo

import dspy
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

# NOTE: This restores the legacy rationale_field behavior after PR #8822.


class ChainOfThought(Module):
    """Generate a step-by-step reasoning chain before producing the final output.

    ``ChainOfThought`` wraps a ``dspy.Predict`` call by automatically prepending a
    ``reasoning`` output field to the signature. The language model is prompted to
    "think step by step" before answering, which often improves accuracy on tasks
    that benefit from intermediate reasoning.

    The generated ``reasoning`` is included in the returned ``Prediction`` alongside
    the original output fields.

    Example:

        Basic usage:

        ```python
        import dspy

        dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

        cot = dspy.ChainOfThought("question -> answer")
        result = cot(question="What is the capital of France?")
        print(result.reasoning)
        print(result.answer)
        ```

        With a typed signature:

        ```python
        class QA(dspy.Signature):
            \"\"\"Answer the question.\"\"\"
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        cot = dspy.ChainOfThought(QA)
        result = cot(question="What is 15 * 7?")
        ```

    Args:
        signature: The input/output signature describing the task. Can be a
            shorthand string like ``"question -> answer"`` or a ``dspy.Signature``
            class.
        rationale_field: Optional custom field info for the reasoning field. When
            ``None``, a default ``dspy.OutputField`` with the prefix
            *"Reasoning: Let's think step by step in order to"* is used.
        rationale_field_type: The Python type annotation for the rationale field.
            Defaults to ``str``.
        **config: Additional keyword arguments forwarded to the underlying
            ``dspy.Predict`` instance (e.g. ``temperature``, ``n``).

    Note:
        **How it works:** ``ChainOfThought`` creates a new signature by prepending a
        ``reasoning`` output field to the original signature. It then delegates to a
        standard ``dspy.Predict`` call with this extended signature. The reasoning
        field appears *before* the other output fields, so the LM generates its
        chain of thought first.

        **Custom rationale fields:** You can override the default reasoning prefix
        and description by passing a custom ``rationale_field``. This is useful when
        you want domain-specific reasoning prompts.

        **Optimiser-friendly:** Like ``Predict``, ``ChainOfThought`` is fully
        compatible with DSPy optimisers such as ``BootstrapFewShot`` and ``MIPROv2``.
    """

    def __init__(
        self,
        signature: str | type[Signature],
        rationale_field: FieldInfo | None = None,
        rationale_field_type: type = str,
        **config: dict[str, Any],
    ):
        super().__init__()
        signature = ensure_signature(signature)
        prefix = "Reasoning: Let's think step by step in order to"
        desc = "${reasoning}"
        rationale_field_type = rationale_field.annotation if rationale_field else rationale_field_type
        rationale_field = rationale_field if rationale_field else dspy.OutputField(prefix=prefix, desc=desc)
        extended_signature = signature.prepend(name="reasoning", field=rationale_field, type_=rationale_field_type)
        self.predict = dspy.Predict(extended_signature, **config)

    def forward(self, **kwargs):
        return self.predict(**kwargs)

    async def aforward(self, **kwargs):
        return await self.predict.acall(**kwargs)
