from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures import InputField, OutputField
from dspy.signatures.signature import ensure_signature


class MultiChainComparison(Module):
    """Compare multiple chain-of-thought attempts and produce a refined answer.

    ``MultiChainComparison`` takes ``M`` candidate reasoning attempts (typically
    from ``dspy.ChainOfThought`` with ``n=M``) and presents them all to the
    language model. The model then synthesises a final, corrected answer by
    holistically evaluating the different reasoning paths.

    This is useful as a simple self-consistency / refinement strategy: generate
    several diverse reasoning chains, then let the model pick and merge the best
    parts.

    Example:

        ```python
        import dspy

        dspy.configure(lm=dspy.LM("groq/moonshotai/kimi-k2-instruct"))

        cot = dspy.ChainOfThought("question -> answer", n=3, temperature=0.7)
        compare = dspy.MultiChainComparison("question -> answer", M=3)

        completions = cot(question="What is 18 * 7?").completions
        result = compare(completions, question="What is 18 * 7?")
        print(result.answer)
        ```

    Args:
        signature: The input/output signature describing the task. Can be a
            shorthand string or a ``dspy.Signature`` class.
        M: The number of reasoning attempts to compare. Must match the ``n``
            parameter used when generating candidates. Defaults to ``3``.
        temperature: Sampling temperature for the comparison call.
            Defaults to ``0.7``.
        **config: Additional keyword arguments forwarded to the underlying
            ``dspy.Predict`` instance.

    Note:
        **Input format:** The ``forward`` method expects a ``completions`` object
        (as returned by ``prediction.completions``) containing ``M`` entries, each
        with either a ``rationale`` or ``reasoning`` field and the final output field.

        **How it works:** Each candidate's reasoning and answer are formatted into
        numbered "Student Attempt" input fields. A ``rationale`` output field is
        prepended so the model first produces corrected reasoning before the final
        answer.
    """

    def __init__(self, signature, M=3, temperature=0.7, **config):  # noqa: N803
        super().__init__()

        self.M = M
        signature = ensure_signature(signature)

        *_, self.last_key = signature.output_fields.keys()

        for idx in range(M):
            signature = signature.append(
                f"reasoning_attempt_{idx+1}",
                InputField(
                    prefix=f"Student Attempt #{idx+1}:",
                    desc="${reasoning attempt}",
                ),
            )

        signature = signature.prepend(
            "rationale",
            OutputField(
                prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
                desc="${corrected reasoning}",
            ),
        )

        self.predict = Predict(signature, temperature=temperature, **config)

    def forward(self, completions, **kwargs):
        attempts = []

        for c in completions:
            rationale = c.get("rationale", c.get("reasoning")).strip().split("\n")[0].strip()
            answer = str(c[self.last_key]).strip().split("\n")[0].strip()
            attempts.append(
                f"«I'm trying to {rationale} I'm not sure but my prediction is {answer}»",
            )

        assert (
            len(attempts) == self.M
        ), f"The number of attempts ({len(attempts)}) doesn't match the expected number M ({self.M}). Please set the correct value for M when initializing MultiChainComparison."

        kwargs = {
            **{f"reasoning_attempt_{idx+1}": attempt for idx, attempt in enumerate(attempts)},
            **kwargs,
        }
        return self.predict(**kwargs)
