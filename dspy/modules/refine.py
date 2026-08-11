"""Refine: BestOfN with a mandatory bar, optionally feeding attempts back.

The sibling macro — one tree builder, two public faces. `Refine` IS
`BestOfN` semantics with the threshold early-exit made mandatory (a
Refine that never stops early is just BestOfN — say which one you mean),
plus an optional feedback channel:

- `feedback=False` (default): identical structure to
  `BestOfN(module, n, reward, threshold)`; only the class name differs
  in the artifact's module tree.
- `feedback=True`: the previous attempt's outputs and score thread into
  the NEXT attempt through the inner module's reserved `hints` input, as
  a Format'd string built from a declared `describe_attempt` tool leaf
  (outputs rendered as compact JSON). The inner module must DECLARE that
  input; if it does not, construction refuses with a teaching error —
  feedback silently dropped on the floor would be a lie.

Everything else — the declared reward leaf, the typed skip-and-continue
attempt handlers, the all-failed refusal, `best_score`/`attempts` as
declared outputs, the printed native twin — is inherited from BestOfN.
"""

from __future__ import annotations

from dspy.modules.best_of_n import BestOfN

__all__ = ["Refine"]


class Refine(BestOfN):
    """Retry a sub-module until its reward clears a mandatory bar.

    Args:
        module: The inner program — any `dspy.Module` (Predict included).
        n: The attempt cap, baked as a literal (`ir_literals` refreshes
            it on edit).
        reward: A plain named function `reward(outputs) -> float`; wrapped
            into the declared reward tool leaf exactly as in `BestOfN`.
        threshold: The mandatory early-exit bar: the loop breaks as soon
            as an attempt's score reaches it.
        feedback: When True, thread the previous attempt (outputs +
            score) into the inner module's `hints` input; the inner
            module must declare that input or construction refuses.

    Examples:
        ```python
        solver = dspy.Predict("question, hints -> answer")

        def scored(outputs) -> float:
            return 1.0 if outputs["answer"].isdigit() else 0.0

        refine = dspy.Refine(solver, 3, scored, threshold=1.0, feedback=True)
        prediction = refine(question="How many moons does Mars have?")
        ```
    """

    def __init__(self, module, n, reward, threshold, feedback=False):
        if threshold is None:
            raise ValueError(
                "Refine requires a threshold (the early-exit bar); a Refine that never stops early is "
                "BestOfN — use dspy.BestOfN when exhausting all N attempts is what you mean"
            )
        self._setup(module, n, reward, threshold, feedback=bool(feedback))
