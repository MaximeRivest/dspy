# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy
from dspy.adapters import parse, strategy  # NEW authoring modules


class QA(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(role="reasoning")  # the ONE role
    answer: str = dspy.OutputField()


program = dspy.ChainOfThought  # conceptually; the signature is the point

# THREE strategies for the SAME role on the SAME program. The adapter
# choice follows the model and the inference strategy, never the program.

# 1. Native provider thinking: hide the field from the token stream,
#    patch the request, route the provider channel back to the field.
native = dspy.ChatAdapter.with_strategies(  # NEW: with_strategies
    reasoning=strategy.rule(
        predicate=strategy.capability("native_reasoning"),
        hides=["reasoning"],
        engine_controls={"request_patch": {"reasoning": {"effort": "medium"}}},
        routings=[strategy.channel("reasoning", field="reasoning")],
    )
)

# 2. Classic prefix-CoT: pure fragments. The field stays visible; the
#    template's own lens parses it. Works on any instruct model.
prefix_cot = dspy.ChatAdapter.with_strategies(
    reasoning=strategy.rule(
        predicate=strategy.capability("instruct"),
        fragments=[
            strategy.fragment(
                "user",
                "Reason step by step in the `reasoning` field before giving the final answer.",
            )
        ],
    )
)

# 3. Interleaved thinking BY INSTRUCTION: fragments tell the model to
#    think in tags after every sentence; a combinator routing collects
#    the tags into `reasoning` and consumes them from the visible text
#    before the lens runs. Pure prompt + parse data — any instruct model.
interleaved = dspy.ChatAdapter.with_strategies(
    reasoning=strategy.rule(
        predicate=strategy.capability("instruct"),
        hides=["reasoning"],
        fragments=[
            strategy.fragment(
                "user",
                "After every sentence of your answer, add a brief thought inside <think>...</think> tags.",
            )
        ],
        routings=[
            strategy.text(
                parse.pipeline(
                    parse.regex(
                        "<think>(?P<t>[^<]*)</think>",
                        dialect="re2",
                        mode="findall",
                        group="t",
                        join="\n",
                    )
                ),
                field="reasoning",
                consume=True,  # matched spans leave the stream the lens sees
            )
        ],
    )
)
