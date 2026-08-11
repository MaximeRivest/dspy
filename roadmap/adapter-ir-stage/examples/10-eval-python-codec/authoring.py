# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import pydantic

import dspy
from dspy.adapters import parse, strategy  # NEW authoring modules


class SalesReport(pydantic.BaseModel):
    region: str
    total: float
    top_products: list[str]


class Summarize(dspy.Signature):
    """Build the sales report from the transcript."""

    transcript: str = dspy.InputField()
    report: SalesReport = dspy.OutputField(role="code")  # conduct via code emission


# Structured output, conducted as "the model writes Python; a sandboxed
# interpreter evaluates it into the promised object". This is the north
# star's codec/strategy hybrid: the CODEC says values are spelled as
# Python literals; the STRATEGY says how the exchange conducts it and
# names the interpreter LEAF that materializes the value. The leaf is
# the trust boundary — it is already representable in the ProgramIR
# trust vocabulary, with an explicit isolation declaration.
eval_python = dspy.ChatAdapter.with_strategies(
    code=strategy.rule(
        predicate=strategy.capability("instruct"),
        fragments=[
            strategy.fragment(
                "user",
                "For the `report` field, emit a fenced ```python block containing a single expression that constructs the value (keyword arguments, literals only).",
            )
        ],
        routings=[
            strategy.text(
                parse.pipeline(parse.fenced_block(language="python", policy="require")),
                field="report",
                materialize={  # NEW: value materialization via a leaf
                    "interpreter": {
                        "leaf": "python_literal_eval",
                        "language": {"name": "python", "requires": ">=3.10"},
                        "isolation": "sandboxed",
                        "effects": [],
                        "network": False,
                    }
                },
            )
        ],
    ),
).with_codecs(per_field={"report": {"family": "python_literal", "options": {}}})

# The artifact's requirement set now truthfully includes a python
# sandbox — receivers without one refuse, naming the leaf.
