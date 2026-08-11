# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy
from dspy.adapters import parse, strategy  # NEW authoring modules


class GroundedQA(dspy.Signature):
    """Answer from the documents, citing your sources."""

    documents: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: dspy.Citations = dspy.OutputField(role="citations")


# 1. Native citations: the provider assembles citations on a typed
#    channel. The strategy renames the text target (answer ->
#    answer_text) so the text lens and the native assembler compose —
#    the field-transform machinery the spec already names.
native = dspy.ChatAdapter.with_strategies(
    citations=strategy.rule(
        predicate=strategy.capability("native_citations"),
        hides=["citations"],
        transforms=[strategy.rename("answer", "answer_text")],
        engine_controls={"request_patch": {"citations": {"enabled": True}}},
        routings=[strategy.channel("citations", field="citations", coerce="Citations")],
    )
)

# 2. Inline markers: any instruct model. Fragments teach [n] markers;
#    a combinator collects marker+span pairs into the citations field
#    and consumes nothing (the markers stay readable in the answer).
inline = dspy.ChatAdapter.with_strategies(
    citations=strategy.rule(
        predicate=strategy.capability("instruct"),
        hides=["citations"],
        fragments=[
            strategy.fragment(
                "user",
                "After each claim, cite the supporting document as [n], where n is the 1-based document index.",
            )
        ],
        routings=[
            strategy.text(
                parse.pipeline(
                    parse.regex(
                        "(?P<span>[^.!?]*[.!?])\\s*\\[(?P<doc>[0-9]+)\\]",
                        dialect="re2",
                        mode="findall",
                    ),
                    parse.citations(span_group="span", doc_group="doc"),
                ),
                field="citations",
                consume=False,
            )
        ],
    )
)
