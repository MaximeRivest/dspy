# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy
from dspy.adapters import parse  # NEW: the parse-combinator authoring module


class Extract(dspy.Signature):
    """Extract the entities and a one-line summary."""

    text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField()
    summary: str = dspy.OutputField()


# Start from the builtin json preset; replace the parser with declared
# parse-data (level 1). Each combinator is data with pinned semantics;
# the pipeline serializes verbatim into the entry.
adapter = dspy.JSONAdapter.with_parser(  # NEW: with_parser
    parse.pipeline(
        parse.fenced_block(language="json", policy="prefer"),
        parse.alternatives(
            parse.json_object(repair="none"),
            parse.json_object(repair="json_repair"),
        ),
        parse.fields_from_object(unknown_keys="exhaust"),
    )
)

program = dspy.Predict(Extract)

# The tolerance choices are now optimizer axes: an optimizer may flip
# repair="json_repair" to repair="none" as a one-field mutation and
# measure the difference.
entry = adapter.dump_entry()
assert entry["parser"]["kind"] == "pipeline"
