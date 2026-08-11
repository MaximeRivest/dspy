# IDEALIZED SURFACE — this file shows target authoring code. Parts marked
# NEW do not exist in dspy today. Nothing here executes.

import dspy


class QA(dspy.Signature):
    """Answer the question concisely."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# The user writes exactly what they write today. No new API is needed for
# the baseline: the change is in what dump_entry() EMITS.
adapter = dspy.ChatAdapter()
program = dspy.Predict(QA)

entry = adapter.dump_entry()

# TODAY: entry["parser"] == "chat"  (a name into a closed enum of four).
# NEW (target): entry["parser"] == {"kind": "lens", "of": "template"}
# — the parser is DERIVED from the bound template: labels become section
# boundaries, slots become captures, values decode through the bound codec.
assert entry["parser"] == {"kind": "lens", "of": "template"}

# NEW: the dual of preview(). Pure, byte-testable, no LM call.
parsed = adapter.parse_preview(
    QA,
    completion="[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]",
)
assert parsed == {"answer": "Paris"}
