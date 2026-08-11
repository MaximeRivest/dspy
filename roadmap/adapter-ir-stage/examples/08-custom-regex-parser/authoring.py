# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy
from dspy.adapters import parse  # NEW authoring module


class Judge(dspy.Signature):
    """Grade the response."""

    response: str = dspy.InputField()
    score: int = dspy.OutputField()
    verdict: str = dspy.OutputField()


# A user's own wire format: the model answers "SCORE: 7/10" and
# "VERDICT: pass" lines. The parser is AUTHORED — but it is authored
# DATA, not authored code. RE2-subset named groups map to fields;
# values coerce through the field's schema. No authored_by, no
# identity hash, no sandbox: the origin collapse means the only load
# question is "do you speak parse_combinators 0.1".
grader = dspy.make_adapter(
    name="grader_lines",
    template=[
        {"role": "system", "content": "{instruction}"},
        {
            "role": "user",
            "content": "{% for f in inputs separator='\\n\\n' %}{f.name}:\n{f.value}{% endfor %}\n\nReply with exactly two lines:\nSCORE: <n>/10\nVERDICT: <pass|fail>",
        },
    ],
    parser=parse.pipeline(
        parse.regex(
            "(?m)^SCORE: (?P<score>[0-9]+)/10$\\n^VERDICT: (?P<verdict>pass|fail)$",
            dialect="re2",
            mode="search",
        ),
        parse.fields_from_groups(),  # named group -> same-named field
    ),
)

# parse_preview is the development loop: pure, no LM.
assert grader.parse_preview(
    Judge, completion="SCORE: 7/10\nVERDICT: pass"
) == {"score": 7, "verdict": "pass"}
