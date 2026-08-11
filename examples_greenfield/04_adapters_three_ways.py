"""04 — adapters: four wire formats, one unchanged program.

The adapter choice follows the MODEL, never the program. Three presets
(chat / json / xml) and one custom entry with a declared pipeline
parser render the same signature into four different byte streams — and
each parses its own convention back. All of it is data: every one of
these adapters dumps to a JSON entry and loads back exactly.

Run: uv run python examples_greenfield/04_adapters_three_ways.py
"""

import dspy
from dspy.adapters import parse


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


class Grade(dspy.Signature):
    """Grade the essay."""

    essay: str = dspy.InputField()
    score: int = dspy.OutputField()


ESSAY = "Dogs are great."

COMPLETIONS = {
    "chat": "[[ ## score ## ]]\n7\n\n[[ ## completed ## ]]",
    "json": '{"score": 7}',
    "xml": "<score>\n7\n</score>",
    "grader": "Persuasive, but thin on evidence.\nSCORE: 7/10",
}

grader = dspy.make_adapter(
    name="grader",
    template=[
        {"role": "system", "content": "{instruction}\nReply with feedback, then a final line 'SCORE: <n>/10'."},
        {"role": "user", "content": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}\n{fragments('user')}"},
    ],
    parser=parse.pipeline(
        parse.regex(r"(?m)^SCORE: (?P<score>[0-9]+)/10$", mode="search"),
        parse.fields_from_groups(),
    ),
)

adapters = [dspy.ChatAdapter(), dspy.JSONAdapter(), dspy.XMLAdapter(), grader]

for adapter in adapters:
    section(f"entry '{adapter.name}' — parser kind: {adapter.parser_spec['kind']}")
    for message in adapter.preview(Grade, {"essay": ESSAY}):
        print(f"--- {message['role']} " + "-" * (64 - len(message["role"])))
        print(message["content"])
    completion = COMPLETIONS[adapter.name]
    fields = adapter.parse_preview(Grade, completion)
    print("--- a completion in this convention " + "-" * 33)
    print(completion)
    print("parsed back:", fields, f"(score is {type(fields['score']).__name__})")

section("the same program runs under any of the four")
for adapter in adapters:
    predictor = dspy.Predict(Grade, adapter=adapter, lm=dspy.DummyLM([COMPLETIONS[adapter.name]]))
    print(f"{adapter.name:>7}: score = {predictor(essay=ESSAY).score}")
