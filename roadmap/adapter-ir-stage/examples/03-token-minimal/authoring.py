# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy


class QA(dspy.Signature):
    """Answer the question concisely."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# A token-efficient adapter for a small/weak model. The whole adapter is
# a template author's act: no scaffold system message, one-line labels,
# no completed marker, no type notes. The parser comes for free — it is
# the lens of this template (and with one output field the lens
# degenerates to full_text: everything after the last label).
tiny = dspy.make_adapter(  # NEW: make_adapter — template in, adapter out
    name="tiny_qa",
    template=[
        {"role": "system", "content": "{instruction}"},
        {
            "role": "demos",
            "user": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}",
            "assistant": "{% for f in outputs separator='\\n' strip %}{f.name}: {f.value}{% endfor %}",
        },
        {
            "role": "user",
            "content": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}\n{fragments('user')}",
        },
    ],
    # parser omitted: NEW default — the derived lens of the template.
)

# Same program, same signature; the adapter choice follows the model
# and the token budget, never the program (the orthogonality rule).
program = dspy.Predict(QA)
program.set_lm(dspy.LM("openai/gpt-tiny"), adapter=tiny)
