# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy


class QA(dspy.Signature):
    """Answer the question concisely."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# A BASE (non-instruct) model adapter: same signatures, same programs,
# different LM family. The whole exchange is one few-shot
# pattern-completion prompt; discipline comes from stop sequences, not
# from chat markers. Chat/JSON/XML adapters simply do not apply here —
# this is the second column of the adapter x LM-family axis.
base = dspy.make_adapter(  # NEW: make_adapter
    name="base_fewshot",
    template=[
        {
            "role": "user",
            "content": "{instruction}\n\n{demos(style='yaml')}\n\n{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}\n{% for f in outputs separator='\\n' %}{f.name}:{% endfor %}",
        },
    ],
    # NEW: engine controls as adapter data — request-side facts the
    # engine applies, generalizing today's native-FC request patch.
    engine_controls={
        "completion_mode": True,
        "stop_sequences": ["\nquestion:", "\n\n"],
    },
    # NEW: declared capability requirement. Loading this entry against
    # an engine that only speaks chat-instruct refuses loudly, naming
    # the missing capability — never a silent wrong render.
    requires={"lm_capabilities": ["completion"]},
)

program = dspy.Predict(QA)
program.set_lm(dspy.LM("together/llama-3-70b-base"), adapter=base)
