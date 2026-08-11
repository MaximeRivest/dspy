"""01 — hello: a signature, a predictor, one exchange, and the X-ray.

The greenfield thesis in its smallest form: declare WHAT (a Signature),
get a predictor (Predict), run it — then ask the program to explain
itself. The explain view is not a wrapper's guess; it is the compiled
ProgramIR, the same value `program.save()` writes to disk.

Run: uv run python examples_greenfield/01_hello.py   (DummyLM, no network)
"""

import dspy


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def chat_completion(**fields: str) -> str:
    """One scripted completion in the chat adapter's convention."""
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


class QA(dspy.Signature):
    """Answer the question in one short sentence."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# One scripted LM, bound through the explicit binding table — no ambient
# settings object anywhere in this world.
lm = dspy.DummyLM([chat_completion(answer="Because the atmosphere scatters blue light most.")])
dspy.configure(lm=lm)

program = dspy.Predict(QA)

section("the exchange")
prediction = program(question="Why is the sky blue?")
print("question :", "Why is the sky blue?")
print("answer   :", prediction.answer)

section("what the LM actually saw (recorded by DummyLM)")
for message in lm.calls[0]["messages"]:
    print(f"--- {message['role']} " + "-" * (64 - len(message["role"])))
    print(message["content"])

section("program.explain() — the compiled IR as one readable view")
print(program.explain())
