"""06 — optimize: the optimizer step as a reviewable diff.

BootstrapFewShot mutates the program's DATA: a teacher runs the
trainset through the engine, and each metric-passing trace becomes a
demo — reasoning included. Score = engine replay; keep = the state.
Because the program is the IR, before and after are two manifests, and
`dspy.diff` shows the demos as the ONLY delta: the whole optimizer
step, reviewable like a code change.

Run: uv run python examples_greenfield/06_optimize.py
"""

import dspy
from dspy import optim


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def chat_completion(**fields: str) -> str:
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


CAPITALS = {"France": "Paris", "Canada": "Ottawa"}


def copycat(messages: list[dict]) -> str:
    """The student LM: answers right only when a demo carries the capital."""
    context = "\n".join(str(m["content"]) for m in messages[:-1])
    current = str(messages[-1]["content"])
    country = next((c for c in CAPITALS if f"capital of {c}?" in current), None)
    answer = CAPITALS[country] if country and CAPITALS[country] in context else "unknown"
    return chat_completion(reasoning="Copy from the demos.", answer=answer)


def oracle(messages: list[dict]) -> str:
    """The teacher LM: knows the table, shows its reasoning."""
    current = str(messages[-1]["content"])
    country = next((c for c in CAPITALS if f"capital of {c}?" in current), None)
    return chat_completion(reasoning=f"Recall {country}.", answer=CAPITALS.get(country, "unknown"))


def exact_answer(example: dspy.Example, prediction) -> bool:
    return example.answer == prediction.answer


trainset = [
    dspy.Example(question=f"What is the capital of {c}?", answer=a).with_inputs("question")
    for c, a in CAPITALS.items()
]

student = dspy.ChainOfThought("question -> answer")
teacher = dspy.ChainOfThought("question -> answer", lm=dspy.DummyLM(oracle))
dspy.configure(lm=dspy.DummyLM(copycat))

section("before: zero-shot, scored by engine replay over the devset")
before_score = optim.evaluate(student, trainset, exact_answer).score
before_manifest = student.to_manifest()
print("score:", before_score, " (the student LM only copies from demos — none yet)")

section("BootstrapFewShot: metric-passing teacher traces become demos")
optimizer = optim.BootstrapFewShot(metric=exact_answer, max_bootstrapped_demos=2, max_labeled_demos=0)
optimizer.compile(student, trainset=trainset, teacher=teacher)
demo = student.demos[0]
print(f"demo[0]: question={demo.question!r}")
print(f"         reasoning={demo.reasoning!r}   <- traced by the teacher")
print(f"         answer={demo.answer!r}")

section("after: same devset, same metric")
after_score = optim.evaluate(student, trainset, exact_answer).score
print(f"score: {before_score} -> {after_score}")

section("dspy.diff(before, after) — the optimizer step as a diff")
for line in dspy.diff(before_manifest, student):
    print(line)
