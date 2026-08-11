"""02 — chain of thought: one program, two reasoning conducts.

ChainOfThought does not wrap Predict — it IS Predict, with a
reasoning-ROLE output field. HOW the model reasons is the adapter's
reasoning strategy: the default binding is "auto", which resolves from
the LM's DECLARED capability facts (data on the binding, never sniffed).
Same program, different model — different rendered bytes, no re-author.

Run: uv run python examples_greenfield/02_chain_of_thought.py
"""

import dspy


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def chat_completion(**fields: str) -> str:
    """One scripted completion in the chat adapter's convention."""
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


cot = dspy.ChainOfThought("question -> answer")
adapter = cot.resolve_adapter()

section("an instruct model: auto -> prefix_cot (a rendered fragment)")
instruct_lm = dspy.DummyLM([chat_completion(reasoning="17 * 4 = 68.", answer="68")])
dspy.configure(lm=instruct_lm)
prediction = cot(question="What is 17 * 4?")
print("resolved :", adapter.explain_strategies(cot.signature, lm=instruct_lm)["reasoning"])
print("reasoning:", prediction.reasoning)
print("answer   :", prediction.answer)

section("a native-thinking model: auto -> native (a request channel)")
native_lm = dspy.DummyLM(["unused"], native_reasoning=True)
print("resolved :", adapter.explain_strategies(cot.signature, lm=native_lm)["reasoning"])
print("(the LM layer surfaces no response channels yet, so a live native")
print(" run parses reasoning=None — see README-GREENFIELD honest limits)")

section("the rendered messages differ — preview is pure, no LM call")
inputs = {"question": "What is 17 * 4?"}
prefix_call = adapter.format(cot.signature, inputs, lm=instruct_lm)
native_call = adapter.format(cot.signature, inputs, lm=native_lm)

print("prefix_cot request:", prefix_call.request, " (conduct rides the prompt)")
print("native request:    ", native_call.request)

for name, call in (("prefix_cot", prefix_call), ("native", native_call)):
    print(f"\n--- user message under {name} " + "-" * (40 - len(name)))
    print(call.messages[-1]["content"])
