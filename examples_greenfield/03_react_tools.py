"""03 — ReAct: a tool-using agent whose artifact is the whole story.

Two declared tools, a literal loop cap, typed failures — and because
the program IS the IR, the run's journal, the static lint, and the
call/token bounds all come from the same compiled value.

Run: uv run python examples_greenfield/03_react_tools.py
"""

import json

import dspy


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def chat_completion(**fields: str) -> str:
    """One scripted completion in the chat adapter's convention."""
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


def react_step(thought: str, tool: str, args: dict) -> str:
    return chat_completion(next_thought=thought, next_tool_name=tool, next_tool_args=json.dumps(args))


def lookup_capital(country: str) -> str:
    """Return the capital of a country, from a fixed table."""
    capitals = {"france": "Paris", "japan": "Tokyo"}
    return capitals.get(country.lower(), "unknown")


def city_population(city: str) -> str:
    """Return a city's population, as one sentence."""
    table = {"paris": "About 2.1 million people live in Paris."}
    return table.get(city.lower(), "No population data.")


script = [
    react_step("First find the capital of France.", "lookup_capital", {"country": "France"}),
    react_step("Now find how many people live in Paris.", "city_population", {"city": "Paris"}),
    react_step("The trajectory has everything.", "finish", {}),
    chat_completion(reasoning="The tools found the capital, then its population.", answer="About 2.1 million."),
]
dspy.configure(lm=dspy.DummyLM(script))

react = dspy.ReAct("question -> answer", tools=[lookup_capital, city_population], max_iters=5)

section("the engine run")
prediction = react(question="How many people live in the capital of France?")
print("answer:", prediction.answer)

section("the trajectory (the engine's per-leaf run journal)")
calls = prediction._trajectory["predictor_calls"]
print("predictor calls:", [call["predictor"] for call in calls])
print(json.dumps(calls[-1]["inputs"]["trajectory"], indent=2))

section("react.lint() — static findings over the compiled program")
findings = react.lint()
print(findings if findings else "no findings — the loop is capped, every branch reachable")

section("react.cost() — static bounds per invocation (before any run)")
estimate = react.cost()
print("total calls  :", estimate["total_calls"])
for path, bounds in sorted(estimate["calls"].items()):
    print(f"calls[{path}] :", bounds)
print("prompt tokens:", estimate["total_prompt_tokens"], "(chars/4 heuristic)")
