"""05 — save, ship, load: one artifact path, nothing lost.

`program.save(dir)` compiles the program and writes the ProgramIR
artifact: manifest, tool source sidecars, a locked PEP 723 environment.
`dspy.load(dir, bindings=...)` reads, links, and materializes it — the
receiver brings the LM, everything else rides in the artifact. The
loaded program replays the direct run byte-for-byte.

Run: uv run python examples_greenfield/05_save_ship_load.py
"""

import json
import tempfile
from pathlib import Path

import dspy


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def chat_completion(**fields: str) -> str:
    """One scripted completion in the chat adapter's convention."""
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


def lookup_capital(country: str) -> str:
    """Return the capital of a country, from a fixed table."""
    capitals = {"france": "Paris", "japan": "Tokyo"}
    return capitals.get(country.lower(), "unknown")


def script() -> list[str]:
    return [
        chat_completion(next_thought="Look it up.", next_tool_name="lookup_capital",
                        next_tool_args='{"country": "France"}'),
        chat_completion(next_thought="Done gathering.", next_tool_name="finish", next_tool_args="{}"),
        chat_completion(reasoning="The observation names the capital.", answer="Paris"),
    ]


dspy.configure(lm=dspy.DummyLM(script()))
react = dspy.ReAct("question -> answer", tools=[lookup_capital], max_iters=3)

section("the direct run")
direct = react(question="What is the capital of France?")
print("answer:", direct.answer)

with tempfile.TemporaryDirectory() as tmp:
    target = Path(tmp) / "artifact"

    section("react.save(dir) — the one save path")
    react.save(target)
    for path in sorted(target.rglob("*")):
        kind = "dir " if path.is_dir() else f"{path.stat().st_size:>6}B"
        print(f"  {kind}  {path.relative_to(target)}")

    manifest = json.loads((target / "manifest.json").read_text())
    print("\nmanifest keys :", ", ".join(sorted(manifest)))
    print("components    :", ", ".join(sorted(manifest["components"])))
    print("\ntools/lookup_capital.py (the declared tool leaf, source sidecar):")
    print((target / "tools" / "lookup_capital.py").read_text())

    section("dspy.load(dir, bindings=...) — read + link + materialize")
    loaded = dspy.load(target, bindings={"lm": {"dummy": dspy.DummyLM(script())}})
    replayed = loaded(question="What is the capital of France?")
    print("replayed answer:", replayed.answer)

    assert replayed.toDict() == direct.toDict()
    print("replayed.toDict() == direct.toDict():", replayed.toDict() == direct.toDict())
    print("(the tool leaf was rebuilt from its sidecar and really ran)")
