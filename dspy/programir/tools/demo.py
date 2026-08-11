"""One-command tour of the ProgramIR workbench tools.

Compiles three corpus programs (tests/programir/corpus_programs.py) with a
declared LM — no weights, no network — then shows all three tools:

- lint over SharedTasks (a thrown-away predictor output) and MiniReAct;
- cost over NestedAnswerer, MiniReAct (For cap), and MiniRLM (While cap);
- diff between NestedAnswerer and a simulated optimizer step (instruction
  rewrite, demo added, config bump, one forward branch edited).

Run from the repository root:

    python -m dspy.programir.tools.demo
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

from dspy.programir.tools import cost, diff, lint
from dspy.programir.tools._common import WIDTH


def _load_corpus():
    """Import the corpus programs module from the source checkout."""
    corpus_dir = Path(__file__).resolve().parents[3] / "tests" / "programir"
    if not (corpus_dir / "corpus_programs.py").is_file():
        raise SystemExit(f"run the demo from a dspy source checkout: missing {corpus_dir / 'corpus_programs.py'}")
    sys.path.insert(0, str(corpus_dir))
    try:
        import corpus_programs
    finally:
        sys.path.pop(0)
    return corpus_programs


def _compile(program_class):
    import dspy
    from dspy.programir import compile as compile_ir

    program = program_class()
    program.set_lm(dspy.LM("openai/demo-model"))
    return compile_ir(program).to_manifest()


def _banner(title: str) -> str:
    bar = "#" * WIDTH
    return f"\n{bar}\n# {title}\n{bar}"


def _optimizer_step(manifest: dict) -> dict:
    """Simulate one optimization step on the NestedAnswerer manifest."""
    step = copy.deepcopy(manifest)
    components = step["components"]
    components["3a_instructions"]["polish"] = "Rewrite the draft into a polished, well-sourced answer. Keep it short."
    components["3b_demos"]["drafter.classifier.classify"] = [
        {"input_keys": ["question"], "question": "Invent a color.", "category": "creative"},
    ]
    components["3c_predictor_config"]["polish"]["max_tokens"] = 400
    # Hand edit: the drafter branch now tests for "imaginative".
    body = components["5_forward"]["drafter"]["body"]
    for statement in body:
        if statement.get("node") == "If":
            statement["test"]["right"] = {"node": "Const", "value": "imaginative"}
    return step


def main() -> int:
    corpus = _load_corpus()

    shared = _compile(corpus.SharedTasks)
    nested = _compile(corpus.NestedAnswerer)
    react = _compile(corpus.MiniReAct)
    rlm = _compile(corpus.MiniRLM)

    print(_banner("1/3  lint — SharedTasks (a result nobody reads)"))
    print(lint.build_text(shared))
    print(_banner("1/3  lint — MiniReAct (dynamic tools, typed raise)"))
    print(lint.build_text(react))

    print(_banner("2/3  cost — NestedAnswerer (branchy pipeline)"))
    print(cost.build_text(nested))
    print(_banner("2/3  cost — MiniReAct (For cap = 3, early break)"))
    print(cost.build_text(react))
    print(_banner("2/3  cost — MiniRLM (While with a break guard)"))
    print(cost.build_text(rlm))

    print(_banner("3/3  diff — NestedAnswerer vs one optimizer step"))
    print(diff.build_text(nested, _optimizer_step(nested)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
