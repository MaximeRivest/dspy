"""Human-explorable Tier 1 tools over ProgramIR manifests.

Three pure views over the manifest JSON (staging-lessons §B, Tier 1):
`lint` (static findings), `diff` (semantic old->new delta), and `cost`
(static call/token bounds). Each is callable from Python and as a CLI:

    python -m dspy.programir.tools.lint <artifact-or-module>
    python -m dspy.programir.tools.diff <old> <new>
    python -m dspy.programir.tools.cost <artifact-or-module>
    python -m dspy.programir.tools.demo

No tool calls an LM or executes authored code. Submodules import on
first use (`from dspy.programir.tools import lint`); the package keeps
no eager imports so `python -m` entry points run without a double-import
warning.
"""

__all__ = ["lint", "diff", "cost"]
