"""BestOfN: sample a sub-module N times, keep the best-scoring attempt.

The first tier-3 IR macro: a module whose forward is BUILT as a tree
(`programir.build` constructors) around ONE declared sub-module leaf.
It wraps ANY module — a Predict, a ChainOfThought, a ReAct — because the
inner program is a leaf call, not inlined structure.

The reward is a DECLARED metric leaf, never an opaque callable: the
plain Python function you pass is wrapped into a generated pure tool
(`def reward(outputs: dict) -> float`) that embeds your source verbatim,
exactly like the dispatch wrappers ReAct builds for its tools. The
artifact carries the reward's real source; a receiver rebuilds and runs
it from the sidecar.

The loop shape (all literals baked, zero reach-back):

- For attempt over the literal ``range(N)``: call the inner leaf, score
  its outputs with the reward leaf; a typed attempt failure
  (`AdapterParseError`, `ToolError`) skips the attempt via the Try
  handlers instead of killing the run;
- track best-so-far with Compare/If (first success seeds it; strict
  improvement replaces it — ties keep the earliest attempt);
- with a threshold: ``if best_score >= threshold: break`` (early exit);
- if EVERY attempt failed, raise a typed `ToolError` — never return an
  empty record silently.

`best_score` and `attempts` are DECLARED outputs of the returned record,
not `_trajectory` exhaust: engine records carry declared outputs only
(PIR-013), so exhaust would vanish the moment another module wraps this
one. The run journal still rides the root prediction's
`_trajectory["predictor_calls"]` as for every engine run.

The native twin is DERIVED: the printer renders the built tree and binds
it on the instance, so `forward_native` agrees with the engine arm by
construction.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

from dspy.modules._generate import load_generated
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.programir.build import (
    Assign,
    BinOp,
    BoolOp,
    Break,
    CallModule,
    CallPredict,
    CallTool,
    Compare,
    Continue,
    Dict,
    Do,
    Except,
    For,
    Format,
    Forward,
    If,
    Method,
    Raise,
    Return,
    SetIndex,
    Try,
    UnaryOp,
    Var,
)
from dspy.programir.leaves import extract_tool
from dspy.programir.printer import bind_forward

__all__ = ["BestOfN"]

#: Names the macro's own loop machinery binds; an inner input colliding
#: with one would be overwritten mid-loop, so it refuses at construction.
_RESERVED_LOCALS = (
    "attempt",
    "attempts",
    "best",
    "best_score",
    "final",
    "found",
    "hints",
    "note",
    "outputs",
    "score",
)

#: Output names the macro adds to the returned record.
_RESERVED_OUTPUTS = ("best_score", "attempts")

#: The feedback renderer is a TOOL leaf on purpose: the engine's Format
#: renders floats under the render_float pin ("0", "1"), native Python
#: under str() ("0.0", "1.0") — a score formatted in-forward would
#: diverge between the arms. Inside a tool leaf both arms run this exact
#: Python, so the rendered bytes agree.
_DESCRIBE_SOURCE = '''def describe_attempt(outputs: dict, score: float) -> str:
    """Render one attempt (score + outputs) as feedback text."""
    import json

    values = {key: outputs[key] for key in sorted(outputs.keys())}
    return "(score " + str(score) + "): " + json.dumps(values, ensure_ascii=False, default=str)
'''


class BestOfN(Module):
    """Run a sub-module up to N times; return the best-scoring outputs.

    Args:
        module: The inner program — any `dspy.Module` (Predict included).
            It becomes a declared sub-module leaf in the artifact.
        n: The attempt cap, baked into the built forward as a literal
            (named in `ir_literals` so edits recompile).
        reward: A plain named function taking the attempt's outputs
            record and returning a number — ``def reward(outputs) ->
            float``. Read fields by subscript (``outputs["answer"]``); it
            becomes a declared pure tool leaf carrying this exact source.
        threshold: Optional early-exit bar: the loop breaks as soon as
            the best score reaches it. `None` always exhausts N.

    The returned record is the best attempt's outputs plus two declared
    fields: `best_score` (the winning reward) and `attempts` (attempts
    consumed, failed ones included).

    Examples:
        ```python
        def has_citation(outputs) -> float:
            return 1.0 if "[" in outputs["answer"] else 0.0

        best = dspy.BestOfN(dspy.Predict("question -> answer"), 5, has_citation, threshold=1.0)
        prediction = best(question="Who proved Fermat's last theorem?")
        ```
    """

    ir_literals = ("n", "threshold")

    def __init__(self, module, n, reward, threshold=None):
        self._setup(module, n, reward, threshold, feedback=False)

    def _setup(self, module, n, reward, threshold, *, feedback):
        owner = type(self).__name__
        if not isinstance(module, Module):
            raise TypeError(
                f"{owner} wraps a dspy.Module (the inner program becomes a declared sub-module leaf), "
                f"got {type(module).__name__}"
            )
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"{owner} n must be an int >= 1 (the literal attempt cap), got {n!r}")
        if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))):
            raise ValueError(f"{owner} threshold must be a number or None, got {threshold!r}")

        inner_inputs = _inner_input_names(module, owner=owner)
        if feedback:
            if "hints" not in inner_inputs:
                raise ValueError(
                    f"{owner}(feedback=True) threads the previous attempt into a reserved `hints` input, "
                    f"but the inner {type(module).__name__} declares no `hints` input field "
                    f"(declared inputs: {inner_inputs}). Add `hints: str = dspy.InputField()` to its "
                    "signature, or pass feedback=False."
                )
            args = [name for name in inner_inputs if name != "hints"]
        else:
            args = list(inner_inputs)
        collisions = sorted(set(args) & set(_RESERVED_LOCALS))
        if collisions:
            raise ValueError(
                f"{owner} reserves the name(s) {collisions} for its own loop machinery; rename them in "
                f"the inner module's inputs (reserved: {', '.join(_RESERVED_LOCALS)})"
            )
        inner_outputs = _inner_output_names(module)
        output_collisions = sorted(set(inner_outputs) & set(_RESERVED_OUTPUTS))
        if output_collisions:
            raise ValueError(
                f"{owner} adds the declared output(s) {list(_RESERVED_OUTPUTS)} to the returned record; "
                f"the inner module's output(s) {output_collisions} would be overwritten — rename them"
            )

        self.inner = module
        self.n = n
        self.threshold = threshold
        self.reward = _build_reward_tool(reward, owner=owner)
        self._feedback = feedback
        self._macro_args = args
        if feedback:
            self.describe_attempt = load_generated(_DESCRIBE_SOURCE, tag="describe-attempt", name="describe_attempt")

        claimed = {"reward": extract_tool(self.reward, name="reward")}
        if feedback:
            claimed["describe_attempt"] = extract_tool(self.describe_attempt, name="describe_attempt")
        _check_tool_collisions(module, claimed, owner=owner)

        self.build_forward_ir()

    def build_forward_ir(self):
        """Build the forward tree (current literals) and refresh the twin.

        The compiler calls this at every compile, so edited `n` /
        `threshold` values bake freshly; the printed native `forward`
        rebinds alongside.
        """
        tree = self._forward_tree()
        bind_forward(self, tree, tag=type(self).__name__.lower())
        return tree

    def _forward_tree(self):
        call = CallPredict if isinstance(self.inner, Predict) else CallModule
        call_kwargs = {name: Var(name) for name in self._macro_args}
        if self._feedback:
            call_kwargs["hints"] = Var("hints")

        keep_if_best = If(
            BoolOp("or", UnaryOp("not", Var("found")), Compare("gt", Var("score"), Var("best_score"))),
            [Assign("best", Var("outputs")), Assign("best_score", Var("score")), Assign("found", True)],
        )
        loop_body = [
            Assign("attempts", BinOp("add", Var("attempts"), 1)),
            Try(
                [
                    Assign("outputs", call("inner", **call_kwargs)),
                    Assign("score", CallTool("reward", outputs=Var("outputs"))),
                ],
                [Except("AdapterParseError", [Continue()]), Except("ToolError", [Continue()])],
            ),
            keep_if_best,
        ]
        if self.threshold is not None:
            loop_body.append(If(Compare("ge", Var("best_score"), self.threshold), [Break()]))
        if self._feedback:
            loop_body.append(Assign("note", CallTool("describe_attempt", outputs=Var("outputs"), score=Var("score"))))
            loop_body.append(Assign("hints", Format("Previous attempt ", Var("note"))))

        body = [
            Assign("attempts", 0),
            Assign("best_score", 0.0),
            Assign("best", Dict()),
            Assign("found", False),
        ]
        if self._feedback:
            body.append(Assign("hints", ""))
        body.extend(
            [
                For("attempt", loop_body, range=self.n),
                If(
                    UnaryOp("not", Var("found")),
                    [
                        Raise(
                            "ToolError",
                            f"{type(self).__name__}: all {self.n} attempt(s) failed; there is no best output to return",
                        )
                    ],
                ),
                Assign("final", Dict()),
                Do(Method(Var("final"), "update", Var("best"))),
                SetIndex("final", "best_score", Var("best_score")),
                SetIndex("final", "attempts", Var("attempts")),
                Return(Var("final")),
            ]
        )
        return Forward(self._macro_args, body)


def _check_tool_collisions(module, claimed, *, owner) -> None:
    """Refuse at construction when the inner tree owns a colliding tool.

    The macro's generated leaves use fixed pool names — `reward`, plus
    `describe_attempt` under feedback. ProgramIR pools tools by NAME,
    one function identity per name per program, so an inner tool wearing
    one of those names with a DIFFERENT identity would refuse deep
    inside the first compile with the pool's non-teaching identity
    error. Refuse HERE instead, naming the owning module and the remedy.
    An inner tool with the IDENTICAL identity — a nested macro built on
    the same reward function — dedupes in the pool and passes.

    Args:
        module: The inner module whose tree is walked.
        claimed: The macro's own leaves, as `{name: AuthoredLeaf}`.
        owner: The macro class name, for the teaching error.
    """
    for path, tool_name, tool in _owned_tools(module):
        macro_leaf = claimed.get(tool_name)
        if macro_leaf is None:
            continue
        try:
            inner_leaf = extract_tool(tool, name=tool_name)
            identical = inner_leaf.entry == macro_leaf.entry and inner_leaf.source == macro_leaf.source
        except ValueError:
            identical = False
        if identical:
            continue
        where = f"the inner {type(module).__name__}" if path == "self" else f"its sub-module at path {path!r}"
        remedy = (
            "pass the SAME reward function so the leaves dedupe, or rename the inner tool"
            if tool_name == "reward"
            else "rename the inner tool"
        )
        raise ValueError(
            f"{owner} declares its own generated tool leaf named {tool_name!r}, but {where} already owns "
            f"a different tool with that name; ProgramIR pools tools by name (one function identity per "
            f"program), so the first compile would refuse. To wrap this module, {remedy}."
        )


def _owned_tools(module):
    """Every tool leaf a module tree owns, as `(path, name, tool)` triples.

    Mirrors the compiler's child classification exactly: a `Tool` or
    plain-function attribute is a tool named by the attribute; a
    non-empty dict of those (ReAct's dispatch table) contributes one
    tool per key.
    """
    from dspy.adapters.types.tool import Tool

    for path, owner_module in module._named_modules():
        for attribute, child in owner_module.__dict__.items():
            if not (attribute.isidentifier() and attribute.isascii()):
                continue
            if isinstance(child, Tool) or inspect.isfunction(child):
                yield path, attribute, child
            elif (
                isinstance(child, dict)
                and child
                and all(
                    isinstance(key, str)
                    and key.isidentifier()
                    and key.isascii()
                    and (isinstance(value, Tool) or inspect.isfunction(value))
                    for key, value in child.items()
                )
            ):
                yield from ((path, key, value) for key, value in child.items())


def _inner_input_names(module, *, owner) -> list[str]:
    """The inner module's input names: its signature, else its forward."""
    signature = getattr(module, "signature", None)
    if signature is not None and hasattr(signature, "input_fields"):
        return list(signature.input_fields)
    parameters = [
        parameter for parameter in inspect.signature(module.forward).parameters.values() if parameter.name != "self"
    ]
    names = []
    for parameter in parameters:
        if parameter.kind in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
            raise ValueError(
                f"{owner} cannot derive the inner {type(module).__name__}'s inputs: its forward takes "
                f"`*{'*' if parameter.kind is inspect.Parameter.VAR_KEYWORD else ''}{parameter.name}`. "
                "Give the module a `signature` attribute or a forward with named parameters."
            )
        names.append(parameter.name)
    if not names:
        raise ValueError(
            f"{owner} cannot derive the inner {type(module).__name__}'s inputs: its forward declares no parameters"
        )
    return names


def _inner_output_names(module) -> list[str]:
    """The inner module's declared outputs, when introspectable (else [])."""
    signature = getattr(module, "signature", None)
    if signature is not None and hasattr(signature, "output_fields"):
        return list(signature.output_fields)
    return []


def _build_reward_tool(reward, *, owner):
    """Wrap one plain reward function as the declared `reward` tool leaf.

    The generated wrapper `def reward(outputs: dict) -> float` embeds the
    user function's source verbatim (the artifact carries what runs),
    calls it with the attempt's outputs record, converts any failure into
    the typed `ToolError`, and coerces the result to a float. Extraction
    runs EAGERLY so an unexportable reward (closure, global reads, no
    source) refuses at construction, not at the first compile.
    """
    if not inspect.isfunction(reward) or reward.__name__ == "<lambda>":
        raise ValueError(
            f"{owner} reward must be a plain named function — it becomes a declared metric leaf whose "
            "source the artifact carries; lambdas and callable objects have no declarable source"
        )
    parameters = list(inspect.signature(reward).parameters.values())
    positional = (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    if len(parameters) != 1 or parameters[0].kind not in positional:
        raise ValueError(
            f"{owner} reward takes the attempt's outputs record as its ONE argument "
            f"(`def {reward.__name__}(outputs) -> float`); read fields by subscript, e.g. outputs['answer']"
        )
    try:
        original = textwrap.dedent(inspect.getsource(reward)).strip()
    except (OSError, TypeError) as error:
        raise ValueError(
            f"{owner} reward {reward.__name__!r} has no introspectable source; declared metric leaves "
            "carry their source into the artifact — author it as a plain named function"
        ) from error
    tree = ast.parse(original)
    if not tree.body or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError(f"{owner} reward {reward.__name__!r} must be a plain named function definition")
    inner_name = tree.body[0].name
    # The wrapper source deliberately carries NO owner-class text: the
    # pool dedupes tools by identity, so the SAME user reward must yield
    # byte-identical wrappers under BestOfN and Refine alike — otherwise
    # Refine(BestOfN(...)) on one shared reward would collide at compile.
    source = (
        "def reward(outputs: dict) -> float:\n"
        '    """Declared reward leaf: score one attempt\'s outputs."""\n'
        f"{textwrap.indent(original, '    ')}\n"
        "    from dspy.core.errors import ToolError\n"
        "    try:\n"
        f"        result = {inner_name}(outputs)\n"
        "    except ToolError:\n"
        "        raise\n"
        "    except Exception as error:\n"
        '        raise ToolError("Execution error in declared reward: " + str(error)) from error\n'
        "    if isinstance(result, bool) or not isinstance(result, (int, float)):\n"
        '        raise ToolError("reward must return a number, got " + type(result).__name__)\n'
        "    return float(result)\n"
    )
    wrapper = load_generated(source, tag=f"reward-{reward.__name__}", name="reward")
    extract_tool(wrapper, name="reward")  # eager: refuse unexportable rewards where they are declared
    return wrapper
