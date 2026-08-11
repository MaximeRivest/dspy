"""ReAct: reason + act, built directly as v0.3 nodes.

The classic loop — think, pick a tool, observe, repeat, then extract —
authored so the compiled artifact is the whole story:

- the loop cap is a LITERAL: `max_iters` bakes into the For range when
  the forward tree is built (`ir_literals` names it so edits recompile);
- tools are DECLARED leaves: every user tool becomes a uniform-args
  dispatch wrapper (`args: dict`) in the tools table, so one dynamic
  call site `self.tools[name](args=...)` serves them all (SEM-8);
- the trajectory is a plain dict grown with SetIndex writes;
- failures are the TYPED table only: a parse failure ends the loop, a
  tool failure becomes an observation, both via `except`;
- termination is the extract predictor (a ChainOfThought leaf) — the
  loop's `finish` move is a name check, not a tool.

The forward is BUILT with the typed node constructors (`programir.build`)
— no source templates. The native twin is DERIVED: the printer renders
the built tree to source, registers it in linecache, and binds it on the
instance, so `forward_native` runs genuine Python that agrees with the
engine arm by construction.
"""

from __future__ import annotations

from dspy.modules._generate import ToolTable, build_dispatch_tool, normalize_tools
from dspy.modules.chain_of_thought import ChainOfThought
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.programir.build import (
    Assign,
    Attr,
    BinOp,
    Break,
    Builtin,
    CallPredict,
    CallToolDynamic,
    Compare,
    Const,
    Dict,
    Except,
    For,
    Forward,
    If,
    Return,
    SetIndex,
    Try,
    Var,
)
from dspy.programir.printer import bind_forward
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import ensure_signature, make_signature

__all__ = ["ReAct"]

_RESERVED_FIELDS = ("trajectory", "next_thought", "next_tool_name", "next_tool_args", "reasoning")


class ReAct(Module):
    """A tool-using agent over any signature.

    Args:
        signature: The task's signature (`"question -> answer"` or a
            `dspy.Signature`); its inputs become the forward's inputs,
            its outputs the extract predictor's contract.
        tools: A list of plain named functions or `dspy.Tool` values —
            full type hints, self-contained bodies (they become declared
            tool leaves in the artifact). At least one.
        max_iters: The loop cap, baked into the built forward as a
            literal (named in `ir_literals` so edits recompile).

    Examples:
        ```python
        def get_weather(city: str) -> str:
            "One city's weather, as a sentence."
            return f"The weather in {city} is sunny."

        react = dspy.ReAct("question -> answer", tools=[get_weather])
        react.set_lm(dspy.LM("openai/gpt-4o-mini"))
        prediction = react(question="What is the weather in Tokyo?")
        ```
    """

    ir_literals = ("max_iters",)

    def __init__(self, signature, tools, max_iters: int = 20):
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        collisions = sorted(set(signature.fields) & set(_RESERVED_FIELDS))
        if collisions:
            raise ValueError(
                f"ReAct reserves the field name(s) {collisions} for its own loop channels; rename them "
                f"in your signature (reserved: {', '.join(_RESERVED_FIELDS)})"
            )
        specs = normalize_tools(tools, owner="ReAct", reserved=("finish",))
        if not specs:
            raise ValueError(
                "ReAct without tools is ChainOfThought wearing a loop; pass at least one tool, or use "
                "dspy.ChainOfThought directly"
            )
        self._tool_specs = specs
        self.tools = ToolTable((spec.name, build_dispatch_tool(spec, owner="ReAct")) for spec in specs)

        self.react = Predict(self._react_signature(specs))
        self.extract = ChainOfThought(self._extract_signature())

        self.build_forward_ir()

    def build_forward_ir(self):
        """Build the forward tree (current literals) and refresh the twin.

        The compiler calls this at every compile, so an edited `max_iters`
        bakes freshly; the printed native `forward` rebinds alongside.
        """
        tree = self._forward_tree()
        bind_forward(self, tree, tag="react")
        return tree

    def _forward_tree(self):
        inputs = list(self.signature.input_fields)
        step_kwargs = {name: Var(name) for name in inputs}

        def journal(prefix):
            return BinOp("add", Const(prefix), Builtin("str", Var("idx")))

        loop_body = [
            Try(
                [Assign("pred", CallPredict("react", **step_kwargs, trajectory=Var("trajectory")))],
                [Except("AdapterParseError", [Break()])],
            ),
            SetIndex("trajectory", journal("thought_"), Attr("pred", "next_thought")),
            SetIndex("trajectory", journal("tool_name_"), Attr("pred", "next_tool_name")),
            SetIndex("trajectory", journal("tool_args_"), Attr("pred", "next_tool_args")),
            If(Compare("eq", Attr("pred", "next_tool_name"), Const("finish")), [Break()]),
            Try(
                [
                    Assign(
                        "observation",
                        CallToolDynamic(Attr("pred", "next_tool_name"), args=Attr("pred", "next_tool_args")),
                    )
                ],
                [
                    Except(
                        "ToolError",
                        [Assign("observation", Const("The tool call failed. Use another tool, or finish."))],
                    )
                ],
            ),
            SetIndex("trajectory", journal("observation_"), Var("observation")),
        ]
        return Forward(
            inputs,
            [
                Assign("trajectory", Dict()),
                For("idx", loop_body, range=self.max_iters),
                Assign("final", CallPredict("extract", **step_kwargs, trajectory=Var("trajectory"))),
                Return(Var("final")),
            ],
        )

    def _react_signature(self, specs):
        signature = self.signature
        inputs = ", ".join(f"`{name}`" for name in signature.input_fields)
        outputs = ", ".join(f"`{name}`" for name in signature.output_fields)
        lines = []
        if signature.instructions:
            lines.append(signature.instructions)
            lines.append("")
        lines.extend(
            [
                f"You are an agent. In each step you see the input field(s) {inputs} and the trajectory so far.",
                f"Your goal is to use the supplied tools to collect everything needed to produce {outputs}.",
                "In each step, produce `next_thought`, `next_tool_name`, and `next_tool_args`; the tool's "
                "observation is appended to the trajectory.",
                "The `next_tool_name` must be one of:",
            ]
        )
        for index, spec in enumerate(specs):
            lines.append(f"({index + 1}) {spec}")
        lines.append(
            f"({len(specs) + 1}) finish, whose args are {{}}: signals that everything needed to produce "
            f"{outputs} is now in the trajectory."
        )
        lines.append("The `next_tool_args` value must be a JSON object with the chosen tool's arguments.")

        fields = {}
        for name, field in signature.input_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, InputField(desc=desc))
        fields["trajectory"] = (dict, InputField(desc="the run so far: thoughts, tool calls, observations"))
        fields["next_thought"] = (str, OutputField(desc="reason about the situation and plan the next step"))
        fields["next_tool_name"] = (str, OutputField(desc="the exact name of the one tool to run next"))
        fields["next_tool_args"] = (dict, OutputField(desc="the chosen tool's arguments, as a JSON object"))
        return make_signature(fields, "\n".join(lines), signature_name="ReActStep")

    def _extract_signature(self):
        signature = self.signature
        fields = {}
        for name, field in signature.input_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, InputField(desc=desc))
        fields["trajectory"] = (dict, InputField(desc="the completed run: thoughts, tool calls, observations"))
        for name, field in signature.output_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, OutputField(desc=desc))
        return make_signature(fields, signature.instructions, signature_name="ReActExtract")
