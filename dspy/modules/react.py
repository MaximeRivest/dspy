"""ReAct: reason + act, authored as an ordinary readable forward.

The classic loop — think, pick a tool, observe, repeat, then extract —
written as PLAIN Python the compiler reads directly (node-set v0.4, the
inputs-bag envelope, D-041). Open the file and read the loop:

- the forward is signature-polymorphic: `def forward(self, **inputs)`
  takes whatever the module's signature declares and splats it into the
  step and extract predictors (`self.react(**inputs, ...)`), so one
  forward serves every task — the v0.4 record envelope;
- the loop cap is a LITERAL: `range(self.max_iters)` bakes into the For
  range at compile time (`ir_literals` names it so edits recompile);
- tools are DECLARED leaves: every user tool becomes a uniform-args
  dispatch wrapper (`args: dict`) in the tools table, so one dynamic
  call site `self.tools[name](args=...)` serves them all (SEM-8);
- the trajectory is a plain dict grown with subscript writes;
- failures are the TYPED table only: a parse failure ends the loop, a
  tool failure becomes an observation, both via `except`;
- termination is the extract predictor (a ChainOfThought leaf) — the
  loop's `finish` move is a name check, not a tool.

Nothing is generated: the forward below IS the artifact's forward. The
compiler lowers this exact source into the node set, and the equivalence
test's control arm (`forward_native`) runs this exact Python — two
genuine implementations, not a tree and its printed twin.
"""

from __future__ import annotations

from dspy.core.errors import AdapterParseError, ToolError
from dspy.modules._generate import ToolTable, build_dispatch_tool, normalize_tools
from dspy.modules.chain_of_thought import ChainOfThought
from dspy.modules.module import Module
from dspy.modules.predict import Predict
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
        max_iters: The loop cap, baked into the compiled forward as a
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

    def forward(self, **inputs):
        trajectory = {}
        for idx in range(self.max_iters):
            try:
                pred = self.react(**inputs, trajectory=trajectory)
            except AdapterParseError:  # the engine's typed error table
                break
            trajectory["thought_" + str(idx)] = pred.next_thought
            trajectory["tool_name_" + str(idx)] = pred.next_tool_name
            trajectory["tool_args_" + str(idx)] = pred.next_tool_args
            if pred.next_tool_name == "finish":
                break
            try:
                observation = self.tools[pred.next_tool_name](args=pred.next_tool_args)
            except ToolError:  # the engine's typed error table
                observation = "The tool call failed. Use another tool, or finish."
            trajectory["observation_" + str(idx)] = observation
        final = self.extract(**inputs, trajectory=trajectory)
        return final

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
