"""ReActV2: the v2 agent shape — history events, submit, termination_reason.

The v2 loop differs from ReAct in its channels, and this rebuild keeps
that shape inside the v0.3 subset:

- each turn the step predictor sees the HISTORY (a plain list of event
  dicts: thought, tool calls, results) instead of a flat trajectory;
- the step emits `tool_calls` — a LIST of `{"name", "args"}` objects —
  and every call in the batch executes (multi-call turns);
- termination is the `submit` convention: a call named `submit` carries
  the final outputs in its args; an empty batch or a parse failure also
  ends the loop;
- `history` and `termination_reason` are DECLARED outputs of the module
  record — engine records carry declared outputs only (PIR-013), so the
  v2 channels are contract, not exhaust.

Deliberate trims from the legacy experimental ReActV2 (noted in
BUILD-STATE): no `dspy.History`/`ToolCalls` typed channels (plain
lists/dicts — native-FC strategies arrive with the LM channel surface),
and no forced-submit second pass.
"""

from __future__ import annotations

from dspy.modules._generate import ToolTable, bind_generated_forward, build_dispatch_tool, normalize_tools
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import ensure_signature, make_signature

__all__ = ["ReActV2"]

_RESERVED_FIELDS = ("history", "next_thought", "tool_calls", "termination_reason")

_FORWARD_TEMPLATE = """\
def forward(self{params}):
    history = []
    outputs = {{}}
    termination_reason = "max_iters"
    turn = 0
    while turn < self.max_iters:
        turn = turn + 1
        try:
            pred = self.react({react_kwargs}, history=history)
        except AdapterParseError:
            termination_reason = "parse_error"
            break
        calls = pred.tool_calls
        if len(calls) == 0:
            termination_reason = "empty_tool_calls"
            break
        results = []
        for call in calls:
            if call["name"] == "submit":
                outputs = call["args"]
                termination_reason = "submit"
                results.append({{"name": "submit", "value": "submitted"}})
            else:
{dispatch}
                results.append({{"name": call["name"], "value": value}})
        event = {{"thought": pred.next_thought, "tool_calls": calls, "results": results}}
        history.append(event)
        if termination_reason == "submit":
            break
    final = {{{output_entries}, "history": history, "termination_reason": termination_reason}}
    return final
"""

#: The dispatch arm when tools exist: the dynamic tool leaf, failures typed.
_DISPATCH_TOOLS = """\
                try:
                    value = self.tools[call["name"]](args=call["args"])
                except ToolError:
                    value = "The tool call failed. Use another tool, or submit."
"""

#: The dispatch arm for a pure submit agent: no tools table is declared,
#: so any non-submit call is answered in-forward.
_DISPATCH_NONE = """\
                value = "Unknown tool. The only valid call is submit."
"""


class ReActV2(Module):
    """The v2 tool-using agent: history events, batched calls, submit.

    Args:
        signature: The task's signature; its outputs are what a `submit`
            call must carry in its args.
        tools: A list of plain named functions or `dspy.Tool` values (the
            same declared-leaf rules as `ReAct`). May be empty — a pure
            submit agent is legal in v2.
        max_iters: The turn cap, baked into the compiled forward as a
            literal (declared via `ir_literals`).

    The prediction carries the signature's outputs plus the v2 channels
    `history` (the event list) and `termination_reason` (`"submit"`,
    `"empty_tool_calls"`, `"parse_error"`, or `"max_iters"`).
    """

    ir_literals = ("max_iters",)

    def __init__(self, signature, tools=(), max_iters: int = 20):
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters

        collisions = sorted(set(signature.fields) & set(_RESERVED_FIELDS))
        if collisions:
            raise ValueError(
                f"ReActV2 reserves the field name(s) {collisions} for its own loop channels; rename "
                f"them in your signature (reserved: {', '.join(_RESERVED_FIELDS)})"
            )
        specs = normalize_tools(tools, owner="ReActV2", reserved=("submit",))
        self._tool_specs = specs
        self.tools = ToolTable((spec.name, build_dispatch_tool(spec, owner="ReActV2")) for spec in specs)

        self.react = Predict(self._react_signature(specs))

        input_names = list(signature.input_fields)
        output_entries = ", ".join(
            f'"{name}": outputs.get("{name}", "")' for name in signature.output_fields
        )
        source = _FORWARD_TEMPLATE.format(
            params="".join(f", {name}" for name in input_names),
            react_kwargs=", ".join(f"{name}={name}" for name in input_names),
            output_entries=output_entries,
            dispatch=(_DISPATCH_TOOLS if specs else _DISPATCH_NONE).rstrip("\n"),
        )
        bind_generated_forward(self, source, tag="react_v2")

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
                f"You are an agent. Use the supplied tools to produce {outputs} from {inputs}.",
                "Each turn you see the history of previous turns: your thought, the tool calls you "
                "made, and their results.",
                "Emit `next_thought` and `tool_calls` — a JSON list of objects, each "
                '`{"name": <tool name>, "args": <JSON object>}`. Every call in the list runs.',
                f"When the final answer is ready, emit one call named `submit` whose args carry {outputs}.",
                "The available tools are:",
            ]
        )
        for index, spec in enumerate(specs):
            lines.append(f"({index + 1}) {spec}")
        lines.append(
            f"({len(specs) + 1}) submit, whose args are the final output field(s) {outputs}."
        )
        fields = {}
        for name, field in signature.input_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, InputField(desc=desc))
        fields["history"] = (list, InputField(desc="previous turns: thoughts, tool calls, results"))
        fields["next_thought"] = (str, OutputField(desc="reason about the situation and plan the turn"))
        fields["tool_calls"] = (list, OutputField(desc='the calls to run: [{"name": ..., "args": {...}}, ...]'))
        return make_signature(fields, "\n".join(lines), signature_name="ReActV2Step")
