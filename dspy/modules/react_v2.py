"""ReActV2: the v2 agent shape — history events, submit, termination_reason.

The v2 loop differs from ReAct in its channels, authored as PLAIN
readable forwards (node-set v0.4, the inputs-bag envelope, D-041):

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

The forward is signature-polymorphic: `def forward(self, **inputs)` splats
the module's own input record into the step predictor (`**inputs`, the
v0.4 record envelope), and the turn cap is a baked literal
(`range`-free `while turn < self.max_iters`, `max_iters` in `ir_literals`).

A toolless ReActV2 is legal (a pure submit agent). It cannot declare an
empty tools table, so it reads a SECOND plain forward (`_forward_toolless`)
that answers any non-submit call in-forward instead of dispatching. Both
forwards are authored Python the compiler reads directly, and the one
`__init__` picks binds on the instance.

Deliberate trims from the legacy experimental ReActV2 (noted in
BUILD-STATE): no `dspy.History`/`ToolCalls` typed channels (plain
lists/dicts — native-FC strategies arrive with the LM channel surface),
and no forced-submit second pass.
"""

from __future__ import annotations

from dspy.core.errors import AdapterParseError, ToolError
from dspy.modules._generate import ToolTable, build_dispatch_tool, normalize_tools
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.signatures.field import InputField, OutputField
from dspy.signatures.signature import ensure_signature, make_signature

__all__ = ["ReActV2"]

_RESERVED_FIELDS = ("history", "next_thought", "tool_calls", "termination_reason")


class ReActV2(Module):
    """The v2 tool-using agent: history events, batched calls, submit.

    Args:
        signature: The task's signature; its outputs are what a `submit`
            call must carry in its args.
        tools: A list of plain named functions or `dspy.Tool` values (the
            same declared-leaf rules as `ReAct`). May be empty — a pure
            submit agent is legal in v2.
        max_iters: The turn cap, baked into the compiled forward as a
            literal (named in `ir_literals` so edits recompile).

    The prediction carries the signature's outputs plus the v2 channels
    `history` (the event list) and `termination_reason` (`"submit"`,
    `"empty_tool_calls"`, `"parse_error"`, or `"max_iters"`).
    """

    #: max_iters bakes as the turn cap; output_fields bakes as the list of
    #: declared output-field names the forward iterates to build the final
    #: record (the output side; the v0.4 record envelope covers inputs).
    ir_literals = ("max_iters", "output_fields")

    def __init__(self, signature, tools=(), max_iters: int = 20):
        self.signature = signature = ensure_signature(signature)
        self.max_iters = max_iters
        self.output_fields = list(signature.output_fields)

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

        # A toolless agent has no tools table to declare, so it cannot
        # dispatch; it reads the second plain forward, which answers any
        # non-submit call in-forward. The one __init__ picks; both are
        # authored Python the compiler reads.
        if not specs:
            self.forward = self._forward_toolless

    def forward(self, **inputs):
        history = []
        outputs = {}
        termination_reason = "max_iters"
        turn = 0
        while turn < self.max_iters:
            turn = turn + 1
            try:
                pred = self.react(**inputs, history=history)
            except AdapterParseError:  # the engine's typed error table
                termination_reason = "parse_error"
                break
            calls = pred.tool_calls
            if len(calls) == 0:
                termination_reason = "empty_tool_calls"
                break
            results = []
            for call in calls:
                name = call["name"]
                args = call["args"]
                if name == "submit":
                    outputs = args
                    termination_reason = "submit"
                    results.append({"name": "submit", "value": "submitted"})
                else:
                    try:
                        value = self.tools[name](args=args)
                    except ToolError:  # the engine's typed error table
                        value = "The tool call failed. Use another tool, or submit."
                    results.append({"name": name, "value": value})
            event = {"thought": pred.next_thought, "tool_calls": calls, "results": results}
            history.append(event)
            if termination_reason == "submit":
                break
        final = {}
        for field in self.output_fields:
            final[field] = outputs.get(field, "")
        final["history"] = history
        final["termination_reason"] = termination_reason
        return final

    def _forward_toolless(self, **inputs):
        history = []
        outputs = {}
        termination_reason = "max_iters"
        turn = 0
        while turn < self.max_iters:
            turn = turn + 1
            try:
                pred = self.react(**inputs, history=history)
            except AdapterParseError:  # the engine's typed error table
                termination_reason = "parse_error"
                break
            calls = pred.tool_calls
            if len(calls) == 0:
                termination_reason = "empty_tool_calls"
                break
            results = []
            for call in calls:
                name = call["name"]
                args = call["args"]
                if name == "submit":
                    outputs = args
                    termination_reason = "submit"
                    results.append({"name": "submit", "value": "submitted"})
                else:
                    value = "Unknown tool. The only valid call is submit."
                    results.append({"name": name, "value": value})
            event = {"thought": pred.next_thought, "tool_calls": calls, "results": results}
            history.append(event)
            if termination_reason == "submit":
                break
        final = {}
        for field in self.output_fields:
            final[field] = outputs.get(field, "")
        final["history"] = history
        final["termination_reason"] = termination_reason
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
        lines.append(f"({len(specs) + 1}) submit, whose args are the final output field(s) {outputs}.")
        fields = {}
        for name, field in signature.input_fields.items():
            desc = (field.json_schema_extra or {}).get("desc")
            fields[name] = (field.annotation, InputField(desc=desc))
        fields["history"] = (list, InputField(desc="previous turns: thoughts, tool calls, results"))
        fields["next_thought"] = (str, OutputField(desc="reason about the situation and plan the turn"))
        fields["tool_calls"] = (list, OutputField(desc='the calls to run: [{"name": ..., "args": {...}}, ...]'))
        return make_signature(fields, "\n".join(lines), signature_name="ReActV2Step")
