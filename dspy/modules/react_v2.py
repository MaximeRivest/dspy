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

The forward is BUILT with the typed node constructors; the native twin
is the printer's rendering of the same tree (linecache-registered, bound
on the instance). A toolless ReActV2 is legal (pure submit agent): its
tree swaps the dispatch arm for an in-forward answer, since an empty
tools table cannot be declared.

Deliberate trims from the legacy experimental ReActV2 (noted in
BUILD-STATE): no `dspy.History`/`ToolCalls` typed channels (plain
lists/dicts — native-FC strategies arrive with the LM channel surface),
and no forced-submit second pass.
"""

from __future__ import annotations

from dspy.modules._generate import ToolTable, build_dispatch_tool, normalize_tools
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
    Do,
    Except,
    For,
    Forward,
    If,
    Index,
    List,
    Method,
    Return,
    Try,
    Var,
    While,
)
from dspy.programir.printer import bind_forward
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
        max_iters: The turn cap, baked into the built forward as a
            literal (named in `ir_literals` so edits recompile).

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

        self.build_forward_ir()

    def build_forward_ir(self):
        """Build the forward tree (current literals) and refresh the twin."""
        tree = self._forward_tree()
        bind_forward(self, tree, tag="react_v2")
        return tree

    def _forward_tree(self):
        signature = self.signature
        inputs = list(signature.input_fields)
        step_kwargs = {name: Var(name) for name in inputs}
        call_name = Index(Var("call"), Const("name"))
        call_args = Index(Var("call"), Const("args"))

        if self._tool_specs:
            dispatch = [
                Try(
                    [Assign("value", CallToolDynamic(call_name, args=call_args))],
                    [
                        Except(
                            "ToolError", [Assign("value", Const("The tool call failed. Use another tool, or submit."))]
                        )
                    ],
                )
            ]
        else:
            # A pure submit agent declares no tools table; any non-submit
            # call is answered in-forward.
            dispatch = [Assign("value", Const("Unknown tool. The only valid call is submit."))]

        per_call = If(
            Compare("eq", call_name, Const("submit")),
            [
                Assign("outputs", call_args),
                Assign("termination_reason", Const("submit")),
                Do(Method(Var("results"), "append", Dict({"name": "submit", "value": "submitted"}))),
            ],
            [
                *dispatch,
                Do(Method(Var("results"), "append", Dict({"name": call_name, "value": Var("value")}))),
            ],
        )
        turn_body = [
            Assign("turn", BinOp("add", Var("turn"), Const(1))),
            Try(
                [Assign("pred", CallPredict("react", **step_kwargs, history=Var("history")))],
                [Except("AdapterParseError", [Assign("termination_reason", Const("parse_error")), Break()])],
            ),
            Assign("calls", Attr("pred", "tool_calls")),
            If(
                Compare("eq", Builtin("len", Var("calls")), Const(0)),
                [Assign("termination_reason", Const("empty_tool_calls")), Break()],
            ),
            Assign("results", List()),
            For("call", [per_call], iter=Var("calls")),
            Assign(
                "event",
                Dict({"thought": Attr("pred", "next_thought"), "tool_calls": Var("calls"), "results": Var("results")}),
            ),
            Do(Method(Var("history"), "append", Var("event"))),
            If(Compare("eq", Var("termination_reason"), Const("submit")), [Break()]),
        ]
        final = Dict(
            [
                *[
                    (Const(name), Method(Var("outputs"), "get", Const(name), Const("")))
                    for name in signature.output_fields
                ],
                (Const("history"), Var("history")),
                (Const("termination_reason"), Var("termination_reason")),
            ]
        )
        return Forward(
            inputs,
            [
                Assign("history", List()),
                Assign("outputs", Dict()),
                Assign("termination_reason", Const("max_iters")),
                Assign("turn", Const(0)),
                While(Compare("lt", Var("turn"), Const(self.max_iters)), turn_body),
                Assign("final", final),
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
