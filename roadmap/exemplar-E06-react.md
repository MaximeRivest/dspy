# E-06 — ReAct: dynamic tools, error flow, turns, and the async twin

**Proves (gaps G2, G9, G16):** model-dispatched tool leaves
(`tools[pred.name]`), `For`/`Break`/`Try`/`ExceptHandler`/`Raise` as
bytes, the `turns` face carrying a tool exchange across two LM calls,
the `history` role, and the D-037 aforward rule (async twin ⇒ identical
graph). Successor to server example 14; the equivalence oracle for this
family is STRUCTURAL (call sequence, tool observations, branch/except
paths), not bit-for-bit — ex-14's ruling, unchanged.

## 1. The program

```python
import dspy

lm = dspy.Model("openai-chat:gpt-4o-mini", native_fc=True)

def lookup(city: str) -> str:
    # deps: httpx
    """Population of a city, from the internal gazetteer."""
    ...

def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression."""
    ...


class Researcher(dspy.Module):
    def __init__(self):
        self.tools = {"lookup": dspy.Tool(lookup),
                      "calculate": dspy.Tool(calculate)}
        self.react = dspy.Predict(
            "question, trajectory: dspy.History -> "
            "next_thought, next_tool_name, next_tool_args: dict",
            model=lm)                     # tools role served native-FC or textual —
                                          # the STRATEGY decides; the program never knows
        self.extract = dspy.Predict("question, trajectory: dspy.History -> answer",
                                    model=lm)

    def forward(self, inputs):
        trajectory = []
        for idx in range(8):                             # For(range) + Break below
            pred = self.react(question=inputs.question, trajectory=trajectory)
            if pred.next_tool_name == "finish":
                break
            try:
                tool = self.tools[pred.next_tool_name]   # DYNAMIC leaf: model picks
                obs = tool(**pred.next_tool_args)        # splat of model-decided args —
            except dspy.ToolError as e:                  #   lawful ONLY through the tool
                obs = "tool failed: " + e.message        #   leaf's declared arg schema
            trajectory.append({"thought": pred.next_thought,
                               "tool": pred.next_tool_name, "obs": obs})
        if len(trajectory) == 8:
            raise dspy.ModuleError("no answer after 8 steps")
        return self.extract(question=inputs.question, trajectory=trajectory)

    async def aforward(self, inputs):                    # D-037: the twin — every leaf
        trajectory = []                                  # call awaited, SAME graph bytes;
        for idx in range(8):                             # compile asserts §3 is identical
            pred = await self.react.acall(question=inputs.question, trajectory=trajectory)
            ...                                          # (body mirrors forward, verbatim)
```

Notes against the drafts: unknown tool name at runtime raises a
catchable `ToolError` (ex-14 semantics — R-02 is the *compile* refusal
for static undeclared calls; dynamic dispatch resolves against the
declared `self.tools` map, so the leaf set is still closed).
`trajectory` is a plain value threaded through calls — carrying is
orchestration (component 5), spelling is the adapter's (`turns`).

## 2. The signature bytes — history + tools roles (component 2)

```jsonc
"2_signature": {
  "react": {"fields": [
    {"name": "question",       "direction": "input",  "shape": {"type": "string"}, "semantic_role": "plain"},
    {"name": "trajectory",     "direction": "input",  "shape": {"$ref": "#/shapes/history"},
     "semantic_role": "history",  "role_resolved_from": "shape:History"},
    {"name": "next_thought",   "direction": "output", "shape": {"type": "string"}, "semantic_role": "reasoning"},
    {"name": "next_tool_name", "direction": "output", "shape": {"type": "string"}, "semantic_role": "tool_calls"},
    {"name": "next_tool_args", "direction": "output", "shape": {"type": "object"}, "semantic_role": "tool_calls"}
  ]}
}
```

The `history` role's law (lm15 `ContinuationState`): strategies must
carry provider continuation state through render, never strip it —
thinking-block replay survives the ping-pong.

## 3. The forward bytes (component 5) — the four nodes E-06 exists for

```jsonc
"5_forward": {"Researcher.forward": {"body": [
  {"node": "Assign", "target": "trajectory", "value": {"node": "List", "items": []}},
  {"node": "For", "target": "idx",
   "iter": {"node": "Call", "builtin": "range", "args": [{"node": "Const", "value": 8}]},
   "body": [
     {"node": "Assign", "target": "pred",
      "value": {"node": "Call", "leaf": "react", "kwargs": {"…": "…"}}},
     {"node": "If",
      "test": {"node": "Compare", "op": "eq",
               "left": {"node": "Attr", "value": {"node": "Var", "name": "pred"}, "attr": "next_tool_name"},
               "right": {"node": "Const", "value": "finish"}},
      "body": [{"node": "Break"}], "orelse": []},
     {"node": "Try",
      "body": [
        {"node": "Assign", "target": "obs",
         "value": {"node": "CallDynamic",                       // ⟂ the dynamic-leaf node:
                   "pool": "tools",                             //   resolves in the DECLARED
                   "name": {"node": "Attr", "value": {"node": "Var", "name": "pred"},
                            "attr": "next_tool_name"},          //   tool map at runtime;
                   "splat": {"node": "Attr", "value": {"node": "Var", "name": "pred"},
                             "attr": "next_tool_args"}}}],      //   unknown name ⇒ ToolError
      "handlers": [{"node": "ExceptHandler", "type": "ToolError", "name": "e",
        "body": [{"node": "Assign", "target": "obs",
                  "value": {"node": "BinOp", "op": "add",
                            "left": {"node": "Const", "value": "tool failed: "},
                            "right": {"node": "Attr", "value": {"node": "Var", "name": "e"}, "attr": "message"}}}]}]},
     {"node": "Expr", "value": {"node": "Call", "method": "append",
        "on": {"node": "Var", "name": "trajectory"}, "args": [{"node": "Dict", "…": "…"}]}}
   ]},
  {"node": "If",
   "test": {"node": "Compare", "op": "eq",
            "left": {"node": "Call", "builtin": "len", "args": [{"node": "Var", "name": "trajectory"}]},
            "right": {"node": "Const", "value": 8}},
   "body": [{"node": "Raise", "type": "ModuleError",
             "message": {"node": "Const", "value": "no answer after 8 steps"}}], "orelse": []},
  {"node": "Return", "value": {"node": "Call", "leaf": "extract", "kwargs": {"…": "…"}}}
]},
"Researcher.aforward": {"same_graph_as": "Researcher.forward"}}   // D-037: the twin is a
                                                                  // POINTER, not a copy —
                                                                  // divergence is a compile error
```

Byte-shape note for ratification: `CallDynamic` (pool + runtime name +
schema-checked splat) is the model-dispatched convention of §d as an
explicit node; ex-14 interpreted the same thing from a subscripted Call.
Pick ONE encoding at ratification — this file proposes the named node
because analyses (flow, cost, lint) then see dispatch without pattern-
matching subscripts. The splat of model-decided args stays lawful ONLY
into a tool leaf's declared arg schema (D-041's line: splat what the
signature declares — the tool's schema IS the declaration here;
arbitrary `**model_output` into a non-leaf refuses).

## 4. Tools with BOTH dispatch modes (component 6)

```jsonc
"6_tools": {
  "lookup":    {"name": "lookup", "parameters": {"…": "…"}, "return_schema": {"type": "string"},
                "source": "tools/lookup.py", "deps": ["httpx"], "language": "python",
                "authored_by": "human", "kind": "call", "dispatch": ["model"],   // ⟂ DRAFT: which
                "placement": {"rung": "in_process", "contract": "tool/call-1"}}, //   modes are lawful
  "calculate": {"…": "…", "dispatch": ["model"]}
}
```

## 5. The turns face, end-to-end (component 4) — the two-call ping-pong

Call N: the model emits tool calls (native channel or textual — the
strategy's routing reads them). Call N+1: the SAME rule's `turns` face
spells yesterday's calls + results into today's messages. One
convention, three places (fragment teaches, routing reads, turns
echoes), probe-locked at registration (R-15).

```jsonc
"4_adapter": {"chat": {"…": "…",
  "strategies": {"tools": {"kind": "rule",
    "predicate": {"capability": "native_function_calling"},
    "routings": [{"kind": "channel", "channel": "tool_calls", "field": "next_tool_name+next_tool_args"}],
    "turns": {"assistant": {"kind": "native"}, "result": {"kind": "native"}},
    "fallback": "textual_heredoc"},
   "history": {"kind": "rule",
    "predicate": {"capability": "instruct"},
    "turns": {"assistant": {"kind": "native"}, "result": {"kind": "native"}},
    "continuation": "carry"}}}}                       // lm15 ContinuationState: never stripped
```

Swap the bound model for one without native FC and ONLY the strategy
resolution changes: `textual_heredoc` renders calls as heredoc text,
its routing regex reads them back, its `turns` echoes them — same
program, same signature, same forward bytes. That is the polyfill loop
as a one-field diff.

## 6. What a conformant engine must reproduce (the structural oracle)

Per ex-14: identical call sequence (react → tool → … → extract), tool
observations and error paths (including the except branch when a tool
raises and the `Raise` when the loop exhausts), branch decisions, the
first call's artifact-determined rendered messages, and raise
type+message. Free-text thought equality is informational. `aforward`
runs must satisfy the SAME oracle against the SAME graph bytes.

## 7. Refusal hooks exercised (cross-ref E-11)

R-02 (a static call to a tool not in `self.tools`), R-04 (a `gather`
variant of aforward), R-15 (a turns-drift rule), plus one new vector
this file contributes: **R-33 · dynamic dispatch outside the declared
pool** — `self.anything[pred.name](...)` where `anything` is not a
declared tool map:
> `forward: dynamic call at modules.py:31 dispatches over 'self.anything' which is not a declared tool pool — model-dispatched calls resolve only within declared leaves`
