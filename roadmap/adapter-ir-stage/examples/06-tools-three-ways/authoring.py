# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy
from dspy.adapters import parse, strategy  # NEW authoring modules


class Research(dspy.Signature):
    """Answer using the available tools."""

    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField(role="tools")
    tool_calls: dspy.ToolCalls = dspy.OutputField(role="tool_calls")
    answer: str = dspy.OutputField()


# The tool-calling FORMAT is a strategy, not a program property.
# Same program (ReAct over Research), three wire conducts.

# 1. Native function calling: the tools leave the token stream and
#    become a request-side array; calls come back on a typed channel.
native_fc = dspy.ChatAdapter.with_strategies(
    tools=strategy.rule(
        predicate=strategy.capability("native_function_calling"),
        hides=["tools", "tool_calls"],
        engine_controls={
            "request_patch": {
                "tools": {"$from": "field:tools"},  # the call's declared tools
                "tool_choice": "auto",
            }
        },
        routings=[strategy.channel("tool_calls", field="tool_calls", coerce="ToolCalls")],
    )
)

# 2. CLI-style text invocation: any instruct model, zero provider
#    features. Fragments teach the line format; a combinator recovers it.
cli_text = dspy.ChatAdapter.with_strategies(
    tools=strategy.rule(
        predicate=strategy.capability("instruct"),
        hides=["tool_calls"],
        fragments=[
            strategy.fragment(
                "user",
                "To call a tool, write a line: !call <tool_name> <json arguments>. Available tools:\n{field('tools')}",
            )
        ],
        routings=[
            strategy.text(
                parse.pipeline(
                    parse.regex(
                        "(?m)^!call (?P<name>[a-zA-Z0-9_]+) (?P<args>\\{.*\\})$",
                        dialect="re2",
                        mode="findall",
                    ),
                    parse.tool_calls(name_group="name", args_group="args", args="json"),
                ),
                field="tool_calls",
                consume=True,
            )
        ],
    )
)

# 3. XML tool blocks: same idea, XML dialect.
xml_blocks = dspy.ChatAdapter.with_strategies(
    tools=strategy.rule(
        predicate=strategy.capability("instruct"),
        hides=["tool_calls"],
        fragments=[
            strategy.fragment(
                "user",
                'To call a tool, emit <tool_call name="NAME">{"arg": ...}</tool_call>. Available tools:\n{field(\'tools\')}',
            )
        ],
        routings=[
            strategy.text(
                parse.pipeline(
                    parse.regex(
                        '<tool_call name="(?P<name>[a-zA-Z0-9_]+)">(?P<args>[^<]*)</tool_call>',
                        dialect="re2",
                        mode="findall",
                    ),
                    parse.tool_calls(name_group="name", args_group="args", args="json"),
                ),
                field="tool_calls",
                consume=True,
            )
        ],
    )
)
