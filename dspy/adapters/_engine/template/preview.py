"""The template walker and ``preview()``: full message lists, no LM call.

``render_template_messages`` interprets a parsed message-list template the
way the engine pipeline does — system content in schema position, the
demos directive expanded through the historical complete/incomplete
classification, the history directive expanded into prior turns, user
content against the call values — as one pure function. ``preview()``
wraps it with codec defaults so the learn-by-looking loop is always
available (spec section 3, discoverability).

Message-role asymmetry, declared (spec section 3, user-turn assembly):
every user message assembles through the historical join-then-strip
(``render_user_content``) and is OMITTED from the list when it assembles
to nothing; system and assistant messages emit their rendered bytes
verbatim, and always emit.

Deliberate limits, documented rather than approximated: history turns
expand as plain user/assistant pairs (the engine's native tool-call replay
machinery lives with the pipeline, not the language), and custom-type
markers are left unexpanded (part splitting is the frontend's job).
"""

from typing import Any

from dspy.adapters._engine.template.parser import (
    ContentMessage,
    DemosDirective,
    HistoryDirective,
    ParsedTemplate,
    directive_pair,
    parse_message_template,
)
from dspy.adapters._engine.template.renderer import (
    COMPLETE_DEMO_MISSING_FIELD_MESSAGE,
    INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE,
    RenderContext,
    classify_demos,
    render_nodes,
    render_user_content,
)


def render_template_messages(
    template: ParsedTemplate,
    signature,
    demos=(),
    inputs: dict[str, Any] | None = None,
    *,
    input_codec,
    output_codec,
    fragments: dict[str, list[str]] | None = None,
    parser: str | None = None,
    turns_spec: dict | None = None,
) -> list[dict[str, Any]]:
    """Walk a parsed template into the full chat message list, purely.

    ``parser`` names the preset's parser binding when rendering through a
    preset context; bare directives then expand through the matching
    default turn patterns (spec section 3). None keeps the chat marker
    fallback.
    """
    inputs = dict(inputs or {})
    fragments = fragments or {}

    _refuse_second_history_field(signature)
    history_field_name = _history_field_name(signature)
    history_turns: list[dict[str, Any]] = []
    values_signature = signature
    if history_field_name and history_field_name in inputs:
        history_value = inputs.pop(history_field_name)
        history_turns = list(getattr(history_value, "messages", history_value) or [])
        values_signature = signature.delete(history_field_name)

    def ctx(mode, values=None, missing=None, sig=None):
        return RenderContext(
            signature=sig if sig is not None else values_signature,
            mode=mode,
            input_codec=input_codec,
            output_codec=output_codec,
            values=values,
            missing_field_message=missing,
            fragments=fragments,
            demos=tuple(demos),
            history=tuple(history_turns),
        )

    rendered: list[dict[str, Any]] = []
    for message in template.messages:
        if isinstance(message, ContentMessage):
            if message.role == "system":
                # Schema position renders against the full signature (the
                # history field is described even though its turns expand)
                # and WITHOUT call values — the same context the engine
                # delegation path builds, so preview bytes are engine bytes.
                content = render_nodes(message.nodes, ctx("schema", sig=signature))
                rendered.append({"role": "system", "content": content})
            elif message.role == "user":
                content = render_user_content(message.nodes, ctx("user_values", values=inputs))
                if content:  # empty user turns are omitted (declared, spec section 3)
                    rendered.append({"role": "user", "content": content})
            else:  # authored assistant content interpolates the call values
                content = render_nodes(message.nodes, ctx("user_values", values=inputs))
                rendered.append({"role": "assistant", "content": content})
        elif isinstance(message, DemosDirective):
            rendered.extend(_expand_demos(message, signature, demos, ctx, parser))
        elif isinstance(message, HistoryDirective):
            rendered.extend(_expand_history(message, history_turns, ctx, parser, turns_spec))
    return rendered


def _history_field_name(signature) -> str | None:
    from dspy.adapters.types.history import History

    for name, info in signature.input_fields.items():
        if info.annotation == History:
            return name
    return None


def _refuse_second_history_field(signature) -> None:
    """A template hosts exactly one conversation history; a second History
    field's turns would silently vanish — refuse naming both fields (D-δ)."""
    from dspy.adapters.types.history import History

    names = [name for name, info in signature.input_fields.items() if info.annotation == History]
    if len(names) > 1:
        raise ValueError(
            f"signature declares {len(names)} History fields ({', '.join(names)}) but a template hosts "
            f"exactly one conversation history — the turns of {', '.join(names[1:])} would silently "
            "vanish; merge the histories into one field"
        )


def _expand_demos(directive: DemosDirective, signature, demos, ctx, parser=None) -> list[dict[str, Any]]:
    user_nodes, assistant_nodes = directive_pair(directive, parser)
    incomplete, complete = classify_demos(signature, demos)
    messages = []
    for demo in incomplete:
        # The incomplete-demo preamble is directive DATA (D-δ): an authored
        # directive without the key gets exactly the author's pattern; the
        # builtin presets carry the historical sentence explicitly.
        user = render_user_content(
            user_nodes, ctx("user_values", values=dict(demo), sig=signature), prefix=directive.preamble or ""
        )
        assistant = render_nodes(
            assistant_nodes,
            ctx("assistant_values", values=dict(demo), missing=INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE, sig=signature),
        )
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    for demo in complete:
        user = render_user_content(user_nodes, ctx("user_values", values=dict(demo), sig=signature))
        assistant = render_nodes(
            assistant_nodes,
            ctx("assistant_values", values=dict(demo), missing=COMPLETE_DEMO_MISSING_FIELD_MESSAGE, sig=signature),
        )
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def _expand_history(
    directive: HistoryDirective, turns, ctx, parser=None, turns_spec=None
) -> list[dict[str, Any]]:
    from dspy.adapters.types.tool import ToolCalls

    user_nodes, assistant_nodes = directive_pair(directive, parser)
    messages = []
    for turn in turns:
        turn = dict(turn)
        calls_value = next((v for v in turn.values() if isinstance(v, ToolCalls)), None)
        if turns_spec is not None and calls_value is not None:
            messages.extend(
                _expand_tool_turn(turn, calls_value, turns_spec, user_nodes, assistant_nodes, ctx)
            )
            continue
        user = render_user_content(user_nodes, ctx("user_values", values=turn))
        if user:
            messages.append({"role": "user", "content": user})
        assistant = render_nodes(assistant_nodes, ctx("assistant_values", values=turn, missing=None))
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    return messages


def _expand_tool_turn(turn, calls_value, turns_spec, user_nodes, assistant_nodes, ctx) -> list[dict[str, Any]]:
    """One past tool exchange, spelled by the active tools rule's `turns`
    face: user inputs, then the calls as the assistant turn, then the
    results — template text or native message shapes."""
    from dspy.adapters._engine.template.turns import normalized_results, render_turn_content, result_text
    from dspy.adapters.types.tool import ToolCalls

    values = {name: value for name, value in turn.items() if not isinstance(value, ToolCalls)}
    messages: list[dict[str, Any]] = []
    user = render_user_content(user_nodes, ctx("user_values", values=values))
    if user:
        messages.append({"role": "user", "content": user})

    assistant_text = render_nodes(assistant_nodes, ctx("assistant_values", values=values, missing=None)) or ""
    calls = [{"id": call.id, "name": call.name, "args": call.args} for call in calls_value.tool_calls]
    assistant_spec = turns_spec.get("assistant")
    if assistant_spec is None:
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
    elif assistant_spec["kind"] == "native":
        messages.append({"role": "assistant", "content": assistant_text, "tool_calls": [dict(c) for c in calls]})
    else:
        spelled = render_turn_content(assistant_spec["content"], calls=calls)
        content = f"{assistant_text}\n{spelled}".strip("\n") if assistant_text else spelled
        messages.append({"role": "assistant", "content": content})

    result_spec = turns_spec.get("result")
    results = normalized_results(calls, calls_value.tool_call_results)
    if result_spec is not None and results:
        if result_spec["kind"] == "native":
            for item in results:
                messages.append(
                    {"role": "tool", "tool_call_id": item.get("id"), "content": result_text(item["value"])}
                )
        else:
            messages.append({"role": "user", "content": render_turn_content(result_spec["content"], results=results)})
    return messages


def preview(
    template,
    signature,
    *,
    demos=(),
    inputs: dict[str, Any] | None = None,
    fragments: dict[str, list[str]] | None = None,
    input_codec=None,
    output_codec=None,
    parser: str | None = None,
) -> list[dict[str, Any]]:
    """Render a template against a signature and values with no LM call.

    ``template`` is either raw message-list data or an already-parsed
    ``ParsedTemplate``. Codecs default to the shared text codec; ``parser``
    keys bare-directive default patterns when previewing a preset.
    """
    from dspy.adapters.codecs import TEXT_PYTHONISH

    parsed = template if isinstance(template, ParsedTemplate) else parse_message_template(template)
    return render_template_messages(
        parsed,
        signature,
        demos=demos,
        inputs=inputs,
        input_codec=input_codec or TEXT_PYTHONISH,
        output_codec=output_codec or TEXT_PYTHONISH,
        fragments=fragments,
        parser=parser,
    )
