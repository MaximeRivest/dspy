"""The template walker and ``preview()``: full message lists, no LM call.

``render_template_messages`` interprets a parsed message-list template the
way the engine pipeline does — system content in schema position, the
demos directive expanded through the historical complete/incomplete
classification, the history directive expanded into prior turns, user
content against the call values — as one pure function. ``preview()``
wraps it with codec defaults so the learn-by-looking loop is always
available (spec section 3, discoverability).

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
    parse_message_template,
)
from dspy.adapters._engine.template.renderer import (
    COMPLETE_DEMO_MISSING_FIELD_MESSAGE,
    INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE,
    INCOMPLETE_DEMO_PREFIX,
    RenderContext,
    TemplateRenderError,
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
) -> list[dict[str, Any]]:
    """Walk a parsed template into the full chat message list, purely."""
    inputs = dict(inputs or {})
    fragments = fragments or {}

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

    parsed_demos = template.demos_directive
    rendered: list[dict[str, Any]] = []
    for message in template.messages:
        if isinstance(message, ContentMessage):
            if message.role == "system":
                # Schema position renders against the full signature (the
                # history field is described even though its turns expand).
                content = render_nodes(message.nodes, ctx("schema", values=inputs, sig=signature))
                rendered.append({"role": "system", "content": content})
            elif message.role == "user":
                content = render_user_content(message.nodes, ctx("user_values", values=inputs))
                if content:
                    rendered.append({"role": "user", "content": content})
            else:  # authored assistant content interpolates the call values
                content = render_nodes(message.nodes, ctx("user_values", values=inputs))
                rendered.append({"role": "assistant", "content": content})
        elif isinstance(message, DemosDirective):
            rendered.extend(_expand_demos(message, signature, demos, ctx))
        elif isinstance(message, HistoryDirective):
            rendered.extend(_expand_history(message, parsed_demos, history_turns, ctx))
    return rendered


def _history_field_name(signature) -> str | None:
    from dspy.adapters.types.history import History

    for name, info in signature.input_fields.items():
        if info.annotation == History:
            return name
    return None


def _pair_nodes(directive, demos_directive):
    """A directive's user/assistant node pair, falling back to the demos
    directive's patterns (history turns follow the demo pair shapes, the
    historical behavior)."""
    if directive.user is not None:
        return directive.user, directive.assistant
    if demos_directive is not None and demos_directive.user is not None:
        return demos_directive.user, demos_directive.assistant
    raise TemplateRenderError(
        "this directive has no user/assistant content templates and no demos directive to inherit them from"
    )


def _expand_demos(directive: DemosDirective, signature, demos, ctx) -> list[dict[str, Any]]:
    if directive.user is None:
        raise TemplateRenderError(
            "the demos directive needs user= and assistant= content templates to expand demos"
        )
    incomplete, complete = classify_demos(signature, demos)
    messages = []
    for demo in incomplete:
        user = render_user_content(
            directive.user, ctx("user_values", values=dict(demo), sig=signature), prefix=INCOMPLETE_DEMO_PREFIX
        )
        assistant = render_nodes(
            directive.assistant,
            ctx("assistant_values", values=dict(demo), missing=INCOMPLETE_DEMO_MISSING_FIELD_MESSAGE, sig=signature),
        )
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    for demo in complete:
        user = render_user_content(directive.user, ctx("user_values", values=dict(demo), sig=signature))
        assistant = render_nodes(
            directive.assistant,
            ctx("assistant_values", values=dict(demo), missing=COMPLETE_DEMO_MISSING_FIELD_MESSAGE, sig=signature),
        )
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


def _expand_history(directive: HistoryDirective, demos_directive, turns, ctx) -> list[dict[str, Any]]:
    user_nodes, assistant_nodes = _pair_nodes(directive, demos_directive)
    messages = []
    for turn in turns:
        user = render_user_content(user_nodes, ctx("user_values", values=dict(turn)))
        if user:
            messages.append({"role": "user", "content": user})
        assistant = render_nodes(assistant_nodes, ctx("assistant_values", values=dict(turn), missing=None))
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
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
) -> list[dict[str, Any]]:
    """Render a template against a signature and values with no LM call.

    ``template`` is either raw message-list data or an already-parsed
    ``ParsedTemplate``. Codecs default to the shared text codec.
    """
    from dspy.adapters._engine.codecs import TEXT_PYTHONISH

    parsed = template if isinstance(template, ParsedTemplate) else parse_message_template(template)
    return render_template_messages(
        parsed,
        signature,
        demos=demos,
        inputs=inputs,
        input_codec=input_codec or TEXT_PYTHONISH,
        output_codec=output_codec or TEXT_PYTHONISH,
        fragments=fragments,
    )
