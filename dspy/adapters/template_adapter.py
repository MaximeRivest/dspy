"""TemplateAdapter: your messages are the prompt.

The template authoring surface (Epic D). Where the class adapters render a
fixed prompt shape, a ``TemplateAdapter`` renders exactly the message list
you author — the signature stays the I/O contract, the template decides
what the model sees, and nothing is added that you didn't write:

    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "You are a concise assistant. {instruction}"},
            {"role": "user", "content": "Summarize:\\n\\n{text}"},
        ],
        parse_mode="full_text",
    )

Content strings speak the constrained template language (adapter IR
contract, section 3): ``{field}`` value slots, ``{instruction}``, aggregate
slots (``{inputs(...)}``, ``{outputs(...)}``, ``{demos(...)}``,
``{history(...)}``), loop blocks over fields, ``{"role": "demos"}`` /
``{"role": "history"}`` directive messages, positional fragment slots, and
``{{``/``}}`` escapes. The language is closed on purpose — every template
is analyzable, diffable data that serializes into a program artifact and
loads with no exec. Templates parse eagerly at construction with teaching
errors, and the adapter is strategy-aware from birth: natively-served
fields leave the token stream, and the bake-time capacity check refuses a
call whose data has no lane in your template rather than degrade silently.
"""

from typing import Any

from dspy.adapters.serde import PresetAdapter


class TemplateAdapter(PresetAdapter):
    """An adapter authored as a literal message-list template.

    Args:
        messages: The prompt as a list of ``{"role": ..., "content": ...}``
            template messages (plus optional ``demos``/``history``
            directive messages). Parsed eagerly; unknown constructs refuse
            at construction naming the valid vocabulary.
        parse_mode: How output fields are recovered from the completion —
            ``"json"`` (default; keys match output fields), ``"chat"``
            (``[[ ## field ## ]]`` markers), ``"xml"``
            (``<field>value</field>`` tags), or ``"full_text"`` (the whole
            completion into exactly one output field).
        name: The component-4 entry name ``dump_entry()`` serializes under.
        codecs: Directional value-codec bindings, e.g.
            ``{"input": "baml"}``. Unbound directions use the shared text
            codec.
        strategies: Per-role strategy bindings, e.g.
            ``{"reasoning": "textual_field"}`` — the same binding surface
            the class adapters take; unbound roles resolve ``"auto"``
            against the LM's capabilities at bake.
        config: Extra entry config, carried through serialization.
        **kwargs: Base :class:`~dspy.adapters.base.Adapter` options
            (``callbacks``, ``use_native_function_calling``, ...).

    ``preview(signature, demos=..., inputs=...)`` renders the exact
    messages with no LM call; ``dump_entry()`` serializes the adapter as
    pure data and ``dspy.adapters.load_entry`` links it back.
    """

    def __init__(
        self,
        messages: list[dict[str, Any]],
        parse_mode: str = "json",
        *,
        name: str = "template",
        codecs: dict[str, str] | None = None,
        strategies: dict[str, str] | None = None,
        config: dict[str, Any] | None = None,
        **kwargs,
    ):
        from dspy.adapters._engine.presets import _AUTO_STRATEGIES, _make_preset
        from dspy.adapters._engine.strategies.vocabulary import check_binding_name
        from dspy.adapters._engine.template.vocabulary import VOCABULARY

        for role, strategy_name in (strategies or {}).items():
            check_binding_name(role, strategy_name)
        if not isinstance(parse_mode, str):
            raise ValueError(
                f"parse_mode takes a parser-binding name — one of: {', '.join(VOCABULARY['parsers'])}. "
                "Authored parsers ship through the registration API (adapter IR contract, section 9), "
                "not the constructor."
            )
        unknown_directions = set(codecs or {}) - {"input", "output"}
        if unknown_directions:
            raise ValueError(
                f"codecs= binds the 'input' and 'output' directions only, got {sorted(unknown_directions)}"
            )
        preset = _make_preset(
            name=name,
            template_messages=messages,
            parser=parse_mode,
            codecs={"input": "text_pythonish", "output": "text_pythonish", **(codecs or {})},
            strategies={**_AUTO_STRATEGIES, **(strategies or {})},
            config=dict(config or {}),
        )
        super().__init__(preset, **kwargs)
