"""Built-in presets: chat, json, xml — templates in, adapters out.

A preset is nothing but an entry: a template in the constrained language,
the lens as its parser (the four historical parser programs are lens
derivations of their own templates now), text codecs both ways, and the
auto strategy block. `ChatAdapter()`, `JSONAdapter()`, `XMLAdapter()` are
thin constructors over those entries — same user syntax, no class bodies.

`make_adapter` is the general door: template-first adapter construction,
the derived lens as the default parser when `parser` is omitted.

The byte-parity traps live in the template data on purpose (`strip` loop
flags, the trailing newline after the completed marker, fragment slots
alone on their lines so an unused slot costs zero bytes).
"""

from typing import Any

from dspy.adapters._engine.template.renderer import INCOMPLETE_DEMO_PREFIX
from dspy.adapters.adapter import DEFAULT_STRATEGIES, DERIVED, Adapter

__all__ = ["ChatAdapter", "JSONAdapter", "XMLAdapter", "make_adapter"]


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

_CHAT_SYSTEM = """\
Your input fields are:
{% for f in inputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
Your output fields are:
{% for f in outputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
All interactions will be structured in the following way, with the appropriate values filled in.

{% for f in inputs separator='\\n\\n' strip %}
[[ ## {f.name} ## ]]
{{{f.name}}}
{% endfor %}

{% for f in outputs separator='\\n\\n' strip %}
[[ ## {f.name} ## ]]
{f.typed_placeholder}
{% endfor %}

[[ ## completed ## ]]
{fragments('system')}
In adhering to this structure, your objective is: {instruction(style='indented')}"""

_CHAT_DEMO_USER = """\
{% for f in inputs separator='\\n\\n' %}
[[ ## {f.name} ## ]]
{f.value}
{% endfor %}"""

_CHAT_DEMO_ASSISTANT = """\
{% for f in outputs separator='\\n\\n' strip %}
[[ ## {f.name} ## ]]
{f.value}
{% endfor %}

[[ ## completed ## ]]
"""

_CHAT_USER = """\
{% for f in inputs separator='\\n\\n' %}
[[ ## {f.name} ## ]]
{f.value}
{% endfor %}

{fragments('user')}
Respond with the corresponding output fields, starting with the field \
{% for f in outputs separator=', then ' %}`[[ ## {f.name} ## ]]`{f.chat_type_hint}{% endfor %}, \
and then ending with the marker for `[[ ## completed ## ]]`."""

CHAT_TEMPLATE: list[dict[str, Any]] = [
    {"role": "system", "content": _CHAT_SYSTEM},
    {
        "role": "demos",
        "user": _CHAT_DEMO_USER,
        "assistant": _CHAT_DEMO_ASSISTANT,
        "preamble": INCOMPLETE_DEMO_PREFIX,
    },
    {"role": "history", "user": _CHAT_DEMO_USER, "assistant": _CHAT_DEMO_ASSISTANT},
    {"role": "user", "content": _CHAT_USER},
]


# ---------------------------------------------------------------------------
# json
# ---------------------------------------------------------------------------

_JSON_SYSTEM = """\
Your input fields are:
{% for f in inputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
Your output fields are:
{% for f in outputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
All interactions will be structured in the following way, with the appropriate values filled in.

Inputs will have the following structure:

{% for f in inputs separator='\\n\\n' strip %}
[[ ## {f.name} ## ]]
{{{f.name}}}
{% endfor %}

Outputs will be a JSON object with the following fields.

{outputs(style='json_object')}
{fragments('system')}
In adhering to this structure, your objective is: {instruction(style='indented')}"""

_JSON_DEMO_ASSISTANT = "{outputs(style='json_object')}"

_JSON_USER = """\
{% for f in inputs separator='\\n\\n' %}
[[ ## {f.name} ## ]]
{f.value}
{% endfor %}

{fragments('user')}
Respond with a JSON object in the following order of fields: \
{% for f in outputs separator=', then ' %}`{f.name}`{f.chat_type_hint}{% endfor %}."""

JSON_TEMPLATE: list[dict[str, Any]] = [
    {"role": "system", "content": _JSON_SYSTEM},
    {
        "role": "demos",
        "user": _CHAT_DEMO_USER,
        "assistant": _JSON_DEMO_ASSISTANT,
        "preamble": INCOMPLETE_DEMO_PREFIX,
    },
    {"role": "history", "user": _CHAT_DEMO_USER, "assistant": _JSON_DEMO_ASSISTANT},
    {"role": "user", "content": _JSON_USER},
]


# ---------------------------------------------------------------------------
# xml
# ---------------------------------------------------------------------------

_XML_SYSTEM = """\
Your input fields are:
{% for f in inputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
Your output fields are:
{% for f in outputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
{% section strip %}
All interactions will be structured in the following way, with the appropriate values filled in.

{% for f in inputs separator='\\n\\n' strip %}
<{f.name}>
{{{f.name}}}
</{f.name}>
{% endfor %}

{% for f in outputs separator='\\n\\n' strip %}
<{f.name}>
{f.typed_placeholder}
</{f.name}>
{% endfor %}
{% endsection %}
{fragments('system')}
In adhering to this structure, your objective is: {instruction(style='indented')}"""

_XML_DEMO_USER = """\
{% for f in inputs separator='\\n\\n' strip %}
<{f.name}>
{f.value}
</{f.name}>
{% endfor %}"""

_XML_DEMO_ASSISTANT = """\
{% for f in outputs separator='\\n\\n' strip %}
<{f.name}>
{f.value}
</{f.name}>
{% endfor %}"""

_XML_USER = """\
{% for f in inputs separator='\\n\\n' strip %}
<{f.name}>
{f.value}
</{f.name}>
{% endfor %}

{fragments('user')}
Respond with the corresponding output fields wrapped in XML tags \
{% for f in outputs separator=', then ' %}`<{f.name}>`{% endfor %}."""

XML_TEMPLATE: list[dict[str, Any]] = [
    {"role": "system", "content": _XML_SYSTEM},
    {
        "role": "demos",
        "user": _XML_DEMO_USER,
        "assistant": _XML_DEMO_ASSISTANT,
        "preamble": INCOMPLETE_DEMO_PREFIX,
    },
    {"role": "history", "user": _XML_DEMO_USER, "assistant": _XML_DEMO_ASSISTANT},
    {"role": "user", "content": _XML_USER},
]


# ---------------------------------------------------------------------------
# The thin constructors
# ---------------------------------------------------------------------------


class _PresetAdapter(Adapter):
    """A preset entry with the same constructor surface everywhere."""

    preset_name: str
    preset_template: list[dict[str, Any]]

    def __init__(
        self,
        *,
        strategies: dict | None = None,
        codecs: dict | None = None,
        parser: dict | None = None,
        config: dict | None = None,
    ):
        merged_strategies = dict(DEFAULT_STRATEGIES)
        if strategies:
            merged_strategies.update(strategies)
        super().__init__(
            name=self.preset_name,
            template=self.preset_template,
            parser=parser,
            codecs=codecs,
            strategies=merged_strategies,
            config=config,
        )


class ChatAdapter(_PresetAdapter):
    """The chat preset: `[[ ## field ## ]]` marker sections, lens-parsed.

    Examples:
        ```python
        adapter = dspy.ChatAdapter()
        assert adapter.dump_entry()["parser"] == {"kind": "lens", "of": "template"}
        ```
    """

    preset_name = "chat"
    preset_template = CHAT_TEMPLATE


class JSONAdapter(_PresetAdapter):
    """The json preset: one JSON object of output fields, lens-parsed."""

    preset_name = "json"
    preset_template = JSON_TEMPLATE


class XMLAdapter(_PresetAdapter):
    """The xml preset: `<field>value</field>` tags, lens-parsed."""

    preset_name = "xml"
    preset_template = XML_TEMPLATE


def make_adapter(
    *,
    name: str,
    template: list[dict[str, Any]],
    parser: dict | None = None,
    codecs: dict | None = None,
    strategies: dict | None = None,
    engine_controls: dict | None = None,
    config: dict | None = None,
    requires: Any = DERIVED,
) -> Adapter:
    """Template in, adapter out — the whole custom adapter is a template
    author's act.

    Args:
        name: The entry name.
        template: The message-list template.
        parser: Omitted = the derived lens of the template.
        codecs: Codec bindings; text codecs both ways by default.
        strategies: Role bindings; the auto block by default.
        engine_controls: Request-side facts the engine applies (stop
            sequences, completion mode); stored in `config.engine_controls`.
        config: Open configuration.
        requires: The declared requirement set (`{"lm_capabilities":
            [...]}`); omitted = derived from the entry body.

    Examples:
        ```python
        tiny = dspy.make_adapter(
            name="tiny_qa",
            template=[
                {"role": "system", "content": "{instruction}"},
                {"role": "user", "content": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}\\n{fragments('user')}"},
            ],
        )
        ```
    """
    config = dict(config or {})
    if engine_controls is not None:
        config["engine_controls"] = dict(engine_controls)
    return Adapter(
        name=name,
        template=template,
        parser=parser,
        codecs=codecs,
        strategies=strategies,
        config=config,
        requires=requires,
    )
