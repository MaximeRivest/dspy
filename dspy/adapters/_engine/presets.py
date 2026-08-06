"""Built-in presets: the historical adapters' rendering as template data.

A preset is the canonical component-4 entry (spec section 4): a template in
the constrained language plus a parser binding, codec bindings, strategy
bindings, and config. The three built-ins here byte-reproduce ChatFormat/
JSONFormat/XMLFormat rendering — the golden corpus adjudicates — and ARE
the render path: the engine Format classes delegate every system/user/
assistant content string to these templates through the delegation
functions at the bottom.

The byte-parity traps live in the template data on purpose:

- ``strip`` loop flags reproduce the historical join-then-strip section
  shapes (field descriptions, structure sections, assistant field blocks);
- the xml structure region sits inside ``{% section strip %}`` because its
  loops run right against the objective tail: legacy join-then-strip
  collapses a trailing empty outputs (or inputs+outputs) section together
  with its blank-line separators, which per-loop strips cannot express;
- the demos directive's assistant template carries the trailing newline
  after the completed marker;
- ``{fragments(...)}`` slots sit alone on their lines and render zero
  bytes while empty, so today's adapters pay nothing for the strategy seam
  (strategy fragments start filling them in D-4);
- ``{instruction(style='indented')}`` is the dedent-then-eight-space
  objective block, placed after the historical trailing space on
  ``objective is: ``.

Templates parse eagerly at preset construction (import time), so a broken
template refuses before anything renders.
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from dspy.adapters._engine.codecs import resolve_codec
from dspy.adapters._engine.template import (
    ContentMessage,
    ParsedTemplate,
    RenderContext,
    parse_content,
    parse_message_template,
    render_nodes,
    render_user_content,
)
from dspy.adapters._engine.template.vocabulary import VOCABULARY


@dataclass(frozen=True)
class Preset:
    """One named adapter-as-data entry.

    Serialization lands in D-5 and must carry ``adapter_ir_version`` plus
    the closed-vocabulary versions block from its FIRST dump/load shape
    (D-024); origin-tagged code entries carry ``language`` (D-025). No
    versionless serde shape may ever exist.
    """

    name: str
    template: ParsedTemplate
    parser: str
    codecs: Mapping[str, str]
    strategies: Mapping[str, str]
    config: Mapping[str, Any]


def _make_preset(name, template_messages, parser, codecs, strategies, config=None) -> Preset:
    parsed = parse_message_template(template_messages)
    valid_parsers = VOCABULARY["parsers"]
    if parser not in valid_parsers:
        raise ValueError(f"unknown parser {parser!r} for preset {name!r} — valid parsers: {', '.join(valid_parsers)}")
    for direction in ("input", "output"):
        resolve_codec(codecs[direction])
    return Preset(
        name=name,
        template=parsed,
        parser=parser,
        codecs=MappingProxyType(dict(codecs)),
        strategies=MappingProxyType(dict(strategies)),
        config=MappingProxyType(dict(config or {})),
    )


_AUTO_STRATEGIES = {
    "reasoning": "auto",
    "citations": "auto",
    "tools": "auto",
    "media": "auto",
    "history": "directive_turns",
}


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

CHAT = _make_preset(
    name="chat",
    template_messages=[
        {"role": "system", "content": _CHAT_SYSTEM},
        {"role": "demos", "user": _CHAT_DEMO_USER, "assistant": _CHAT_DEMO_ASSISTANT},
        {"role": "history"},
        {"role": "user", "content": _CHAT_USER},
    ],
    parser="chat",
    codecs={"input": "text_pythonish", "output": "text_pythonish"},
    strategies=_AUTO_STRATEGIES,
)


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

JSON = _make_preset(
    name="json",
    template_messages=[
        {"role": "system", "content": _JSON_SYSTEM},
        {"role": "demos", "user": _CHAT_DEMO_USER, "assistant": _JSON_DEMO_ASSISTANT},
        {"role": "history"},
        {"role": "user", "content": _JSON_USER},
    ],
    parser="json",
    codecs={"input": "text_pythonish", "output": "text_pythonish"},
    strategies=_AUTO_STRATEGIES,
)


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

XML = _make_preset(
    name="xml",
    template_messages=[
        {"role": "system", "content": _XML_SYSTEM},
        {"role": "demos", "user": _XML_DEMO_USER, "assistant": _XML_DEMO_ASSISTANT},
        {"role": "history"},
        {"role": "user", "content": _XML_USER},
    ],
    parser="xml",
    codecs={"input": "text_pythonish", "output": "text_pythonish"},
    strategies=_AUTO_STRATEGIES,
)


PRESETS: dict[str, Preset] = {preset.name: preset for preset in (CHAT, JSON, XML)}


def get_preset(name: str) -> Preset:
    """A preset by name; a dangling ref refuses naming itself (ADP-005)."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset {name!r} — available presets: {', '.join(PRESETS)}") from None


# ---------------------------------------------------------------------------
# The BAML pairing (D-019, spec section 4): preset json + the baml codec
# bindings + this system arrangement. Not a named preset — the codec owns
# the schema-prose SPELLING ({f.typed_placeholder} through the baml codec),
# this template owns the ARRANGEMENT. The empty-collapse shape is the
# legacy "\n".join(sections) exactly: each loop chunk brackets itself in
# newlines and joins on the empty separator, so an empty field set
# contributes zero bytes, not stray blank lines.
# ---------------------------------------------------------------------------

BAML_SYSTEM = """\
Your input fields are:
{% for f in inputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
Your output fields are:
{% for f in outputs strip %}
{f.i}. `{f.name}` ({f.type}):{f.desc_suffix}
{% endfor %}
All interactions will be structured in the following way, with the appropriate values filled in.
{% for f in inputs separator='' %}

[[ ## {f.name} ## ]]
{{{f.name}}}

{% endfor %}{% for f in outputs separator='' %}

[[ ## {f.name} ## ]]
{f.typed_placeholder}

{% endfor %}
[[ ## completed ## ]]
{fragments('system')}
In adhering to this structure, your objective is: {instruction(style='indented')}"""

BAML_SYSTEM_MESSAGE = ContentMessage(role="system", nodes=parse_content(BAML_SYSTEM), raw=BAML_SYSTEM)


# ---------------------------------------------------------------------------
# Delegation: the Format render surface, executed from preset templates
# ---------------------------------------------------------------------------


def _preset_for(fmt) -> Preset:
    if not getattr(fmt, "preset_name", None):
        raise KeyError(f"{type(fmt).__name__} declares no preset_name — it cannot render through a preset")
    return get_preset(fmt.preset_name)


def _content_message(preset: Preset, role: str) -> ContentMessage:
    for message in preset.template.messages:
        if isinstance(message, ContentMessage) and message.role == role:
            return message
    raise KeyError(f"preset {preset.name!r} has no {role!r} template message")


def _context(fmt, signature, mode, values=None, missing=None) -> RenderContext:
    return RenderContext(
        signature=signature,
        mode=mode,
        input_codec=fmt.input_codec,
        output_codec=fmt.output_codec,
        values=values,
        missing_field_message=missing,
        fragments={},
    )


def render_preset_system(fmt, signature) -> str:
    """The system message, rendered from template data in schema position.

    A format carrying a ``system_template_message`` (the BAML pairing's
    schema-prose arrangement) renders that; otherwise the preset's own
    system message renders.
    """
    message = getattr(fmt, "system_template_message", None) or _content_message(_preset_for(fmt), "system")
    return render_nodes(message.nodes, _context(fmt, signature, "schema"))


def render_preset_user_content(fmt, signature, inputs, prefix="", suffix="", main_request=False) -> str:
    """User-turn content from the preset template.

    The main request renders the template's user message (which carries the
    output-requirements literal and the ``user`` fragment slot); demo and
    history turns render the demos directive's user pattern. The
    prefix/suffix join-then-strip reproduces the historical shape for every
    call the pipeline makes (prefix only on incomplete demos, suffix never).
    """
    preset = _preset_for(fmt)
    nodes = _content_message(preset, "user").nodes if main_request else preset.template.demos_directive.user
    ctx = _context(fmt, signature, "user_values", values=inputs)
    return render_user_content(nodes, ctx, prefix=prefix, suffix=suffix)


def render_preset_assistant_content(fmt, signature, outputs, missing_field_message=None) -> str:
    """Assistant-turn content from the demos directive's assistant pattern."""
    preset = _preset_for(fmt)
    nodes = preset.template.demos_directive.assistant
    ctx = _context(fmt, signature, "assistant_values", values=outputs, missing=missing_field_message)
    return render_nodes(nodes, ctx)
