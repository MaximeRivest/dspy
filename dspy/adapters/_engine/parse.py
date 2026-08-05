"""Engine parser core: labeled-section text parsing, bug-for-bug.

Reproduces the legacy ChatAdapter parse mechanics exactly, including its
latent quirks: the header pattern is matched against the STRIPPED line but
the same-line trailing content is sliced from the ORIGINAL line at the
stripped match's end offset (observable whenever a header line has leading
whitespace); first occurrence of a header wins; the strict
``fields.keys() == output_fields.keys()`` completeness check; and the exact
``AdapterParseError`` shapes (per-field failure message, ``parsed_result``
on incompleteness).

Value coercion is delegated to the format's output codec
(``codecs.ValueCodec.parse_value``); the default codec wraps
``dspy.adapters.utils.parse_value`` unchanged.
"""

from typing import Any

from dspy.utils.exceptions import AdapterParseError


def parse_labeled_sections(completion: str, header_pattern) -> list[tuple[str | None, str]]:
    """Split a completion into ``(header, content)`` sections."""
    sections: list[tuple[str | None, list[str]]] = [(None, [])]

    for line in completion.splitlines():
        match = header_pattern.match(line.strip())
        if match:
            header = match.group(1)
            remaining_content = line[match.end() :].strip()
            sections.append((header, [remaining_content] if remaining_content else []))
        else:
            sections[-1][1].append(line)

    return [(k, "\n".join(v).strip()) for k, v in sections]


def parse_fields(
    sections: list[tuple[str | None, str]],
    signature,
    completion: str,
    adapter_name: str,
    codec=None,
) -> dict[str, Any]:
    """Extract and coerce output fields from parsed sections."""
    if codec is None:
        from dspy.adapters._engine.codecs import TEXT_PYTHONISH

        codec = TEXT_PYTHONISH
    fields = {}
    for k, v in sections:
        if (k not in fields) and (k in signature.output_fields):
            try:
                fields[k] = codec.parse_value(v, signature.output_fields[k].annotation)
            except Exception as e:
                raise AdapterParseError(
                    adapter_name=adapter_name,
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
                )
    if fields.keys() != signature.output_fields.keys():
        raise AdapterParseError(
            adapter_name=adapter_name,
            signature=signature,
            lm_response=completion,
            parsed_result=fields,
        )

    return fields


class FormatParserHook:
    """The plan-carried parser derived from the resolved Format.

    Recorded on every engine-built plan so the IR is complete (rendering and
    parsing share one Format source of truth); consumed directly once the
    engine postprocess parses LMResponse objects.
    """

    name = "format.text"

    def __init__(self, fmt):
        self.format = fmt

    def parse(self, response_view, ctx) -> dict[str, Any]:
        return self.format.parse(ctx.signature, response_view.text)
