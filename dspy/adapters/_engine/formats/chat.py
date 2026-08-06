"""ChatFormat: ChatAdapter's wire behavior, rendered from the ``chat`` preset.

Since D-2 the rendered content (system message, user turns, assistant
turns) executes from the preset template in ``_engine/presets.py`` — data,
not method bodies. The granular methods kept below (field description,
field structure, task description, output requirements, field blocks) are
the frozen legacy literals: BAML still composes its system message from
them, and the preset parity tests diff them against the template renders
so the two sources cannot drift. Shared value-rendering rules (guillemet
list blobs, type notes, description strings) stay in
``dspy.adapters.utils``.

Byte-parity traps live in the preset template on purpose: the
``\\n\\n``-join-then-``strip()`` shape of section rendering (``strip`` loop
flags), the trailing-space demo messages (defined on the Format base), the
8-space task-description indentation (``{instruction(style='indented')}``),
and the trailing newline after the assistant completed marker.

The format also owns parsing: the header regex, the parse mechanics (via
the shared engine core), and even the historical ``adapter_name`` string in
parse errors — error identity is wire-format behavior, pinned by the parse
corpus.
"""

import re
from typing import Any

from dspy.adapters._engine.formats import Format
from dspy.adapters._engine.parse import parse_fields, parse_labeled_sections
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    get_annotation_name,
    get_field_description_string,
    translate_field_type,
)


class ChatFormat(Format):
    field_header_pattern = re.compile(r"\[\[ ## (\w+) ## \]\]")

    #: Legacy parse errors hardcode this name regardless of subclass; it is
    #: pinned wire behavior, not a reflection of the calling class.
    parse_error_adapter_name = "ChatAdapter"

    preset_name = "chat"

    def parse(self, signature, completion: str) -> dict[str, Any]:
        sections = parse_labeled_sections(completion, self.field_header_pattern)
        return parse_fields(sections, signature, completion, self.parse_error_adapter_name, codec=self.output_codec)

    # --- Rendered content: the preset template is the render path ----------
    # The three delegators below execute the preset named by
    # ``preset_name`` (json/xml override only the name), so every content
    # string on the engine path renders from template data. The granular
    # methods further down keep the frozen legacy literals: BAML still
    # composes from them, and the preset parity tests diff the two sources
    # so they cannot drift.

    def render_system(self, signature) -> str:
        from dspy.adapters._engine.presets import render_preset_system

        return render_preset_system(self, signature)

    def render_user_content(
        self,
        signature,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        from dspy.adapters._engine.presets import render_preset_user_content

        return render_preset_user_content(self, signature, inputs, prefix, suffix, main_request)

    def render_assistant_content(self, signature, outputs: dict[str, Any], missing_field_message=None) -> str:
        from dspy.adapters._engine.presets import render_preset_assistant_content

        return render_preset_assistant_content(self, signature, outputs, missing_field_message)

    def render_field_description(self, signature) -> str:
        return (
            f"Your input fields are:\n{get_field_description_string(signature.input_fields)}\n"
            f"Your output fields are:\n{get_field_description_string(signature.output_fields)}"
        )

    def render_field_structure(self, signature) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields):
            return self.render_fields_with_values(
                {
                    field_name: (field_info, translate_field_type(field_name, field_info))
                    for field_name, field_info in fields.items()
                }
            )

        parts.append(format_signature_fields_for_instructions(signature.input_fields))
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        parts.append("[[ ## completed ## ]]\n")
        return "\n\n".join(parts).strip()

    def render_task_description(self, signature) -> str:
        import textwrap

        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        return f"In adhering to this structure, your objective is: {objective}"

    def output_requirements(self, signature) -> str | None:
        def type_info(v):
            if v.annotation == ToolCalls:
                return ' (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})'
            if v.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            else:
                return ""

        message = "Respond with the corresponding output fields, starting with the field "
        message += ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message

    def render_fields_with_values(self, fields_with_values: dict[str, tuple], codec=None) -> str:
        """Field blocks: ``[[ ## name ## ]]\\nvalue`` joined by blank lines.

        Takes ``{name: (field_info, value)}`` — the engine-internal shape of
        legacy ``format_field_with_value``'s ``FieldInfoWithName`` mapping.
        ``codec`` selects the value syntax; defaults to the output binding
        (this method's main callers render assistant/demo output blocks).
        """
        codec = codec or self.output_codec
        output = []
        for name, (field_info, field_value) in fields_with_values.items():
            formatted_field_value = codec.render_value(field_value, field_info)
            output.append(f"[[ ## {name} ## ]]\n{formatted_field_value}")

        return "\n\n".join(output).strip()
