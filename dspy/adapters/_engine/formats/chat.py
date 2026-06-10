"""ChatFormat: every ChatAdapter literal, frozen.

Each method reproduces the corresponding legacy ChatAdapter body
byte-for-byte (the dual-run harness and golden corpus adjudicate). Shared
value-rendering rules (guillemet list blobs, type notes, description
strings) stay in ``dspy.adapters.utils`` — they were never adapter methods
and remain the single source of those strings.

Byte-parity traps preserved on purpose: the ``\\n\\n``-join-then-``strip()``
shape of section rendering, the trailing-space demo messages (defined on the
Format base), the 8-space task-description indentation, and the trailing
newline after the assistant completed marker.
"""

from typing import Any

from dspy.adapters._engine.formats import Format
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    translate_field_type,
)


class ChatFormat(Format):
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

    def render_user_content(
        self,
        signature,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]
        for k, v in signature.input_fields.items():
            if k in inputs:
                value = inputs.get(k)
                formatted_field_value = format_field_value(field_info=v, value=value)
                messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        if main_request:
            output_requirements = self.output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

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

    def render_assistant_content(self, signature, outputs: dict[str, Any], missing_field_message=None) -> str:
        assistant_message_content = self.render_fields_with_values(
            {k: (v, outputs.get(k, missing_field_message)) for k, v in signature.output_fields.items()}
        )
        assistant_message_content += "\n\n[[ ## completed ## ]]\n"
        return assistant_message_content

    def render_fields_with_values(self, fields_with_values: dict[str, tuple]) -> str:
        """Field blocks: ``[[ ## name ## ]]\\nvalue`` joined by blank lines.

        Takes ``{name: (field_info, value)}`` — the engine-internal shape of
        legacy ``format_field_with_value``'s ``FieldInfoWithName`` mapping.
        """
        output = []
        for name, (field_info, field_value) in fields_with_values.items():
            formatted_field_value = format_field_value(field_info=field_info, value=field_value)
            output.append(f"[[ ## {name} ## ]]\n{formatted_field_value}")

        return "\n\n".join(output).strip()
