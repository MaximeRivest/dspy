"""XMLFormat: XMLAdapter's literals and quirks, frozen.

Layered on ChatFormat exactly as XMLAdapter subclasses ChatAdapter: field
description, task description, and demo machinery are inherited; tag
wrapping, the schema section (NO completed marker), the single-call user
content rendering, and parsing are overridden.

Quirks reproduced verbatim, not improved: the DOTALL first-match-wins
parser matches tags anywhere in the completion (nested tags and CDATA fail
exactly as before); the per-field coercion error message interpolates the
FieldInfo repr rather than the field name; the output-requirements sentence
keeps its historical wording.
"""

import re
from typing import Any

from dspy.adapters._engine.formats.chat import ChatFormat
from dspy.adapters.utils import translate_field_type
from dspy.utils.exceptions import AdapterParseError


class XMLFormat(ChatFormat):
    field_pattern = re.compile(r"<(?P<name>\w+)>((?P<content>.*?))</\1>", re.DOTALL)
    parse_error_adapter_name = "XMLAdapter"

    def render_fields_with_values(self, fields_with_values: dict[str, tuple], codec=None) -> str:
        codec = codec or self.output_codec
        output = []
        for name, (field_info, field_value) in fields_with_values.items():
            formatted = codec.render_value(field_value, field_info)
            output.append(f"<{name}>\n{formatted}\n</{name}>")
        return "\n\n".join(output).strip()

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
        return "\n\n".join(parts).strip()

    def render_user_content(
        self,
        signature,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]

        messages.append(
            self.render_fields_with_values(
                {k: (v, inputs.get(k)) for k, v in signature.input_fields.items() if k in inputs},
                codec=self.input_codec,
            )
        )

        if main_request:
            output_requirements = self.output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(messages).strip()

    def render_assistant_content(self, signature, outputs: dict[str, Any], missing_field_message=None) -> str:
        return self.render_fields_with_values(
            {k: (v, outputs.get(k, missing_field_message)) for k, v in signature.output_fields.items()}
        )

    def output_requirements(self, signature) -> str | None:
        message = "Respond with the corresponding output fields wrapped in XML tags "
        message += ", then ".join(f"`<{f}>`" for f in signature.output_fields)
        message += "."
        return message

    def parse(self, signature, completion: str) -> dict[str, Any]:
        fields = {}
        for match in self.field_pattern.finditer(completion):
            name = match.group("name")
            content = match.group("content").strip()
            if name in signature.output_fields and name not in fields:
                fields[name] = content
        for k, v in fields.items():
            fields[k] = self._parse_field_value(signature.output_fields[k], v, completion, signature)
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name=self.parse_error_adapter_name,
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )
        return fields

    def _parse_field_value(self, field_info, raw, completion, signature):
        try:
            return self.output_codec.parse_value(raw, field_info.annotation)
        except Exception as e:
            # Legacy quirk preserved: the message interpolates the FieldInfo
            # repr, not the field name.
            raise AdapterParseError(
                adapter_name=self.parse_error_adapter_name,
                signature=signature,
                lm_response=completion,
                message=f"Failed to parse field {field_info} with value {raw}: {e}",
            )
