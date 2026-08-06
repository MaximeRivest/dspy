"""JSONFormat: the second engine personality, layered on ChatFormat.

JSONAdapter inherits ChatAdapter's user-side rendering (marker headers) and
overrides the schema sentence, the assistant/demo side (JSON objects), and
parsing — so JSONFormat subclasses ChatFormat and overrides exactly those
surfaces, mirroring the legacy class relationship.

Parse semantics preserved bug-for-bug, including the asymmetry with chat:
per-field ``parse_value`` failures propagate RAW here (legacy JSONAdapter
has no try/except around coercion — the orchestration-level fallback
machinery catches them), while the not-a-JSON-object and incomplete-fields
cases raise ``AdapterParseError`` with ``adapter_name="JSONAdapter"``.

The structured-output decision tree (response_format selection and the
json_object retry) deliberately stays in json_adapter.py: it is two-call
ORCHESTRATION, and control flow is excluded from the v1 IR by design.
"""

import json
from typing import Any

import json_repair
import regex

from dspy.adapters._engine.formats.chat import ChatFormat
from dspy.adapters.types.tool import ToolCalls
from dspy.adapters.utils import (
    get_annotation_name,
    serialize_for_json,
    translate_field_type,
)
from dspy.utils.exceptions import AdapterParseError


class JSONFormat(ChatFormat):
    parse_error_adapter_name = "JSONAdapter"

    #: Rendered content executes ChatFormat's preset delegators against the
    #: json preset (marker-block user side, JSON-object assistant side).
    preset_name = "json"

    def render_field_structure(self, signature) -> str:
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        parts.append("Inputs will have the following structure:")
        parts.append(
            self.render_fields_with_values(
                {
                    field_name: (field_info, translate_field_type(field_name, field_info))
                    for field_name, field_info in signature.input_fields.items()
                }
            )
        )
        parts.append("Outputs will be a JSON object with the following fields.")
        parts.append(
            self._render_json_object(
                {
                    field_name: translate_field_type(field_name, field_info)
                    for field_name, field_info in signature.output_fields.items()
                }
            )
        )
        return "\n\n".join(parts).strip()

    def output_requirements(self, signature) -> str | None:
        def type_info(v):
            if v.annotation == ToolCalls:
                return ' (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})'
            return (
                f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
                if v.annotation is not str
                else ""
            )

        message = "Respond with a JSON object in the following order of fields: "
        message += ", then ".join(f"`{f}`{type_info(v)}" for f, v in signature.output_fields.items())
        message += "."
        return message

    def _render_json_object(self, values: dict[str, Any]) -> str:
        return json.dumps(serialize_for_json(values), indent=2, ensure_ascii=False)

    def parse(self, signature, completion: str) -> dict[str, Any]:
        fields = json_repair.loads(completion)

        if not isinstance(fields, dict):
            pattern = r"\{(?:[^{}]|(?R))*\}"
            match = regex.search(pattern, completion, regex.DOTALL)
            if match:
                completion = match.group(0)
                fields = json_repair.loads(completion)

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name=self.parse_error_adapter_name,
                signature=signature,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            )

        fields = {k: v for k, v in fields.items() if k in signature.output_fields}

        # NOTE: coercion failures propagate raw on purpose (legacy parity);
        # see module docstring.
        for k, v in fields.items():
            if k in signature.output_fields:
                fields[k] = self.output_codec.parse_value(v, signature.output_fields[k].annotation)

        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name=self.parse_error_adapter_name,
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    # The user side inherits ChatFormat's input_codec routing; the JSON
    # object rendering above is format-owned structure (whole-object
    # serialization), not per-value codec territory.
