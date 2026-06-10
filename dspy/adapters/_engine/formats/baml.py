"""BAMLFormat: BAMLAdapter's two overridden surfaces, frozen.

Layered on JSONFormat exactly as BAMLAdapter subclasses JSONAdapter. Only
the schema section ({name} placeholders plus simplified-type lines, joined
with single newlines) and the user content (pydantic inputs rendered as
``model_dump_json(indent=2, by_alias=True)``, with the empty-segment
filter) differ; parsing and the assistant side are inherited from
JSONFormat.

The simplified-schema helpers are deliberately IMPORTED from
``dspy.adapters.baml_adapter`` rather than duplicated: they are pure
module-level functions and the single source of those strings; the
consolidation PR decides their final home.
"""

from typing import Any

from pydantic import BaseModel

from dspy.adapters._engine.formats.json import JSONFormat
from dspy.adapters.baml_adapter import _render_type_str
from dspy.adapters.utils import format_field_value


class BAMLFormat(JSONFormat):
    def render_field_structure(self, signature) -> str:
        sections = []

        sections.append(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n"
        )

        if signature.input_fields:
            for name in signature.input_fields.keys():
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"{{{name}}}")
                sections.append("")

        if signature.output_fields:
            for name, field in signature.output_fields.items():
                field_type = field.annotation
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"Output field `{name}` should be of type: {_render_type_str(field_type, indent=0)}\n")

        sections.append("[[ ## completed ## ]]")

        return "\n".join(sections)

    def render_user_content(
        self,
        signature,
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        messages = [prefix]
        for key, field_info in signature.input_fields.items():
            if key in inputs:
                value = inputs.get(key)
                if isinstance(value, BaseModel):
                    formatted_value = value.model_dump_json(indent=2, by_alias=True)
                else:
                    formatted_value = format_field_value(field_info=field_info, value=value)

                messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if main_request:
            output_requirements = self.output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(m for m in messages if m).strip()
