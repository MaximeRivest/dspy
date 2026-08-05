"""BAMLFormat: BAMLAdapter's two overridden surfaces, frozen.

Layered on JSONFormat exactly as BAMLAdapter subclasses JSONAdapter — and
under the codec factoring, BAML decomposes visibly: it is JSONFormat plus
an **input codec preference** (``PydanticJSONCodec`` — models as indented
JSON) plus a schema-prose structure section. The empty-segment filter in
user content is the one genuine format-level difference on that side;
parsing and the assistant side are inherited from JSONFormat.

The simplified-schema helpers are deliberately IMPORTED from
``dspy.adapters.baml_adapter`` rather than duplicated: they are pure
module-level functions and the single source of those strings; the
consolidation PR decides their final home. Schema prose is codec-adjacent
(it describes what syntax is requested back) but stays a format literal
until the codec pool exists — see roadmap/epic-B-shapes-codecs.md
non-goals.
"""

from typing import Any

from dspy.adapters._engine.formats.json import JSONFormat
from dspy.adapters.baml_adapter import _render_type_str


class BAMLFormat(JSONFormat):
    @property
    def input_codec(self):
        from dspy.adapters._engine.codecs import PYDANTIC_JSON

        return PYDANTIC_JSON
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
                formatted_value = self.input_codec.render_value(value, field_info)
                messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if main_request:
            output_requirements = self.output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(m for m in messages if m).strip()
