"""BAMLFormat: the BAML pairing — preset json + baml codec bindings.

Layered on JSONFormat exactly as BAMLAdapter subclasses JSONAdapter, and
since D-3 the class body is the pairing declaration itself: the ``baml``
codec bound in both directions (indented-pydantic values in, schema-prose
placeholders, shared text parsing) and the schema-prose system arrangement
carried as template data (``presets.BAML_SYSTEM_MESSAGE``). User turns,
demos, the assistant side, and parsing all inherit the json preset's
delegation; nothing renders from a method body.

``render_field_structure`` below is the frozen legacy composition kept as
the reference the parity tests diff the template against — it goes with the
``format_*`` zoo in Epic H.
"""

from dspy.adapters._engine.codecs import render_schema_prose
from dspy.adapters._engine.formats.json import JSONFormat
from dspy.adapters._engine.presets import BAML_SYSTEM_MESSAGE


class BAMLFormat(JSONFormat):
    codec_binding_overrides = {"input": "baml", "output": "baml"}
    system_template_message = BAML_SYSTEM_MESSAGE
    entry_name = "baml"

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
                sections.append(f"Output field `{name}` should be of type: {render_schema_prose(field_type, indent=0)}\n")

        sections.append("[[ ## completed ## ]]")

        return "\n".join(sections)
