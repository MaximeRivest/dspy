"""
Custom adapter for improving structured outputs using the information from Pydantic models.
Based on the format used by BAML: https://github.com/BoundaryML/baml
"""

from typing import Any

from pydantic import BaseModel

# The schema-prose machinery is the baml codec's schema spelling and lives
# in the engine (D-3); these aliases keep the historical import surface.
from dspy.adapters._engine.codecs import (  # noqa: F401
    COMMENT_SYMBOL,
    INDENTATION,
    _build_simplified_schema,
)
from dspy.adapters._engine.codecs import (
    render_schema_prose as _render_type_str,
)
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import format_field_value as original_format_field_value
from dspy.signatures.signature import Signature


class BAMLAdapter(JSONAdapter):
    """
    A DSPy adapter that improves the rendering of complex/nested Pydantic models to help LMs.

    This adapter generates a compact, human-readable schema representation for nested Pydantic output
    fields, inspired by the BAML project's JSON formatter (https://github.com/BoundaryML/baml).
    The resulting rendered schema is more token-efficient and easier for smaller LMs to follow than a
    raw JSON schema. It also includes Pydantic field descriptions as comments in the schema, which
    provide valuable additional context for the LM to understand the expected output.

    Example Usage:
    ```python
    import dspy
    from pydantic import BaseModel, Field
    from typing import Literal
    from baml_adapter import BAMLAdapter  # Import from your module

    # 1. Define your Pydantic models
    class PatientAddress(BaseModel):
        street: str
        city: str
        country: Literal["US", "CA"]

    class PatientDetails(BaseModel):
        name: str = Field(description="Full name of the patient.")
        age: int
        address: PatientAddress | None

    # 2. Define a signature using the Pydantic model as an output field
    class ExtractPatientInfo(dspy.Signature):
        '''Extract patient information from the clinical note.'''
        clinical_note: str = dspy.InputField()
        patient_info: PatientDetails = dspy.OutputField()

    # 3. Configure dspy to use the new adapter
    llm = dspy.OpenAI(model="gpt-4.1-mini")
    dspy.configure(lm=llm, adapter=BAMLAdapter())

    # 4. Run your program
    extractor = dspy.Predict(ExtractPatientInfo)
    note = "John Doe, 45 years old, lives at 123 Main St, Anytown. Resident of the US."
    result = extractor(clinical_note=note)
    print(result.patient_info)

    # Expected output:
    # PatientDetails(name='John Doe', age=45, address=PatientAddress(street='123 Main St', city='Anytown', country='US'))
    ```
    """

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Overrides the base method to generate a simplified schema for Pydantic models."""

        sections = []

        # Add structural explanation
        sections.append(
            "All interactions will be structured in the following way, with the appropriate values filled in.\n"
        )

        # Add input structure section
        if signature.input_fields:
            for name in signature.input_fields.keys():
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"{{{name}}}")
                sections.append("")  # Empty line after each input

        # Add output structure section
        if signature.output_fields:
            for name, field in signature.output_fields.items():
                field_type = field.annotation
                sections.append(f"[[ ## {name} ## ]]")
                sections.append(f"Output field `{name}` should be of type: {_render_type_str(field_type, indent=0)}\n")

        # Add completed section
        sections.append("[[ ## completed ## ]]")

        return "\n".join(sections)

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """Overrides the base method to render Pydantic input instances as clean JSON."""
        messages = [prefix]
        for key, field_info in signature.input_fields.items():
            if key in inputs:
                value = inputs.get(key)
                formatted_value = ""
                if isinstance(value, BaseModel):
                    # Use clean, indented JSON for Pydantic instances
                    formatted_value = value.model_dump_json(indent=2, by_alias=True)
                else:
                    # Fallback to the original dspy formatter for other types
                    formatted_value = original_format_field_value(field_info=field_info, value=value)

                messages.append(f"[[ ## {key} ## ]]\n{formatted_value}")

        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        messages.append(suffix)
        return "\n\n".join(m for m in messages if m).strip()
