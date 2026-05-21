"""
Custom adapter for improving structured outputs using the information from Pydantic models.
Based on the format used by BAML: https://github.com/BoundaryML/baml
"""

import inspect
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from dspy.adapters.json_adapter import JSONAdapter
from dspy.clients.language_models.types import LMTextPart
from dspy.signatures.signature import Signature

# Changing the comment symbol to Python's # rather than other languages' // seems to help
COMMENT_SYMBOL = "#"
INDENTATION = "  "


def _render_type_str(
    annotation: Any,
    depth: int = 0,
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Recursively renders a type annotation into a simplified string.

    Args:
        annotation: The type annotation to render
        depth: Current recursion depth (prevents infinite recursion)
        indent: Current indentation level for nested structures
    """
    # Non-nested types
    if annotation is str:
        return "string"
    if annotation is int:
        return "int"
    if annotation is float:
        return "float"
    if annotation is bool:
        return "boolean"
    if inspect.isclass(annotation) and issubclass(annotation, BaseModel):
        return _build_simplified_schema(annotation, indent, seen_models)

    try:
        origin = get_origin(annotation)
        args = get_args(annotation)
    except Exception:
        return str(annotation)

    # Optional[T] or T | None
    if origin in (types.UnionType, Union):
        non_none_args = [arg for arg in args if arg is not type(None)]
        # Render the non-None part of the union
        type_render = " or ".join([_render_type_str(arg, depth + 1, indent, seen_models) for arg in non_none_args])
        # Add "or null" if None was part of the union
        if len(non_none_args) < len(args):
            return f"{type_render} or null"
        return type_render

    # Literal[T1, T2, ...]
    if origin is Literal:
        return " or ".join(f'"{arg}"' for arg in args)

    # list[T]
    if origin is list:
        # For Pydantic models in lists, use bracket notation
        inner_type = args[0]
        if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
            # Build inner schema - the Pydantic model inside should use indent level for array contents
            inner_schema = _build_simplified_schema(inner_type, indent + 1, seen_models)
            # Format with proper bracket notation and indentation
            current_indent = INDENTATION * indent
            return f"[\n{inner_schema}\n{current_indent}]"
        else:
            return f"{_render_type_str(inner_type, depth + 1, indent, seen_models)}[]"

    # dict[T1, T2]
    if origin is dict:
        return f"dict[{_render_type_str(args[0], depth + 1, indent, seen_models)}, {_render_type_str(args[1], depth + 1, indent, seen_models)}]"

    # fallback
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


def _build_simplified_schema(
    pydantic_model: type[BaseModel],
    indent: int = 0,
    seen_models: set[type] | None = None,
) -> str:
    """Builds a simplified, human-readable schema from a Pydantic model.

    Args:
        pydantic_model: The Pydantic model to build schema for
        indent: Current indentation level
        seen_models: Set to track visited pydantic models (prevents infinite recursion)
    """
    seen_models = seen_models or set()

    if pydantic_model in seen_models:
        raise ValueError("BAMLAdapter cannot handle recursive pydantic models, please use a different adapter.")

    # Add `pydantic_model` to `seen_models` with a placeholder value to avoid infinite recursion.
    seen_models.add(pydantic_model)

    lines = []
    current_indent = INDENTATION * indent
    next_indent = INDENTATION * (indent + 1)

    # Add model docstring as a comment above the object if it exists
    # Only do this for top-level schemas (indent=0), since nested field docstrings
    # are already added before the field name in the parent schema
    if indent == 0 and pydantic_model.__doc__:
        docstring = pydantic_model.__doc__.strip()
        # Handle multiline docstrings by prefixing each line with the comment symbol
        for line in docstring.split("\n"):
            line = line.strip()
            if line:
                lines.append(f"{current_indent}{COMMENT_SYMBOL} {line}")

    lines.append(f"{current_indent}{{")

    fields = pydantic_model.model_fields
    if not fields:
        lines.append(f"{next_indent}{COMMENT_SYMBOL} No fields defined")
    for name, field in fields.items():
        if field.description:
            lines.append(f"{next_indent}{COMMENT_SYMBOL} {field.description}")
        elif field.alias and field.alias != name:
            # If there's an alias but no description, show the alias as a comment
            lines.append(f"{next_indent}{COMMENT_SYMBOL} alias: {field.alias}")

        # If the field type is a BaseModel, add its docstring as a comment before the field
        field_annotation = field.annotation
        # Handle Optional types
        origin = get_origin(field_annotation)
        if origin in (types.UnionType, Union):
            args = get_args(field_annotation)
            non_none_args = [arg for arg in args if arg is not type(None)]
            if len(non_none_args) == 1:
                field_annotation = non_none_args[0]

        if inspect.isclass(field_annotation) and issubclass(field_annotation, BaseModel):
            if field_annotation.__doc__:
                docstring = field_annotation.__doc__.strip()
                for line in docstring.split("\n"):
                    line = line.strip()
                    if line:
                        lines.append(f"{next_indent}{COMMENT_SYMBOL} {line}")

        rendered_type = _render_type_str(field.annotation, indent=indent + 1, seen_models=seen_models)
        line = f"{next_indent}{name}: {rendered_type},"

        lines.append(line)

    lines.append(f"{current_indent}}}")
    return "\n".join(lines)


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

    def render_system_message(self, signature: type[Signature]) -> str:
        s = ""
        s += "Your input fields are:\n"
        i = 1
        for name, field in signature.input_fields.items():
            annotation = field.annotation
            type_name = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
            desc = field.json_schema_extra["desc"] if field.json_schema_extra["desc"] != f"${{{name}}}" else ""
            s += f"{i}. `{name}` ({type_name}):"
            if desc:
                s += f" {desc}"
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.input_fields):
                s += "\n"
            i += 1

        s += "\nYour output fields are:\n"
        i = 1
        for name, field in signature.output_fields.items():
            annotation = field.annotation
            type_name = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
            desc = field.json_schema_extra["desc"] if field.json_schema_extra["desc"] != f"${{{name}}}" else ""
            s += f"{i}. `{name}` ({type_name}):"
            if desc:
                s += f" {desc}"
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.output_fields):
                s += "\n"
            i += 1

        s += "\n"
        s += self.render_baml_field_structure(signature)
        s += "\n"
        instructions = signature.instructions
        import textwrap

        instructions = textwrap.dedent(instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        s += f"In adhering to this structure, your objective is: {objective}"
        return s

    def render_baml_field_structure(self, signature: type[Signature]) -> str:
        s = ""
        s += "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        for name in signature.input_fields.keys():
            s += f"[[ ## {name} ## ]]\n"
            s += f"{{{name}}}\n\n"
        for name, field in signature.output_fields.items():
            s += f"[[ ## {name} ## ]]\n"
            s += f"Output field `{name}` should be of type: {_render_type_str(field.annotation, indent=0)}\n\n"
        s += "[[ ## completed ## ]]"
        return s

    def render_current_user_message(self, signature: type[Signature], inputs: dict[str, Any]) -> str | list[Any]:
        parts = []
        i = 1
        for name, field in signature.input_fields.items():
            if name in inputs:
                if i > 1:
                    parts.append(LMTextPart(text="\n\n"))
                value = inputs[name]
                parts.append(LMTextPart(text=f"[[ ## {name} ## ]]\n"))
                if isinstance(value, BaseModel):
                    rendered = value.model_dump_json(indent=2, by_alias=True)
                else:
                    try:
                        from pydantic import TypeAdapter

                        rendered = TypeAdapter(type(value)).dump_python(value, mode="json")
                    except Exception:
                        rendered = str(value)
                    if isinstance(rendered, dict) or isinstance(rendered, list):
                        import json

                        rendered = json.dumps(rendered, ensure_ascii=False)
                    else:
                        rendered = str(rendered)
                parts.append(LMTextPart(text=rendered))
                i += 1
        if parts:
            parts.append(LMTextPart(text="\n\n"))
        suffix = "Respond with a JSON object in the following order of fields: "
        i = 1
        for name, field in signature.output_fields.items():
            if i > 1:
                suffix += ", then "
            suffix += f"`{name}`"
            if field.annotation is not str:
                annotation = field.annotation
                type_name = annotation.__name__ if hasattr(annotation, "__name__") else str(annotation)
                suffix += f" (must be formatted as a valid Python {type_name})"
            i += 1
        suffix += "."
        parts.append(LMTextPart(text=suffix))
        return parts
