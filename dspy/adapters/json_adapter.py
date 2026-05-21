import json
import logging
import textwrap
from typing import Any, Literal, get_args, get_origin

import json_repair
import pydantic
import regex
from pydantic import TypeAdapter

from dspy.adapters.base import Adapter
from dspy.adapters.types.history import History
from dspy.adapters.types.tool import ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.types import LMMessage, LMRequestPatch, LMTextPart
from dspy.signatures.signature import Signature, SignatureMeta
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

logger = logging.getLogger(__name__)


def _has_open_ended_mapping(signature: SignatureMeta) -> bool:
    for field in signature.output_fields.values():
        if get_origin(field.annotation) is dict:
            return True
    return False


def _consume_json_field_value(buffer: str) -> tuple[str, bool]:
    text = buffer.lstrip()
    if not text:
        return "", False
    if text[0] == '"':
        escaped = False
        for index, char in enumerate(text[1:], start=1):
            if escaped:
                escaped = False
                continue
            if char == "\\":
                escaped = True
                continue
            if char == '"':
                try:
                    return json.loads(text[: index + 1]), True
                except json.JSONDecodeError:
                    return text[1:index], True
        return "", False
    try:
        value = json.JSONDecoder().raw_decode(text)[0]
        return json.dumps(value) if not isinstance(value, str) else value, True
    except json.JSONDecodeError:
        return "", False


def _best_effort_json_field_value(buffer: str) -> str:
    text = buffer.strip()
    if text.startswith('"'):
        return text[1:].rstrip('"')
    return text.rstrip(",}").strip()


class JSONAdapter(Adapter):
    def __init__(self, callbacks: list[BaseCallback] | None = None, use_native_function_calling: bool = True):
        super().__init__(callbacks=callbacks, use_native_function_calling=use_native_function_calling)
        self.use_native_function_calling = use_native_function_calling

    def stream_start_identifier(self, field_name: str) -> str:
        return f'"{field_name}":'

    def consume_stream_field_buffer(self, field_name: str, buffer: str, *, final: bool) -> tuple[str, str, bool]:
        parsed, ended = _consume_json_field_value(buffer)
        if ended:
            return parsed, "", True
        if final:
            return _best_effort_json_field_value(buffer), "", False
        return "", buffer, False

    def _json_adapter_call_common(self, lm, lm_kwargs, signature, demos, inputs, call_fn):
        if isinstance(lm, BaseLM):
            if "response_format" not in lm.supported_params:
                return call_fn(lm, lm_kwargs, signature, demos, inputs)
        elif not isinstance(lm, LanguageModel) or not lm.capabilities.response_schema:
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

        has_tool_calls = any(field.annotation == ToolCalls for field in signature.output_fields.values())
        supports_response_schema = (
            lm.supports_response_schema if isinstance(lm, BaseLM) else lm.capabilities.response_schema
        )
        if (
            _has_open_ended_mapping(signature)
            or (not self.use_native_function_calling and has_tool_calls)
            or not supports_response_schema
        ):
            lm_kwargs["response_format"] = {"type": "json_object"}
            return call_fn(lm, lm_kwargs, signature, demos, inputs)

    def __call__(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().__call__)
        if result:
            return result

        try:
            schema_signature = self._response_format_signature(lm, lm_kwargs, signature, inputs)
            lm_kwargs["response_format"] = _get_structured_outputs_response_format(
                schema_signature, self.use_native_function_calling
            )
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception:
            logger.warning("Failed to use structured output format, falling back to JSON mode.")
            lm_kwargs["response_format"] = {"type": "json_object"}
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(lm, lm_kwargs, signature, demos, inputs, super().acall)
        if result:
            return await result

        try:
            schema_signature = self._response_format_signature(lm, lm_kwargs, signature, inputs)
            lm_kwargs["response_format"] = _get_structured_outputs_response_format(
                schema_signature, self.use_native_function_calling
            )
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception:
            logger.warning("Failed to use structured output format, falling back to JSON mode.")
            lm_kwargs["response_format"] = {"type": "json_object"}
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)

    def _response_format_signature(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        inputs: dict[str, Any],
    ) -> type[Signature]:
        patch = self.collect_type_strategy_patches(signature, lm, dict(lm_kwargs), inputs)
        processed = self.signature_without_patch_fields(signature, patch)
        if isinstance(lm, LanguageModel):
            processed = self._call_preprocess_language_model_builtin_types(lm, dict(lm_kwargs), processed, inputs)
        return processed

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        patch: LMRequestPatch | None = None,
    ) -> list[LMMessage]:
        inputs = dict(inputs)
        current_signature = signature
        history_turns = []
        for name, field in signature.input_fields.items():
            if field.annotation == History:
                current_signature = signature.delete(name)
                if name in inputs:
                    history_turns = inputs[name].messages
                    del inputs[name]
                break

        messages = [LMMessage(role="system", parts=[LMTextPart(text=self.render_system_message(signature))])]

        for turn in history_turns:
            content = self.render_demo_user_message(current_signature, turn, True)
            messages.append(
                LMMessage(
                    role="user",
                    parts=[LMTextPart(text=content)] if isinstance(content, str) else list(content),
                )
            )
            messages.append(
                LMMessage(
                    role="assistant",
                    parts=[
                        LMTextPart(
                            text=self.render_demo_assistant_message(
                                current_signature, turn, "Not supplied for this conversation history message. "
                            )
                        )
                    ],
                )
            )

        for demo in demos:
            demo_complete = True
            for name in signature.fields:
                if name not in demo or demo[name] is None:
                    demo_complete = False
                    break
            has_input = False
            for name in signature.input_fields:
                if name in demo:
                    has_input = True
                    break
            has_output = False
            for name in signature.output_fields:
                if name in demo:
                    has_output = True
                    break
            if not demo_complete and not (has_input and has_output):
                continue
            missing = "Not supplied for this conversation history message. "
            if not demo_complete:
                missing = "Not supplied for this particular example. "
            content = self.render_demo_user_message(signature, demo, demo_complete)
            messages.append(
                LMMessage(
                    role="user",
                    parts=[LMTextPart(text=content)] if isinstance(content, str) else list(content),
                )
            )
            messages.append(
                LMMessage(
                    role="assistant",
                    parts=[LMTextPart(text=self.render_demo_assistant_message(signature, demo, missing))],
                )
            )

        content = self.render_current_user_message(current_signature, inputs)
        messages.append(
            LMMessage(
                role="user",
                parts=[LMTextPart(text=content)] if isinstance(content, str) else list(content),
            )
        )
        return _place_lm_request_patch(messages, patch)

    def render_system_message(self, signature: type[Signature]) -> str:
        s = ""
        s += "Your input fields are:\n"
        i = 1
        for name, field in signature.input_fields.items():
            annotation = field.annotation
            if annotation is str:
                type_name = "str"
            elif get_origin(annotation) is Literal:
                args = []
                for arg in get_args(annotation):
                    if isinstance(arg, str):
                        if "'" in arg and '"' not in arg:
                            args.append(f'"{arg}"')
                        elif '"' in arg and "'" not in arg:
                            args.append(f"'{arg}'")
                        elif "'" in arg and '"' in arg:
                            args.append("'" + arg.replace("'", "\\'") + "'")
                        else:
                            args.append(f"'{arg}'")
                    elif hasattr(arg, "__name__"):
                        args.append(arg.__name__)
                    else:
                        args.append(str(arg))
                type_name = "Literal[" + ", ".join(args) + "]"
            elif get_origin(annotation) is not None:
                origin = get_origin(annotation)
                args = []
                for arg in get_args(annotation):
                    if arg is str:
                        args.append("str")
                    elif hasattr(arg, "__name__"):
                        args.append(arg.__name__)
                    else:
                        args.append(str(arg))
                type_name = (origin.__name__ if hasattr(origin, "__name__") else str(origin)) + "[" + ", ".join(args) + "]"
            elif hasattr(annotation, "__name__"):
                type_name = annotation.__name__
            else:
                type_name = str(annotation)

            desc = field.json_schema_extra["desc"] if field.json_schema_extra["desc"] != f"${{{name}}}" else ""
            s += f"{i}. `{name}` ({type_name}):"
            if desc:
                s += f" {desc}"
            for custom_type in [] if not hasattr(annotation, "extract_custom_type_from_annotation") else annotation.extract_custom_type_from_annotation(annotation):
                if custom_type.description():
                    custom_type_name = custom_type.__name__ if hasattr(custom_type, "__name__") else str(custom_type)
                    s += f"\n    Type description of {custom_type_name}: {custom_type.description()}"
            try:
                from dspy.adapters.types.base_type import Type as DspyType

                for custom_type in DspyType.extract_custom_type_from_annotation(annotation):
                    if custom_type.description():
                        custom_type_name = custom_type.__name__ if hasattr(custom_type, "__name__") else str(custom_type)
                        s += f"\n    Type description of {custom_type_name}: {custom_type.description()}"
            except Exception:
                pass
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.input_fields):
                s += "\n"
            i += 1

        s += "\nYour output fields are:\n"
        i = 1
        for name, field in signature.output_fields.items():
            annotation = field.annotation
            if annotation is str:
                type_name = "str"
            elif get_origin(annotation) is Literal:
                args = []
                for arg in get_args(annotation):
                    if isinstance(arg, str):
                        if "'" in arg and '"' not in arg:
                            args.append(f'"{arg}"')
                        elif '"' in arg and "'" not in arg:
                            args.append(f"'{arg}'")
                        elif "'" in arg and '"' in arg:
                            args.append("'" + arg.replace("'", "\\'") + "'")
                        else:
                            args.append(f"'{arg}'")
                    elif hasattr(arg, "__name__"):
                        args.append(arg.__name__)
                    else:
                        args.append(str(arg))
                type_name = "Literal[" + ", ".join(args) + "]"
            elif get_origin(annotation) is not None:
                origin = get_origin(annotation)
                args = []
                for arg in get_args(annotation):
                    if arg is str:
                        args.append("str")
                    elif hasattr(arg, "__name__"):
                        args.append(arg.__name__)
                    else:
                        args.append(str(arg))
                type_name = (origin.__name__ if hasattr(origin, "__name__") else str(origin)) + "[" + ", ".join(args) + "]"
            elif hasattr(annotation, "__name__"):
                type_name = annotation.__name__
            else:
                type_name = str(annotation)

            desc = field.json_schema_extra["desc"] if field.json_schema_extra["desc"] != f"${{{name}}}" else ""
            s += f"{i}. `{name}` ({type_name}):"
            if desc:
                s += f" {desc}"
            try:
                from dspy.adapters.types.base_type import Type as DspyType

                for custom_type in DspyType.extract_custom_type_from_annotation(annotation):
                    if custom_type.description():
                        custom_type_name = custom_type.__name__ if hasattr(custom_type, "__name__") else str(custom_type)
                        s += f"\n    Type description of {custom_type_name}: {custom_type.description()}"
            except Exception:
                pass
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.output_fields):
                s += "\n"
            i += 1

        s += "\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        s += "Inputs will have the following structure:\n"
        i = 1
        for name, field in signature.input_fields.items():
            s += f"[[ ## {name} ## ]]\n"
            s += f"{{{name}}}"
            if i < len(signature.input_fields):
                s += "\n\n"
            i += 1

        s += "\n\nOutputs will be a JSON object with the following fields.\n"
        s += "{\n"
        i = 1
        for name, field in signature.output_fields.items():
            annotation = field.annotation
            note = ""
            if annotation is str:
                note = ""
            elif annotation is bool:
                note = "        # note: the value you produce must be True or False"
            elif annotation is int:
                note = "        # note: the value you produce must be a single int value"
            elif annotation is float:
                note = "        # note: the value you produce must be a single float value"
            elif get_origin(annotation) is Literal:
                note = "        # note: the value you produce must exactly match (no extra characters) one of: "
                note += "; ".join([str(arg) for arg in get_args(annotation)])
            else:
                schema = pydantic.TypeAdapter(annotation).json_schema()

                def move_type_to_front(value):
                    if isinstance(value, dict):
                        out = {}
                        for key in sorted(value.keys(), key=lambda k: (k != "type", k)):
                            out[key] = move_type_to_front(value[key])
                        return out
                    if isinstance(value, list):
                        return [move_type_to_front(item) for item in value]
                    return value

                schema = move_type_to_front(schema)
                note = "        # note: the value you produce must adhere to the JSON schema: "
                note += json.dumps(schema, ensure_ascii=False)
            comma = "," if i < len(signature.output_fields) else ""
            rendered_field = f"{{{name}}}{note}".replace('"', '\\"')
            s += f'  "{name}": "{rendered_field}"{comma}\n'
            i += 1
        s += "}\n"

        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        s += f"In adhering to this structure, your objective is: {objective}"
        return s

    def render_demo_user_message(self, signature: type[Signature], demo: dict[str, Any], demo_complete: bool) -> str | list[Any]:
        parts = []
        if not demo_complete:
            parts.append(LMTextPart(text="This is an example of the task, though some input or output fields are not supplied.\n\n"))
        i = 1
        for name, field in signature.input_fields.items():
            if name in demo:
                if i > 1:
                    parts.append(LMTextPart(text="\n\n"))
                value = demo[name]
                parts.append(LMTextPart(text=f"[[ ## {name} ## ]]\n"))
                try:
                    rendered = pydantic.TypeAdapter(type(value)).dump_python(value, mode="json")
                except Exception:
                    rendered = str(value)
                if isinstance(rendered, dict) or isinstance(rendered, list):
                    rendered = json.dumps(rendered, ensure_ascii=False)
                else:
                    rendered = str(rendered)
                parts.append(LMTextPart(text=rendered))
                i += 1
        return parts

    def render_demo_assistant_message(self, signature: type[Signature], demo: dict[str, Any], missing_message: str) -> str:
        s = "{\n"
        i = 1
        for name, field in signature.output_fields.items():
            value = demo.get(name, missing_message)
            try:
                rendered = pydantic.TypeAdapter(type(value)).dump_python(value, mode="json")
            except Exception:
                rendered = str(value)
            rendered = json.dumps(rendered, ensure_ascii=False)
            comma = "," if i < len(signature.output_fields) else ""
            s += f'  "{name}": {rendered}{comma}\n'
            i += 1
        s += "}"
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
                try:
                    rendered = pydantic.TypeAdapter(type(value)).dump_python(value, mode="json")
                except Exception:
                    rendered = str(value)
                if isinstance(rendered, dict) or isinstance(rendered, list):
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
                if annotation is bool:
                    type_name = "bool"
                elif annotation is int:
                    type_name = "int"
                elif annotation is float:
                    type_name = "float"
                elif hasattr(annotation, "__name__"):
                    type_name = annotation.__name__
                else:
                    type_name = str(annotation)
                suffix += f" (must be formatted as a valid Python {type_name})"
            i += 1
        suffix += "."
        parts.append(LMTextPart(text=suffix))
        return parts

    def parse_json_completion(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = json_repair.loads(completion)
        if not isinstance(fields, dict):
            match = regex.search(r"\{(?:[^{}]|(?R))*\}", completion, regex.DOTALL)
            if match:
                completion = match.group(0)
                fields = json_repair.loads(completion)

        if not isinstance(fields, dict):
            raise AdapterParseError(
                adapter_name="JSONAdapter",
                signature=signature,
                lm_response=completion,
                message="LM response cannot be serialized to a JSON object.",
            )

        parsed = {}
        for name, field in signature.output_fields.items():
            if name not in fields:
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed,
                )
            value = fields[name]
            try:
                if field.annotation is str:
                    parsed[name] = str(value)
                else:
                    parsed[name] = pydantic.TypeAdapter(field.annotation).validate_python(value)
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="JSONAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse field {name} with value {value}. Error message: {e}",
                )
        return parsed

    def prepare_response_format(self, signature: SignatureMeta, lm: BaseLM | LanguageModel, lm_kwargs: dict[str, Any]):
        for name, field in signature.output_fields.items():
            if get_origin(field.annotation) is dict:
                raise ValueError(
                    f"Field '{name}' has an open-ended mapping type which is not supported by Structured Outputs."
                )

        fields = {}
        for name, field in signature.output_fields.items():
            if self.use_native_function_calling and field.annotation == ToolCalls:
                continue
            default = field.default if hasattr(field, "default") else ...
            fields[name] = (field.annotation, default)

        model = pydantic.create_model(
            "DSPyProgramOutputs",
            __config__=pydantic.ConfigDict(extra="forbid"),
            **fields,
        )
        schema = model.model_json_schema()
        for prop in schema.get("properties", {}).values():
            prop.pop("json_schema_extra", None)

        def enforce_required(schema_part: dict):
            if schema_part.get("type") == "object":
                props = schema_part.get("properties")
                if props is not None:
                    schema_part["required"] = list(props.keys())
                    schema_part["additionalProperties"] = False
                    for sub_schema in props.values():
                        if isinstance(sub_schema, dict):
                            enforce_required(sub_schema)
                else:
                    schema_part["properties"] = {}
                    schema_part["required"] = []
                    schema_part["additionalProperties"] = False
            if schema_part.get("type") == "array" and isinstance(schema_part.get("items"), dict):
                enforce_required(schema_part["items"])
            for key in ("$defs", "definitions"):
                if key in schema_part:
                    for def_schema in schema_part[key].values():
                        enforce_required(def_schema)

        enforce_required(schema)
        model.model_json_schema = lambda *args, **kwargs: schema
        lm_kwargs["response_format"] = model

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        return self.parse_json_completion(signature, completion)

    def format_finetune_data(
        self, signature: type[Signature], demos: list[dict[str, Any]], inputs: dict[str, Any], outputs: dict[str, Any]
    ) -> dict[str, list[Any]]:
        raise NotImplementedError


def _place_lm_request_patch(messages: list[LMMessage], patch: LMRequestPatch | None) -> list[LMMessage]:
    if patch is None:
        return messages

    final_user_index = _last_user_message_index(messages)
    if patch.system_parts and messages:
        _append_parts(messages[0], patch.system_parts)
    if patch.messages:
        insert_at = final_user_index if final_user_index is not None else len(messages)
        messages[insert_at:insert_at] = patch.messages
        if final_user_index is not None:
            final_user_index += len(patch.messages)
    if patch.assistant_parts:
        insert_at = final_user_index if final_user_index is not None else len(messages)
        messages.insert(insert_at, LMMessage(role="assistant", parts=list(patch.assistant_parts)))
        if final_user_index is not None:
            final_user_index += 1
    if patch.user_parts:
        if final_user_index is None:
            messages.append({"role": "user", "content": list(patch.user_parts)})
        else:
            _append_parts(messages[final_user_index], patch.user_parts)
    return messages


def _last_user_message_index(messages: list[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role == "user":
            return index
    return None


def _append_parts(message: Any, parts: list[Any]) -> None:
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif content is None:
            content = []
        else:
            content = list(content)
        content.extend(parts)
        message["content"] = content
        return
    message.parts.extend(parts)


def _get_structured_outputs_response_format(
    signature: SignatureMeta,
    use_native_function_calling: bool = True,
) -> type[pydantic.BaseModel]:
    for name, field in signature.output_fields.items():
        if get_origin(field.annotation) is dict:
            raise ValueError(
                f"Field '{name}' has an open-ended mapping type which is not supported by Structured Outputs."
            )

    fields = {}
    for name, field in signature.output_fields.items():
        if use_native_function_calling and field.annotation == ToolCalls:
            continue
        default = field.default if hasattr(field, "default") else ...
        fields[name] = (field.annotation, default)

    model = pydantic.create_model(
        "DSPyProgramOutputs",
        __config__=pydantic.ConfigDict(extra="forbid"),
        **fields,
    )
    schema = model.model_json_schema()
    for prop in schema.get("properties", {}).values():
        prop.pop("json_schema_extra", None)

    def enforce_required(schema_part: dict):
        if schema_part.get("type") == "object":
            props = schema_part.get("properties")
            if props is not None:
                schema_part["required"] = list(props.keys())
                schema_part["additionalProperties"] = False
                for sub_schema in props.values():
                    if isinstance(sub_schema, dict):
                        enforce_required(sub_schema)
            else:
                schema_part["properties"] = {}
                schema_part["required"] = []
                schema_part["additionalProperties"] = False
        if schema_part.get("type") == "array" and isinstance(schema_part.get("items"), dict):
            enforce_required(schema_part["items"])
        for key in ("$defs", "definitions"):
            if key in schema_part:
                for def_schema in schema_part[key].values():
                    enforce_required(def_schema)

    enforce_required(schema)
    model.model_json_schema = lambda *args, **kwargs: schema
    return model
