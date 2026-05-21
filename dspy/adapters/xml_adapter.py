import ast
import enum
import inspect
import json
import re
import textwrap
import types
from collections.abc import Mapping
from typing import Any, Literal, Union, get_args, get_origin

import json_repair
import pydantic
from pydantic import TypeAdapter

from dspy.adapters.base import Adapter
from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.types.code import Code
from dspy.adapters.types.history import History
from dspy.adapters.types.reasoning import Reasoning
from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.types import LMMessage, LMRequestPatch, LMTextPart
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError


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


class XMLAdapter(Adapter):
    def __init__(self, callbacks: list[BaseCallback] | None = None):
        super().__init__(callbacks)
        self.field_pattern = re.compile(r"<(?P<name>\w+)>((?P<content>.*?))</\1>", re.DOTALL)

    def stream_start_identifier(self, field_name: str) -> str:
        return f"<{field_name}>"

    def consume_stream_field_buffer(self, field_name: str, buffer: str, *, final: bool) -> tuple[str, str, bool]:
        close_tag = f"</{field_name}>"
        boundary = buffer.find(close_tag)
        if boundary != -1:
            return buffer[:boundary].rstrip(), "", True
        if final:
            return buffer, "", False
        if buffer.endswith("<") or "</" in buffer:
            return "", buffer, False
        return buffer, "", False

    def __call__(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return await super().acall(lm, lm_kwargs, signature, demos, inputs)

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
            if annotation is Reasoning:
                type_name = "str"
            elif annotation is str:
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
                    elif arg is type(None):
                        args.append("NoneType")
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
            for custom_type in DspyType.extract_custom_type_from_annotation(annotation):
                if custom_type.description():
                    custom_type_name = custom_type.__name__ if hasattr(custom_type, "__name__") else str(custom_type)
                    s += f"\n    Type description of {custom_type_name}: {custom_type.description()}"
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.input_fields):
                s += "\n"
            i += 1

        s += "\nYour output fields are:\n"
        i = 1
        for name, field in signature.output_fields.items():
            annotation = field.annotation
            if annotation is Reasoning:
                type_name = "str"
            elif annotation is str:
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
                    elif arg is type(None):
                        args.append("NoneType")
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
            for custom_type in DspyType.extract_custom_type_from_annotation(annotation):
                if custom_type.description():
                    custom_type_name = custom_type.__name__ if hasattr(custom_type, "__name__") else str(custom_type)
                    s += f"\n    Type description of {custom_type_name}: {custom_type.description()}"
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.output_fields):
                s += "\n"
            i += 1

        s += "\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        i = 1
        for name, field in signature.input_fields.items():
            s += f"<{name}>\n"
            s += f"{{{name}}}\n"
            s += f"</{name}>"
            if i < len(signature.input_fields) or len(signature.output_fields) > 0:
                s += "\n\n"
            i += 1

        i = 1
        for name, field in signature.output_fields.items():
            annotation = field.annotation
            note = ""
            if annotation is str or annotation is Reasoning:
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
            elif inspect.isclass(annotation) and issubclass(annotation, Code) and annotation.description():
                note = ""
            else:
                schema = pydantic.TypeAdapter(annotation).json_schema()

                def move_type_to_front(value):
                    if isinstance(value, Mapping):
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
            s += f"<{name}>\n"
            s += f"{{{name}}}{note}\n"
            s += f"</{name}>"
            if i < len(signature.output_fields):
                s += "\n\n"
            i += 1

        instructions = textwrap.dedent(signature.instructions)
        objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
        s += f"\nIn adhering to this structure, your objective is: {objective}"
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
                parts.append(LMTextPart(text=f"<{name}>\n"))
                try:
                    rendered = pydantic.TypeAdapter(type(value)).dump_python(value, mode="json")
                except Exception:
                    rendered = str(value)
                if isinstance(rendered, dict) or isinstance(rendered, list):
                    rendered = json.dumps(rendered, ensure_ascii=False)
                else:
                    rendered = str(rendered)
                parts.append(LMTextPart(text=rendered))
                parts.append(LMTextPart(text=f"\n</{name}>"))
                i += 1
        return parts

    def render_demo_assistant_message(self, signature: type[Signature], demo: dict[str, Any], missing_message: str) -> str:
        s = ""
        i = 1
        for name, field in signature.output_fields.items():
            if i > 1:
                s += "\n\n"
            value = demo.get(name, missing_message)
            try:
                rendered = pydantic.TypeAdapter(type(value)).dump_python(value, mode="json")
            except Exception:
                rendered = str(value)
            if isinstance(rendered, dict) or isinstance(rendered, list):
                rendered = json.dumps(rendered, ensure_ascii=False)
            else:
                rendered = str(rendered)
            s += f"<{name}>\n"
            s += rendered + "\n"
            s += f"</{name}>"
            i += 1
        return s

    def render_current_user_message(self, signature: type[Signature], inputs: dict[str, Any]) -> str | list[Any]:
        parts = []
        i = 1
        for name, field in signature.input_fields.items():
            if name in inputs:
                if i > 1:
                    parts.append(LMTextPart(text="\n\n"))
                value = inputs[name]
                parts.append(LMTextPart(text=f"<{name}>\n"))
                try:
                    rendered = pydantic.TypeAdapter(type(value)).dump_python(value, mode="json")
                except Exception:
                    rendered = str(value)
                if isinstance(rendered, dict) or isinstance(rendered, list):
                    rendered = json.dumps(rendered, ensure_ascii=False)
                else:
                    rendered = str(rendered)
                parts.append(LMTextPart(text=rendered))
                parts.append(LMTextPart(text=f"\n</{name}>"))
                i += 1
        if parts:
            parts.append(LMTextPart(text="\n\n"))
        suffix = "Respond with the corresponding output fields wrapped in XML tags "
        i = 1
        for name in signature.output_fields:
            if i > 1:
                suffix += ", then "
            suffix += f"`<{name}>`"
            i += 1
        suffix += "."
        parts.append(LMTextPart(text=suffix))
        return parts

    def parse_xml_completion(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        fields = {}
        for match in self.field_pattern.finditer(completion):
            name = match.group("name")
            content = match.group("content").strip()
            if name in signature.output_fields and name not in fields:
                fields[name] = content

        parsed = {}
        for name, field in signature.output_fields.items():
            if name not in fields:
                raise AdapterParseError(
                    adapter_name="XMLAdapter",
                    signature=signature,
                    lm_response=completion,
                    parsed_result=parsed,
                )
            raw = fields[name]
            annotation = field.annotation
            try:
                if annotation is str:
                    parsed[name] = str(raw)
                elif isinstance(annotation, enum.EnumMeta):
                    found = False
                    for member in annotation:
                        if member.value == raw:
                            parsed[name] = member
                            found = True
                            break
                    if not found and raw in annotation.__members__:
                        parsed[name] = annotation[raw]
                        found = True
                    if not found:
                        raise ValueError(f"{raw!r} is not a valid name or value for the enum {annotation.__name__}")
                else:
                    origin = get_origin(annotation)
                    if origin is Literal:
                        allowed = get_args(annotation)
                        value = raw.strip()
                        if value.startswith(("Literal[", "str[")) and value.endswith("]"):
                            value = value[value.find("[") + 1 : -1]
                        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
                            value = value[1:-1]
                        if value in allowed:
                            parsed[name] = value
                        else:
                            raise ValueError(f"{raw!r} is not one of {allowed!r}")
                    elif origin in (Union, types.UnionType) and type(None) in get_args(annotation) and str in get_args(annotation):
                        parsed[name] = pydantic.TypeAdapter(annotation).validate_python(raw)
                    else:
                        candidate = json_repair.loads(raw)
                        if candidate == "" and raw != "":
                            try:
                                candidate = ast.literal_eval(raw)
                            except (ValueError, SyntaxError):
                                candidate = raw
                        try:
                            parsed[name] = pydantic.TypeAdapter(annotation).validate_python(candidate)
                        except pydantic.ValidationError as e:
                            try:
                                if inspect.isclass(annotation) and issubclass(annotation, DspyType):
                                    parsed[name] = pydantic.TypeAdapter(annotation).validate_python(raw)
                                else:
                                    raise e
                            except TypeError:
                                raise e
            except Exception as e:
                raise AdapterParseError(
                    adapter_name="XMLAdapter",
                    signature=signature,
                    lm_response=completion,
                    message=f"Failed to parse field {name} with value {raw}: {e}",
                )
        return parsed

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        return self.parse_xml_completion(signature, completion)
