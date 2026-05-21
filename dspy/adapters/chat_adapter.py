import ast
import enum
import inspect
import json
import re
import textwrap
import types
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, Union, get_args, get_origin

import json_repair
import pydantic
from pydantic import TypeAdapter
from pydantic.fields import FieldInfo

from dspy.adapters.base import Adapter
from dspy.adapters.types.base_type import Type as DspyType
from dspy.adapters.types.code import Code
from dspy.adapters.types.history import History
from dspy.adapters.types.reasoning import Reasoning
from dspy.clients.base_lm import BaseLM
from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.types import LMMessage, LMPart, LMRequestPatch, LMTextPart
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError, ContextWindowExceededError

field_header_pattern: re.Pattern[str] = re.compile(r"\[\[ ## (\w+) ## \]\]")

AdapterContent: TypeAlias = str | list[LMPart]


class ChatAdapter(Adapter):
    """Default Adapter for most language models.

    The ChatAdapter formats DSPy signatures into a format compatible with most language models.
    It uses delimiter patterns like `[[ ## field_name ## ]]` to clearly separate input and output fields in
    the message content.

    Key features:
        - Structures inputs and outputs using field header markers for clear field delineation.
        - Provides automatic fallback to JSONAdapter if the chat format fails.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type[DspyType]] | None = None,
        use_json_adapter_fallback: bool = True,
        adapter_types: list[Any] | None = None,
    ):
        """
        Args:
            callbacks: List of callback functions to execute during adapter methods.
            use_native_function_calling: Whether to enable native function calling capabilities.
            native_response_types: List of output field types handled by native LM features.
            use_json_adapter_fallback: Whether to automatically fallback to JSONAdapter if the ChatAdapter fails.
                If True, when an error occurs (except ContextWindowExceededError), the adapter will retry using
                JSONAdapter. Defaults to True.
        """
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            native_response_types=native_response_types,
            adapter_types=adapter_types,
        )
        self.use_json_adapter_fallback = use_json_adapter_fallback

    def __call__(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if (
                isinstance(e, ContextWindowExceededError)
                or isinstance(self, JSONAdapter)
                or not self.use_json_adapter_fallback
            ):
                # On context window exceeded error, already using JSONAdapter, or use_json_adapter_fallback is False
                # we don't want to retry with a different adapter. Raise the original error instead of the fallback error.
                raise e
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    async def acall(
        self,
        lm: BaseLM | LanguageModel,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            # fallback to JSONAdapter
            from dspy.adapters.json_adapter import JSONAdapter

            if (
                isinstance(e, ContextWindowExceededError)
                or isinstance(self, JSONAdapter)
                or not self.use_json_adapter_fallback
            ):
                # On context window exceeded error, already using JSONAdapter, or use_json_adapter_fallback is False
                # we don't want to retry with a different adapter. Raise the original error instead of the fallback error.
                raise e
            return await JSONAdapter().acall(lm, lm_kwargs, signature, demos, inputs)

    def stream_start_identifier(self, field_name: str) -> str:
        return f"[[ ## {field_name} ## ]]"

    def consume_stream_field_buffer(self, field_name: str, buffer: str, *, final: bool) -> tuple[str, str, bool]:
        match = field_header_pattern.search(buffer)
        if match:
            return buffer[: match.start()].rstrip(), "", True
        if final or not self._could_form_field_header(buffer):
            return buffer, "", False
        return "", buffer, False

    def _could_form_field_header(self, buffer: str) -> bool:
        prefixes = ("[", "[[", "[[ ", "[[ #", "[[ ##")
        return any(buffer.endswith(prefix) for prefix in prefixes) or "[[ ##" in buffer

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        patch: LMRequestPatch | None = None,
    ) -> list[LMMessage]:
        inputs = dict(inputs)
        current_signature = signature
        history_turns: list[dict[str, Any]] = []
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
            messages.append(LMMessage(role="user", parts=_adapter_content_to_parts(content)))
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
            messages.append(LMMessage(role="user", parts=_adapter_content_to_parts(content)))
            messages.append(
                LMMessage(
                    role="assistant",
                    parts=[LMTextPart(text=self.render_demo_assistant_message(signature, demo, missing))],
                )
            )

        content = self.render_current_user_message(current_signature, inputs)
        messages.append(LMMessage(role="user", parts=_adapter_content_to_parts(content)))
        return _place_lm_request_patch(messages, patch)

    def render_system_message(self, signature: type[Signature]) -> str:
        s = ""
        s += "Your input fields are:\n"
        i = 1
        for name, field in signature.input_fields.items():
            s += f"{i}. `{name}` ({_annotation_name(field.annotation)}):"
            desc = field.json_schema_extra["desc"] if field.json_schema_extra["desc"] != f"${{{name}}}" else ""
            if desc:
                s += f" {desc}"
            else:
                s += ""
            for custom_type in DspyType.extract_custom_type_from_annotation(field.annotation):
                if custom_type.description():
                    s += f"\n    Type description of {_annotation_name(custom_type)}: {custom_type.description()}"
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.input_fields):
                s += "\n"
            i += 1

        s += "\nYour output fields are:\n"
        i = 1
        for name, field in signature.output_fields.items():
            s += f"{i}. `{name}` ({_annotation_name(field.annotation)}):"
            desc = field.json_schema_extra["desc"] if field.json_schema_extra["desc"] != f"${{{name}}}" else ""
            if desc:
                s += f" {desc}"
            for custom_type in DspyType.extract_custom_type_from_annotation(field.annotation):
                if custom_type.description():
                    s += f"\n    Type description of {_annotation_name(custom_type)}: {custom_type.description()}"
            if field.json_schema_extra.get("constraints"):
                s += f"\nConstraints: {field.json_schema_extra['constraints']}"
            if i < len(signature.output_fields):
                s += "\n"
            i += 1

        s += "\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n"
        i = 1
        for name, field in signature.input_fields.items():
            s += f"[[ ## {name} ## ]]\n"
            s += f"{{{name}}}"
            if i < len(signature.input_fields) or len(signature.output_fields) > 0:
                s += "\n\n"
            i += 1

        i = 1
        for name, field in signature.output_fields.items():
            s += f"[[ ## {name} ## ]]\n"
            s += _translate_field_type(name, field, role="output")
            if i < len(signature.output_fields):
                s += "\n\n"
            i += 1

        s += "\n\n[[ ## completed ## ]]\n"
        s += f"In adhering to this structure, your objective is: {_indented_objective(signature.instructions)}"
        return s

    def render_demo_user_message(self, signature: type[Signature], demo: dict[str, Any], demo_complete: bool) -> AdapterContent:
        parts: list[LMPart] = []
        if not demo_complete:
            parts.append(LMTextPart(text="This is an example of the task, though some input or output fields are not supplied.\n\n"))
        i = 1
        for name, field in signature.input_fields.items():
            if name in demo:
                if i > 1:
                    parts.append(LMTextPart(text="\n\n"))
                parts.append(LMTextPart(text=f"[[ ## {name} ## ]]\n"))
                value = demo[name]
                parts.append(LMTextPart(text=str(_format_field_value(field_info=field, value=value))))
                i += 1
        return parts

    def render_demo_assistant_message(self, signature: type[Signature], demo: dict[str, Any], missing_message: str | None) -> str:
        s = ""
        i = 1
        for name, field in signature.output_fields.items():
            if i > 1:
                s += "\n\n"
            s += f"[[ ## {name} ## ]]\n"
            s += str(_format_field_value(field_info=field, value=demo.get(name, missing_message)))
            i += 1
        s += "\n\n[[ ## completed ## ]]\n"
        return s

    def render_current_user_message(self, signature: type[Signature], inputs: dict[str, Any]) -> AdapterContent:
        parts: list[LMPart] = []
        i = 1
        for name, field in signature.input_fields.items():
            if name in inputs:
                if i > 1:
                    parts.append(LMTextPart(text="\n\n"))
                parts.append(LMTextPart(text=f"[[ ## {name} ## ]]\n"))
                value = inputs[name]
                parts.append(LMTextPart(text=str(_format_field_value(field_info=field, value=value))))
                i += 1
        if parts:
            parts.append(LMTextPart(text="\n\n"))
        suffix = "Respond with the corresponding output fields, starting with the field "
        i = 1
        for name, field in signature.output_fields.items():
            if i > 1:
                suffix += ", then "
            suffix += f"`[[ ## {name} ## ]]`"
            if field.annotation is not str:
                suffix += f" (must be formatted as a valid Python {_annotation_name(field.annotation)})"
            i += 1
        suffix += ", and then ending with the marker for `[[ ## completed ## ]]`."
        parts.append(LMTextPart(text=suffix))
        return parts

    def parse_chat_completion(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        sections: list[tuple[str | None, list[str]]] = [(None, [])]

        for line in completion.splitlines():
            match = field_header_pattern.match(line.strip())
            if match:
                header = match.group(1)
                remaining_content = line[match.end() :].strip()
                sections.append((header, [remaining_content] if remaining_content else []))
            else:
                sections[-1][1].append(line)

        sections = [(k, "\n".join(v).strip()) for k, v in sections]

        fields: dict[str, Any] = {}
        for k, v in sections:
            if (k not in fields) and (k in signature.output_fields):
                try:
                    fields[k] = _parse_value(v, signature.output_fields[k].annotation)
                except Exception as e:
                    raise AdapterParseError(
                        adapter_name="ChatAdapter",
                        signature=signature,
                        lm_response=completion,
                        message=f"Failed to parse field {k} with value {v} from the LM response. Error message: {e}",
                    )
        if fields.keys() != signature.output_fields.keys():
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=signature,
                lm_response=completion,
                parsed_result=fields,
            )

        return fields

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        return self.parse_chat_completion(signature, completion)

    def format_finetune_data(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Format the call data into finetuning data according to the OpenAI API specifications.

        For the chat adapter, this means formatting the data as a list of messages, where each message is a dictionary
        with a "role" and "content" key. The role can be "system", "user", or "assistant". Then, the messages are
        wrapped in a dictionary with a "messages" key.
        """
        from dspy.adapters.legacy_bridge import legacy_messages_from_typed_messages

        system_user_messages = self.format(signature=signature, demos=demos, inputs=inputs)
        assistant_message_content = self.render_demo_assistant_message(signature, outputs, "Not supplied for this conversation history message. ")
        assistant_message = LMMessage(role="assistant", parts=[LMTextPart(text=assistant_message_content)])
        messages = legacy_messages_from_typed_messages(system_user_messages + [assistant_message])
        return {"messages": messages}


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
            messages.append(LMMessage(role="user", parts=list(patch.user_parts)))
        else:
            _append_parts(messages[final_user_index], patch.user_parts)
    return messages


def _adapter_content_to_parts(content: AdapterContent) -> list[LMPart]:
    return [LMTextPart(text=content)] if isinstance(content, str) else list(content)


def _last_user_message_index(messages: list[LMMessage]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.role == "user":
            return index
    return None


def _append_parts(message: LMMessage, parts: list[LMPart]) -> None:
    message.parts.extend(parts)


def _indented_objective(instructions: str) -> str:
    instructions = textwrap.dedent(instructions)
    return ("\n" + " " * 8).join([""] + instructions.splitlines())


def _translate_field_type(field_name: str, field_info: FieldInfo, *, role: str) -> str:
    field_type = field_info.annotation

    if role == "input" or field_type is str or field_type is Reasoning:
        desc = ""
    elif field_type is bool:
        desc = "must be True or False"
    elif field_type in (int, float):
        desc = f"must be a single {field_type.__name__} value"
    elif _annotation_is_subclass(field_type, enum.Enum):
        enum_vals = "; ".join(str(member.value) for member in field_type)
        desc = f"must be one of: {enum_vals}"
    elif get_origin(field_type) is Literal:
        desc = f"must exactly match (no extra characters) one of: {'; '.join([str(x) for x in get_args(field_type)])}"
    elif _annotation_is_subclass(field_type, Code) and field_type.description():
        desc = ""
    else:
        desc = f"must adhere to the JSON schema: {json.dumps(_json_schema(field_type), ensure_ascii=False)}"

    desc = (" " * 8) + f"# note: the value you produce {desc}" if desc else ""
    return f"{{{field_name}}}{desc}"


def _format_field_value(field_info: FieldInfo, value: Any, assume_text: bool = True) -> str | dict[str, str]:
    if isinstance(value, list) and field_info.annotation is str:
        string_value = _format_input_list_field_value(value)
    else:
        jsonable_value = _serialize_for_json(value)
        if isinstance(jsonable_value, dict | list):
            string_value = json.dumps(jsonable_value, ensure_ascii=False)
        else:
            string_value = str(jsonable_value)

    if assume_text:
        return string_value
    return {"type": "text", "text": string_value}


def _serialize_for_json(value: Any) -> Any:
    try:
        return TypeAdapter(type(value)).dump_python(value, mode="json")
    except Exception:
        return str(value)


def _parse_value(value: Any, annotation: Any) -> Any:
    if annotation is str:
        return str(value)

    if isinstance(annotation, enum.EnumMeta):
        return _find_enum_member(annotation, value)

    origin = get_origin(annotation)

    if origin is Literal:
        allowed = get_args(annotation)
        if value in allowed:
            return value

        if isinstance(value, str):
            v = value.strip()
            if v.startswith(("Literal[", "str[")) and v.endswith("]"):
                v = v[v.find("[") + 1 : -1]
            if len(v) > 1 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]

            if v in allowed:
                return v

        raise ValueError(f"{value!r} is not one of {allowed!r}")

    if not isinstance(value, str):
        return TypeAdapter(annotation).validate_python(value)

    if origin in (Union, types.UnionType) and type(None) in get_args(annotation) and str in get_args(annotation):
        return TypeAdapter(annotation).validate_python(value)

    candidate = json_repair.loads(value)
    if candidate == "" and value != "":
        try:
            candidate = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            candidate = value

    try:
        return TypeAdapter(annotation).validate_python(candidate)
    except pydantic.ValidationError as e:
        if _annotation_is_subclass(annotation, DspyType):
            try:
                return TypeAdapter(annotation).validate_python(value)
            except Exception:
                raise e
        raise


def _annotation_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is None:
        if annotation is Reasoning:
            return "str"
        if hasattr(annotation, "__name__"):
            return annotation.__name__
        return str(annotation)

    if origin is Literal:
        args_str = ", ".join(
            _quoted_string_for_literal_type_annotation(arg) if isinstance(arg, str) else _annotation_name(arg)
            for arg in args
        )
    else:
        args_str = ", ".join(_annotation_name(arg) for arg in args)
    return f"{_annotation_name(origin)}[{args_str}]"


def _annotation_is_subclass(annotation: Any, expected_base: type) -> bool:
    try:
        return inspect.isclass(annotation) and issubclass(annotation, expected_base)
    except TypeError:
        return False


def _json_schema(field_type: Any) -> dict[str, Any]:
    def move_type_to_front(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: move_type_to_front(item)
                for key, item in sorted(value.items(), key=lambda item: (item[0] != "type", item[0]))
            }
        if isinstance(value, list):
            return [move_type_to_front(item) for item in value]
        return value

    return move_type_to_front(pydantic.TypeAdapter(field_type).json_schema())


def _find_enum_member(enum_type: enum.EnumMeta, identifier: Any) -> enum.Enum:
    for member in enum_type:
        if member.value == identifier:
            return member
    if identifier in enum_type.__members__:
        return enum_type[identifier]
    raise ValueError(f"{identifier} is not a valid name or value for the enum {enum_type.__name__}")


def _format_input_list_field_value(value: list[Any]) -> str:
    if len(value) == 0:
        return "N/A"
    if len(value) == 1:
        return _format_blob(value[0])
    return "\n".join([f"[{idx + 1}] {_format_blob(txt)}" for idx, txt in enumerate(value)])


def _format_blob(blob: str) -> str:
    if "\n" not in blob and "«" not in blob and "»" not in blob:
        return f"«{blob}»"
    modified_blob = blob.replace("\n", "\n    ")
    return f"«««\n    {modified_blob}\n»»»"


def _quoted_string_for_literal_type_annotation(value: str) -> str:
    has_single = "'" in value
    has_double = '"' in value
    if has_single and not has_double:
        return f'"{value}"'
    if has_double and not has_single:
        return f"'{value}'"
    if has_single and has_double:
        escaped = value.replace("'", "\\'")
        return f"'{escaped}'"
    return f"'{value}'"
