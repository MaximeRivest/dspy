"""The custom-type base: host objects that render as structured content.

A `Type` subclass is a host-language value with a declared wire rendering:
`format()` returns either a plain string or a list of content parts (the
OpenAI-style content-part dicts). Values render into message text through
the reserved part markers; `split_message_content_for_custom_types` then
splits each user message around them so providers receive real content
parts. The streaming/native-feature hooks of the legacy type died with the
carve — strategies conduct native features now, as data.
"""

import json
import re
from typing import Any, get_args, get_origin

import json_repair
import pydantic

CUSTOM_TYPE_START_IDENTIFIER = "<<CUSTOM-TYPE-START-IDENTIFIER>>"
CUSTOM_TYPE_END_IDENTIFIER = "<<CUSTOM-TYPE-END-IDENTIFIER>>"


class Type(pydantic.BaseModel):
    """Base class for custom types carried in DSPy signatures.

    Subclasses implement `format()` to return either a string or a list of
    content-part dicts (the OpenAI user-message content array shape).

    Examples:
        ```python
        class Image(Type):
            url: str

            def format(self) -> list[dict[str, Any]]:
                return [{"type": "image_url", "image_url": {"url": self.url}}]
        ```
    """

    def format(self) -> list[dict[str, Any]] | str:
        raise NotImplementedError

    @classmethod
    def description(cls) -> str:
        """Description of the custom type, appended to field descriptions."""
        return ""

    @classmethod
    def extract_custom_type_from_annotation(cls, annotation):
        """Extract all custom types from an annotation, at any nesting depth."""
        # Direct match. Nested aliases like `list[dict[str, Event]]` pass
        # `isinstance(annotation, type)` on 3.10 but fail on 3.11+; ignore.
        try:
            if isinstance(annotation, type) and issubclass(annotation, cls):
                return [annotation]
        except TypeError:
            pass

        origin = get_origin(annotation)
        if origin is None:
            return []

        result = []
        for arg in get_args(annotation):
            result.extend(cls.extract_custom_type_from_annotation(arg))

        return result

    @pydantic.model_serializer()
    def serialize_model(self):
        formatted = self.format()
        if isinstance(formatted, list):
            return (
                f"{CUSTOM_TYPE_START_IDENTIFIER}{json.dumps(formatted, ensure_ascii=False)}{CUSTOM_TYPE_END_IDENTIFIER}"
            )
        return formatted


def split_message_content_for_custom_types(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Split user message content into content-part lists around custom types.

    Finds the reserved custom-type markers in each user message's string
    content and splits the content around them, so values like `dspy.Image`
    reach the provider as native content parts:

    ```
    [
        {"type": "text", "text": "{text_before_image}"},
        {"type": "image_url", "image_url": {"url": "{image_url}"}},
        {"type": "text", "text": "{text_after_image}"},
    ]
    ```

    Messages without markers pass through unchanged.
    """
    for message in messages:
        if message["role"] != "user":
            # Custom type parts live in user messages only.
            continue

        pattern = rf"{CUSTOM_TYPE_START_IDENTIFIER}(.*?){CUSTOM_TYPE_END_IDENTIFIER}"
        result = []
        last_end = 0
        content: str = message["content"]

        for match in re.finditer(pattern, content, re.DOTALL):
            start, end = match.span()

            if start > last_end:
                result.append({"type": "text", "text": content[last_end:start]})

            custom_type_content = match.group(1).strip()
            parsed = None

            for parse_fn in [json.loads, _parse_doubly_quoted_json, json_repair.loads]:
                try:
                    parsed = parse_fn(custom_type_content)
                    break
                except json.JSONDecodeError:
                    continue

            if parsed:
                for part in parsed:
                    result.append(part)
            else:
                result.append({"type": "text", "text": custom_type_content})

            last_end = end

        if last_end == 0:
            continue

        if last_end < len(content):
            result.append({"type": "text", "text": content[last_end:]})

        message["content"] = result

    return messages


def _parse_doubly_quoted_json(json_str: str) -> Any:
    """Parse a doubly quoted JSON string (a `Type` nested in a list or dict
    is json-encoded twice)."""
    return json.loads(json.loads(f'"{json_str}"'))
