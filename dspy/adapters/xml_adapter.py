"""XMLAdapter — outputs as XML tags, now a TemplateAdapter configuration."""

from __future__ import annotations

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.template_adapter import TemplateAdapter
from dspy.utils.callback import BaseCallback

_XML_MESSAGES = [
    {
        "role": "system",
        "content": "{field_descriptions()}\n{field_structure(style='xml')}\n{task_description()}",
    },
    {"role": "demos"},
    {"role": "history"},
    {
        "role": "user",
        "content": "{inputs(style='xml')}\n\n{output_requirements(style='xml')}",
    },
]


class XMLAdapter(ChatAdapter):
    """Adapter that expects XML-tagged output from the LM.

    Inherits from ChatAdapter for fallback compatibility.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type] | None = None,
    ):
        TemplateAdapter.__init__(
            self,
            messages=_XML_MESSAGES,
            parse_mode="xml",
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            native_response_types=native_response_types,
        )
        self.use_json_adapter_fallback = False
