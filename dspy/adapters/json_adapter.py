"""JSONAdapter — outputs as JSON objects, now a TemplateAdapter configuration."""

from __future__ import annotations

from typing import Any

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.template_adapter import TemplateAdapter
from dspy.utils.callback import BaseCallback

_JSON_MESSAGES = [
    {
        "role": "system",
        "content": "{field_descriptions()}\n{field_structure(style='json')}\n{task_description()}",
    },
    {"role": "demos"},
    {"role": "history"},
    {
        "role": "user",
        "content": "{inputs(style='chat')}\n\n{output_requirements(style='json')}",
    },
]


class JSONAdapter(ChatAdapter):
    """Adapter that expects JSON output from the LM.

    Inherits from ChatAdapter so it serves as the fallback target —
    ``isinstance(self, JSONAdapter)`` prevents infinite fallback loops.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type] | None = None,
    ):
        # Skip ChatAdapter.__init__ — go straight to TemplateAdapter
        TemplateAdapter.__init__(
            self,
            messages=_JSON_MESSAGES,
            parse_mode="json",
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            native_response_types=native_response_types,
        )
        self.use_json_adapter_fallback = False
