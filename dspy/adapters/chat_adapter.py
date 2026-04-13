"""ChatAdapter — DSPy's default adapter, now a TemplateAdapter configuration.

Uses ``[[ ## field ## ]]`` delimiters for field separation.
Falls back to JSONAdapter on parse errors.
"""

from __future__ import annotations

import re
from typing import Any, NamedTuple

from pydantic.fields import FieldInfo

from dspy.adapters.template_adapter import TemplateAdapter, _CHAT_HEADER
from dspy.clients.base_lm import BaseLM
from dspy.signatures.signature import Signature
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import ContextWindowExceededError

# Re-exports for backward compatibility
field_header_pattern = _CHAT_HEADER


class FieldInfoWithName(NamedTuple):
    name: str
    info: FieldInfo

_CHAT_MESSAGES = [
    {
        "role": "system",
        "content": "{field_descriptions()}\n{field_structure(style='chat')}\n{task_description()}",
    },
    {"role": "demos"},
    {"role": "history"},
    {
        "role": "user",
        "content": "{inputs(style='chat')}\n\n{output_requirements(style='chat')}",
    },
]


class ChatAdapter(TemplateAdapter):
    """Default Adapter using ``[[ ## field ## ]]`` markers.

    Falls back to JSONAdapter when parsing fails.
    """

    def __init__(
        self,
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type] | None = None,
        use_json_adapter_fallback: bool = True,
    ):
        super().__init__(
            messages=_CHAT_MESSAGES,
            parse_mode="chat",
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            native_response_types=native_response_types,
        )
        self.use_json_adapter_fallback = use_json_adapter_fallback

    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, ContextWindowExceededError) or isinstance(self, JSONAdapter) or not self.use_json_adapter_fallback:
                raise
            return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)

    async def acall(self, lm, lm_kwargs, signature, demos, inputs):
        try:
            return await super().acall(lm, lm_kwargs, signature, demos, inputs)
        except Exception as e:
            from dspy.adapters.json_adapter import JSONAdapter

            if isinstance(e, ContextWindowExceededError) or isinstance(self, JSONAdapter) or not self.use_json_adapter_fallback:
                raise
            return await JSONAdapter().acall(lm, lm_kwargs, signature, demos, inputs)
