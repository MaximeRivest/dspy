"""The type frontend: host-facing custom types and their role derivations.

Importing this package registers the type -> semantic-role derivations
with the signature layer (`dspy.signatures.roles`), so a bare
`tools: list[dspy.Tool]` field derives the `tools` role without the
signature layer ever importing the adapter world.
"""

from dspy.adapters.types.base_type import Type, split_message_content_for_custom_types
from dspy.adapters.types.citation import Citations
from dspy.adapters.types.history import History
from dspy.adapters.types.image import Image
from dspy.adapters.types.tool import Tool, ToolCallResults, ToolCalls
from dspy.signatures.roles import register_role_derivation

register_role_derivation(Tool, "tools")
register_role_derivation(ToolCalls, "tool_calls")
register_role_derivation(Citations, "citations")
register_role_derivation(History, "history")
register_role_derivation(Image, "media")

__all__ = [
    "Citations",
    "History",
    "Image",
    "Tool",
    "ToolCallResults",
    "ToolCalls",
    "Type",
    "split_message_content_for_custom_types",
]
