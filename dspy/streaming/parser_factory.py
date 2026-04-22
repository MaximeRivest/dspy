from __future__ import annotations

from typing import TYPE_CHECKING

from dspy.streaming.parsers.base import BaseStreamParser

if TYPE_CHECKING:
    from dspy.adapters.base import Adapter
    from dspy.signatures.signature import Signature


def create_stream_parser(adapter: Adapter, signature: type[Signature]) -> BaseStreamParser:
    """Create the right stream parser for the given adapter and signature.

    Falls back to ``ChatStreamParser`` for unrecognized adapters (best-effort
    field tagging based on ``[[ ## field ## ]]`` delimiters).
    """
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.adapters.xml_adapter import XMLAdapter
    from dspy.streaming.parsers import ChatStreamParser, JSONStreamParser, XMLStreamParser

    output_fields = list(signature.output_fields.keys())

    if isinstance(adapter, JSONAdapter):
        return JSONStreamParser(output_fields)
    if isinstance(adapter, XMLAdapter):
        return XMLStreamParser(output_fields)
    if isinstance(adapter, ChatAdapter) or adapter is None:
        return ChatStreamParser(output_fields)

    # Unknown adapter — best-effort with ChatAdapter format
    return ChatStreamParser(output_fields)
