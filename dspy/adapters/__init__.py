"""Adapters v2: the adapter IR made executable — the entry IS the adapter.

Public surface:

- `ChatAdapter` / `JSONAdapter` / `XMLAdapter` — thin preset constructors.
- `make_adapter` — template in, adapter out (lens parser by default).
- `Adapter` — the entry-backed class every constructor returns.
- `load_entry` / `dump_entry` (as `Adapter.dump_entry`) — exact serde of
  the extended 0.3.0-draft entry shape.
- `parse` — the parse-combinator authoring module (level-1 parse-data).
- `strategy` — the strategy-rule authoring module (rules as data).
- `register_strategy` / `register_codec` — the registries' public doors.
"""

from dspy.adapters import parse, strategy
from dspy.adapters.adapter import Adapter, AdapterCall
from dspy.adapters.codecs import register_codec
from dspy.adapters.errors import AdapterError, AdapterParseError, EntryError, UnserializableTypeError
from dspy.adapters.presets import ChatAdapter, JSONAdapter, XMLAdapter, make_adapter
from dspy.adapters.serde import ADAPTER_IR_VERSION, dumps_entry, load_entry
from dspy.adapters.strategies import register_strategy, unregister_strategy
from dspy.adapters.types import (
    Citations,
    History,
    Image,
    Tool,
    ToolCallResults,
    ToolCalls,
    Type,
)

__all__ = [
    "ADAPTER_IR_VERSION",
    "Adapter",
    "AdapterCall",
    "AdapterError",
    "AdapterParseError",
    "ChatAdapter",
    "Citations",
    "EntryError",
    "History",
    "Image",
    "JSONAdapter",
    "Tool",
    "ToolCallResults",
    "ToolCalls",
    "Type",
    "UnserializableTypeError",
    "XMLAdapter",
    "dumps_entry",
    "load_entry",
    "make_adapter",
    "parse",
    "register_codec",
    "register_strategy",
    "strategy",
    "unregister_strategy",
]
