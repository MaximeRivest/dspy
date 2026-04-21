"""dspy.lm_types — typed LM interface (parts, messages, config, completion,
errors, streaming, styles, adapter).

The core primitives for a provider-agnostic, typed DSPy LM contract.
This is the "north star" implementation layered on top of existing DSPy;
it coexists with the legacy ``dspy.BaseLM`` / ``dspy.adapters.Adapter``
surface via the ``AdapterV2.as_legacy()`` shim.
"""

from dspy.lm_types.adapter_v2 import AdapterV2, from_preset
from dspy.lm_types.base_lm_v2 import BaseLMv2
from dspy.lm_types.completion import LMCompletion, LMResponse, LMUsage
from dspy.lm_types.config import LMConfig, LMToolDef
from dspy.lm_types.errors import (
    AdapterError,
    AdapterParseError,
    AuthError,
    ContextLengthError,
    DSPyError,
    ErrorCode,
    InvalidRequestError,
    LMError,
    RateLimitError,
    RETRYABLE_ERRORS,
    ServerError,
    TimeoutError,
    TransportError,
    error_class,
    error_code,
    map_http_status,
)
from dspy.lm_types.messages import LMMessage
from dspy.lm_types.parts import (
    AudioPart,
    CitationPart,
    DocumentPart,
    ImagePart,
    Part,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from dspy.lm_types.preset import (
    CHAT_PRESET,
    CSV_PRESET,
    CallableFragment,
    Content,
    Demos,
    Element,
    ForEach,
    Fragment,
    History,
    InputFields,
    Inputs,
    Instruction,
    JSON_PRESET,
    Message,
    OutputFields,
    OutputRequest,
    Outputs,
    PRESETS,
    Preset,
    Structure,
    Text,
    XML_PRESET,
)
from dspy.lm_types.streaming import (
    LMStreamError,
    LMStreamEvent,
    PartAccumulator,
    PartDelta,
)
from dspy.lm_types.styles import (
    CSVStyle,
    ChatStyle,
    FieldChunk,
    FieldLayout,
    JSONStyle,
    StreamParser,
    Style,
    StyleLike,
    TSVStyle,
    XMLStyle,
    available_styles,
    get_style,
    register_style,
)

__all__ = [
    # Parts
    "Part",
    "TextPart",
    "ImagePart",
    "AudioPart",
    "DocumentPart",
    "ToolCallPart",
    "ToolResultPart",
    "ThinkingPart",
    "CitationPart",
    # Messages
    "LMMessage",
    # Config
    "LMConfig",
    "LMToolDef",
    # Completion
    "LMCompletion",
    "LMResponse",
    "LMUsage",
    # Errors
    "DSPyError",
    "LMError",
    "AuthError",
    "RateLimitError",
    "ContextLengthError",
    "InvalidRequestError",
    "TimeoutError",
    "ServerError",
    "TransportError",
    "RETRYABLE_ERRORS",
    "AdapterError",
    "AdapterParseError",
    "ErrorCode",
    "error_code",
    "error_class",
    "map_http_status",
    # Streaming
    "PartDelta",
    "LMStreamError",
    "LMStreamEvent",
    "PartAccumulator",
    # Styles
    "Style",
    "StyleLike",
    "FieldLayout",
    "FieldChunk",
    "StreamParser",
    "ChatStyle",
    "JSONStyle",
    "XMLStyle",
    "CSVStyle",
    "TSVStyle",
    "register_style",
    "get_style",
    "available_styles",
    # Preset fragments
    "Text",
    "Instruction",
    "InputFields",
    "OutputFields",
    "Structure",
    "Inputs",
    "Outputs",
    "OutputRequest",
    "ForEach",
    "CallableFragment",
    "Fragment",
    "Content",
    "Message",
    "Demos",
    "History",
    "Element",
    "Preset",
    "CHAT_PRESET",
    "JSON_PRESET",
    "XML_PRESET",
    "CSV_PRESET",
    "PRESETS",
    # LM / Adapter
    "BaseLMv2",
    "AdapterV2",
    "from_preset",
]
