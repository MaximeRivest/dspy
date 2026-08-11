"""dspy rebuilt on the IR foundation (branch greenfield-ir).

Stage A1: core (signatures, examples, predictions, typed errors) and
lm (LM, DummyLM, explicit bindings). Stage A2: adapters v2 — the entry
IS the adapter (lens parsers, rule strategies, codec families). The
execution spine, modules, and optimizers land in later stages.
"""

from dspy.core import (
    CATCHABLE_NAMES,
    HANDLER_NAMES,
    RAISEABLE,
    AdapterParseError,
    CatchableError,
    Completions,
    Example,
    InputField,
    InterpreterArithmeticError,
    InterpreterError,
    InterpreterKeyError,
    InterpreterTypeError,
    LMError,
    LoopCapError,
    MalformedNodeError,
    OutputField,
    PirError,
    Prediction,
    Signature,
    SignatureMeta,
    ToolError,
    UncatchableError,
    ensure_signature,
    handler_matches,
    make_signature,
)
from dspy.lm import LM, BindingError, DummyLM, LMCapabilities, configure
from dspy import roles
from dspy import adapters
from dspy.adapters import (
    Adapter,
    ChatAdapter,
    Citations,
    History,
    Image,
    JSONAdapter,
    Tool,
    ToolCallResults,
    ToolCalls,
    XMLAdapter,
    load_entry,
    make_adapter,
)

from dspy.__metadata__ import __author__, __author_email__, __description__, __name__, __url__, __version__
