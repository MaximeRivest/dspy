"""The core surface: signatures, examples, predictions, and typed errors.

One import point for the intent layer. `Signature` keeps the historical
metaclass authoring surface (class signatures, string signatures, role
shorthand); `Prediction` carries declared outputs plus the
`_trajectory` exhaust channel; `dspy.core.errors` is the canonical
typed-error table from the programir contract. The normalized LM
request/response types stay in `dspy.core.types` (import that module
directly; they are the adapter engine's substrate, not top-level API).
"""

from dspy.core.errors import (
    CATCHABLE_NAMES,
    HANDLER_NAMES,
    RAISEABLE,
    AdapterParseError,
    CatchableError,
    InterpreterArithmeticError,
    InterpreterError,
    InterpreterKeyError,
    InterpreterTypeError,
    LMError,
    LoopCapError,
    MalformedNodeError,
    PirError,
    ToolError,
    UncatchableError,
    handler_matches,
)
from dspy.core.example import Example
from dspy.core.prediction import Completions, Prediction
from dspy.signatures import (
    InputField,
    OutputField,
    Signature,
    SignatureMeta,
    ensure_signature,
    make_signature,
)

__all__ = [
    # signatures
    "InputField",
    "OutputField",
    "Signature",
    "SignatureMeta",
    "ensure_signature",
    "make_signature",
    # data containers
    "Example",
    "Completions",
    "Prediction",
    # typed-error table
    "PirError",
    "CatchableError",
    "ToolError",
    "InterpreterError",
    "AdapterParseError",
    "LMError",
    "InterpreterTypeError",
    "InterpreterKeyError",
    "InterpreterArithmeticError",
    "UncatchableError",
    "LoopCapError",
    "MalformedNodeError",
    "RAISEABLE",
    "CATCHABLE_NAMES",
    "HANDLER_NAMES",
    "handler_matches",
]
