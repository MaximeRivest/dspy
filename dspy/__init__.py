"""dspy rebuilt on the IR foundation (branch greenfield-ir).

Stage A1: core (signatures, examples, predictions, typed errors) and
lm (LM, DummyLM, explicit bindings). Stage A2: adapters v2 — the entry
IS the adapter (lens parsers, rule strategies, codec families). Stage
A3: the execution spine — the program IS the IR: `Module.forward`
compiles to the node set, `program(**inputs)` runs through the engine,
`program.save(dir)` / `dspy.load(dir, bindings=...)` is the one
serialization path. Stage A4: the module library (ChainOfThought, ReAct,
ReActV2, RLM), subset-clean by construction. Stage A5: optimizers as IR
mutations — propose mutates demos/instructions as data, score replays
the program through the engine, keep is the artifact.
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

from dspy.modules import RLM, BestOfN, ChainOfThought, Module, Predict, ReAct, ReActV2, Refine
from dspy.programir.interpreters import InProcessInterpreter
from dspy import optim
from dspy.optim import BootstrapFewShot, BootstrapFewShotWithRandomSearch, LabeledFewShot
from dspy.tooling import Broker, Envelope, SecretFromEnv, Session, confined, retrust, tool

from dspy.__metadata__ import __author__, __author_email__, __description__, __name__, __url__, __version__


def load(path, bindings=None, *, envelope=None):
    """Load a saved program artifact: read + link + materialize.

    Args:
        path: Artifact directory written by `program.save(path)`.
        bindings: Receiver bindings, keyed by kind (`"lm"`, `"adapter"`,
            `"tool"`, `"interpreter"`), then by pool entry name. LM and
            interpreter entries always need a binding; a missing one
            refuses naming the entry.
        envelope: An optional `dspy.Envelope` — the receiver's isolation
            envelope (D-042). An envelope meeting an `isolation_required`
            leaf's floor IS the grant, so the leaf runs from its sidecar
            without a hand-bound callable.

    Returns:
        A callable program running the artifact through the engine.

    Examples:
        ```python
        program = dspy.load("artifacts/qa", bindings={"lm": {"openai-gpt-4o-mini": lm}})
        prediction = program(question="Why is the sky blue?")
        ```
    """
    from dspy.programir import load as _load

    return _load(path, bindings, envelope=envelope)


def diff(old, new):
    """Semantic diff of two programs or artifacts as report lines.

    Accepts `Module`s, `ProgramIR` values, manifest dicts, or artifact
    paths on either side; returns rendered lines (empty = equivalent).
    """
    from dspy.programir.tools.diff import diff as _diff

    return _diff(old, new)
