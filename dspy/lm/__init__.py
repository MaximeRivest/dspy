"""The LM layer: one lm15-routed LM, a scripted DummyLM, explicit bindings.

Capability facts (instruct/base, native_reasoning, native_fc,
native_citations) are declared constructor data — strategies predicate
on them, never on live-object sniffing. Bindings are a plain
module-level dict written by `dspy.configure(lm=...)`; per-predictor
bindings override it. No thread-locals, no context managers, no ambient
settings object.
"""

from dspy.lm.bindings import BINDINGS, BindingError, configure, resolve
from dspy.lm.dummy import DummyLM
from dspy.lm.lm import LM, LMCapabilities, default_router


def __getattr__(name: str):
    # `ROUTER` hydrates its model catalog on first use — importing it
    # eagerly would put ~1s of catalog loading on `import dspy`.
    if name == "ROUTER":
        from dspy.lm import lm as _lm

        return _lm.ROUTER
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BINDINGS",
    "BindingError",
    "DummyLM",
    "LM",
    "LMCapabilities",
    "ROUTER",
    "configure",
    "resolve",
]
