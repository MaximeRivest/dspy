"""The LM layer: one litellm-backed LM, a scripted DummyLM, explicit bindings.

Capability facts (instruct/base, native_reasoning, native_fc,
native_citations) are declared constructor data — strategies predicate
on them, never on live-object sniffing. Bindings are a plain
module-level dict written by `dspy.configure(lm=...)`; per-predictor
bindings override it. No thread-locals, no context managers, no ambient
settings object.
"""

from dspy.lm.bindings import BINDINGS, BindingError, configure, resolve
from dspy.lm.dummy import DummyLM
from dspy.lm.lm import LM, LMCapabilities

__all__ = [
    "BINDINGS",
    "BindingError",
    "DummyLM",
    "LM",
    "LMCapabilities",
    "configure",
    "resolve",
]
