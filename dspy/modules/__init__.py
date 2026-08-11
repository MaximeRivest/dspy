"""IR-native modules: the program IS the IR.

`Module` compiles its authored `forward` into the ProgramIR node set and
executes through the engine interpreter; `Predict` is the leaf that runs
one LM exchange. Composed modules (ChainOfThought, ReAct, RLM) arrive in
the next stage.
"""

from dspy.modules.module import Module
from dspy.modules.predict import Predict

__all__ = ["Module", "Predict"]
