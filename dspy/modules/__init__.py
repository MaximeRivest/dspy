"""IR-native modules: the program IS the IR.

`Module` compiles its authored `forward` into the ProgramIR node set and
executes through the engine interpreter; `Predict` is the leaf that runs
one LM exchange. `ChainOfThought` is a Predict with the reasoning role
engaged; `ReAct`, `ReActV2`, and `RLM` are composed modules whose
forwards compile clean into node-set v0.3 by construction.
"""

from dspy.modules.chain_of_thought import ChainOfThought
from dspy.modules.module import Module
from dspy.modules.predict import Predict
from dspy.modules.react import ReAct
from dspy.modules.react_v2 import ReActV2
from dspy.modules.rlm import RLM

__all__ = ["RLM", "ChainOfThought", "Module", "Predict", "ReAct", "ReActV2"]
