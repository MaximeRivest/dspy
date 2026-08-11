"""Interpreter primitives: the sandboxed-code machinery RLM-style modules use.

`Example` and `Prediction` moved to `dspy.core`; `Module` is rebuilt by
the execution spine (stage A3). What remains here is the code-interpreter
protocol and its local Deno/Pyodide implementation.
"""

from dspy.primitives.code_interpreter import CodeExecutionError, CodeInterpreter, CodeInterpreterError, FinalOutput
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.primitives.sandbox_serializable import SandboxSerializable

__all__ = [
    "CodeExecutionError",
    "CodeInterpreter",
    "CodeInterpreterError",
    "FinalOutput",
    "PythonInterpreter",
    "SandboxSerializable",
]
