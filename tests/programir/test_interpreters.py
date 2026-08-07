import pytest

from dspy.primitives.module import Module
from dspy.primitives.python_interpreter import PythonInterpreter
from dspy.programir import compile, read, write
from dspy.programir.interpreters import validate_interpreter_profile


class DuckDBInterpreter:
    def programir_profile(self):
        return {
            "language": "sql",
            "runtime": {"identity": "duckdb", "version": "1.2.0"},
            "contract": {
                "operation": "execute",
                "inputs": ["code", "vars"],
                "output": "result",
            },
            "namespace_policy": "one isolated in-memory database per program run",
            "result_convention": "query rows as JSON arrays",
            "vars_marshaling": "named JSON values registered as one-row tables",
            "packages": [],
            "resource_limits": {"memory_mb": 256},
            "isolation_floor": "none",
            "placement": {
                "rung": "in_process",
                "contract": "execute(code,vars)->result",
                "endpoint_ref": None,
                "isolation": "none",
                "credential_ref": None,
            },
        }


class InterpreterProgram(Module):
    def __init__(self, interpreter):
        self.sql = interpreter

    def forward(self, code):
        return self.sql(code=code)


def test_compile_structural_interpreter_profile(tmp_path):
    ir = compile(InterpreterProgram(DuckDBInterpreter()))
    components = ir.manifest["components"]

    assert ir.manifest["versions"]["interpreter_profile"] == "1.0"
    assert components["1_module_tree"]["uses_interpreter"] is True
    assert components["7_interpreter"]["sql"]["runtime"] == {
        "identity": "duckdb",
        "version": "1.2.0",
    }
    assert components["5_forward"]["self"]["body"][0]["value"]["leaf"] == {
        "kind": "interpreter",
        "ref": "sql",
    }

    destination = tmp_path / "interpreter.ir"
    write(ir, destination)
    assert read(destination) == ir


def test_structural_profile_refuses_placement_inside_isolation_floor():
    profile = DuckDBInterpreter().programir_profile()
    profile["isolation_floor"] = "wasm"
    profile["placement"]["isolation"] = "process"

    with pytest.raises(ValueError, match="inward"):
        validate_interpreter_profile(profile, name="sql")


def test_custom_interpreter_profile_must_be_complete():
    class IncompleteInterpreter:
        def programir_profile(self):
            return {"language": "python"}

    with pytest.raises(ValueError, match="missing keys"):
        compile(InterpreterProgram(IncompleteInterpreter()))


def test_current_python_interpreter_refuses_until_runtime_is_pinned():
    interpreter = object.__new__(PythonInterpreter)

    with pytest.raises(ValueError, match="Deno.*pyodide.*not pinned"):
        compile(InterpreterProgram(interpreter))
