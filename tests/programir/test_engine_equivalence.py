"""Trace equivalence: native execution vs compile -> materialize -> interpret.

The ex-13 method made a standing test. Each corpus program runs twice with
the same scripted DummyLM answers: once natively as a Python `dspy.Module`,
once through the ProgramIR engine. The oracle demands identical predict-call
traces — which predictor, what inputs, what order — and identical final
outputs. Both branches of the conditional program are covered.
"""

from copy import deepcopy

import dspy
from dspy.programir import compile, materialize
from dspy.programir.engine.errors import InterpreterError, ToolError
from dspy.utils.dummies import DummyLM
from tests.programir import corpus_programs
from tests.programir.corpus_programs import InProcessInterpreter, MiniReAct, MiniRLM, NestedAnswerer


def _compile_ir(make_program):
    program = make_program()
    program.set_lm(dspy.LM("openai/dummy"))
    return compile(program)


def _run_with_trace(callable_program, predictor_names, inputs):
    with dspy.settings.context(trace=[], adapter=None):
        output = callable_program(**inputs)
        trace = [
            (predictor_names[id(predictor)], dict(kwargs), prediction.toDict())
            for predictor, kwargs, prediction in dspy.settings.trace
        ]
    return output.toDict(), trace


def _run_native(make_program, answers, inputs):
    program = make_program()
    program.set_lm(DummyLM(deepcopy(answers)))
    names = {id(predictor): name for name, predictor in program.named_predictors()}
    return _run_with_trace(program, names, inputs)


def _run_engine(ir, answers, inputs, *, lm_entry="openai-dummy", extra_bindings=None):
    bindings = {"lm": {lm_entry: DummyLM(deepcopy(answers))}}
    bindings.update(extra_bindings or {})
    executable = materialize(ir, bindings=bindings)
    names = {id(predictor): path for path, predictor in executable.predictors.items()}
    return _run_with_trace(executable, names, inputs)


def _assert_equivalent(make_program, answers, inputs, *, extra_bindings=None, ir=None):
    ir = ir if ir is not None else _compile_ir(make_program)
    native_output, native_trace = _run_native(make_program, answers, inputs)
    engine_output, engine_trace = _run_engine(ir, answers, inputs, extra_bindings=extra_bindings)
    assert engine_trace == native_trace
    assert engine_output == native_output
    return native_trace


def test_nested_conditional_factual_branch_traces_identically():
    answers = [
        {"category": "factual"},
        {"draft_answer": "Paris."},
        {"answer": "Paris is the capital of France."},
    ]
    trace = _assert_equivalent(NestedAnswerer, answers, {"question": "capital of France?"})
    assert [name for name, _, _ in trace] == [
        "drafter.classifier.classify",
        "drafter.draft_factual",
        "polish",
    ]


def test_nested_conditional_creative_branch_traces_identically():
    answers = [
        {"category": "creative"},
        {"draft_answer": "A shimmering city of glass."},
        {"answer": "A shimmering city of glass, reborn."},
    ]
    trace = _assert_equivalent(NestedAnswerer, answers, {"question": "invent a city"})
    assert [name for name, _, _ in trace] == [
        "drafter.classifier.classify",
        "drafter.draft_creative",
        "polish",
    ]


def test_react_loop_with_dynamic_tool_dispatch_traces_identically():
    answers = [
        {"thought": "multiply", "tool_name": "calculator", "tool_arg": "6*7"},
        {"thought": "look it up", "tool_name": "lookup", "tool_arg": "France"},
        {"thought": "done", "tool_name": "finish", "tool_arg": ""},
        {"answer": "42; Paris"},
    ]
    trace = _assert_equivalent(MiniReAct, answers, {"question": "what is 6*7 and the capital of France?"})
    assert [name for name, _, _ in trace] == ["react", "react", "react", "extract"]
    # The tool observations feed the next react call: the trajectories match
    # only when both branches dispatched the same tools with the same results.
    assert "observation: 42" in trace[1][1]["trajectory"]
    assert "observation: Paris" in trace[2][1]["trajectory"]


def test_rlm_while_loop_with_interpreter_leaf_traces_identically():
    answers = [
        {"code": "result = 6*7"},
        {"answer": "42"},
    ]
    trace = _assert_equivalent(
        MiniRLM,
        answers,
        {"question": "6*7?"},
        extra_bindings={"interpreter": {"interpreter": InProcessInterpreter()}},
    )
    assert [name for name, _, _ in trace] == ["gen", "answer"]
    assert trace[1][1]["result"] == "42"


def test_materialize_refuses_unresolved_lm_and_interpreter_by_entry_name():
    import pytest

    ir = _compile_ir(MiniRLM)
    with pytest.raises(ValueError, match="LM pool entry 'openai-dummy'"):
        materialize(ir)
    with pytest.raises(ValueError, match="interpreter pool entry 'interpreter'"):
        materialize(ir, bindings={"lm": {"openai-dummy": DummyLM([])}})


def test_read_write_roundtrip_still_materializes(tmp_path):
    from dspy.programir import read, write

    ir = _compile_ir(NestedAnswerer)
    write(ir, tmp_path / "nested.ir")
    reread = read(tmp_path / "nested.ir")
    answers = [
        {"category": "factual"},
        {"draft_answer": "Ottawa."},
        {"answer": "Ottawa is the capital of Canada."},
    ]
    native_output, native_trace = _run_native(NestedAnswerer, answers, {"question": "capital of Canada?"})
    engine_output, engine_trace = _run_engine(reread, answers, {"question": "capital of Canada?"})
    assert engine_trace == native_trace
    assert engine_output == native_output


# Keep native corpus forwards runnable when a tool or interpreter branch
# raises: the classes reference these names as module globals.
corpus_programs.ToolError = ToolError
corpus_programs.InterpreterError = InterpreterError
