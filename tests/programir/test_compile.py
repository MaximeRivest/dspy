import json
from pathlib import Path

import pytest

import dspy
from dspy.programir import ProgramIR, compile
from dspy.programir.compile import build_program_ir
from dspy.programir.versions import IMPLEMENTED_VERSIONS


def test_compile_bare_predict_produces_programir_directly():
    predictor = dspy.Predict("question -> answer", temperature=0.2)
    predictor.set_lm(dspy.LM("openai/example-model", api_key="live-secret"))
    predictor.demos = [dspy.Example(question="2+2?", answer="4").with_inputs("question")]

    ir = compile(predictor)
    manifest = ir.to_manifest()

    assert isinstance(ir, ProgramIR)
    assert manifest["versions"] == IMPLEMENTED_VERSIONS
    assert manifest["components"]["1_module_tree"]["bindings"] == {
        "adapter": "chat",
        "lm": "openai-example-model",
        "delta": None,
    }
    assert manifest["components"]["2_signature"]["self"]["fields"] == [
        {
            "name": "question",
            "direction": "input",
            "prefix": "Question:",
            "desc": "${question}",
            "shape": {"type": "string"},
            "semantic_role": "plain",
        },
        {
            "name": "answer",
            "direction": "output",
            "prefix": "Answer:",
            "desc": "${answer}",
            "shape": {"type": "string"},
            "semantic_role": "plain",
        },
    ]
    assert manifest["components"]["3b_demos"]["self"] == [
        {"question": "2+2?", "answer": "4", "input_keys": ["question"]}
    ]
    assert manifest["components"]["3c_predictor_config"]["self"] == {"temperature": 0.2}
    assert "live-secret" not in json.dumps(manifest)


def test_compile_uses_the_predictor_adapter_binding_before_ambient():
    predictor = dspy.Predict("question -> answer")
    predictor.set_lm(dspy.LM("openai/example-model"))
    predictor.set_adapter(dspy.JSONAdapter())

    with dspy.context(adapter=dspy.ChatAdapter()):
        manifest = compile(predictor).to_manifest()

    assert manifest["components"]["1_module_tree"]["bindings"]["adapter"] == "json"
    assert list(manifest["components"]["4_adapter"]) == ["json"]


def test_compile_refuses_predict_without_lm():
    with pytest.raises(ValueError, match="predictor 'self'"):
        compile(dspy.Predict("question -> answer"))


def test_compile_refuses_non_module_frontend_values():
    with pytest.raises(TypeError, match="does not support object"):
        compile(object())


def test_plain_component_builder_refuses_non_json_values():
    with pytest.raises(ValueError, match="non-JSON"):
        build_program_ir(
            versions=dict(IMPLEMENTED_VERSIONS),
            components={"bad": object()},
        )


def test_frontend_program_is_not_a_public_or_internal_representation():
    import dspy.programir as programir

    assert not hasattr(programir, "FrontendProgram")
    assert "FrontendProgram" not in Path("dspy/programir/model.py").read_text()


def test_contract_pin_is_one_full_sha_with_newline():
    content = Path("CONTRACT_PIN").read_text()
    assert content.endswith("\n")
    assert len(content) == 41
    int(content.strip(), 16)
