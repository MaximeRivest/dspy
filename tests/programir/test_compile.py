import ast
import json
from pathlib import Path

import pytest

from dspy.programir import FrontendProgram, compile

VERSIONS = {
    "ir_version": "test",
    "node_set": "test",
    "roles": "test",
    "strategies": "test",
    "codecs": "test",
    "adapter_ir": "test",
    "lm15": "test",
}


def snapshot(**overrides):
    values = {
        "versions": VERSIONS,
        "module_tree": {
            "kind": "Predict",
            "name": "self",
            "children": [],
            "bindings": {"adapter": "chat", "lm": "model", "delta": None},
        },
        "signatures": {"self": {"fields": []}},
        "instructions": {"self": "Answer."},
        "demos": {"self": []},
        "predictor_config": {"self": {}},
        "adapters": {"chat": {"versions": {}}},
        "forwards": {
            "self": {
                "language": "restricted-python-ast",
                "args": ["question"],
                "body": [{"node": "Return", "value": {"node": "Const", "value": None}}],
            }
        },
        "tools": {},
        "interpreters": {},
        "lms": {"model": {"weights_identity": "provider/model"}},
        "environment": {},
        "credentials": (),
        "ambient_policy": {},
    }
    values.update(overrides)
    return FrontendProgram(**values)


def test_compile_assembles_components_without_mutating_snapshot():
    frontend = snapshot(evaluation={"metrics": {}, "devset": []})

    ir = compile(frontend)
    manifest = ir.to_manifest()

    assert manifest["versions"] == VERSIONS
    assert manifest["components"]["12_metric"] == {"metrics": {}, "devset": []}
    manifest["components"]["3a_instructions"]["self"] = "changed"
    assert frontend.instructions["self"] == "Answer."


def test_compile_refuses_missing_versions():
    with pytest.raises(ValueError, match="lm15"):
        compile(snapshot(versions={key: value for key, value in VERSIONS.items() if key != "lm15"}))


def test_compile_refuses_non_json_values():
    with pytest.raises(ValueError, match="non-JSON"):
        compile(snapshot(predictor_config={"self": {"bad": object()}}))


def test_neutral_compiler_has_no_runtime_framework_imports():
    source = Path("dspy/programir/compile.py").read_text()
    imports = {
        node.module
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    forbidden = ("dspy.clients", "dspy.dsp", "dspy.predict", "dspy.primitives", "dspy.teleprompt")
    assert not any(module.startswith(forbidden) for module in imports)


def test_contract_pin_is_one_full_sha_with_newline():
    content = Path("CONTRACT_PIN").read_text()
    assert content.endswith("\n")
    assert len(content) == 41
    int(content.strip(), 16)


def test_manifest_is_json_serializable():
    json.dumps(compile(snapshot()).to_manifest(), allow_nan=False)
