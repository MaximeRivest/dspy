import json

import dspy
from dspy.programir import compile
from dspy.programir.shim import handle_line


def request(op, **payload):
    return handle_line(json.dumps({"id": "test", "op": op, **payload}))


def test_shim_advertises_grade1_and_interpreter_profiles():
    reply = request("capabilities")

    assert reply["ok"] is True
    assert reply["result"]["grades"] == [1]
    assert reply["result"]["versions"]["interpreter_profile"] == "1.0"
    assert {"load_manifest", "check_versions", "link", "profile_check", "node_compile"} <= set(
        reply["result"]["ops"]
    )


def test_shim_load_link_and_explain_compiler_output():
    predictor = dspy.Predict("question -> answer")
    predictor.set_lm(dspy.LM("openai/model"))
    manifest = compile(predictor).to_manifest()

    assert request("load_manifest", manifest=manifest)["result"] == {"manifest": manifest}
    assert request("link", manifest=manifest)["result"] == {
        "bindings": {
            "self": {"adapter": "chat", "lm": "openai-model", "delta": None}
        }
    }
    assert request("explain", manifest=manifest)["result"]["view"] == 1


def test_shim_uses_contract_refusal_codes():
    manifest = compile(
        _predictor_with_lm()
    ).to_manifest()
    manifest["unexpected"] = True

    reply = request("load_manifest", manifest=manifest)

    assert reply["ok"] is False
    assert reply["error"]["code"] == "PIR-E-MANIFEST-002"


def _predictor_with_lm():
    predictor = dspy.Predict("question -> answer")
    predictor.set_lm(dspy.LM("openai/model"))
    return predictor
