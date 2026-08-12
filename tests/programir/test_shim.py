import json

import dspy
from dspy.programir import compile
from dspy.programir.shim import handle_line


def request(op, **payload):
    return handle_line(json.dumps({"id": "test", "op": op, **payload}))


def test_shim_advertises_grades_and_interpreter_profiles():
    reply = request("capabilities")

    assert reply["ok"] is True
    # Grade 2 arrived with node_execute (scripted execution, PROTOCOL.md).
    assert reply["result"]["grades"] == [1, 2]
    assert reply["result"]["versions"]["interpreter_profile"] == "1.0"
    assert {
        "load_manifest",
        "check_versions",
        "link",
        "profile_check",
        "node_compile",
        "node_execute",
    } <= set(reply["result"]["ops"])


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


def test_shim_node_execute_traces_calls_and_attribution():
    forwards = {
        "self": {
            "language": "restricted-python-ast",
            "args": [{"name": "inputs", "record": "self"}],
            "body": [
                {
                    "node": "Assign",
                    "target": "pred",
                    "value": {
                        "node": "Call",
                        "leaf": {"kind": "predict", "ref": "inner"},
                        "splat": "inputs",
                        "kwargs": {},
                    },
                },
                {"node": "Return", "value": {"node": "Var", "name": "pred"}},
            ],
        }
    }
    leaves = {"predicts": {"inner": [{"value": {"answer": "4"}}]}}

    reply = request(
        "node_execute",
        forwards=forwards,
        inputs={"question": "2+2?"},
        leaves=leaves,
        record_attribution=True,
    )

    assert reply["ok"] is True
    result = reply["result"]
    assert result["outcome"] == {"kind": "prediction", "prediction": {"answer": "4"}}
    assert result["calls"] == [
        {
            "kind": "predict",
            "target": "inner",
            "kwargs": {"question": "2+2?"},
            "outcome": {"value": {"answer": "4"}},
        }
    ]
    # View-2 attribution (PIR-021): total counts each call once; scripted
    # execution has no live bridge, so each call labels its own target.
    assert result["attribution"] == {"inner": 1}


def _predictor_with_lm():
    predictor = dspy.Predict("question -> answer")
    predictor.set_lm(dspy.LM("openai/model"))
    return predictor
