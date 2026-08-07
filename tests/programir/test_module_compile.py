import runpy
from pathlib import Path

import dspy
from dspy.clients.openai_compat_lm import _OpenAICompatLM
from dspy.programir import compile, link, read, write


class TwoStage(dspy.Module):
    def __init__(self):
        self.draft = dspy.Predict("question -> draft")
        self.finish = dspy.Predict("question, draft -> answer")

    def forward(self, question):
        draft = self.draft(question=question)
        answer = self.finish(question=question, draft=draft.draft)
        return answer


class NestedStage(dspy.Module):
    def __init__(self):
        self.generate = dspy.Predict("question -> draft")

    def forward(self, question):
        return self.generate(question=question)


class NestedProgram(dspy.Module):
    def __init__(self):
        self.stage = NestedStage()
        self.finish = dspy.Predict("draft -> answer")

    def forward(self, question):
        draft = self.stage(question=question)
        return self.finish(draft=draft.draft)


class SharedPredictors(dspy.Module):
    def __init__(self):
        predictor = dspy.Predict("question -> answer")
        self.left = predictor
        self.right = predictor

    def forward(self, question):
        return self.left(question=question)


def configured_two_stage():
    program = TwoStage()
    lm = dspy.LM("openai/shared-model")
    program.set_lm(lm)
    program.set_adapter(dspy.JSONAdapter())
    program.finish.set_adapter(dspy.ChatAdapter())
    return program


def test_compile_composite_module_builds_tree_pools_and_forwards():
    manifest = compile(configured_two_stage()).to_manifest()
    components = manifest["components"]

    assert components["1_module_tree"] == {
        "kind": "TwoStage",
        "name": "self",
        "module_class": "TwoStage",
        "forward_ref": "5_forward/self",
        "children": [
            {
                "kind": "Predict",
                "name": "draft",
                "children": [],
                "bindings": {"adapter": "json", "lm": "openai-shared-model", "delta": None},
            },
            {
                "kind": "Predict",
                "name": "finish",
                "children": [],
                "bindings": {"adapter": "chat", "lm": "openai-shared-model", "delta": None},
            },
        ],
    }
    assert list(components["2_signature"]) == ["draft", "finish"]
    assert list(components["4_adapter"]) == ["json", "chat"]
    assert list(components["8_lm"]) == ["openai-shared-model"]
    assert components["5_forward"]["self"]["body"][1] == {
        "node": "Assign",
        "target": "answer",
        "value": {
            "node": "Call",
            "leaf": {"kind": "predict", "ref": "finish"},
            "kwargs": {
                "question": {"node": "Var", "name": "question"},
                "draft": {"node": "Attr", "obj": "draft", "attr": "draft"},
            },
        },
    }


def test_openai_compatible_lm_emits_packaged_local_endpoint():
    program = dspy.Predict("question -> answer")
    program.set_lm(
        _OpenAICompatLM(
            model="org/canonical-weights",
            base_url="http://localhost:8000/v1",
            api_key="not-written",
            require_auth=True,
        )
    )

    components = compile(program).to_manifest()["components"]
    entry = components["8_lm"]["org-canonical-weights"]

    assert entry["weights_identity"] == "org/canonical-weights"
    assert entry["class"] == {
        "identity": "dspy.clients.openai_compat_lm._OpenAICompatLM",
        "origin": "packaged",
        "language": "python",
        "deps": ["dspy"],
    }
    assert entry["placement"] == {
        "rung": "http_local",
        "contract": "forward(LMRequest)->LMResponse",
        "endpoint_ref": "LM_ENDPOINT",
        "default_endpoint": "http://localhost:8000/v1",
        "isolation": "none",
        "credential_ref": "LM_API_KEY",
    }
    assert components["9_environment"]["python"]["dependencies"] == ["dspy==3.3.0"]


def test_pooling_deduplicates_by_object_not_equal_configuration():
    program = TwoStage()
    program.draft.set_lm(dspy.LM("openai/same-model"))
    program.finish.set_lm(dspy.LM("openai/same-model"))
    program.draft.set_adapter(dspy.ChatAdapter())
    program.finish.set_adapter(dspy.ChatAdapter())

    components = compile(program).to_manifest()["components"]

    assert list(components["8_lm"]) == ["openai-same-model", "openai-same-model-2"]
    assert list(components["4_adapter"]) == ["chat", "chat-2"]
    assert components["8_lm"]["openai-same-model-2"]["placement"] == {
        "rung": "http_remote",
        "contract": "forward(LMRequest)->LMResponse",
        "endpoint_ref": "LM_ENDPOINT_2",
        "isolation": "none",
        "credential_ref": "LM_API_KEY_2",
    }


def test_composite_module_roundtrips_and_links(tmp_path):
    original = compile(configured_two_stage())
    destination = tmp_path / "two-stage.ir"

    finalized = write(original, destination)
    restored = read(destination)

    assert restored == finalized
    assert link(restored) == {
        "draft": {"adapter": "json", "lm": "openai-shared-model", "delta": None},
        "finish": {"adapter": "chat", "lm": "openai-shared-model", "delta": None},
    }


def test_compile_nested_module_uses_dotted_predictor_paths():
    program = NestedProgram()
    program.set_lm(dspy.LM("openai/model"))
    program.set_adapter(dspy.ChatAdapter())

    manifest = compile(program).to_manifest()

    assert list(manifest["components"]["2_signature"]) == ["stage.generate", "finish"]
    assert list(manifest["components"]["5_forward"]) == ["stage", "self"]
    assert manifest["components"]["5_forward"]["self"]["body"][0]["value"]["leaf"] == {
        "kind": "module",
        "ref": "stage",
    }


def test_v01_exemplar_compiles_as_authored():
    namespace = runpy.run_path(Path("roadmap/exemplar-program-v01.py"))
    program = namespace["TicketAssistant"]()
    program.set_lm(dspy.LM("openai/model"))
    program.set_adapter(dspy.JSONAdapter())

    ir = compile(program)

    assert list(ir.manifest["components"]["5_forward"]) == ["policy", "self"]
    assert ir.manifest["components"]["5_forward"]["self"]["body"][3]["node"] == "If"


def test_compile_refuses_shared_predictor_instance():
    program = SharedPredictors()
    program.set_lm(dspy.LM("openai/model"))

    try:
        compile(program)
    except ValueError as error:
        assert "shared by 'left' and 'right'" in str(error)
    else:
        raise AssertionError("shared predictor should refuse")
