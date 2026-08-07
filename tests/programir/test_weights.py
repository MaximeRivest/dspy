import json
from pathlib import Path

import pytest

import dspy
from dspy.programir import compile
from dspy.programir.validate import validate_manifest


class _Config:
    def to_dict(self):
        return {"architectures": ["TinyModel"], "hidden_size": 2}


class _Model:
    config = _Config()

    def state_dict(self):
        shared = object()
        return {"embed.weight": shared, "lm_head.weight": shared, "layer.weight": object()}


class _Tokenizer:
    def save_pretrained(self, path):
        path = Path(path)
        (path / "tokenizer.json").write_text('{"version":"1.0"}\n')
        (path / "tokenizer_config.json").write_text('{"model_max_length":32}\n')


class WeightOwningLM(dspy.BaseLM):
    forward_contract = "typed_lm"

    def __init__(self, transformer, tokenizer):
        # deps: torch, transformers, safetensors
        super().__init__(model="test/tiny")
        self.transformer = transformer
        self.tokenizer = tokenizer

    def programir_weight_spec(self):
        return {
            "model_attribute": "transformer",
            "tokenizer_attribute": "tokenizer",
            "weights_identity": "test/tiny",
            "engine": "transformers",
            "device": "cpu",
            "frozen": False,
            "weight_ref": "base",
            "ties": [{"target": "lm_head.weight", "source": "embed.weight"}],
        }

    def forward(self, request):
        raise NotImplementedError


class SharedWeightProgram(dspy.Module):
    def __init__(self, lm):
        self.left = dspy.Predict("question -> answer")
        self.right = dspy.Predict("question -> answer")
        self.set_lm(lm)

    def forward(self, question):
        left = self.left(question=question)
        return self.right(question=left.answer)


def test_weight_protocol_bakes_one_shared_entry_and_sidecar_family(monkeypatch):
    captured = {}

    def fake_save(tensors):
        captured.update(tensors)
        return b"safe tensor bytes"

    monkeypatch.setattr("dspy.programir.weights._save_safetensors", fake_save)
    ir = compile(SharedWeightProgram(WeightOwningLM(_Model(), _Tokenizer())))
    components = ir.manifest["components"]
    validate_manifest(ir.to_manifest())

    assert list(components["8_lm"]) == ["test-tiny"]
    assert components["1_module_tree"]["children"][0]["bindings"]["lm"] == "test-tiny"
    assert components["1_module_tree"]["children"][1]["bindings"]["lm"] == "test-tiny"
    entry = components["8_lm"]["test-tiny"]
    assert entry["engine"] == "transformers"
    assert entry["class"]["deps"] == ["torch", "transformers", "safetensors"]
    assert entry["weights"] == {
        "format": "safetensors",
        "files": {
            "tensors": "weights/model.safetensors",
            "rebuild_config": "weights/rebuild_config.json",
            "tying": "weights/tying.json",
            "tokenizer": "weights/tokenizer/",
            "device": "weights/device.json",
        },
        "frozen": False,
        "weight_ref": "base",
        "placement": entry["placement"],
    }
    assert set(captured) == {"embed.weight", "layer.weight"}
    assert ir.sidecars["weights/model.safetensors"] == b"safe tensor bytes"
    assert json.loads(ir.sidecars["weights/tying.json"]) == [
        {"source": "embed.weight", "target": "lm_head.weight"}
    ]
    assert entry["class"]["identity"] == "weight_owning_lm.WeightOwningLM"
    assert entry["class"]["source"] == "lm/weight_owning_lm.py"
    assert ir.sidecars["lm/weight_owning_lm.py"].decode().startswith("import dspy\n")
    assert "class WeightOwningLM" in ir.sidecars["lm/weight_owning_lm.py"].decode()
    assert components["9_environment"]["python"]
    assert components["10_credentials"] == []


def test_weight_protocol_refuses_implicit_ties(monkeypatch):
    lm = WeightOwningLM(_Model(), _Tokenizer())
    lm.programir_weight_spec = lambda: {
        **WeightOwningLM.programir_weight_spec(lm),
        "ties": [{"target": "missing.weight", "source": "embed.weight"}],
    }
    program = dspy.Predict("question -> answer")
    program.set_lm(lm)
    monkeypatch.setattr("dspy.programir.weights._save_safetensors", lambda tensors: b"bytes")

    with pytest.raises(ValueError, match="absent tensor"):
        compile(program)


def test_custom_lm_without_weight_protocol_refuses():
    class UndeclaredLM(dspy.BaseLM):
        pass

    program = dspy.Predict("question -> answer")
    program.set_lm(UndeclaredLM("test/undeclared"))

    with pytest.raises(ValueError, match=r"must declare programir_weight_spec\(\)"):
        compile(program)
