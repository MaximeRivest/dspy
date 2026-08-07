import json

import pytest

import dspy
from dspy.programir import ProgramIR, compile, link, read, write
from dspy.programir.errors import ProgramIRRefusal


def compiled_predict():
    predictor = dspy.Predict("question -> answer")
    predictor.set_lm(dspy.LM("openai/example-model", api_key="secret"))
    return compile(predictor)


def test_read_write_roundtrip_returns_same_programir(tmp_path):
    original = compiled_predict()
    destination = tmp_path / "program.ir"

    write(original, destination)
    restored = read(destination)

    assert restored == original
    assert link(restored) == {
        "self": {"adapter": "chat", "lm": "openai-example-model", "delta": None}
    }


def test_read_preserves_sidecars_as_uninterpreted_bytes(tmp_path):
    destination = tmp_path / "program.ir"
    original = compiled_predict()
    with_sidecar = ProgramIR(manifest=original.manifest, sidecars={"tools/a.py": b"raise SystemExit\n"})

    write(with_sidecar, destination)
    restored = read(destination)

    assert restored.sidecars == {"tools/a.py": b"raise SystemExit\n"}


def test_read_refuses_unknown_manifest_key(tmp_path):
    destination = tmp_path / "program.ir"
    write(compiled_predict(), destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["mystery"] = True
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ProgramIRRefusal) as caught:
        read(destination)

    assert caught.value.code == "PIR-E-MANIFEST-001"
    assert caught.value.detail == {"unknown": ["mystery"]}


def test_read_refuses_incompatible_version_before_linking(tmp_path):
    destination = tmp_path / "program.ir"
    write(compiled_predict(), destination)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["versions"]["ir_version"] = "0.2"
    manifest["components"]["1_module_tree"]["bindings"]["lm"] = "missing"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ProgramIRRefusal) as caught:
        read(destination)

    assert caught.value.code == "PIR-E-VERSION-001"
    assert caught.value.detail["entry"] == "ir_version"


def test_link_refuses_dangling_pool_reference():
    ir = compiled_predict()
    manifest = ir.to_manifest()
    manifest["components"]["1_module_tree"]["bindings"]["adapter"] = "missing"

    with pytest.raises(ProgramIRRefusal) as caught:
        link(ProgramIR(manifest=manifest))

    assert caught.value.code == "PIR-E-LINK-002"
    assert caught.value.detail == {
        "predictor": "self",
        "binding": "adapter",
        "pool": "4_adapter",
        "entry": "missing",
    }


def test_read_requires_an_artifact_directory(tmp_path):
    with pytest.raises(ValueError, match="artifact directory"):
        read(tmp_path / "missing.ir")
