import importlib
import subprocess
from pathlib import Path

import pytest

import dspy
from dspy.clients.openai_compat_lm import _OpenAICompatLM
from dspy.programir import ProgramIR


def leaking_tool(query: str) -> str:
    """Return a value that must never survive credential scanning."""
    return "super-secret-value"


class SecretProgram(dspy.Module):
    def __init__(self):
        self.lookup = leaking_tool
        self.answer = dspy.Predict("value -> answer")

    def forward(self, query):
        value = self.lookup(query=query)
        return self.answer(value=value)


def fake_uv_lock(command, **kwargs):
    Path(f"{command[-1]}.lock").write_text("version = 1\n")
    return subprocess.CompletedProcess(command, 0)


def test_dspy_export_is_public():
    assert dspy.export is dspy.programir.export


def test_export_delegates_to_compile_and_write(monkeypatch, tmp_path):
    module = importlib.import_module("dspy.programir.export")
    program = dspy.Predict("question -> answer")
    sentinel = ProgramIR(manifest={"sentinel": True})
    calls = []

    def fake_compile(value, *, metric, devset):
        calls.append(("compile", value, metric, devset))
        return sentinel

    def fake_write(ir, path, *, credential_values):
        calls.append(("write", ir, path, credential_values))
        return sentinel

    monkeypatch.setattr(module, "compile", fake_compile)
    monkeypatch.setattr(module, "write", fake_write)

    result = dspy.export(program, tmp_path / "program.ir", metric="metric", devset=["example"])

    assert result is sentinel
    assert calls[0] == ("compile", program, "metric", ["example"])
    assert calls[1][:3] == ("write", sentinel, tmp_path / "program.ir")


def test_export_scans_live_lm_credentials_across_finalized_bytes(monkeypatch, tmp_path):
    program = SecretProgram()
    program.set_lm(dspy.LM("openai/model", api_key="super-secret-value"))
    write_module = importlib.import_module("dspy.programir.write")
    monkeypatch.setattr(write_module.subprocess, "run", fake_uv_lock)
    destination = tmp_path / "secret.ir"

    with pytest.raises(ValueError, match="LM_API_KEY.*tools/lookup.py"):
        dspy.export(program, destination)

    assert not destination.exists()


def test_export_scans_direct_openai_compatible_credentials(monkeypatch, tmp_path):
    program = SecretProgram()
    program.set_lm(
        _OpenAICompatLM(
            model="org/model",
            base_url="http://localhost:8000/v1",
            api_key="super-secret-value",
        )
    )
    write_module = importlib.import_module("dspy.programir.write")
    monkeypatch.setattr(write_module.subprocess, "run", fake_uv_lock)

    with pytest.raises(ValueError, match="LM_API_KEY.*tools/lookup.py"):
        dspy.export(program, tmp_path / "secret.ir")


def test_export_returns_the_finalized_artifact(monkeypatch, tmp_path):
    program = SecretProgram()
    program.set_lm(dspy.LM("openai/model", api_key="different-live-key"))
    write_module = importlib.import_module("dspy.programir.write")
    monkeypatch.setattr(write_module.subprocess, "run", fake_uv_lock)

    finalized = dspy.export(program, tmp_path / "program.ir")

    assert "env_entry.py.lock" in finalized.sidecars
    assert (tmp_path / "program.ir" / "manifest.json").is_file()
