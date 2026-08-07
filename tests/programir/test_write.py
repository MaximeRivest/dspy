import importlib
import subprocess
from pathlib import Path

import pytest

from dspy.programir import ProgramIR, write
from dspy.programir.write import canonical_json_bytes


def test_canonical_json_sorts_keys_and_preserves_number_types():
    value = {"z": 1, "a": [1.0, True, "é"]}

    assert canonical_json_bytes(value) == '{"a":[1.0,true,"é"],"z":1}\n'.encode()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.0])
def test_canonical_json_refuses_unsupported_floats(value):
    with pytest.raises(ValueError):
        canonical_json_bytes(value)


def test_write_materializes_manifest_and_sidecars(tmp_path):
    destination = tmp_path / "program.ir"
    ir = ProgramIR(manifest={"versions": {}, "components": {}}, sidecars={"tools/lookup.py": b"pass\n"})

    write(ir, destination)

    assert (destination / "manifest.json").read_bytes() == b'{"components":{},"versions":{}}\n'
    assert (destination / "tools" / "lookup.py").read_bytes() == b"pass\n"


def test_write_scans_every_emitted_file_for_credentials(tmp_path):
    destination = tmp_path / "program.ir"
    ir = ProgramIR(
        manifest={"versions": {}, "components": {}},
        sidecars={"tools/lookup.py": b'TOKEN = "secret-value"\n'},
    )

    with pytest.raises(ValueError, match="API_KEY.*tools/lookup.py"):
        write(ir, destination, credential_values={"API_KEY": "secret-value"})

    assert not destination.exists()


def test_write_materializes_declared_python_lock(tmp_path, monkeypatch):
    destination = tmp_path / "program.ir"
    manifest = {
        "versions": {},
        "components": {
            "9_environment": {
                "python": {
                    "pep723_entry": "env_entry.py",
                    "lock": "env_entry.py.lock",
                }
            }
        },
    }
    ir = ProgramIR(manifest=manifest, sidecars={"env_entry.py": b"# /// script\n# ///\n"})

    def fake_uv_lock(command, **kwargs):
        assert command[:3] == ["uv", "lock", "--script"]
        Path(f"{command[-1]}.lock").write_bytes(b"revision = 1\n")
        return subprocess.CompletedProcess(command, 0)

    write_module = importlib.import_module("dspy.programir.write")
    monkeypatch.setattr(write_module.subprocess, "run", fake_uv_lock)

    finalized = write(ir, destination)

    assert finalized.sidecars["env_entry.py.lock"] == b"revision = 1\n"
    assert (destination / "env_entry.py.lock").read_bytes() == b"revision = 1\n"


def test_write_refuses_lock_failure_atomically(tmp_path, monkeypatch):
    destination = tmp_path / "program.ir"
    manifest = {
        "components": {
            "9_environment": {
                "python": {"pep723_entry": "env_entry.py", "lock": "env_entry.py.lock"}
            }
        }
    }
    ir = ProgramIR(manifest=manifest, sidecars={"env_entry.py": b"# /// script\n# ///\n"})
    write_module = importlib.import_module("dspy.programir.write")
    monkeypatch.setattr(
        write_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1),
    )

    with pytest.raises(ValueError, match="failed to produce"):
        write(ir, destination)

    assert not destination.exists()


def test_write_refuses_unsafe_sidecar_paths(tmp_path):
    ir = ProgramIR(manifest={}, sidecars={"../outside": b"bad"})

    with pytest.raises(ValueError, match="safe relative"):
        write(ir, tmp_path / "program.ir")


def test_write_refuses_existing_destination(tmp_path):
    destination = tmp_path / "program.ir"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        write(ProgramIR(manifest={}), destination)


def test_writer_has_no_implicit_clock_or_environment_reads():
    source = Path("dspy/programir/write.py").read_text()

    assert "datetime" not in source
    assert "SOURCE_DATE_EPOCH" not in source
    assert "os.environ" not in source
