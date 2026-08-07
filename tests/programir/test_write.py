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
