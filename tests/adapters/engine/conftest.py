import sys
from pathlib import Path

# Make the golden corpus package (tests/adapters/golden) importable from the
# engine test directory, mirroring pytest's rootdir insertion for
# tests/adapters itself.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def skip_if_python_sensitive(case):
    """Skip fixture-byte assertions for python-minor-dependent cases.

    Mirrors test_golden_parity's guard: some fixtures embed bytes that vary
    with the CPython minor (docstring dedent, union JSON schemas); the
    pinned parity CI job is the byte authority for those.
    """
    import pytest
    from golden.generate_fixtures import _python_minor_differs

    if getattr(case, "python_sensitive", False) and _python_minor_differs():
        pytest.skip("python-minor-dependent fixture bytes; the pinned parity CI job is the byte authority")
