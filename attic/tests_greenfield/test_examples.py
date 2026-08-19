"""Stage A6 tests: every demo script runs clean, end to end.

The six scripts in `examples_greenfield/` are the morning tour — all
DummyLM, zero network. Each must exit 0 when run exactly as the README
says to run it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples_greenfield"

EXPECTED = [
    "01_hello.py",
    "02_chain_of_thought.py",
    "03_react_tools.py",
    "04_adapters_three_ways.py",
    "05_save_ship_load.py",
    "06_optimize.py",
]


def test_the_tour_is_exactly_six_scripts():
    assert sorted(path.name for path in EXAMPLES.glob("*.py")) == EXPECTED


@pytest.mark.parametrize("name", EXPECTED)
def test_example_runs_clean(name):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=300,
    )
    assert result.returncode == 0, f"{name} failed:\n--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    assert result.stdout.strip(), f"{name} printed nothing"
