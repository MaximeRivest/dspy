"""Build deterministic language environment declarations."""

from __future__ import annotations

import json
from typing import Iterable

from dspy.__metadata__ import __version__


def python_environment(dependencies: Iterable[str]) -> tuple[dict, bytes]:
    """Return the Python environment block and its PEP 723 entry bytes."""
    resolved = sorted({dependency for dependency in dependencies if dependency} | {f"dspy=={__version__}"})
    rendered_dependencies = "\n".join(f"#   {json.dumps(dependency)}," for dependency in resolved)
    source = (
        "# /// script\n"
        '# requires-python = ">=3.10,<3.15"\n'
        "# dependencies = [\n"
        f"{rendered_dependencies}\n"
        "# ]\n"
        "# ///\n"
        "\n"
        '"""Materialize the Python environment for this ProgramIR artifact."""\n'
        "\n"
        "import dspy  # noqa: F401\n"
    ).encode()
    block = {
        "requires_python": ">=3.10,<3.15",
        "dependencies": resolved,
        "pep723_entry": "env_entry.py",
        "lock": "env_entry.py.lock",
        "system_deps": [],
    }
    return block, source
