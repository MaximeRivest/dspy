"""Prepare Python environments for FlexIR candidates.

Two rungs of engine-owned environment preparation live here:

- Rung 2, install-on-admit (`ensure_deps`): after admission accepts a
  dep-carrying authored leaf and BEFORE the candidate is evaluated,
  install any missing granted package into the CURRENT interpreter's
  environment via `uv pip install`. An install failure is a teaching
  refusal string (the caller feeds it to the ledger), never a crash —
  a bad package name is the proposal's fault, not infrastructure.
- Rung 3, artifact environments (`EnvCache`): build (and cache, keyed by
  the artifact's lockfile hash) a virtualenv per distinct dependency set
  so a candidate can be SCORED under its own artifact's environment.
  Only candidates that change deps pay a resolve.

The `uv` subprocess here is OPTIMIZER machinery, not generated code: the
arguments are package names that already passed the `allowed_deps` gate.
`_run_uv` is the seam tests monkeypatch to fake uv outcomes.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

__all__ = ["EnvCache", "ensure_deps", "missing_deps"]

#: The repository root of the RUNNING dspy — the default source for the
#: `dspy` override when building an artifact environment (the manifest
#: pins `dspy==<greenfield version>`, which is not the PyPI dspy; the
#: child env must get THIS tree instead).
REPO_ROOT = Path(__file__).resolve().parents[2]

_NAME_SPLIT = re.compile(r"[<>=!~\[;\s]")


def _run_uv(args: list[str]) -> subprocess.CompletedProcess:
    """Run one `uv` command (THE seam; tests monkeypatch this)."""
    return subprocess.run(["uv", *args], capture_output=True, text=True)


def _base_name(dep: str) -> str:
    """The distribution name of one dep spec ("pkg>=1" -> "pkg")."""
    return _NAME_SPLIT.split(dep, 1)[0].strip()


def missing_deps(deps: list[str]) -> list[str]:
    """The subset of `deps` whose distribution is not installed here."""
    import importlib.metadata as metadata

    absent = []
    for dep in deps:
        try:
            metadata.distribution(_base_name(dep))
        except metadata.PackageNotFoundError:
            absent.append(dep)
    return absent


def ensure_deps(deps: list[str]) -> str | None:
    """Install missing deps into THIS interpreter's env; refuse teaching.

    Returns None when every dep is (or becomes) installed, else the
    teaching refusal text the caller records in the ledger. The install
    is `uv pip install --python sys.executable <missing...>` — the
    current environment, deliberately: this is rung 2, the user opted in
    with `FlexIR(auto_install=True)`. Installs are NOT unwound when the
    candidate is later rejected; the env drifts from the lock until the
    next export re-locks.
    """
    missing = missing_deps(deps)
    if not missing:
        return None
    result = _run_uv(["pip", "install", "--python", sys.executable, *missing])
    if result.returncode != 0:
        tail = " | ".join((result.stderr or "").strip().splitlines()[-4:])
        return (
            f"auto_install could not install dep(s) {missing}: `uv pip install` exited "
            f"{result.returncode}. uv said: {tail}"
        )
    still = missing_deps(missing)
    if still:
        return (
            f"auto_install ran `uv pip install` but dep(s) {still} are still not installed; "
            "check that each # deps: name is a real distribution name"
        )
    return None


class EnvCache:
    """Virtualenvs for artifact-mode scoring, keyed by lockfile hash.

    One environment per distinct dependency lock: `interpreter_for` hashes
    the artifact's `env_entry.py.lock` (falling back to `env_entry.py`)
    and reuses a cached interpreter when the hash was seen — in memory
    first, then on disk under `cache_dir/<hash>/`. Only a candidate that
    CHANGES deps pays a resolve; `provisions` counts real builds so tests
    can pin the reuse.

    `same_env=True` is the TEST-ONLY escape hatch: instead of building a
    uv venv it returns the running interpreter, so the export/harness/
    protocol/caching logic is exercised without creating environments.
    It is never set by user-facing code paths.

    `overrides` maps distribution names to local paths installed editable
    in place of the locked release (default: `{"dspy": REPO_ROOT}` — the
    manifest's dspy pin is not the PyPI dspy).
    """

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        overrides: dict[str, str | Path] | None = None,
        same_env: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.overrides = {name: Path(path) for name, path in (overrides or {"dspy": REPO_ROOT}).items()}
        self.same_env = same_env
        self.provisions = 0
        self._interpreters: dict[str, str] = {}

    def interpreter_for(self, artifact_dir: str | Path) -> str:
        """The python executable for this artifact's environment."""
        artifact_dir = Path(artifact_dir)
        key = self._lock_hash(artifact_dir)
        cached = self._interpreters.get(key)
        if cached is not None:
            return cached
        env_python = self.cache_dir / key / "bin" / "python"
        if env_python.exists():
            self._interpreters[key] = str(env_python)
            return str(env_python)
        self.provisions += 1
        if self.same_env:
            self._interpreters[key] = sys.executable
            return sys.executable
        interpreter = self._provision(self.cache_dir / key, artifact_dir)
        self._interpreters[key] = interpreter
        return interpreter

    def _lock_hash(self, artifact_dir: Path) -> str:
        lock = artifact_dir / "env_entry.py.lock"
        source = lock if lock.is_file() else artifact_dir / "env_entry.py"
        return hashlib.sha256(source.read_bytes()).hexdigest()[:16]

    def _provision(self, env_dir: Path, artifact_dir: Path) -> str:
        """Build one venv: locked deps minus overrides, plus editables.

        A failure here is INFRASTRUCTURE (the artifact's environment
        could not be built), so it raises — it is never scored against
        the candidate.
        """
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        created = _run_uv(["venv", str(env_dir)])
        if created.returncode != 0:
            raise RuntimeError(f"FlexIR could not create the candidate environment: {created.stderr.strip()}")
        python = env_dir / "bin" / "python"
        deps = self._locked_deps(artifact_dir)
        arguments = ["pip", "install", "--python", str(python), *deps]
        for path in self.overrides.values():
            arguments += ["--editable", str(path)]
        installed = _run_uv(arguments)
        if installed.returncode != 0:
            raise RuntimeError(
                "FlexIR could not install the candidate environment: "
                + " | ".join((installed.stderr or "").strip().splitlines()[-4:])
            )
        return str(python)

    def _locked_deps(self, artifact_dir: Path) -> list[str]:
        manifest = json.loads((artifact_dir / "manifest.json").read_text())
        declared = manifest["components"]["9_environment"]["python"]["dependencies"]
        return [dep for dep in declared if _base_name(dep) not in self.overrides]
