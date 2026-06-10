"""Generate (or check) the golden parity fixtures.

Usage, from the repository root:

    python tests/adapters/golden/generate_fixtures.py            # write fixtures
    python tests/adapters/golden/generate_fixtures.py --check    # verify zero drift

Three corpora are generated: ``request/`` (adapter-to-LM payloads and
intermediate render surfaces), ``parse/`` (parsed values, exception
semantics, fallback chains), and ``callbacks/`` (ordered adapter callback
event streams).

Fixture regeneration is governed by the rebase protocol in README.md:
regeneration is only allowed in dedicated corpus-update commits containing
zero ``dspy/`` source changes, so any fixture diff in a feature PR is, by
construction, a behavior change.
"""

import argparse
import json
import platform
import sys
from pathlib import Path

GOLDEN_DIR = Path(__file__).resolve().parent
REQUEST_DIR = GOLDEN_DIR / "request"

# Allow running as a plain script: make the `golden` package importable the
# same way pytest's rootdir insertion does for tests/adapters.
sys.path.insert(0, str(GOLDEN_DIR.parent))

from golden.cases import CASES, case_fixture  # noqa: E402
from golden.parse_cases import (  # noqa: E402
    CALLBACK_CASE_IDS,
    PARSE_CASES,
    callback_case_fixture,
    parse_case_fixture,
)


def _versions() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    def _installed(package: str) -> str:
        try:
            return version(package)
        except PackageNotFoundError:
            return "unknown"

    return {
        "python": platform.python_version(),
        "dspy": _installed("dspy"),
        "pydantic": _installed("pydantic"),
        "litellm": _installed("litellm"),
        "json_repair": _installed("json_repair"),
    }


def render_fixture(document: dict) -> str:
    return json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def generate() -> dict[str, str]:
    """Render every fixture document to its serialized form, keyed by relative path."""
    rendered = {}
    for case_id in sorted(CASES):
        rendered[f"request/{case_id}.json"] = render_fixture(case_fixture(CASES[case_id]))
    for case_id in sorted(PARSE_CASES):
        rendered[f"parse/{case_id}.json"] = render_fixture(parse_case_fixture(PARSE_CASES[case_id]))
    for case_id in sorted(CALLBACK_CASE_IDS):
        rendered[f"callbacks/{case_id}.json"] = render_fixture(callback_case_fixture(PARSE_CASES[case_id]))
    rendered["request/_metadata.json"] = render_fixture(
        {
            "corpora": {
                "request": len(CASES),
                "parse": len(PARSE_CASES),
                "callbacks": len(CALLBACK_CASE_IDS),
            },
            "versions": _versions(),
        }
    )
    return rendered


def write(rendered: dict[str, str]) -> None:
    for corpus in ("request", "parse", "callbacks"):
        (GOLDEN_DIR / corpus).mkdir(exist_ok=True)
    on_disk = {
        str(path.relative_to(GOLDEN_DIR))
        for corpus in ("request", "parse", "callbacks")
        for path in (GOLDEN_DIR / corpus).glob("*.json")
    }
    for name in sorted(on_disk - set(rendered)):
        (GOLDEN_DIR / name).unlink()
        print(f"removed stale fixture: {name}")
    for name, content in sorted(rendered.items()):
        (GOLDEN_DIR / name).write_text(content, encoding="utf-8")
    print(f"wrote {len(rendered)} fixture files under {GOLDEN_DIR}")


def check(rendered: dict[str, str]) -> int:
    on_disk = {
        str(path.relative_to(GOLDEN_DIR)): path.read_text(encoding="utf-8")
        for corpus in ("request", "parse", "callbacks")
        for path in (GOLDEN_DIR / corpus).glob("*.json")
    }
    drifted = sorted(
        set(on_disk) ^ set(rendered)
        | {name for name in set(on_disk) & set(rendered) if on_disk[name] != rendered[name]}
    )
    # Metadata records environment provenance (library AND python versions),
    # so it legitimately differs on other machines. Python-sensitive cases
    # (class docstrings embedded in prompts dedent differently on CPython
    # 3.13+) only byte-reproduce on the corpus python minor. Behavior
    # fixtures must always reproduce.
    skippable = _python_sensitive_fixture_names() if _python_minor_differs() else set()

    def _is_sensitive(name):
        return name.rsplit("/", 1)[-1] in skippable

    metadata_drift = [name for name in drifted if name.endswith("_metadata.json")]
    sensitive_drift = [name for name in drifted if _is_sensitive(name)]
    behavior_drift = [name for name in drifted if not _is_sensitive(name) and not name.endswith("_metadata.json")]
    if behavior_drift:
        print("fixture drift detected (regeneration does not reproduce disk state):")
        for name in behavior_drift:
            print(f"  {name}")
        return 1
    if metadata_drift or sensitive_drift:
        print(
            f"warning: skipped {len(metadata_drift)} metadata + {len(sensitive_drift)} python-sensitive "
            "fixtures (environment provenance / docstring dedent differs from the corpus python; "
            "the pinned CI job is their byte authority)"
        )
    print(f"all {len(rendered) - len(metadata_drift) - len(sensitive_drift)} behavior fixtures reproduce exactly")
    return 0


def _python_minor_differs() -> bool:
    metadata_path = REQUEST_DIR / "_metadata.json"
    if not metadata_path.exists():
        return False
    recorded = json.loads(metadata_path.read_text(encoding="utf-8"))["versions"]["python"]
    return recorded.split(".")[:2] != platform.python_version().split(".")[:2]


def _python_sensitive_fixture_names() -> set[str]:
    """Basenames of fixtures whose bytes depend on the python minor (case
    ids are unique across corpora, so basename matching is unambiguous)."""
    from golden.cases import CASES

    names = {f"{case_id}.json" for case_id, case in CASES.items() if case.python_sensitive}
    try:
        from golden.parse_cases import PARSE_CASES

        names |= {
            f"{case_id}.json" for case_id, case in PARSE_CASES.items() if getattr(case, "python_sensitive", False)
        }
    except ImportError:
        pass
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify fixtures reproduce exactly")
    args = parser.parse_args()

    rendered = generate()
    # Determinism gate: a second in-process generation must be byte-identical.
    second = generate()
    if rendered != second:
        unstable = sorted(name for name in rendered if rendered.get(name) != second.get(name))
        print("nondeterministic fixture generation detected:")
        for name in unstable:
            print(f"  {name}")
        return 2

    if args.check:
        return check(rendered)
    write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
