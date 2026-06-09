"""Golden parity gate: adapters must reproduce the recorded corpus exactly.

Every case in ``golden/cases.py`` is re-executed against the current adapter
code and compared byte-for-byte against its checked-in fixture. Any diff means
adapter behavior changed. Fixtures may only be regenerated in dedicated
corpus-update commits containing zero ``dspy/`` source changes — see
``golden/README.md`` for the protocol.
"""

import json
import warnings

import pytest
from golden.cases import CASES, case_fixture
from golden.generate_fixtures import GOLDEN_DIR, REQUEST_DIR, _versions, render_fixture
from golden.parse_cases import (
    CALLBACK_CASE_IDS,
    PARSE_CASES,
    callback_case_fixture,
    parse_case_fixture,
)

MISSING_FIXTURE_HINT = (
    "fixture missing — generate the corpus with "
    "`python tests/adapters/golden/generate_fixtures.py` in a dedicated corpus commit"
)


def _python_minor_matches_corpus():
    import platform

    metadata = json.loads((REQUEST_DIR / "_metadata.json").read_text(encoding="utf-8"))
    recorded = metadata["versions"]["python"]
    return recorded.split(".")[:2] == platform.python_version().split(".")[:2]


def _skip_if_python_sensitive(case):
    if getattr(case, "python_sensitive", False) and not _python_minor_matches_corpus():
        pytest.skip(
            "fixture bytes depend on the python minor version (docstring dedent, CPython 3.13+); "
            "the pinned parity CI job is the byte authority for this case"
        )


def _assert_parity(fixture_path, actual_text):
    assert fixture_path.exists(), MISSING_FIXTURE_HINT
    expected_text = fixture_path.read_text(encoding="utf-8")
    # Structured comparison first for a readable diff on failure, then exact
    # byte equality.
    assert json.loads(actual_text) == json.loads(expected_text)
    assert actual_text == expected_text


@pytest.mark.parametrize("case_id", sorted(PARSE_CASES))
def test_parse_parity(case_id):
    _skip_if_python_sensitive(PARSE_CASES[case_id])
    actual = render_fixture(parse_case_fixture(PARSE_CASES[case_id]))
    _assert_parity(GOLDEN_DIR / "parse" / f"{case_id}.json", actual)


@pytest.mark.parametrize("case_id", sorted(CALLBACK_CASE_IDS))
def test_callback_sequence_parity(case_id):
    actual = render_fixture(callback_case_fixture(PARSE_CASES[case_id]))
    _assert_parity(GOLDEN_DIR / "callbacks" / f"{case_id}.json", actual)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_request_parity(case_id):
    _skip_if_python_sensitive(CASES[case_id])
    fixture_path = REQUEST_DIR / f"{case_id}.json"
    assert fixture_path.exists(), MISSING_FIXTURE_HINT

    expected_text = fixture_path.read_text(encoding="utf-8")
    actual_text = render_fixture(case_fixture(CASES[case_id]))

    # Structured comparison first for a readable diff on failure, then exact
    # byte equality (sorted-key rendering makes the two equivalent unless
    # serialization itself drifted, which must also fail).
    assert json.loads(actual_text) == json.loads(expected_text)
    assert actual_text == expected_text


def test_fixture_files_match_registry():
    expected_files = (
        {f"request/{case_id}.json" for case_id in CASES}
        | {f"parse/{case_id}.json" for case_id in PARSE_CASES}
        | {f"callbacks/{case_id}.json" for case_id in CALLBACK_CASE_IDS}
        | {"request/_metadata.json"}
    )
    on_disk = {
        str(path.relative_to(GOLDEN_DIR))
        for corpus in ("request", "parse", "callbacks")
        for path in (GOLDEN_DIR / corpus).glob("*.json")
    }
    assert on_disk == expected_files, (
        "fixture files out of sync with the case registry "
        f"(missing: {sorted(expected_files - on_disk)}, stale: {sorted(on_disk - expected_files)})"
    )


def test_metadata_versions():
    """Warn (not fail) when the floating environment drifts from the corpus.

    Schema-derived strings can legitimately change across pydantic/litellm
    versions; the dedicated pinned parity CI job is the authoritative
    byte-level gate, while the floating test matrix gets a loud signal.
    """
    metadata = json.loads((REQUEST_DIR / "_metadata.json").read_text(encoding="utf-8"))
    current = _versions()
    recorded = metadata["versions"]
    drifted = {
        name: (recorded.get(name), current.get(name)) for name in current if recorded.get(name) != current.get(name)
    }
    if drifted:
        warnings.warn(
            f"golden corpus was recorded under different versions: {drifted}; "
            "byte-level parity is authoritative only in the pinned CI job",
            stacklevel=1,
        )
