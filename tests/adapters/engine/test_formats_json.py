"""QC-07 gates: JSONAdapter cutover dual-run (request + parse), BAML legacy
routing, shared-machinery regression on chat, and JSON parse fuzzing."""

import json
import random

import pytest
from conftest import skip_if_python_sensitive
from golden.cases import CASES, case_fixture, execute_case
from golden.generate_fixtures import GOLDEN_DIR, REQUEST_DIR, render_fixture
from golden.harness import canonical_error, canonicalize
from golden.parse_cases import PARSE_CASES, parse_case_fixture

import dspy
from dspy.adapters._engine.formats import resolve_format
from dspy.adapters._engine.formats.json import JSONFormat
from dspy.adapters._engine.overrides import resolve_override_verdict
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.json_adapter import JSONAdapter

JSON_REQUEST_CASE_IDS = sorted(case_id for case_id, case in CASES.items() if case.adapter == "json")
JSON_PARSE_CASE_IDS = sorted(case_id for case_id, case in PARSE_CASES.items() if case.adapter == "json")
BAML_REQUEST_CASE_IDS = sorted(case_id for case_id, case in CASES.items() if case.adapter == "baml")
CHAT_REQUEST_CASE_IDS = sorted(case_id for case_id, case in CASES.items() if case.adapter == "chat")


class _ForcedLegacyJSON(JSONAdapter):
    """Pass-through format override routes the whole instance legacy."""

    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


def _forced_legacy_factory(case, recorder):
    return _ForcedLegacyJSON(**case.adapter_kwargs)


def test_json_adapter_is_engine_backed_with_json_format():
    adapter = JSONAdapter()
    assert resolve_override_verdict(adapter).engine_eligible
    assert isinstance(resolve_format(adapter), JSONFormat)
    # ChatAdapter still resolves ChatFormat, not JSONFormat.
    from dspy.adapters._engine.formats.chat import ChatFormat
    from dspy.adapters.chat_adapter import ChatAdapter

    assert type(resolve_format(ChatAdapter())) is ChatFormat


def test_baml_adapter_migration_state_and_stays_unexported():
    """Updated per migration PR: BAMLAdapter became engine-backed in QC-08.
    It must never be exported regardless of migration state."""
    from dspy.adapters._engine.formats.baml import BAMLFormat

    assert resolve_override_verdict(BAMLAdapter()).engine_eligible
    assert isinstance(resolve_format(BAMLAdapter()), BAMLFormat)
    import dspy.adapters

    assert not hasattr(dspy.adapters, "BAMLAdapter")


@pytest.mark.parametrize("case_id", JSON_REQUEST_CASE_IDS)
def test_json_request_dual_run_engine_matches_fixture(case_id):
    skip_if_python_sensitive(CASES[case_id])
    actual = json.loads(render_fixture(case_fixture(CASES[case_id])))
    expected = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", JSON_REQUEST_CASE_IDS)
def test_json_request_dual_run_legacy_matches_fixture(case_id):
    skip_if_python_sensitive(CASES[case_id])
    case = CASES[case_id]
    legacy_expected = canonicalize(execute_case(case, adapter_factory=_forced_legacy_factory))
    fixture = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert legacy_expected == fixture["expected"]


@pytest.mark.parametrize("case_id", JSON_PARSE_CASE_IDS)
def test_json_parse_dual_run_engine_matches_fixture(case_id):
    actual = json.loads(render_fixture(parse_case_fixture(PARSE_CASES[case_id])))
    expected = json.loads((GOLDEN_DIR / "parse" / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", BAML_REQUEST_CASE_IDS)
def test_baml_request_corpus_unchanged(case_id):
    skip_if_python_sensitive(CASES[case_id])
    """BAML must reproduce its pre-engine fixtures exactly (it routed legacy
    until QC-08, and via BAMLFormat since)."""
    actual = json.loads(render_fixture(case_fixture(CASES[case_id])))
    expected = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", CHAT_REQUEST_CASE_IDS)
def test_chat_corpus_unchanged_by_shared_machinery(case_id):
    """JSONFormat subclasses ChatFormat: prove the shared machinery did not
    shift a single chat byte."""
    from golden.generate_fixtures import _python_minor_differs

    if getattr(CASES[case_id], "python_sensitive", False) and _python_minor_differs():
        pytest.skip("python-minor-dependent docstring bytes; the pinned parity CI job is the byte authority")
    actual = json.loads(render_fixture(case_fixture(CASES[case_id])))
    expected = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


# --- JSON parse fuzzing ---------------------------------------------------------

SEED = 0x77AB1
N_CASES = 100

WRAPPERS = [
    lambda text: text,
    lambda text: f"```json\n{text}\n```",
    lambda text: f"Here is the answer:\n{text}\nHope that helps!",
    lambda text: text.replace('"', "'", 3),
    lambda text: text[:-3],
    lambda text: "[" + text + "]",
    lambda text: text + ' {"sneaky": "second object"}',
    lambda text: "not json at all",
    lambda text: "",
]


def _fuzz_case(index):
    rng = random.Random(SEED + index)
    names = rng.sample(["alpha", "beta", "gamma", "delta"], rng.randint(1, 3))
    fields = {name: (str, dspy.OutputField()) for name in names}
    fields["seed_input"] = (str, dspy.InputField())
    signature = dspy.make_signature(fields, None, signature_name="FuzzJSONSignature")

    payload = {
        name: rng.choice(["value", "multi\nline", "", "unicode «π» 🧪", '"quoted"'])
        for name in names
        if rng.random() < 0.9
    }
    completion = rng.choice(WRAPPERS)(json.dumps(payload, indent=rng.choice([None, 2])))
    return signature, completion


@pytest.mark.parametrize("index", range(N_CASES))
def test_json_parse_fuzz_engine_equals_legacy(index):
    signature, completion = _fuzz_case(index)

    def run(adapter):
        try:
            return ("completed", canonicalize(adapter.parse(signature, completion)))
        except Exception as error:
            return (type(error).__name__, canonicalize(canonical_error(error)))

    assert run(JSONAdapter()) == run(_ForcedLegacyJSON()), f"json parse fuzz divergence at index {index}"
