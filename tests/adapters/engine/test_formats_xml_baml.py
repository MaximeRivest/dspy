"""QC-08 gates: XML and BAML cutover dual-runs, the renderer-purity proof
(two new wire formats with zero render.py changes), and XML parse fuzzing."""

import json
import random

import pytest
from golden.cases import CASES, case_fixture, execute_case
from golden.generate_fixtures import GOLDEN_DIR, REQUEST_DIR, render_fixture
from golden.harness import canonical_error, canonicalize
from golden.parse_cases import PARSE_CASES, parse_case_fixture

import dspy
from dspy.adapters._engine.formats import resolve_format
from dspy.adapters._engine.formats.baml import BAMLFormat
from dspy.adapters._engine.formats.xml import XMLFormat
from dspy.adapters._engine.overrides import resolve_override_verdict
from dspy.adapters.baml_adapter import BAMLAdapter
from dspy.adapters.xml_adapter import XMLAdapter

XML_REQUEST_CASE_IDS = sorted(case_id for case_id, case in CASES.items() if case.adapter == "xml")
BAML_REQUEST_CASE_IDS = sorted(case_id for case_id, case in CASES.items() if case.adapter == "baml")
XML_PARSE_CASE_IDS = sorted(case_id for case_id, case in PARSE_CASES.items() if case.adapter == "xml")


class _ForcedLegacyXML(XMLAdapter):
    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


class _ForcedLegacyBAML(BAMLAdapter):
    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


def test_xml_and_baml_are_engine_backed_with_their_formats():
    assert resolve_override_verdict(XMLAdapter()).engine_eligible
    assert isinstance(resolve_format(XMLAdapter()), XMLFormat)
    assert resolve_override_verdict(BAMLAdapter()).engine_eligible
    assert isinstance(resolve_format(BAMLAdapter()), BAMLFormat)
    assert not resolve_override_verdict(_ForcedLegacyXML()).engine_eligible
    assert not resolve_override_verdict(_ForcedLegacyBAML()).engine_eligible


@pytest.mark.parametrize("case_id", XML_REQUEST_CASE_IDS)
def test_xml_request_dual_run_engine_matches_fixture(case_id):
    actual = json.loads(render_fixture(case_fixture(CASES[case_id])))
    expected = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", XML_REQUEST_CASE_IDS)
def test_xml_request_dual_run_legacy_matches_fixture(case_id):
    case = CASES[case_id]
    legacy = canonicalize(execute_case(case, adapter_factory=lambda c, r: _ForcedLegacyXML(**c.adapter_kwargs)))
    fixture = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert legacy == fixture["expected"]


@pytest.mark.parametrize("case_id", BAML_REQUEST_CASE_IDS)
def test_baml_request_dual_run_legacy_matches_fixture(case_id):
    case = CASES[case_id]
    legacy = canonicalize(execute_case(case, adapter_factory=lambda c, r: _ForcedLegacyBAML(**c.adapter_kwargs)))
    fixture = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert legacy == fixture["expected"]


@pytest.mark.parametrize("case_id", XML_PARSE_CASE_IDS)
def test_xml_parse_dual_run_engine_matches_fixture(case_id):
    actual = json.loads(render_fixture(parse_case_fixture(PARSE_CASES[case_id])))
    expected = json.loads((GOLDEN_DIR / "parse" / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


# --- XML parse fuzzing -----------------------------------------------------------

SEED = 0x31A07
N_CASES = 80

MUTATIONS = [
    lambda text: text,
    lambda text: f"Some prose before.\n{text}\nAnd after.",
    lambda text: text + "\n" + text,  # duplicates: first match wins
    lambda text: text.replace("</", "</ ", 1),  # broken closing tag
    lambda text: text[: len(text) // 2],
    lambda text: f"<wrapper>{text}</wrapper>",  # nesting: known failure mode, pinned
    lambda text: text.replace(">", ">\n  ", 2),
    lambda text: "no tags at all",
]


def _fuzz_case(index):
    rng = random.Random(SEED + index)
    names = rng.sample(["alpha", "beta", "gamma"], rng.randint(1, 3))
    fields = {name: (str, dspy.OutputField()) for name in names}
    fields["seed_input"] = (str, dspy.InputField())
    signature = dspy.make_signature(fields, None, signature_name="FuzzXMLSignature")

    blocks = [
        f"<{name}>\n{rng.choice(['value', 'multi' + chr(10) + 'line', '', 'a < b > c'])}\n</{name}>" for name in names
    ]
    rng.shuffle(blocks)
    completion = MUTATIONS[rng.randrange(len(MUTATIONS))]("\n\n".join(blocks))
    return signature, completion


@pytest.mark.parametrize("index", range(N_CASES))
def test_xml_parse_fuzz_engine_equals_legacy(index):
    signature, completion = _fuzz_case(index)

    def run(adapter):
        try:
            return ("completed", canonicalize(adapter.parse(signature, completion)))
        except Exception as error:
            return (type(error).__name__, canonicalize(canonical_error(error)))

    assert run(XMLAdapter()) == run(_ForcedLegacyXML()), f"xml parse fuzz divergence at index {index}"
