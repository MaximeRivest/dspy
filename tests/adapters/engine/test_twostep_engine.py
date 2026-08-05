"""QC-09 gates: the last adapter migration. TwoStep dual-runs (request +
parse incl. the tagged legacy-async-quirk fixtures), the dedicated-assembly
proof (render.py untouched by a structurally different pipeline), and the
extraction parser hook."""

import json

import pytest
from conftest import skip_if_python_sensitive
from golden.cases import CASES, case_fixture, execute_case
from golden.generate_fixtures import GOLDEN_DIR, REQUEST_DIR, render_fixture
from golden.harness import Recorder, StubLM, canonicalize
from golden.parse_cases import PARSE_CASES, parse_case_fixture

import dspy
from dspy.adapters._engine.builder import build_plan
from dspy.adapters._engine.formats import resolve_format
from dspy.adapters._engine.formats.twostep import TwoStepExtractionParserHook, TwoStepFormat
from dspy.adapters._engine.overrides import resolve_override_verdict
from dspy.adapters.two_step_adapter import TwoStepAdapter

TWOSTEP_REQUEST_CASE_IDS = sorted(case_id for case_id, case in CASES.items() if case.adapter == "two_step")
TWOSTEP_PARSE_CASE_IDS = sorted(case_id for case_id, case in PARSE_CASES.items() if case.adapter == "two_step")
QUIRK_PARSE_CASE_IDS = sorted(case_id for case_id, case in PARSE_CASES.items() if "legacy-async-quirk" in case.tags)


class _ForcedLegacyTwoStep(TwoStepAdapter):
    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


def _stub():
    return StubLM(Recorder())


def test_two_step_is_engine_backed_with_its_format():
    adapter = TwoStepAdapter(_stub())
    assert resolve_override_verdict(adapter).engine_eligible
    assert isinstance(resolve_format(adapter), TwoStepFormat)
    assert not resolve_override_verdict(_ForcedLegacyTwoStep(_stub())).engine_eligible


@pytest.mark.parametrize("case_id", TWOSTEP_REQUEST_CASE_IDS)
def test_two_step_request_dual_run_engine_matches_fixture(case_id):
    skip_if_python_sensitive(CASES[case_id])
    actual = json.loads(render_fixture(case_fixture(CASES[case_id])))
    expected = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", TWOSTEP_REQUEST_CASE_IDS)
def test_two_step_request_dual_run_legacy_matches_fixture(case_id):
    skip_if_python_sensitive(CASES[case_id])
    case = CASES[case_id]

    def factory(c, recorder):
        extraction = StubLM(recorder, name="extraction", **c.extraction_lm)
        return _ForcedLegacyTwoStep(extraction, **c.adapter_kwargs)

    legacy = canonicalize(execute_case(case, adapter_factory=factory))
    fixture = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert legacy == fixture["expected"]


@pytest.mark.parametrize("case_id", TWOSTEP_PARSE_CASE_IDS)
def test_two_step_parse_dual_run_engine_matches_fixture(case_id):
    actual = json.loads(render_fixture(parse_case_fixture(PARSE_CASES[case_id])))
    expected = json.loads((GOLDEN_DIR / "parse" / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", QUIRK_PARSE_CASE_IDS)
def test_legacy_async_quirks_reproduce_exactly(case_id):
    """Each tagged quirk fixture (extraction-on-empty-text, id-less tool
    calls, str(output) error payloads, tolerance differences) must reproduce
    with ZERO regeneration. Unification is the tracked follow-up — see
    TwoStepAdapter._legacy_async_quirks_postprocess's docstring (the kill
    list for the legacy-retirement epic)."""
    actual = json.loads(render_fixture(parse_case_fixture(PARSE_CASES[case_id])))
    expected = json.loads((GOLDEN_DIR / "parse" / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


def test_async_extraction_stays_awaited():
    """The fifth, corpus-invisible quirk: in the async path the extraction
    model must be called via acall, never the sync path (which would block
    the event loop in real programs)."""
    import asyncio

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    calls = {"sync": 0, "async": 0}

    class ProbeLM(StubLM):
        def __call__(self, prompt=None, messages=None, **kwargs):
            calls["sync"] += 1
            return ["[[ ## answer ## ]]\nBerlin\n\n[[ ## completed ## ]]"]

        async def acall(self, prompt=None, messages=None, **kwargs):
            calls["async"] += 1
            return ["[[ ## answer ## ]]\nBerlin\n\n[[ ## completed ## ]]"]

    main = ProbeLM(Recorder(), name="main")
    extraction = ProbeLM(Recorder(), name="extraction")
    adapter = TwoStepAdapter(extraction)
    values = asyncio.run(adapter.acall(main, {}, QA, [], {"question": "Q?"}))
    assert values == [{"answer": "Berlin"}]
    assert calls["sync"] == 0 and calls["async"] == 2


def test_extraction_parser_hook_recorded_on_plans():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    extraction = StubLM(Recorder(), name="extraction")
    adapter = TwoStepAdapter(extraction)
    built = build_plan(adapter, StubLM(Recorder()), {}, QA, {"question": "Q?"})
    hooks = [parser for parser in built.plan.parsers if isinstance(parser, TwoStepExtractionParserHook)]
    assert len(hooks) == 1
    assert hooks[0].extraction_model is extraction


def test_render_module_untouched_by_two_step():
    """TwoStep's structurally different assembly must live in
    formats/twostep.py; render.py keeps its base-pipeline purity."""
    import inspect as inspect_module

    import dspy.adapters._engine.render as render_module

    source = inspect_module.getsource(render_module)
    assert "two_step" not in source and "TwoStep" not in source
