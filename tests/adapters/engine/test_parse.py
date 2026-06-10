"""QC-06 gates: parse cutover dual-run, the format/parse coupling invariant,
fallback-adapter routing, and parse round-trip fuzzing."""

import json
import random

import pytest
from golden.cases import CASES
from golden.generate_fixtures import GOLDEN_DIR, render_fixture
from golden.harness import Recorder, StubLM, canonical_error, canonicalize
from golden.parse_cases import PARSE_CASES, parse_case_fixture

import dspy
from dspy.adapters._engine.builder import build_plan
from dspy.adapters._engine.formats import resolve_format
from dspy.adapters._engine.parse import FormatParserHook
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.utils.exceptions import AdapterParseError

CHAT_PARSE_CASE_IDS = sorted(case_id for case_id, case in PARSE_CASES.items() if case.adapter == "chat")


def _skip_if_python_sensitive(case):
    from golden.generate_fixtures import _python_minor_differs

    if getattr(case, "python_sensitive", False) and _python_minor_differs():
        pytest.skip("python-minor-dependent docstring bytes; the pinned parity CI job is the byte authority")


class _ForcedLegacyChat(ChatAdapter):
    """Routes the whole instance through the legacy pipeline (QC-05 pattern)."""

    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


@pytest.mark.parametrize("case_id", CHAT_PARSE_CASE_IDS)
def test_parse_dual_run_engine_matches_fixture(case_id):
    _skip_if_python_sensitive(PARSE_CASES[case_id])
    actual = json.loads(render_fixture(parse_case_fixture(PARSE_CASES[case_id])))
    expected = json.loads((GOLDEN_DIR / "parse" / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", CHAT_PARSE_CASE_IDS)
def test_parse_dual_run_legacy_matches_fixture(case_id):
    """Forced-legacy parse runners against the same pre-engine fixtures."""
    _skip_if_python_sensitive(PARSE_CASES[case_id])
    case = PARSE_CASES[case_id]
    if "parse" not in case.runners:
        pytest.skip("case has no direct-parse runner")
    payload = case.payload()
    fixture = json.loads((GOLDEN_DIR / "parse" / f"{case_id}.json").read_text(encoding="utf-8"))
    adapter = _ForcedLegacyChat(**case.adapter_kwargs)
    try:
        values = adapter.parse(payload["signature"], case.parse_text)
        actual = {"outcome": "completed", "values": canonicalize(values)}
    except Exception as error:
        actual = {"outcome": f"raised:{type(error).__name__}", "error": canonical_error(error)}
    assert actual == fixture["expected"]["parse"]


def test_format_and_parse_share_one_format_instance():
    """The coupling invariant: the demo/assistant renderer and the parser
    must derive from the SAME Format object, and the plan records it."""
    adapter = ChatAdapter()
    fmt = resolve_format(adapter)
    assert fmt is resolve_format(adapter)

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    built = build_plan(adapter, StubLM(Recorder()), {}, QA, {"question": "Q?"})
    hooks = [parser for parser in built.plan.parsers if isinstance(parser, FormatParserHook)]
    assert len(hooks) == 1
    assert hooks[0].format is fmt


def test_legacy_routed_plan_records_no_format_parser():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    built = build_plan(_ForcedLegacyChat(), StubLM(Recorder()), {}, QA, {"question": "Q?"})
    assert not any(isinstance(parser, FormatParserHook) for parser in built.plan.parsers)


def test_fallback_json_adapter_still_routes_legacy():
    """The fresh JSONAdapter constructed inside ChatAdapter's fallback is an
    unmigrated class: it must keep using the legacy pipeline end-to-end."""
    from dspy.adapters._engine.overrides import resolve_override_verdict

    adapter = ChatAdapter()._make_json_adapter_fallback()
    assert not resolve_override_verdict(adapter).engine_eligible


def test_parse_error_identity_preserved_for_subclasses():
    """Legacy parse errors hardcode adapter_name='ChatAdapter'; the engine
    path must reproduce that even for engine-eligible no-op subclasses."""

    class NoOp(ChatAdapter):
        pass

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    with pytest.raises(AdapterParseError) as engine_error:
        NoOp().parse(QA, "garbage")
    with pytest.raises(AdapterParseError) as legacy_error:
        _ForcedLegacyChat().parse(QA, "garbage")
    assert str(engine_error.value) == str(legacy_error.value)


# --- parse round-trip fuzzing -------------------------------------------------

SEED = 0x9A45E
N_CASES = 100

GARBAGE_MUTATIONS = [
    lambda text: text,
    lambda text: text.replace("[[ ## completed ## ]]", ""),
    lambda text: "Sure! Here you go:\n\n" + text,
    lambda text: text + "\n\ntrailing commentary",
    lambda text: "   " + text.replace("\n[[", "\n   [["),  # leading whitespace on headers
    lambda text: text.replace("## ]]", "## ]] same-line content", 1),
    lambda text: text + "\n" + text,  # duplicated: first-occurrence-wins
    lambda text: text.upper(),
    lambda text: text[: len(text) // 2],
]


def _fuzz_completion(index):
    rng = random.Random(SEED + index)
    field_names = rng.sample(["alpha", "beta", "gamma", "delta"], rng.randint(1, 3))
    fields = {name: (str, dspy.OutputField()) for name in field_names}
    fields["seed_input"] = (str, dspy.InputField())
    signature = dspy.make_signature(fields, None, signature_name="FuzzParseSignature")

    values = ["value", "multi\nline", "", "  spaced  "]
    blocks = [f"[[ ## {name} ## ]]\n{rng.choice(values)}" for name in field_names]
    rng.shuffle(blocks)
    completion = "\n\n".join(blocks) + "\n\n[[ ## completed ## ]]"
    completion = rng.choice(GARBAGE_MUTATIONS)(completion)
    return signature, completion


@pytest.mark.parametrize("index", range(N_CASES))
def test_parse_fuzz_engine_equals_legacy(index):
    signature, completion = _fuzz_completion(index)

    def run(adapter):
        try:
            return ("completed", canonicalize(adapter.parse(signature, completion)))
        except Exception as error:
            return (type(error).__name__, canonicalize(canonical_error(error)))

    assert run(ChatAdapter()) == run(_ForcedLegacyChat()), f"parse fuzz divergence at index {index}"


def test_request_corpus_unchanged_spot_check():
    """Cheap guard inside this module: a representative request fixture still
    reproduces (the full corpus runs in the main parity suite)."""
    from golden.cases import case_fixture

    case_id = "base--chat--qa-complete-demos"
    actual = json.loads(render_fixture(case_fixture(CASES[case_id])))
    expected = json.loads((GOLDEN_DIR / "request" / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected
