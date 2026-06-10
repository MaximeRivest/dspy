"""The QC-05 cutover gates: dual-run parity, override routing end-to-end,
and renderer purity.

The dual-run harness executes every ChatAdapter golden case on BOTH paths:
the engine path (plain ChatAdapter, now engine-backed) and the legacy path
(a forced-legacy subclass whose pass-through ``format`` override routes it
through the byte-untouched legacy body). Both must equal the checked-in
fixture, which was recorded from pre-engine code.
"""

import inspect as inspect_module
import json

import pytest
from golden.cases import CASES, case_fixture, execute_case
from golden.generate_fixtures import REQUEST_DIR, render_fixture
from golden.harness import Recorder, StubLM, canonicalize

import dspy.adapters._engine.render as render_module
from dspy.adapters._engine.formats import resolve_format
from dspy.adapters._engine.overrides import resolve_override_verdict
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.two_step_adapter import TwoStepAdapter

CHAT_CASE_IDS = sorted(case_id for case_id, case in CASES.items() if case.adapter == "chat")


def _skip_if_python_sensitive(case):
    from golden.generate_fixtures import _python_minor_differs

    if getattr(case, "python_sensitive", False) and _python_minor_differs():
        pytest.skip("python-minor-dependent docstring bytes; the pinned parity CI job is the byte authority")


class _ForcedLegacyChat(ChatAdapter):
    """Pass-through format override: detection routes the WHOLE instance
    through the legacy pipeline, and super().format() re-checks the verdict
    (still legacy) so the legacy body executes."""

    def format(self, signature, demos, inputs):
        return super().format(signature, demos, inputs)


def _forced_legacy_factory(case, recorder):
    assert case.adapter == "chat"
    return _ForcedLegacyChat(**case.adapter_kwargs)


def test_chat_adapter_is_engine_backed_and_format_resolves():
    adapter = ChatAdapter()
    assert resolve_override_verdict(adapter).engine_eligible
    assert resolve_format(adapter) is not None
    assert not resolve_override_verdict(_ForcedLegacyChat()).engine_eligible


@pytest.mark.parametrize("case_id", CHAT_CASE_IDS)
def test_dual_run_engine_path_matches_fixture(case_id):
    """Engine path vs the recorded pre-engine fixture (also covered by the
    main parity suite; kept here so a renderer regression points at QC-05)."""
    _skip_if_python_sensitive(CASES[case_id])
    actual = json.loads(render_fixture(case_fixture(CASES[case_id])))
    expected = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert actual == expected


@pytest.mark.parametrize("case_id", CHAT_CASE_IDS)
def test_dual_run_legacy_path_matches_fixture(case_id):
    """Forced-legacy subclass vs the same fixture: proves the legacy body is
    intact and byte-equal, i.e. the cutover changed nothing observable."""
    _skip_if_python_sensitive(CASES[case_id])
    case = CASES[case_id]
    legacy_expected = canonicalize(execute_case(case, adapter_factory=_forced_legacy_factory))
    fixture = json.loads((REQUEST_DIR / f"{case_id}.json").read_text(encoding="utf-8"))
    assert legacy_expected == fixture["expected"]


def test_unmigrated_core_adapters_still_route_legacy():
    from dspy.adapters.baml_adapter import BAMLAdapter
    from dspy.adapters.json_adapter import JSONAdapter
    from dspy.adapters.xml_adapter import XMLAdapter

    for cls in (JSONAdapter, XMLAdapter, BAMLAdapter):
        assert not resolve_override_verdict(cls()).engine_eligible, cls.__name__
    assert not resolve_override_verdict(TwoStepAdapter(StubLM(Recorder()))).engine_eligible


def test_call_only_wrapper_uses_engine_path(monkeypatch):
    """refine.py's WrapperAdapter pattern must take the engine path."""
    calls = {"engine": 0}
    original = render_module.render_messages

    def counting(*args, **kwargs):
        calls["engine"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(render_module, "render_messages", counting)

    class Wrapper(ChatAdapter):
        def __call__(self, lm, lm_kwargs, signature, demos, inputs):
            return super().__call__(lm, lm_kwargs, signature, demos, inputs)

    import dspy

    class QA(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    Wrapper(use_json_adapter_fallback=False).format(QA, [], {"question": "Q?"})
    assert calls["engine"] == 1


def test_render_module_contains_no_format_literals():
    """Mechanical approximation of the review gate: the renderer must hold
    zero format-specific literals — markers, schema sentences, completed
    markers all belong to Format objects."""
    source = inspect_module.getsource(render_module)
    for forbidden in ("[[ ##", "## ]]", "completed", "Respond with", "Your input fields", "objective"):
        assert forbidden not in source, f"format literal {forbidden!r} leaked into render.py"
