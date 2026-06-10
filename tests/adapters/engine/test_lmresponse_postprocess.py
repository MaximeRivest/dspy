"""QC-11 gates: engine postprocess consumes the typed LMResponse directly
and is value-equal to the legacy round-trip path on every output shape."""

import pytest
from golden.harness import Recorder, StubLM, canonical_error, canonicalize

import dspy
from dspy.adapters._engine.builder import build_plan
from dspy.adapters._engine.parser_hook import ResponseView
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.types.reasoning import Reasoning
from dspy.clients.openai_format import legacy_outputs_from_lm_response, lm_response_from_legacy_outputs
from dspy.core.types import LMRequest

CHAT_OK = "[[ ## answer ## ]]\nBerlin\n\n[[ ## completed ## ]]"


def _response(outputs):
    return lm_response_from_legacy_outputs(outputs, LMRequest(model="stub/golden-model", messages=[]))


class QA(dspy.Signature):
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class ThoughtfulQA(dspy.Signature):
    """Think then answer."""

    question: str = dspy.InputField()
    reasoning: Reasoning = dspy.OutputField()
    answer: str = dspy.OutputField()


OUTPUT_SHAPES = {
    "plain-str": [CHAT_OK],
    "dict-with-text": [{"text": CHAT_OK}],
    "dict-with-logprobs": [{"text": CHAT_OK, "logprobs": {"content": [{"token": "B", "logprob": -0.2}]}}],
    "n-2-mixed-shapes": [CHAT_OK, {"text": "[[ ## answer ## ]]\nMunich\n\n[[ ## completed ## ]]"}],
    "reasoning-channel": [{"text": CHAT_OK, "reasoning_content": "considered geography"}],
    "citations-channel": [
        {
            "text": "[[ ## answer ## ]]\nAt 100C.\n\n[[ ## completed ## ]]",
            "citations": [
                {"cited_text": "Water boils.", "document_index": 0, "start_char_index": 0, "end_char_index": 12}
            ],
        }
    ],
    "garbage-str": ["no markers here"],
    "empty-str": [""],
}


def _dual_postprocess(signature, legacy_outputs, lm):
    """Run the SAME LMResponse through both postprocess paths."""
    adapter = ChatAdapter(use_json_adapter_fallback=False)
    lm_kwargs = {}
    built = build_plan(adapter, lm, lm_kwargs, signature, {"question": "Q?"})
    response = _response(legacy_outputs)

    def run(**kwargs):
        try:
            return (
                "completed",
                canonicalize(adapter._call_postprocess(built.render_signature, signature, **kwargs)),
            )
        except Exception as error:
            return (type(error).__name__, canonicalize(canonical_error(error)))

    engine = run(outputs=[], lm=lm, lm_kwargs={}, plan=built.plan, response=response)
    legacy = run(outputs=legacy_outputs_from_lm_response(response), lm=lm, lm_kwargs={})
    return engine, legacy


@pytest.mark.parametrize("shape", sorted(OUTPUT_SHAPES))
def test_engine_postprocess_equals_legacy(shape):
    lm = StubLM(Recorder(), supports_reasoning=True)
    signature = ThoughtfulQA if shape in ("reasoning-channel",) else QA
    engine, legacy = _dual_postprocess(signature, OUTPUT_SHAPES[shape], lm)
    assert engine == legacy, f"postprocess divergence for shape {shape!r}"


def test_provider_output_str_vs_dict_fidelity():
    """The view derives its backing through the canonical conversion: str
    provider outputs stay str, dict outputs stay the same dict object."""
    response = _response([CHAT_OK, {"text": CHAT_OK}])
    views = ResponseView.from_lm_response(response)
    assert isinstance(views[0].raw, str)
    assert isinstance(views[1].raw, dict)
    assert views[1].raw["text"] == CHAT_OK


def test_parser_hook_signature_unchanged_since_qc03():
    """The facade swap must not have re-signatured any hook: parse takes
    (response_view, ctx), exactly as frozen in QC-03."""
    import inspect

    from dspy.adapters._engine.parse import FormatParserHook

    parameters = list(inspect.signature(FormatParserHook.parse).parameters)
    assert parameters == ["self", "response_view", "ctx"]
