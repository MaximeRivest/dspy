"""Tests verifying the new interface coexists with existing DSPy code.

These tests ensure that:
- Existing adapters still work
- Existing BaseLM subclasses still work
- New and old LMs can be used interchangeably with Predict
"""


import pytest

import dspy
from dspy.lm_types import (
    AdapterV2,
    BaseLMv2,
    LMCompletion,
    LMConfig,
    LMMessage,
    LMResponse,
    TextPart,
    from_preset,
)
from dspy.utils.dummies import DummyLM


class TestExistingAdaptersStillWork:
    def test_chat_adapter_unchanged(self):
        """Existing ChatAdapter should still work the same way."""
        adapter = dspy.ChatAdapter()
        assert adapter is not None

    def test_json_adapter_unchanged(self):
        adapter = dspy.JSONAdapter()
        assert adapter is not None


class TestPredictWithLMv2:
    def test_predict_with_new_lm(self):
        class Sig(dspy.Signature):
            """Answer."""
            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        class MyLM(BaseLMv2):
            def forward(self, messages, config):
                return LMResponse(completions=[LMCompletion(parts=[TextPart(
                    text="[[ ## a ## ]]\nParis\n\n[[ ## completed ## ]]"
                )])])
            async def aforward(self, messages, config):
                return self.forward(messages, config)

        lm = MyLM(model="mymodel")
        with dspy.context(lm=lm, adapter=from_preset("chat").as_legacy()):
            program = dspy.Predict(Sig)
            result = program(q="What?")
            assert result.a == "Paris"


class TestBackwardCompatShim:
    def test_adapter_v2_exposes_legacy_api(self):
        """AdapterV2.as_legacy() should wrap it as a dspy.Adapter subclass."""
        adapter = from_preset("chat").as_legacy()
        # Should be usable wherever dspy.Adapter is expected
        import dspy.adapters
        assert isinstance(adapter, dspy.adapters.Adapter)

    def test_legacy_shim_with_dummy_lm(self):
        """A DummyLM (old interface) should work with the legacy shim."""
        class Sig(dspy.Signature):
            q: str = dspy.InputField()
            a: str = dspy.OutputField()

        lm = DummyLM([{"a": "Paris"}])
        adapter = from_preset("chat").as_legacy()

        with dspy.context(lm=lm, adapter=adapter):
            program = dspy.Predict(Sig)
            result = program(q="What?")
            assert result.a == "Paris"
