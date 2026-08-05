"""Tests for the Prediction `_trajectory` observability channel.

A Prediction's fields are exactly the signature's declared outputs; mechanism
exhaust (trajectories, undeclared reasoning) travels on `_trajectory`. Legacy
attribute access is shimmed with a DeprecationWarning.
"""

import warnings

import pytest

import dspy
from dspy.utils.dummies import DummyLM


def test_trajectory_channel_defaults_empty():
    pred = dspy.Prediction(answer="x")
    assert pred._trajectory == {}
    assert "answer" in pred._store


def test_trajectory_channel_not_in_store_or_serialization():
    pred = dspy.Prediction(answer="x")
    pred._trajectory["trajectory"] = {"thought_0": "hmm"}
    assert "trajectory" not in pred._store
    assert pred.toDict() == {"answer": "x"}
    assert list(pred.keys()) == ["answer"]


def test_trajectory_shim_warns_and_forwards():
    pred = dspy.Prediction(answer="x")
    pred._trajectory["trajectory"] = {"thought_0": "hmm"}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pred.trajectory == {"thought_0": "hmm"}
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert "mechanism exhaust" in str(caught[0].message)


def test_declared_field_shadows_shim_without_warning():
    pred = dspy.Prediction(trajectory="declared-value")
    pred._trajectory["trajectory"] = "exhaust-value"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert pred.trajectory == "declared-value"
    assert not caught


def test_missing_attribute_still_raises():
    pred = dspy.Prediction(answer="x")
    with pytest.raises(AttributeError):
        _ = pred.nonexistent


def test_cot_reasoning_moves_to_channel():
    lm = DummyLM([{"reasoning": "step by step", "answer": "42"}])
    dspy.configure(lm=lm)
    cot = dspy.ChainOfThought("question -> answer")
    result = cot(question="q")
    assert "reasoning" not in result._store
    assert result._trajectory["reasoning"] == "step by step"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert result.reasoning == "step by step"
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_cot_declared_reasoning_stays_contractual():
    lm = DummyLM([{"reasoning": "because", "answer": "42"}])
    dspy.configure(lm=lm)
    cot = dspy.ChainOfThought("question -> reasoning, answer")
    result = cot(question="q")
    assert result._store["reasoning"] == "because"
    assert "reasoning" not in result._trajectory
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert result.reasoning == "because"
    assert not caught


@pytest.mark.anyio
async def test_cot_async_reasoning_moves_to_channel():
    lm = DummyLM([{"reasoning": "async step", "answer": "42"}])
    dspy.configure(lm=lm)
    cot = dspy.ChainOfThought("question -> answer")
    result = await cot.acall(question="q")
    assert "reasoning" not in result._store
    assert result._trajectory["reasoning"] == "async step"
