"""Tests for LMConfig — typed generation parameters."""


import pytest

from dspy.lm_types import LMConfig, LMToolDef


class TestLMConfig:
    def test_defaults(self):
        c = LMConfig()
        assert c.temperature is None
        assert c.max_tokens is None
        assert c.n == 1
        assert c.tools is None
        assert c.response_format is None
        assert c.reasoning_effort is None
        assert c.extensions == {}

    def test_with_values(self):
        c = LMConfig(temperature=0.7, max_tokens=500, n=3)
        assert c.temperature == 0.7
        assert c.max_tokens == 500
        assert c.n == 3

    def test_tools(self):
        tools = [LMToolDef(name="search", description="search the web")]
        c = LMConfig(tools=tools)
        assert len(c.tools) == 1
        assert c.tools[0].name == "search"

    def test_response_format(self):
        c = LMConfig(response_format={"type": "json_object"})
        assert c.response_format == {"type": "json_object"}

    def test_reasoning_effort(self):
        c = LMConfig(reasoning_effort="high")
        assert c.reasoning_effort == "high"

    def test_extensions(self):
        c = LMConfig(extensions={"provider_specific": "value"})
        assert c.extensions["provider_specific"] == "value"

    def test_merge_overrides(self):
        base = LMConfig(temperature=0.0, max_tokens=100)
        override = LMConfig(temperature=1.0)
        merged = base.merge(override)
        assert merged.temperature == 1.0
        assert merged.max_tokens == 100

    def test_merge_preserves_none(self):
        base = LMConfig(temperature=0.5)
        override = LMConfig()  # everything None
        merged = base.merge(override)
        assert merged.temperature == 0.5

    def test_merge_combines_extensions(self):
        base = LMConfig(extensions={"a": 1})
        override = LMConfig(extensions={"b": 2})
        merged = base.merge(override)
        assert merged.extensions == {"a": 1, "b": 2}

    def test_merge_override_in_extensions(self):
        base = LMConfig(extensions={"k": "v1"})
        override = LMConfig(extensions={"k": "v2"})
        merged = base.merge(override)
        assert merged.extensions == {"k": "v2"}

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            LMConfig(unknown_field=42)


class TestLMToolDef:
    def test_basic(self):
        t = LMToolDef(name="search")
        assert t.name == "search"
        assert t.description is None
        assert t.parameters == {"type": "object", "properties": {}}

    def test_with_description_and_params(self):
        t = LMToolDef(
            name="get_weather",
            description="Get the weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert t.description == "Get the weather"
        assert "city" in t.parameters["properties"]
