"""Tests verifying preset definitions produce expected prompts."""


import pytest

import dspy
from dspy.lm_types import (
    AdapterV2,
    CHAT_PRESET,
    Demos,
    History as HistoryDirective,
    InputFields,
    Inputs,
    Instruction,
    JSON_PRESET,
    Message,
    OutputFields,
    OutputRequest,
    Outputs,
    PRESETS,
    Preset,
    Structure,
    Text,
    XML_PRESET,
    from_preset,
)


@pytest.fixture
def signature():
    class Sig(dspy.Signature):
        """Do the task."""
        query: str = dspy.InputField(desc="The query")
        answer: str = dspy.OutputField(desc="The answer")
    return Sig


class TestPresetStructure:
    def test_chat_preset_has_messages(self):
        assert len(CHAT_PRESET.messages) >= 2

    def test_chat_preset_style(self):
        assert CHAT_PRESET.style == "chat"

    def test_json_preset_style(self):
        assert JSON_PRESET.style == "json"

    def test_xml_preset_style(self):
        assert XML_PRESET.style == "xml"

    def test_presets_dict_contains_builtins(self):
        assert "chat" in PRESETS
        assert "json" in PRESETS
        assert "xml" in PRESETS

    def test_all_presets_are_immutable(self):
        with pytest.raises(Exception):
            CHAT_PRESET.messages = ()


class TestTypedFragments:
    def test_text_fragment(self):
        t = Text("hello")
        assert t.text == "hello"

    def test_instruction_fragment(self):
        i = Instruction()
        assert i is not None

    def test_input_fields_fragment(self):
        f = InputFields()
        assert f is not None

    def test_output_fields_fragment(self):
        f = OutputFields()
        assert f is not None

    def test_structure_default_style(self):
        s = Structure()
        assert s.style == "chat"

    def test_structure_with_style(self):
        s = Structure(style="xml")
        assert s.style == "xml"

    def test_inputs_fragment(self):
        i = Inputs(style="json")
        assert i.style == "json"

    def test_outputs_fragment(self):
        o = Outputs(style="schema")
        assert o.style == "schema"

    def test_output_request_fragment(self):
        o = OutputRequest(style="chat")
        assert o.style == "chat"


class TestDemosDirective:
    def test_plain_demos(self):
        d = Demos()
        assert d.user is None
        assert d.assistant is None

    def test_custom_demos_template(self):
        d = Demos(
            user=[Text("Q: "), Inputs()],
            assistant=[Text("A: "), Outputs()],
        )
        assert d.user is not None
        assert d.assistant is not None


class TestHistoryDirective:
    def test_plain_history(self):
        h = HistoryDirective()
        assert h.user is None

    def test_custom_history(self):
        h = HistoryDirective(user=[Text("X")], assistant=[Text("Y")])
        assert h.user is not None


class TestCustomPreset:
    def test_build_custom_preset(self):
        preset = Preset(
            messages=(
                Message(role="system", content=[Text("I am a helper."), Instruction()]),
                Message(role="user", content=Inputs(style="chat")),
            ),
            style="chat",
        )
        assert len(preset.messages) == 2
        assert preset.style == "chat"

    def test_custom_preset_in_adapter(self, signature):
        preset = Preset(
            messages=(
                Message(role="system", content=[Text("Custom system"), Instruction()]),
                Message(role="user", content=[Text("Go: "), Inputs()]),
            ),
            style="chat",
        )
        adapter = AdapterV2.from_preset_obj(preset)
        messages = adapter.format(signature, [], {"query": "X"})
        # System should contain custom text + instructions
        sys_text = "".join(m.text or "" for m in messages if m.role == "system")
        assert "Custom system" in sys_text


class TestRendering:
    def test_chat_preset_renders_field_markers(self, signature):
        adapter = from_preset("chat")
        messages = adapter.format(signature, [], {"query": "testquery"})
        full_prompt = "\n".join(m.text or "" for m in messages)
        assert "[[ ## query ## ]]" in full_prompt
        assert "testquery" in full_prompt
        assert "answer" in full_prompt

    def test_json_preset_mentions_json(self, signature):
        adapter = from_preset("json")
        messages = adapter.format(signature, [], {"query": "x"})
        full_prompt = "\n".join(m.text or "" for m in messages)
        assert "JSON" in full_prompt or "json" in full_prompt

    def test_xml_preset_renders_xml_tags(self, signature):
        adapter = from_preset("xml")
        messages = adapter.format(signature, [], {"query": "x"})
        full_prompt = "\n".join(m.text or "" for m in messages)
        assert "<answer>" in full_prompt or "<query>" in full_prompt
