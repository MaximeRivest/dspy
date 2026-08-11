"""Stage A2 tests: presets, lens derivation, entry serde, media shapes."""

import json
from pathlib import Path

import pytest

import dspy
from dspy.adapters import load_entry, make_adapter
from dspy.adapters.errors import AdapterError, AdapterParseError, EntryError

EXAMPLES = Path(__file__).resolve().parents[1] / "roadmap" / "adapter-ir-stage" / "examples"


class QA(dspy.Signature):
    """Answer the question concisely."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class TwoOut(dspy.Signature):
    """Extract."""

    text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField()
    summary: str = dspy.OutputField()


def example_entries(name: str) -> list[dict]:
    data = json.loads((EXAMPLES / name / "adapter-ir.json").read_text())
    return data if isinstance(data, list) else [data]


# ---------------------------------------------------------------------------
# Presets: thin constructors over entries
# ---------------------------------------------------------------------------


class TestPresets:
    def test_chat_entry_is_example_01_byte_for_byte(self):
        (want,) = example_entries("01-chat-baseline")
        assert dspy.ChatAdapter().dump_entry() == want

    def test_all_presets_carry_the_lens(self):
        for cls in (dspy.ChatAdapter, dspy.JSONAdapter, dspy.XMLAdapter):
            assert cls().dump_entry()["parser"] == {"kind": "lens", "of": "template"}

    def test_chat_preview_bytes(self):
        messages = dspy.ChatAdapter().preview(QA, {"question": "Why?"})
        assert messages[0]["role"] == "system"
        assert "1. `question` (str):" in messages[0]["content"]
        assert "[[ ## answer ## ]]" in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert "[[ ## question ## ]]\nWhy?" in messages[-1]["content"]
        assert "ending with the marker for `[[ ## completed ## ]]`" in messages[-1]["content"]

    def test_demos_expand_to_turn_pairs(self):
        demos = [{"question": "Q1", "answer": "A1"}]
        messages = dspy.ChatAdapter().preview(QA, {"question": "Q2"}, demos=demos)
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user"]
        assert "[[ ## answer ## ]]\nA1" in messages[2]["content"]
        assert messages[2]["content"].endswith("[[ ## completed ## ]]\n")

    def test_format_returns_call_with_empty_request(self):
        call = dspy.ChatAdapter().format(QA, {"question": "Why?"})
        assert call.request == {}
        assert isinstance(call.messages, list)

    def test_preview_is_pure_no_lm(self):
        first = dspy.ChatAdapter().preview(QA, {"question": "Why?"})
        second = dspy.ChatAdapter().preview(QA, {"question": "Why?"})
        assert first == second


# ---------------------------------------------------------------------------
# The lens: parser derived from the template
# ---------------------------------------------------------------------------


class TestLensDerivation:
    def test_chat_lens_is_labeled_markers(self):
        lens = dspy.ChatAdapter().lens()
        assert lens["mode"] == "labeled"
        assert lens["named"] is True
        assert lens["source"] == "demos.assistant"

    def test_chat_lens_parses_markers(self):
        out = dspy.ChatAdapter().parse_preview(QA, "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]")
        assert out == {"answer": "Paris"}

    def test_chat_lens_ignores_unknown_markers(self):
        completion = "[[ ## noise ## ]]\nignored\n\n[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"
        assert dspy.ChatAdapter().parse_preview(QA, completion) == {"answer": "Paris"}

    def test_json_lens_is_object_mode(self):
        assert dspy.JSONAdapter().lens()["mode"] == "json_object"

    def test_json_lens_parses_object(self):
        out = dspy.JSONAdapter().parse_preview(QA, '{"answer": "Paris"}')
        assert out == {"answer": "Paris"}

    def test_json_lens_prefers_fenced_block_and_repairs(self):
        completion = 'Sure!\n```json\n{"answer": "Paris",}\n```\nDone.'
        assert dspy.JSONAdapter().parse_preview(QA, completion) == {"answer": "Paris"}

    def test_json_lens_coerces_types(self):
        out = dspy.JSONAdapter().parse_preview(TwoOut, '{"entities": ["a", "b"], "summary": "s"}')
        assert out == {"entities": ["a", "b"], "summary": "s"}

    def test_xml_lens_cuts_closing_tags(self):
        out = dspy.XMLAdapter().parse_preview(QA, "<answer>\nParis\n</answer>")
        assert out == {"answer": "Paris"}

    def test_authored_template_lens_reads_name_colon_labels(self):
        tiny = make_adapter(
            name="tiny_qa",
            template=[
                {"role": "system", "content": "{instruction}"},
                {
                    "role": "demos",
                    "user": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}",
                    "assistant": "{% for f in outputs separator='\\n' strip %}{f.name}: {f.value}{% endfor %}",
                },
                {
                    "role": "user",
                    "content": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}\n{fragments('user')}",
                },
            ],
        )
        assert tiny.parse_preview(QA, "answer: Paris") == {"answer": "Paris"}

    def test_single_output_degenerates_to_full_text(self):
        tiny = load_entry(example_entries("03-token-minimal")[0])
        assert tiny.parse_preview(QA, "Paris, of course.") == {"answer": "Paris, of course."}

    def test_full_text_refuses_two_output_fields(self):
        bare = make_adapter(
            name="bare",
            template=[{"role": "user", "content": "{instruction}"}],
        )
        assert bare.lens()["mode"] == "full_text"
        with pytest.raises(AdapterParseError, match="exactly one output field"):
            bare.parse_preview(TwoOut, "no labels anywhere")

    def test_labeled_lens_without_labels_refuses_two_output_fields(self):
        tiny = load_entry(example_entries("03-token-minimal")[0])
        with pytest.raises(AdapterParseError, match="did not carry output field"):
            tiny.parse_preview(TwoOut, "no labels anywhere")

    def test_base_model_template_lens_reads_trailing_label(self):
        base = load_entry(example_entries("04-base-model")[0])
        assert base.lens()["mode"] == "labeled"
        # The completion continues after the rendered "answer:" label.
        assert base.parse_preview(QA, " Paris") == {"answer": "Paris"}
        # An echoed label parses too.
        assert base.parse_preview(QA, "answer: Paris") == {"answer": "Paris"}

    def test_weak_lens_refuses_naming_the_attribute(self):
        from dspy.adapters.lens import LensError

        with pytest.raises(LensError, match=r"\{f.desc\}"):
            make_adapter(
                name="weak",
                template=[
                    {
                        "role": "demos",
                        "user": "{% for f in inputs %}{f.name}: {f.value}{% endfor %}",
                        "assistant": "{% for f in outputs %}{f.desc}: {f.value}{% endfor %}",
                    },
                    {"role": "user", "content": "{% for f in inputs %}{f.name}: {f.value}{% endfor %}"},
                ],
            )

    def test_missing_field_refuses_naming_it(self):
        with pytest.raises(AdapterParseError, match="entities"):
            dspy.ChatAdapter().parse_preview(TwoOut, "[[ ## summary ## ]]\nonly one section")


# ---------------------------------------------------------------------------
# Entry serde: exact, versioned, loud
# ---------------------------------------------------------------------------

ROUND_TRIP_EXAMPLES = [
    "01-chat-baseline",
    "02-json-tolerant",
    "03-token-minimal",
    "04-base-model",
    "05-reasoning-three-ways",
    "06-tools-three-ways",
    "07-citations-native-vs-inline",
    "08-custom-regex-parser",
    "09-media-shapes",
]


class TestEntrySerde:
    @pytest.mark.parametrize("example", ROUND_TRIP_EXAMPLES)
    def test_example_entries_round_trip(self, example):
        for entry in example_entries(example):
            adapter = load_entry(entry)
            assert adapter.dump_entry() == entry, f"{example}/{entry['name']} did not round-trip"

    def test_constructed_adapter_round_trips(self):
        adapter = dspy.ChatAdapter.with_strategies(reasoning="prefix_cot")
        entry = adapter.dump_entry()
        assert load_entry(entry).dump_entry() == entry

    def test_unknown_entry_key_refuses(self):
        entry = dspy.ChatAdapter().dump_entry()
        entry["surprise"] = 1
        with pytest.raises(EntryError, match="unknown entry keys.*surprise"):
            load_entry(entry)

    def test_missing_key_refuses(self):
        entry = dspy.ChatAdapter().dump_entry()
        del entry["versions"]
        with pytest.raises(EntryError, match="missing.*versions"):
            load_entry(entry)

    def test_string_parser_refuses_with_teaching_error(self):
        entry = dspy.ChatAdapter().dump_entry()
        entry["parser"] = "chat"
        with pytest.raises(EntryError, match="lens derivations of their own templates"):
            load_entry(entry)

    def test_dangling_codec_ref_refuses_naming_it(self):
        entry = dspy.ChatAdapter().dump_entry()
        entry["codecs"]["output"] = "guillemet_deluxe"
        with pytest.raises(EntryError, match="guillemet_deluxe"):
            load_entry(entry)

    def test_unknown_strategy_name_refuses_listing_bindables(self):
        entry = dspy.ChatAdapter().dump_entry()
        entry["strategies"]["reasoning"] = "telepathy"
        with pytest.raises(EntryError, match="telepathy.*bindable"):
            load_entry(entry)

    def test_incompatible_version_refuses_naming_both(self):
        entry = dspy.ChatAdapter().dump_entry()
        entry["adapter_ir_version"] = "0.9.0"
        with pytest.raises(EntryError, match="0.9.0.*0.3.0-draft"):
            load_entry(entry)

    def test_used_vocabulary_must_be_pinned(self):
        (entry,) = example_entries("02-json-tolerant")
        entry = json.loads(json.dumps(entry))
        del entry["versions"]["parse_combinators"]
        with pytest.raises(EntryError, match="parse_combinators"):
            load_entry(entry)

    def test_unknown_vocabulary_refuses(self):
        entry = dspy.ChatAdapter().dump_entry()
        entry["versions"]["sonnets"] = "1.0.0"
        with pytest.raises(EntryError, match="sonnets"):
            load_entry(entry)

    def test_authored_parser_refuses_naming_the_requirement(self):
        entry = example_entries("11-authored-code-parser")[0]
        with pytest.raises(EntryError, match=r"requires python>=3.12 sidecar for `4_adapter/ledger_recovery/parser`"):
            load_entry(entry)

    def test_leaf_codec_refuses_naming_the_requirement(self):
        entry = example_entries("12-duckdb-boundary")[0]
        with pytest.raises(EntryError, match=r"requires python>=3.11 sidecar for `4_adapter/chat_duckdb/codecs/db`"):
            load_entry(entry)

    def test_unknown_codec_family_refuses(self):
        entry = example_entries("10-eval-python-codec")[0]
        with pytest.raises(EntryError, match="python_literal"):
            load_entry(entry)

    def test_rule_strategies_derive_requires(self):
        from dspy.adapters import strategy

        adapter = dspy.ChatAdapter.with_strategies(
            reasoning=strategy.rule(
                predicate=strategy.capability("native_reasoning"),
                hides=["reasoning"],
                routings=[strategy.channel("reasoning", field="reasoning")],
            )
        )
        entry = adapter.dump_entry()
        assert entry["requires"] == {"lm_capabilities": ["native_reasoning"]}
        assert entry["versions"]["strategies"] == "1.1.0-draft"
        assert entry["versions"]["lm_capabilities"] == "0.1.0"

    def test_name_only_strategies_omit_requires(self):
        entry = dspy.ChatAdapter().dump_entry()
        assert "requires" not in entry
        assert entry["versions"]["strategies"] == "1.0.0"

    def test_engine_controls_travel_in_config(self):
        base = make_adapter(
            name="base_fewshot",
            template=[{"role": "user", "content": "{instruction}\n{% for f in outputs separator='\\n' %}{f.name}:{% endfor %}"}],
            engine_controls={"completion_mode": True, "stop_sequences": ["\nquestion:", "\n\n"]},
            requires={"lm_capabilities": ["completion"]},
        )
        entry = base.dump_entry()
        assert entry["config"]["engine_controls"]["stop_sequences"] == ["\nquestion:", "\n\n"]
        assert entry["requires"] == {"lm_capabilities": ["completion"]}


# ---------------------------------------------------------------------------
# Requires and request-side data
# ---------------------------------------------------------------------------


class TestRequiresAndRequest:
    def base_adapter(self):
        return make_adapter(
            name="base_fewshot",
            template=[
                {
                    "role": "user",
                    "content": "{instruction}\n\n{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}\n{% for f in outputs separator='\\n' %}{f.name}:{% endfor %}",
                }
            ],
            engine_controls={"completion_mode": True, "stop_sequences": ["\nquestion:", "\n\n"]},
            requires={"lm_capabilities": ["completion"]},
        )

    def test_engine_controls_become_request_data(self):
        caps = dspy.LMCapabilities(instruct=False)
        call = self.base_adapter().format(QA, {"question": "Why?"}, capabilities=caps)
        assert call.request["stop"] == ["\nquestion:", "\n\n"]
        assert call.request["completion_mode"] is True

    def test_unmet_requirement_refuses_naming_the_capability(self):
        with pytest.raises(AdapterError, match="requires LM capability 'completion'"):
            self.base_adapter().format(QA, {"question": "Why?"}, capabilities=dspy.LMCapabilities())

    def test_capabilities_read_from_the_lm(self):
        lm = dspy.DummyLM(["x"], instruct=False)
        call = self.base_adapter().format(QA, {"question": "Why?"}, lm=lm)
        assert call.request["stop"] == ["\nquestion:", "\n\n"]


# ---------------------------------------------------------------------------
# Media shapes: the two-layer rule
# ---------------------------------------------------------------------------


class TestMediaShapes:
    def caption_signature(self):
        PIL = pytest.importorskip("PIL.Image")

        class Caption(dspy.Signature):
            """Describe the photo."""

            photo: PIL.Image = dspy.InputField(role="media")
            caption: str = dspy.OutputField()

        return Caption

    def test_dump_lowers_host_type_to_shape_and_wire(self):
        Caption = self.caption_signature()
        entry = dspy.ChatAdapter().dump_entry(for_signature=Caption)
        assert entry["codecs"]["per_field"]["photo"] == {
            "kind": "shape",
            "shape": "image",
            "wire": {"encoding": "base64", "media_type": "image/png"},
            "frontend_bindings": {"python": "PIL.Image.Image"},
        }
        assert entry["versions"]["shapes"] == "0.1.0"

    def test_format_emits_image_parts_at_the_slot(self):
        import PIL.Image

        Caption = self.caption_signature()
        image = PIL.Image.new("RGB", (2, 2), "red")
        call = dspy.ChatAdapter().format(Caption, {"photo": image})
        content = call.messages[-1]["content"]
        assert isinstance(content, list)
        kinds = [part["type"] for part in content]
        assert "image_url" in kinds
        (image_part,) = [part for part in content if part["type"] == "image_url"]
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_dspy_image_values_work_without_pil_signature(self):
        class Card(dspy.Signature):
            photo: dspy.Image = dspy.InputField()
            caption: str = dspy.OutputField()

        image = dspy.Image("data:image/png;base64,QUJD")
        call = dspy.ChatAdapter().format(Card, {"photo": image})
        content = call.messages[-1]["content"]
        assert isinstance(content, list)
        assert any(part["type"] == "image_url" for part in content)


# ---------------------------------------------------------------------------
# End to end with a DummyLM (the A3 wiring shape)
# ---------------------------------------------------------------------------


class TestDummyLMLoop:
    def test_format_call_parse(self):
        adapter = dspy.ChatAdapter()
        lm = dspy.DummyLM(["[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"])
        call = adapter.format(QA, {"question": "Capital of France?"}, lm=lm)
        outputs = lm(messages=call.messages, **call.request)
        fields = adapter.parse(QA, outputs[0], lm=lm)
        assert fields == {"answer": "Paris"}
        assert lm.calls[0]["messages"] == call.messages
