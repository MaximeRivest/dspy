"""Stage A2 tests: strategies as rule objects — the three trios, working."""

import pytest

import dspy
from dspy.adapters import strategy
from dspy.adapters.errors import AdapterError, EntryError
from dspy.adapters.strategies import (
    capability_fact,
    evaluate_predicate,
    register_strategy,
    unregister_strategy,
    validate_rule,
)

INSTRUCT = dspy.LMCapabilities()
BASE = dspy.LMCapabilities(instruct=False)
NATIVE_REASONING = dspy.LMCapabilities(native_reasoning=True)
NATIVE_FC = dspy.LMCapabilities(native_fc=True)
NATIVE_CITATIONS = dspy.LMCapabilities(native_citations=True)


class CoT(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    reasoning: str = dspy.OutputField(role="reasoning")
    answer: str = dspy.OutputField()


class Research(dspy.Signature):
    """Answer using the available tools."""

    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField(role="tools")
    tool_calls: dspy.ToolCalls = dspy.OutputField(role="tool_calls")
    answer: str = dspy.OutputField()


class GroundedQA(dspy.Signature):
    """Answer from the documents, citing your sources."""

    documents: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citations: dspy.Citations = dspy.OutputField(role="citations")


def search(query: str) -> str:
    """Search the web."""
    return "result"


# ---------------------------------------------------------------------------
# Capability facts and predicates
# ---------------------------------------------------------------------------


class TestPredicates:
    def test_declared_facts(self):
        assert capability_fact(NATIVE_REASONING, "native_reasoning")
        assert not capability_fact(INSTRUCT, "native_reasoning")
        assert capability_fact(BASE, "completion")
        assert not capability_fact(INSTRUCT, "completion")

    def test_no_facts_means_false(self):
        assert not capability_fact(None, "instruct")

    def test_unknown_fact_refuses(self):
        with pytest.raises(AdapterError, match="unknown LM capability"):
            capability_fact(INSTRUCT, "telepathy")

    def test_boolean_structure(self):
        pred = {"all": [{"capability": "instruct"}, {"not": {"capability": "native_reasoning"}}]}
        assert evaluate_predicate(pred, INSTRUCT)
        assert not evaluate_predicate(pred, NATIVE_REASONING)

    def test_rule_validation_refuses_unknown_capability(self):
        rule = strategy.rule(predicate=strategy.capability("native_reasoning"))
        rule["predicate"] = {"capability": "telepathy"}
        with pytest.raises(EntryError, match="telepathy"):
            validate_rule(rule, where="test")

    def test_rule_validation_refuses_missing_faces(self):
        with pytest.raises(EntryError, match="missing faces"):
            validate_rule({"kind": "rule", "predicate": {"capability": "instruct"}}, where="test")

    def test_rule_validation_refuses_unknown_faces(self):
        rule = strategy.rule(predicate=strategy.capability("instruct"))
        rule["mood"] = "bold"
        with pytest.raises(EntryError, match="unknown rule faces"):
            validate_rule(rule, where="test")

    def test_materialize_routing_refuses_naming_the_leaf(self):
        from dspy.adapters import parse

        rule = strategy.rule(
            predicate=strategy.capability("instruct"),
            routings=[
                strategy.text(
                    parse.pipeline(parse.fenced_block(language="python", policy="require")),
                    field="answer",
                    materialize={"interpreter": {"leaf": "python_literal_eval"}},
                )
            ],
        )
        with pytest.raises(EntryError, match="requires interpreter leaf 'python_literal_eval'"):
            validate_rule(rule, where="test")


# ---------------------------------------------------------------------------
# Reasoning: native channel / prefix-CoT / interleaved tags
# ---------------------------------------------------------------------------


class TestReasoningThreeWays:
    def test_native_hides_patches_and_routes_the_channel(self):
        adapter = dspy.ChatAdapter(strategies={"reasoning": "native"})
        call = adapter.format(CoT, {"question": "Why?"}, capabilities=NATIVE_REASONING)
        assert call.request == {"reasoning": {"effort": "medium"}}
        assert all("[[ ## reasoning ## ]]" not in str(m["content"]) for m in call.messages)
        out = adapter.parse(
            CoT,
            "[[ ## answer ## ]]\nBecause.\n\n[[ ## completed ## ]]",
            channels={"reasoning": "thinking..."},
            capabilities=NATIVE_REASONING,
        )
        assert out == {"reasoning": "thinking...", "answer": "Because."}

    def test_native_without_the_channel_yields_none(self):
        adapter = dspy.ChatAdapter(strategies={"reasoning": "native"})
        out = adapter.parse(
            CoT,
            "[[ ## answer ## ]]\nBecause.\n\n[[ ## completed ## ]]",
            capabilities=NATIVE_REASONING,
        )
        assert out["reasoning"] is None

    def test_native_refuses_an_lm_without_the_capability(self):
        adapter = dspy.ChatAdapter(strategies={"reasoning": "native"})
        with pytest.raises(AdapterError, match="native_reasoning"):
            adapter.format(CoT, {"question": "Why?"}, capabilities=INSTRUCT)

    def test_prefix_cot_is_fragments_only(self):
        adapter = dspy.ChatAdapter(strategies={"reasoning": "prefix_cot"})
        call = adapter.format(CoT, {"question": "Why?"}, capabilities=INSTRUCT)
        assert call.request == {}
        user = call.messages[-1]["content"]
        assert "Reason step by step in the `reasoning` field" in user
        # The field stays visible; the template's own lens parses it.
        out = adapter.parse(
            CoT,
            "[[ ## reasoning ## ]]\nstep one\n\n[[ ## answer ## ]]\nBecause.\n\n[[ ## completed ## ]]",
            capabilities=INSTRUCT,
        )
        assert out == {"reasoning": "step one", "answer": "Because."}

    def test_interleaved_collects_and_consumes_think_tags(self):
        adapter = dspy.ChatAdapter(strategies={"reasoning": "interleaved"})
        call = adapter.format(CoT, {"question": "Why?"}, capabilities=INSTRUCT)
        assert "<think>" in call.messages[-1]["content"]
        completion = "[[ ## answer ## ]]\nBecause.<think>hmm</think> Indeed.<think>ok</think>\n\n[[ ## completed ## ]]"
        out = adapter.parse(CoT, completion, capabilities=INSTRUCT)
        assert out["reasoning"] == "hmm\nok"
        assert out["answer"] == "Because. Indeed."

    def test_the_template_stays_constant_across_the_trio(self):
        entries = [
            dspy.ChatAdapter(strategies={"reasoning": name}).dump_entry()
            for name in ("native", "prefix_cot", "interleaved")
        ]
        assert entries[0]["template"] == entries[1]["template"] == entries[2]["template"]
        assert entries[0]["parser"] == entries[1]["parser"] == entries[2]["parser"]


# ---------------------------------------------------------------------------
# Tools: native FC / CLI text / XML blocks
# ---------------------------------------------------------------------------


class TestToolsThreeWays:
    def tools(self):
        return [dspy.Tool(search)]

    def test_native_fc_splices_the_tools_into_the_request(self):
        adapter = dspy.ChatAdapter(strategies={"tools": "native_fc"})
        call = adapter.format(Research, {"question": "Who?", "tools": self.tools()}, capabilities=NATIVE_FC)
        assert call.request["tool_choice"] == "auto"
        assert call.request["tools"][0]["function"]["name"] == "search"
        # Both tool fields leave the token stream.
        assert all("[[ ## tools ## ]]" not in str(m["content"]) for m in call.messages)
        assert all("[[ ## tool_calls ## ]]" not in str(m["content"]) for m in call.messages)

    def test_native_fc_routes_the_channel(self):
        adapter = dspy.ChatAdapter(strategies={"tools": "native_fc"})
        out = adapter.parse(
            Research,
            "[[ ## answer ## ]]\nX\n\n[[ ## completed ## ]]",
            channels={"tool_calls": [{"name": "search", "args": {"query": "q"}}]},
            capabilities=NATIVE_FC,
        )
        assert out["tool_calls"].tool_calls[0].name == "search"
        assert out["answer"] == "X"

    def test_native_fc_splice_without_a_value_refuses(self):
        adapter = dspy.ChatAdapter(strategies={"tools": "native_fc"})
        with pytest.raises(AdapterError, match="provides no value"):
            adapter.format(Research, {"question": "Who?"}, capabilities=NATIVE_FC)

    def test_cli_text_teaches_and_parses_the_line_format(self):
        adapter = dspy.ChatAdapter(strategies={"tools": "cli_text"})
        call = adapter.format(Research, {"question": "Who?", "tools": self.tools()}, capabilities=INSTRUCT)
        user = call.messages[-1]["content"]
        assert "!call <tool_name> <json arguments>" in user
        assert "search" in user  # {field('tools')} rendered the declarations
        completion = '[[ ## answer ## ]]\nLooking.\n!call search {"query": "who"}\n\n[[ ## completed ## ]]'
        out = adapter.parse(Research, completion, capabilities=INSTRUCT)
        assert out["tool_calls"].tool_calls[0].args == {"query": "who"}
        assert out["answer"] == "Looking."  # the call line was consumed

    def test_xml_blocks_parse_the_tagged_call(self):
        adapter = dspy.ChatAdapter(strategies={"tools": "xml_blocks"})
        completion = '[[ ## answer ## ]]\nHm <tool_call name="search">{"query": "who"}</tool_call>\n\n[[ ## completed ## ]]'
        out = adapter.parse(Research, completion, capabilities=INSTRUCT)
        assert out["tool_calls"].tool_calls[0].name == "search"
        assert out["answer"] == "Hm"

    def test_no_calls_parses_to_empty_tool_calls(self):
        adapter = dspy.ChatAdapter(strategies={"tools": "cli_text"})
        out = adapter.parse(
            Research,
            "[[ ## answer ## ]]\nDone.\n\n[[ ## completed ## ]]",
            capabilities=INSTRUCT,
        )
        assert out["tool_calls"].tool_calls == []


# ---------------------------------------------------------------------------
# Citations: native channel / inline markers
# ---------------------------------------------------------------------------


class TestCitationsTwoWays:
    def test_native_renames_patches_and_routes(self):
        adapter = dspy.ChatAdapter(strategies={"citations": "native"})
        call = adapter.format(GroundedQA, {"documents": ["d"], "question": "Q?"}, capabilities=NATIVE_CITATIONS)
        assert call.request == {"citations": {"enabled": True}}
        system = call.messages[0]["content"]
        assert "answer_text" in system  # the rename composes lens and channel
        out = adapter.parse(
            GroundedQA,
            "[[ ## answer_text ## ]]\nWater boils.\n\n[[ ## completed ## ]]",
            channels={"citations": [{"span": "Water boils.", "doc": 1}]},
            capabilities=NATIVE_CITATIONS,
        )
        assert out["answer"] == "Water boils."
        assert out["citations"].citations[0].doc == 1

    def test_inline_markers_stay_visible(self):
        adapter = dspy.ChatAdapter(strategies={"citations": "inline"})
        call = adapter.format(GroundedQA, {"documents": ["d"], "question": "Q?"}, capabilities=INSTRUCT)
        assert "cite the supporting document as [n]" in call.messages[-1]["content"]
        completion = "[[ ## answer ## ]]\nSure. Water boils at 100C. [1] Ice floats. [2]\n\n[[ ## completed ## ]]"
        out = adapter.parse(GroundedQA, completion, capabilities=INSTRUCT)
        docs = [c.doc for c in out["citations"]]
        assert docs == [1, 2]
        # consume: false — the markers stay readable in the answer.
        assert "[1]" in out["answer"] and "[2]" in out["answer"]


# ---------------------------------------------------------------------------
# Resolution, bindings, the registry
# ---------------------------------------------------------------------------


class TestResolution:
    def test_auto_resolves_by_capability_and_records_it(self):
        adapter = dspy.ChatAdapter()
        assert adapter.explain_strategies(CoT, capabilities=NATIVE_REASONING)["reasoning"] == "auto->native"
        assert adapter.explain_strategies(CoT, capabilities=INSTRUCT)["reasoning"] == "auto->prefix_cot"
        assert adapter.explain_strategies(CoT)["reasoning"] == "auto->plain"

    def test_roles_without_fields_are_skipped(self):
        class Plain(dspy.Signature):
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        resolutions = dspy.ChatAdapter().explain_strategies(Plain, capabilities=NATIVE_REASONING)
        assert "reasoning" not in resolutions

    def test_rule_objects_bind_directly(self):
        adapter = dspy.ChatAdapter.with_strategies(
            reasoning=strategy.rule(
                predicate=strategy.capability("instruct"),
                fragments=[strategy.fragment("user", "Think first.")],
            )
        )
        call = adapter.format(CoT, {"question": "Why?"}, capabilities=INSTRUCT)
        assert "Think first." in call.messages[-1]["content"]

    def test_unknown_binding_name_refuses(self):
        with pytest.raises(EntryError, match="bindable"):
            dspy.ChatAdapter(strategies={"reasoning": "telepathy"})

    def test_unknown_role_refuses(self):
        with pytest.raises(EntryError, match="unknown strategy role"):
            dspy.ChatAdapter(strategies={"sonnets": "auto"})

    def test_rule_naming_a_missing_field_refuses(self):
        class Renamed(dspy.Signature):
            question: str = dspy.InputField()
            thoughts: str = dspy.OutputField(role="reasoning")
            answer: str = dspy.OutputField()

        adapter = dspy.ChatAdapter(strategies={"reasoning": "native"})
        with pytest.raises(AdapterError, match="hides field 'reasoning'"):
            adapter.format(Renamed, {"question": "Why?"}, capabilities=NATIVE_REASONING)

    def test_register_strategy_is_bindable_by_name(self):
        rule = strategy.rule(
            predicate=strategy.capability("instruct"),
            fragments=[strategy.fragment("user", "Answer in haiku form.")],
        )
        register_strategy("reasoning", "haiku", rule)
        try:
            adapter = dspy.ChatAdapter(strategies={"reasoning": "haiku"})
            call = adapter.format(CoT, {"question": "Why?"}, capabilities=INSTRUCT)
            assert "Answer in haiku form." in call.messages[-1]["content"]
        finally:
            unregister_strategy("reasoning", "haiku")

    def test_register_strategy_refuses_shadowing_builtins(self):
        rule = strategy.rule(predicate=strategy.capability("instruct"))
        with pytest.raises(AdapterError, match="collides"):
            register_strategy("reasoning", "native", rule)

    def test_unregister_unknown_refuses(self):
        with pytest.raises(AdapterError, match="no strategy named"):
            unregister_strategy("reasoning", "never_registered")

    def test_fragment_referencing_an_unknown_field_refuses(self):
        adapter = dspy.ChatAdapter.with_strategies(
            reasoning=strategy.rule(
                predicate=strategy.capability("instruct"),
                fragments=[strategy.fragment("user", "Use {field('sources')} wisely.")],
            )
        )
        with pytest.raises(AdapterError, match="fragment references field 'sources'"):
            adapter.format(CoT, {"question": "Why?"}, capabilities=INSTRUCT)
