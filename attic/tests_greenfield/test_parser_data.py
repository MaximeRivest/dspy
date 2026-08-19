"""Stage A2 tests: the parse-combinator vocabulary (level-1 parse-data)."""

import json
from pathlib import Path

import pytest

import dspy
from dspy.adapters import load_entry, make_adapter, parse
from dspy.adapters.codecs import TEXT_PYTHONISH, coerce_shape
from dspy.adapters.errors import AdapterError, AdapterParseError, EntryError
from dspy.adapters.parse import remove_spans, run_pipeline, validate_pipeline, validate_re2_pattern

EXAMPLES = Path(__file__).resolve().parents[1] / "roadmap" / "adapter-ir-stage" / "examples"


class Judge(dspy.Signature):
    """Grade the response."""

    response: str = dspy.InputField()
    score: int = dspy.OutputField()
    verdict: str = dspy.OutputField()


class Extract(dspy.Signature):
    """Extract the entities and a one-line summary."""

    text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField()
    summary: str = dspy.OutputField()


def run(spec, completion, signature=None):
    return run_pipeline(spec, completion, signature=signature, output_codec=TEXT_PYTHONISH)


# ---------------------------------------------------------------------------
# Authoring helpers build the exact data spellings
# ---------------------------------------------------------------------------


class TestAuthoring:
    def test_pipeline_builder_matches_example_02(self):
        built = parse.pipeline(
            parse.fenced_block(language="json", policy="prefer"),
            parse.alternatives(
                parse.json_object(repair="none"),
                parse.json_object(repair="json_repair"),
            ),
            parse.fields_from_object(unknown_keys="exhaust"),
        )
        want = json.loads((EXAMPLES / "02-json-tolerant" / "adapter-ir.json").read_text())["parser"]
        assert built == want

    def test_regex_builder_matches_example_05(self):
        built = parse.regex(
            "<think>(?P<t>[^<]*)</think>", dialect="re2", mode="findall", group="t", join="\n"
        )
        assert built == {
            "op": "regex",
            "dialect": "re2",
            "pattern": "<think>(?P<t>[^<]*)</think>",
            "mode": "findall",
            "group": "t",
            "join": "\n",
        }


# ---------------------------------------------------------------------------
# Validation: closed vocabulary, RE2 subset
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_op_refuses_listing_the_vocabulary(self):
        with pytest.raises(EntryError, match="unknown combinator 'levitate'"):
            validate_pipeline({"kind": "pipeline", "steps": [{"op": "levitate"}]})

    def test_unknown_option_refuses(self):
        with pytest.raises(EntryError, match="unknown json_object options"):
            validate_pipeline({"kind": "pipeline", "steps": [{"op": "json_object", "mood": "hopeful"}]})

    def test_unknown_enum_value_refuses(self):
        with pytest.raises(EntryError, match="valid values: none, json_repair"):
            validate_pipeline({"kind": "pipeline", "steps": [{"op": "json_object", "repair": "prayer"}]})

    def test_empty_pipeline_refuses(self):
        with pytest.raises(EntryError, match="non-empty 'steps'"):
            validate_pipeline({"kind": "pipeline", "steps": []})

    @pytest.mark.parametrize(
        ("pattern", "construct"),
        [
            ("(?=next)x", "lookahead"),
            ("(?<!no)x", "negative lookbehind"),
            (r"(a)\1", "backreference"),
            (r"(?P<a>x)(?P=a)", "named backreference"),
        ],
    )
    def test_non_re2_patterns_refuse_naming_the_construct(self, pattern, construct):
        with pytest.raises(EntryError, match=construct):
            validate_re2_pattern(pattern, where="test")

    def test_non_compiling_pattern_refuses(self):
        with pytest.raises(EntryError, match="does not compile"):
            validate_re2_pattern("([unclosed", where="test")

    def test_entry_pipeline_requires_a_fields_terminal(self):
        with pytest.raises(EntryError, match="fields terminal"):
            make_adapter(
                name="valueless",
                template=[{"role": "user", "content": "{% for f in inputs %}{f.name}: {f.value}{% endfor %}"}],
                parser=parse.pipeline(parse.fenced_block(language="json")),
            )


# ---------------------------------------------------------------------------
# Combinator semantics
# ---------------------------------------------------------------------------


class TestCombinators:
    def test_fenced_block_prefer_uses_block_when_present(self):
        spec = parse.pipeline(parse.fenced_block(language="json"), parse.json_object(), parse.fields_from_object())
        state = run(spec, 'noise\n```json\n{"summary": "s", "entities": []}\n```\n', signature=Extract)
        assert state.fields == {"summary": "s", "entities": []}

    def test_fenced_block_prefer_falls_back_to_whole_text(self):
        spec = parse.pipeline(parse.fenced_block(language="json"), parse.json_object(), parse.fields_from_object())
        state = run(spec, '{"summary": "s", "entities": ["e"]}', signature=Extract)
        assert state.fields == {"summary": "s", "entities": ["e"]}

    def test_fenced_block_require_refuses_when_absent(self):
        spec = parse.pipeline(parse.fenced_block(language="python", policy="require"))
        with pytest.raises(AdapterParseError, match="no fenced python block"):
            run(spec, "no fences here")

    def test_json_object_strict_refuses_sloppy_json(self):
        with pytest.raises(AdapterParseError, match="not strict JSON"):
            run(parse.pipeline(parse.json_object(repair="none")), '{"a": 1,}')

    def test_json_object_repair_recovers(self):
        state = run(parse.pipeline(parse.json_object(repair="json_repair")), '{"a": 1,}')
        assert state.obj == {"a": 1}

    def test_alternatives_first_success_wins(self):
        spec = parse.pipeline(
            parse.alternatives(parse.json_object(repair="none"), parse.json_object(repair="json_repair")),
            parse.fields_from_object(),
        )
        state = run(spec, '{"summary": "s", "entities": []}', signature=Extract)
        assert state.fields == {"summary": "s", "entities": []}

    def test_alternatives_all_fail_names_each_branch(self):
        spec = parse.pipeline(parse.alternatives(parse.json_object(repair="none"), parse.fenced_block(policy="require")))
        with pytest.raises(AdapterParseError, match="every alternative failed.*json_object.*fenced_block"):
            run(spec, "not json, not fenced")

    def test_fields_from_object_exhausts_unknown_keys(self):
        spec = parse.pipeline(parse.json_object(), parse.fields_from_object(unknown_keys="exhaust"))
        state = run(spec, '{"summary": "s", "entities": [], "mood": "great"}', signature=Extract)
        assert state.fields == {"summary": "s", "entities": []}
        assert state.exhaust == {"mood": "great"}

    def test_fields_from_object_refuse_policy(self):
        spec = parse.pipeline(parse.json_object(), parse.fields_from_object(unknown_keys="refuse"))
        with pytest.raises(AdapterParseError, match="unknown key 'mood'"):
            run(spec, '{"summary": "s", "mood": "great"}', signature=Extract)

    def test_regex_search_fills_named_groups(self):
        spec = parse.pipeline(
            parse.regex("(?m)^SCORE: (?P<score>[0-9]+)/10$", mode="search"),
            parse.fields_from_groups(),
        )
        state = run(spec, "SCORE: 7/10", signature=Judge)
        assert state.fields == {"score": 7}
        assert isinstance(state.fields["score"], int)

    def test_regex_search_miss_refuses(self):
        spec = parse.pipeline(parse.regex("(?P<x>never)", mode="search"))
        with pytest.raises(AdapterParseError, match="matched nothing"):
            run(spec, "text without the pattern")

    def test_regex_findall_group_join(self):
        spec = parse.pipeline(parse.regex("<think>(?P<t>[^<]*)</think>", mode="findall", group="t", join="\n"))
        state = run(spec, "a<think>one</think>b<think>two</think>")
        assert state.value == "one\ntwo"
        assert len(state.spans) == 2

    def test_coerce_step(self):
        spec = parse.pipeline(
            parse.regex("(?P<n>[0-9]+)", mode="findall", group="n", join=""),
            parse.coerce("int"),
        )
        assert run(spec, "answer 42").value == 42

    def test_coerce_unknown_shape_refuses(self):
        with pytest.raises(AdapterError, match="unknown coercion shape"):
            coerce_shape("x", "hologram")

    def test_tool_calls_terminal(self):
        spec = parse.pipeline(
            parse.regex(r"(?m)^!call (?P<name>[a-zA-Z0-9_]+) (?P<args>\{.*\})$", mode="findall"),
            parse.tool_calls(name_group="name", args_group="args"),
        )
        state = run(spec, '!call search {"query": "cats"}\n!call fetch {"url": "x"}')
        calls = state.value.tool_calls
        assert [call.name for call in calls] == ["search", "fetch"]
        assert calls[0].args == {"query": "cats"}

    def test_citations_terminal(self):
        spec = parse.pipeline(
            parse.regex(r"(?P<span>[^.!?]*[.!?])\s*\[(?P<doc>[0-9]+)\]", mode="findall"),
            parse.citations(span_group="span", doc_group="doc"),
        )
        state = run(spec, "Water boils. [1] Ice floats. [2]")
        assert [(c.span, c.doc) for c in state.value] == [("Water boils.", 1), ("Ice floats.", 2)]

    def test_remove_spans(self):
        assert remove_spans("abcdef", [(1, 3), (4, 5)]) == "adf"


# ---------------------------------------------------------------------------
# Whole pipelines as entry parsers
# ---------------------------------------------------------------------------


class TestPipelineParsers:
    def grader(self):
        # Example 08: the authored wire format, as authored parse-DATA.
        return make_adapter(
            name="grader_lines",
            template=[
                {"role": "system", "content": "{instruction}"},
                {
                    "role": "user",
                    "content": "{% for f in inputs separator='\\n\\n' %}{f.name}:\n{f.value}{% endfor %}\n\nReply with exactly two lines:\nSCORE: <n>/10\nVERDICT: <pass|fail>",
                },
            ],
            parser=parse.pipeline(
                parse.regex(
                    "(?m)^SCORE: (?P<score>[0-9]+)/10$\\n^VERDICT: (?P<verdict>pass|fail)$",
                    dialect="re2",
                    mode="search",
                ),
                parse.fields_from_groups(),
            ),
        )

    def test_example_08_parse_preview(self):
        assert self.grader().parse_preview(Judge, "SCORE: 7/10\nVERDICT: pass") == {
            "score": 7,
            "verdict": "pass",
        }

    def test_example_08_matches_the_file(self):
        entry = json.loads((EXAMPLES / "08-custom-regex-parser" / "adapter-ir.json").read_text())
        assert self.grader().dump_entry() == entry

    def test_example_02_loaded_entry_parses_sloppy_output(self):
        entry = json.loads((EXAMPLES / "02-json-tolerant" / "adapter-ir.json").read_text())
        adapter = load_entry(entry)
        completion = 'Here you go:\n```json\n{"entities": ["Ada"], "summary": "One line", "extra": 1,}\n```'
        assert adapter.parse_preview(Extract, completion) == {
            "entities": ["Ada"],
            "summary": "One line",
        }

    def test_pipeline_parse_failure_is_loud(self):
        with pytest.raises(AdapterParseError, match="matched nothing"):
            self.grader().parse_preview(Judge, "I refuse to grade.")
