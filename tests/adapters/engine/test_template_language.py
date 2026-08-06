"""The constrained template language: vocabulary-as-data, teaching errors,
eager parsing, pure rendering, capacity derivation, preview.

Discoverability is normative (spec section 3): the validator, the error
messages, and ``describe_template_language()`` must all read the ONE
vocabulary structure — several tests here assert the error text enumerates
exactly the sets the data structure declares.
"""

import pytest

import dspy
from dspy.adapters._engine.codecs import TEXT_PYTHONISH
from dspy.adapters._engine.template import (
    VOCABULARY,
    ParsedTemplate,
    TemplateError,
    TemplateRenderError,
    declared_capacity,
    describe_template_language,
    parse_content,
    parse_message_template,
    preview,
    render_nodes,
)
from dspy.adapters._engine.template.parser import (
    Text,
)
from dspy.adapters._engine.template.renderer import RenderContext
from dspy.adapters._engine.template.vocabulary import spell_out


class QA(dspy.Signature):
    """Answer the question briefly."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


class Typed(dspy.Signature):
    """Extract."""

    question: str = dspy.InputField(desc="The question.")
    count: int = dspy.OutputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()
    answer: str = dspy.OutputField()


def ctx(signature=QA, mode="schema", **kwargs):
    kwargs.setdefault("input_codec", TEXT_PYTHONISH)
    kwargs.setdefault("output_codec", TEXT_PYTHONISH)
    return RenderContext(signature=signature, mode=mode, **kwargs)


def render(text, **ctx_kwargs):
    return render_nodes(parse_content(text), ctx(**ctx_kwargs))


# ---------------------------------------------------------------------------
# Discoverability: one structure, read everywhere
# ---------------------------------------------------------------------------


def test_describe_template_language_returns_the_vocabulary_copy():
    described = describe_template_language()
    assert described == VOCABULARY
    described["loop_variables"]["bogus"] = "x"
    assert "bogus" not in VOCABULARY["loop_variables"]


def test_vocabulary_matches_spec_loop_variable_set():
    assert set(VOCABULARY["loop_variables"]) == {
        "i",
        "index",
        "name",
        "type",
        "desc",
        "desc_suffix",
        "value",
        "placeholder",
        "typed_placeholder",
        "marker",
        "chat_type_hint",
    }


@pytest.mark.parametrize(
    ("template", "category_key"),
    [
        ("{% for f in outpts %}{f.name}{% endfor %}", "loop_collections"),
        ("{% for f in inputs %}{f.nome}{% endfor %}", "loop_variables"),
        ("{outputs(style='markdown')}", ("aggregate_styles", "outputs")),
        ("{fragments('assistant')}", "fragment_targets"),
        ("{instruction(style='fancy')}", "instruction_styles"),
    ],
)
def test_errors_enumerate_the_vocabulary_set(template, category_key):
    """Every unknown construct names itself AND the valid set from the data."""
    if isinstance(category_key, tuple):
        valid = VOCABULARY[category_key[0]][category_key[1]]
    else:
        valid = VOCABULARY[category_key]
    with pytest.raises(TemplateError) as excinfo:
        parse_content(template)
    assert spell_out(valid) in str(excinfo.value)


def test_unknown_slot_function_names_the_valid_set():
    with pytest.raises(TemplateError, match="instruction, inputs, outputs, demos, history, fragments"):
        parse_content("{shenanigans()}")


def test_unknown_directive_role_names_both_sets():
    with pytest.raises(TemplateError) as excinfo:
        parse_message_template([{"role": "demoz"}])
    message = str(excinfo.value)
    assert spell_out(VOCABULARY["message_roles"]) in message
    assert spell_out(VOCABULARY["directive_roles"]) in message


# ---------------------------------------------------------------------------
# Eager parsing and refusals
# ---------------------------------------------------------------------------


def test_plain_text_parses_to_text_node():
    assert parse_content("hello world") == (Text("hello world"),)


def test_brace_escapes():
    assert render("{{question}} and }}") == "{question} and }"


def test_lone_closing_brace_refuses():
    with pytest.raises(TemplateError, match="literal"):
        parse_content("oops }")


def test_open_brace_without_slot_refuses():
    with pytest.raises(TemplateError, match="literal brace"):
        parse_content("{ not a slot")


def test_unclosed_loop_refuses():
    with pytest.raises(TemplateError, match="endfor"):
        parse_content("{% for f in inputs %}{f.name}")


def test_endfor_without_for_refuses():
    with pytest.raises(TemplateError, match="without an open"):
        parse_content("{% endfor %}")


def test_nested_loops_refuse():
    with pytest.raises(TemplateError, match="nested"):
        parse_content("{% for f in inputs %}{% for g in outputs %}{g.name}{% endfor %}{% endfor %}")


def test_unknown_block_tag_refuses():
    with pytest.raises(TemplateError, match="valid block tags: for, endfor"):
        parse_content("{% if x %}")


def test_unknown_loop_option_refuses():
    with pytest.raises(TemplateError, match="separator, strip"):
        parse_content("{% for f in inputs reverse %}{f.name}{% endfor %}")


def test_unknown_separator_escape_refuses():
    with pytest.raises(TemplateError, match=r"unknown escape sequence"):
        parse_content(r"{% for f in inputs separator='\q' %}{f.name}{% endfor %}")


def test_loop_variable_reference_outside_loop_refuses():
    with pytest.raises(TemplateError, match="outside a loop block"):
        parse_content("{f.name}")


def test_wrong_loop_variable_name_refuses():
    with pytest.raises(TemplateError, match="iterates as 'f'"):
        parse_content("{% for f in inputs %}{g.name}{% endfor %}")


def test_bare_aggregate_slot_teaches_the_call_form():
    with pytest.raises(TemplateError, match=r"\{inputs\(style='...'\)\}"):
        parse_content("{inputs}")


def test_fragments_requires_quoted_target():
    with pytest.raises(TemplateError, match="one quoted target"):
        parse_content("{fragments(system)}")


def test_duplicate_fragment_target_refuses_across_messages():
    with pytest.raises(TemplateError, match="duplicate fragments target"):
        parse_message_template(
            [
                {"role": "system", "content": "{fragments('system')}"},
                {"role": "user", "content": "{fragments('system')}"},
            ]
        )


def test_fragments_inside_loop_refuses():
    with pytest.raises(TemplateError, match="positional"):
        parse_content("{% for f in inputs %}{fragments('user')}{% endfor %}")


def test_wrap_only_valid_with_xml_style():
    with pytest.raises(TemplateError, match="wrap= is only meaningful"):
        parse_content("{outputs(style='chat', wrap='response')}")


def test_directive_requires_user_and_assistant_together():
    with pytest.raises(TemplateError, match="together, or neither"):
        parse_message_template([{"role": "demos", "user": "{question}"}])


def test_unknown_message_keys_refuse():
    with pytest.raises(TemplateError, match="valid keys: role, content"):
        parse_message_template([{"role": "system", "content": "x", "contnet": "y"}])


def test_at_most_one_demos_directive():
    with pytest.raises(TemplateError, match="at most one"):
        parse_message_template([{"role": "demos"}, {"role": "demos"}])


# ---------------------------------------------------------------------------
# Rendering semantics
# ---------------------------------------------------------------------------


def test_value_slot_renders_codec_value_and_missing_renders_empty():
    assert render("Q: {question}", mode="user_values", values={"question": "hi"}) == "Q: hi"
    assert render("Q: {question}", mode="user_values", values={}) == "Q: "


def test_unknown_value_slot_refuses_naming_fields():
    with pytest.raises(TemplateRenderError, match="question, answer"):
        render("{nonfield}", mode="user_values", values={})


def test_instruction_raw_and_indented():
    assert render("{instruction}") == "Answer the question briefly."
    assert render("X{instruction(style='indented')}") == "X\n        Answer the question briefly."


def test_loop_schema_mode_iterates_all_fields_with_index():
    text = "{% for f in outputs separator=', ' %}{f.i}:{f.name}{% endfor %}"
    assert render(text, signature=Typed) == "1:count, 2:tool_calls, 3:answer"


def test_loop_user_values_mode_iterates_present_fields_only():
    text = "{% for f in inputs separator='\\n\\n' %}{f.marker}\n{f.value}{% endfor %}"
    out = render(text, signature=Typed, mode="user_values", values={"question": "hi"})
    assert out == "[[ ## question ## ]]\nhi"
    assert render(text, signature=Typed, mode="user_values", values={}) == ""


def test_outputs_loop_in_user_position_iterates_all_fields():
    """The output-requirements enumeration names every output field even
    though the user turn's values carry only inputs."""
    text = "{% for f in outputs separator=', then ' %}`[[ ## {f.name} ## ]]`{f.chat_type_hint}{% endfor %}"
    out = render(text, signature=Typed, mode="user_values", values={"question": "hi"})
    assert out.startswith("`[[ ## count ## ]]` (must be formatted as a valid Python int), then `[[ ## tool_calls ## ]]`")
    assert out.endswith(", then `[[ ## answer ## ]]`")


def test_loop_assistant_values_mode_renders_missing_field_message():
    text = "{% for f in outputs separator='; ' %}{f.name}={f.value}{% endfor %}"
    out = render(
        text,
        signature=QA,
        mode="assistant_values",
        values={},
        missing_field_message="Not supplied. ",
    )
    assert out == "answer=Not supplied. "


def test_loop_strip_flag_strips_the_joined_result():
    text = "{% for f in outputs strip %}{f.name}={f.value}{% endfor %}"
    out = render(text, signature=QA, mode="assistant_values", values={}, missing_field_message="pad ")
    assert out == "answer=pad"


def test_loop_value_in_schema_position_without_values_refuses():
    with pytest.raises(TemplateRenderError, match="needs call values"):
        render("{% for f in inputs %}{f.value}{% endfor %}", values=None)


def test_triple_brace_literal_placeholder():
    text = "{% for f in inputs %}{{{f.name}}}{% endfor %}"
    assert render(text) == "{question}"


def test_typed_placeholder_and_chat_type_hint():
    text = "{% for f in outputs separator='|' %}{f.typed_placeholder}{% endfor %}"
    out = render(text, signature=Typed)
    assert out.startswith("{count}")
    assert "# note: the value you produce must be a single int value" in out

    hint = "{% for f in outputs separator='|' %}{f.chat_type_hint}{% endfor %}"
    rendered = render(hint, signature=Typed)
    parts = rendered.split("|")
    assert parts[0] == " (must be formatted as a valid Python int)"
    assert parts[1] == ' (must be a JSON object like {"tool_calls": [{"name": "...", "args": {...}}]})'
    assert parts[2] == ""


def test_desc_and_desc_suffix():
    text = "{% for f in inputs %}[{f.desc}]({f.desc_suffix}){% endfor %}"
    assert render(text, signature=Typed) == "[The question.]( The question.)"
    # Unset descriptions: bare desc is empty, suffix keeps the historical space.
    assert render(text, signature=QA) == "[]( )"


def test_empty_fragment_slot_costs_zero_bytes_line_swallow():
    text = "[[ ## completed ## ]]\n{fragments('system')}\nIn adhering"
    assert render(text) == "[[ ## completed ## ]]\nIn adhering"


def test_filled_fragment_slot_renders_on_its_line():
    text = "[[ ## completed ## ]]\n{fragments('system')}\nIn adhering"
    out = render(text, fragments={"system": ["Cite your sources.", "Be brief."]})
    assert out == "[[ ## completed ## ]]\nCite your sources.\nBe brief.\nIn adhering"


def test_inline_empty_fragment_renders_empty():
    assert render("a {fragments('user')} b") == "a  b"


# ---------------------------------------------------------------------------
# Aggregate slots
# ---------------------------------------------------------------------------


def test_inputs_aggregate_styles():
    values = {"question": "hi"}
    assert render("{inputs()}", mode="user_values", values=values) == "question: hi"
    assert render("{inputs(style='xml')}", mode="user_values", values=values) == "<question>hi</question>"
    assert render("{inputs(style='chat')}", mode="user_values", values=values) == "[[ ## question ## ]]\nhi"
    assert render("{inputs(style='json')}", mode="user_values", values=values) == '{\n  "question": "hi"\n}'


def test_outputs_aggregate_styles():
    assert render("{outputs()}") == "1. `answer` (str):"
    assert render("{outputs(style='xml')}", signature=Typed).splitlines()[-1] == "<answer>answer</answer>"
    wrapped = render("{outputs(style='xml', wrap='response')}")
    assert wrapped == "<response>\n  <answer>answer</answer>\n</response>"
    assert render("{outputs(style='chat')}") == "[[ ## answer ## ]]\n{answer}"
    assert '"type": "integer"' in render("{outputs(style='schema')}", signature=Typed)


def test_outputs_json_object_is_placeholders_in_schema_and_values_in_assistant():
    schema_side = render("{outputs(style='json_object')}")
    assert schema_side == '{\n  "answer": "{answer}"\n}'
    value_side = render(
        "{outputs(style='json_object')}", mode="assistant_values", values={"answer": "Paris"}
    )
    assert value_side == '{\n  "answer": "Paris"\n}'


def test_demos_aggregate_styles():
    demos = ({"question": "1+1?", "answer": "2"},)
    assert render("{demos()}", demos=demos) == "Example 1:\n  question: 1+1?\n  answer: 2"
    assert render("{demos(style='chat')}", demos=demos) == "[[ ## question ## ]]\n1+1?\n\n[[ ## answer ## ]]\n2"
    assert '"question": "1+1?"' in render("{demos(style='json')}", demos=demos)
    assert render("{demos()}", demos=()) == ""


def test_history_aggregate_styles():
    history = ({"question": "hi", "answer": "hello"},)
    assert render("{history()}", history=history) == "Turn 1:\n  question: hi\n  answer: hello"
    assert render("{history(style='xml')}", history=history) == (
        "<turn>\n  <question>hi</question>\n  <answer>hello</answer>\n</turn>"
    )


# ---------------------------------------------------------------------------
# Capacity derivation
# ---------------------------------------------------------------------------

CHAT_LIKE = [
    {"role": "system", "content": "{% for f in outputs %}{f.name}{% endfor %}\n{fragments('system')}\n{instruction}"},
    {"role": "demos", "user": "{% for f in inputs %}{f.value}{% endfor %}", "assistant": "x"},
    {"role": "history"},
    {"role": "user", "content": "{% for f in inputs %}{f.value}{% endfor %}\n\n{fragments('user')}\nRespond."},
]


def test_declared_capacity_on_a_chat_like_template():
    capacity = declared_capacity(parse_message_template(CHAT_LIKE))
    assert capacity.iterates_inputs and capacity.iterates_outputs
    assert capacity.fragment_targets == {"system", "user"}
    assert capacity.hosts_demos and capacity.hosts_history
    assert capacity.hosts_role_textually("reasoning")
    assert capacity.hosts_role_textually("history")


def test_declared_capacity_on_a_static_template():
    template = parse_message_template(
        [
            {"role": "system", "content": "Summarize concisely."},
            {"role": "user", "content": "{question}"},
        ]
    )
    capacity = declared_capacity(template)
    assert not capacity.iterates_outputs
    assert capacity.field_slots == {"question"}
    assert not capacity.hosts_role_textually("reasoning")
    assert capacity.hosts_role_textually("media")
    assert not capacity.hosts_role_textually("history")
    with pytest.raises(ValueError, match="valid roles"):
        capacity.hosts_role_textually("vibes")


# ---------------------------------------------------------------------------
# Preview: full message lists, no LM
# ---------------------------------------------------------------------------


def test_preview_renders_full_message_list_purely():
    template = [
        {"role": "system", "content": "{instruction}"},
        {
            "role": "demos",
            "user": "{% for f in inputs separator='\\n\\n' %}{f.marker}\n{f.value}{% endfor %}",
            "assistant": "{% for f in outputs separator='\\n\\n' %}{f.marker}\n{f.value}{% endfor %}",
        },
        {"role": "history"},
        {"role": "user", "content": "{% for f in inputs separator='\\n\\n' %}{f.marker}\n{f.value}{% endfor %}"},
    ]

    class Chatty(dspy.Signature):
        """Chat."""

        history: dspy.History = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    messages = preview(
        template,
        Chatty,
        demos=[{"history": dspy.History(messages=[]), "question": "Hi?", "answer": "Hello!"}],
        inputs={"question": "Now?", "history": dspy.History(messages=[{"question": "Q1", "answer": "A1"}])},
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert messages[0]["content"] == "Chat."
    # The demo pair renders through the directive templates.
    assert messages[1]["content"].startswith("[[ ## history ## ]]")
    assert messages[2]["content"] == "[[ ## answer ## ]]\nHello!"
    # History turns expand through the demo patterns, minus the history field.
    assert messages[3]["content"] == "[[ ## question ## ]]\nQ1"
    assert messages[4]["content"] == "[[ ## answer ## ]]\nA1"
    assert messages[5]["content"] == "[[ ## question ## ]]\nNow?"

    assert messages == preview(
        template,
        Chatty,
        demos=[{"history": dspy.History(messages=[]), "question": "Hi?", "answer": "Hello!"}],
        inputs={"question": "Now?", "history": dspy.History(messages=[{"question": "Q1", "answer": "A1"}])},
    )


def test_preview_accepts_parsed_templates():
    parsed = parse_message_template([{"role": "user", "content": "{question}"}])
    assert isinstance(parsed, ParsedTemplate)
    messages = preview(parsed, QA, inputs={"question": "hi"})
    assert messages == [{"role": "user", "content": "hi"}]
