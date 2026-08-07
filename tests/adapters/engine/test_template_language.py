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
from dspy.adapters._engine.template.vocabulary import RESERVED_SLOT_NAMES, spell_out


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
        "role",
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


def test_same_quote_escapes_lex_in_loop_separators():
    """The vocabulary documents \\' and \\\" — they must work in the quote
    position where they are needed, not only cross-quoted."""
    (loop,) = parse_content(r"{% for f in inputs separator='don\'t' %}{f.name}{% endfor %}")
    assert loop.separator == "don't"
    (loop,) = parse_content('{% for f in inputs separator="say \\"hi\\"" %}{f.name}{% endfor %}')
    assert loop.separator == 'say "hi"'
    (loop,) = parse_content("{% for f in inputs separator='a\\'b\"c' %}{f.name}{% endfor %}")
    assert loop.separator == "a'b\"c"


def test_percent_is_legal_inside_quoted_loop_options():
    (loop,) = parse_content("{% for f in inputs separator=' % ' %}{f.name}{% endfor %}")
    assert loop.separator == " % "
    (loop,) = parse_content("{% for f in inputs separator='100%' %}{f.name}{% endfor %}")
    assert loop.separator == "100%"


def test_loop_option_arity_errors_state_the_actual_problem():
    with pytest.raises(TemplateError, match="'separator' requires a quoted value"):
        parse_content("{% for f in inputs separator %}{f.name}{% endfor %}")
    with pytest.raises(TemplateError, match="'separator' requires a quoted value"):
        parse_content("{% for f in inputs separator=- %}{f.name}{% endfor %}")
    with pytest.raises(TemplateError, match="'strip' takes no value"):
        parse_content("{% for f in inputs strip='x' %}{f.name}{% endfor %}")


def test_every_declared_loop_option_is_accepted():
    """Acceptance reads the vocabulary data: every declared option parses
    in its declared arity."""
    for key, entry in VOCABULARY["loop_options"].items():
        spelled = f"{key}='x'" if entry["takes_value"] else key
        parse_content(f"{{% for f in inputs {spelled} %}}{{f.name}}{{% endfor %}}")


def test_same_quote_escapes_lex_in_call_kwargs():
    (slot,) = parse_content(r"{outputs(style='xml', wrap='don\'t')}")
    assert slot.wrap == "don't"


def test_loop_variable_reference_outside_loop_refuses():
    with pytest.raises(TemplateError, match="outside a loop block"):
        parse_content("{f.name}")


def test_wrong_loop_variable_name_refuses():
    with pytest.raises(TemplateError, match="iterates as 'f'"):
        parse_content("{% for f in inputs %}{g.name}{% endfor %}")


def test_bare_aggregate_slot_teaches_the_call_form():
    with pytest.raises(TemplateError, match=r"\{inputs\(style='...'\)\}"):
        parse_content("{inputs}")


def test_bare_aggregate_and_fragments_errors_name_the_reserved_collision():
    """A field named after a reserved slot is unreferenceable as a bare
    value slot — the parse refusal names the collision, not just the call
    form (spec section 3, reserved names)."""
    for spelling in ("{outputs}", "{demos}", "{history}", "{inputs}", "{fragments}"):
        with pytest.raises(TemplateError, match="reserved name") as excinfo:
            parse_content(spelling)
        assert spell_out(RESERVED_SLOT_NAMES) in str(excinfo.value)


def test_field_escape_spelling_renders_any_field():
    """{field('name')} is the unambiguous value-slot spelling (spec
    section 3): equivalent to the bare form for ordinary names, the ONLY
    spelling for reserved-colliding names."""
    assert render("{field('question')}", mode="user_values", values={"question": "hm?"}) == "hm?"
    sig = dspy.Signature("inputs, instruction -> outputs", "Do it.")
    rendered = render(
        "{field('inputs')}|{field('instruction')}",
        signature=sig,
        mode="user_values",
        values={"inputs": "A", "instruction": "B"},
    )
    assert rendered == "A|B"


def test_field_escape_spelling_unknown_name_refuses_at_render():
    with pytest.raises(TemplateRenderError, match="unknown value slot"):
        render("{field('mystery')}", mode="user_values", values={})


def test_bare_field_teaches_the_call_form():
    with pytest.raises(TemplateError, match=r"\{field\('name'\)\}"):
        parse_content("{field}")


def test_field_call_form_argument_errors_teach():
    with pytest.raises(TemplateError, match="one quoted field name"):
        parse_content("{field(question)}")
    with pytest.raises(TemplateError, match="field identifier"):
        parse_content("{field('not a name')}")


def test_reserved_collision_errors_name_the_field_escape():
    """Every reserved-collision refusal must teach {field('name')} as the
    way out (spec section 3)."""
    for spelling in ("{outputs}", "{fragments}", "{field}"):
        with pytest.raises(TemplateError, match=r"\{field\(") :
            parse_content(spelling)
    sig = dspy.Signature("instruction -> response", "Follow the instruction.")
    with pytest.raises(TemplateRenderError, match=r"\{field\('instruction'\)\}"):
        preview([{"role": "user", "content": "{instruction}"}], sig, inputs={"instruction": "Write a haiku."})


def test_bare_instruction_refuses_when_a_field_is_named_instruction():
    """Alpaca-shape signatures: bare {instruction} must not silently render
    the docstring where the author plausibly meant the field."""
    sig = dspy.Signature("instruction -> response", "Follow the instruction.")
    with pytest.raises(TemplateRenderError, match="field named 'instruction'") as excinfo:
        preview([{"role": "user", "content": "{instruction}"}], sig, inputs={"instruction": "Write a haiku."})
    assert spell_out(RESERVED_SLOT_NAMES) in str(excinfo.value)


def test_instruction_call_form_stays_unambiguous_under_collision():
    sig = dspy.Signature("instruction -> response", "Follow the instruction.")
    messages = preview(
        [{"role": "user", "content": "{instruction(style='raw')}"}], sig, inputs={"instruction": "Write a haiku."}
    )
    assert messages == [{"role": "user", "content": "Follow the instruction."}]


def test_bare_instruction_renders_the_docstring_without_a_collision():
    assert render("{instruction}") == "Answer the question briefly."


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


# ---------------------------------------------------------------------------
# Section blocks: the join-then-strip region shape
# ---------------------------------------------------------------------------

_SECTION_TEXT = (
    "{% section strip %}\n"
    "Head.\n"
    "\n"
    "{% for f in inputs separator='\\n\\n' strip %}\n"
    "<{f.name}>\n"
    "{% endfor %}\n"
    "\n"
    "{% for f in outputs separator='\\n\\n' strip %}\n"
    "<{f.name}>\n"
    "{% endfor %}\n"
    "{% endsection %}\n"
    "Tail"
)


def test_section_strip_collapses_a_trailing_empty_loop_with_its_separators():
    no_outputs = dspy.make_signature({"question": (str, dspy.InputField())}, "x")
    assert render(_SECTION_TEXT, signature=no_outputs) == "Head.\n\n<question>\nTail"
    no_fields = dspy.make_signature({}, "x")
    assert render(_SECTION_TEXT, signature=no_fields) == "Head.\nTail"


def test_section_keeps_interior_empty_loop_separators():
    """An interior empty subsection keeps its blank-line separators — the
    legacy join inserts them regardless; only leading/trailing collapse."""
    no_inputs = dspy.make_signature({"answer": (str, dspy.OutputField())}, "x")
    assert render(_SECTION_TEXT, signature=no_inputs) == "Head.\n\n\n\n<answer>\nTail"
    assert render(_SECTION_TEXT, signature=QA) == "Head.\n\n<question>\n\n<answer>\nTail"


def test_section_refusals_teach():
    with pytest.raises(TemplateError, match="endsection"):
        parse_content("{% section strip %}x")
    with pytest.raises(TemplateError, match="without an open"):
        parse_content("x {% endsection %}")
    with pytest.raises(TemplateError, match=r"expected \{% section strip %\}"):
        parse_content("{% section %}x{% endsection %}")
    with pytest.raises(TemplateError, match="do not nest"):
        parse_content("{% section strip %}{% section strip %}x{% endsection %}{% endsection %}")
    with pytest.raises(TemplateError, match="not valid inside loop blocks"):
        parse_content("{% for f in inputs %}{% section strip %}x{% endsection %}{% endfor %}")
    with pytest.raises(TemplateError, match="section blocks"):
        parse_content("{% section strip %}{fragments('system')}{% endsection %}")


def test_section_appears_in_capacity_derivation():
    template = parse_message_template([{"role": "user", "content": "{% section strip %}\n{% for f in inputs %}{f.name}{% endfor %}\n{% endsection %}"}])
    assert declared_capacity(template).iterates_inputs


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


def test_every_declared_aggregate_style_renders():
    """The renderer covers exactly the vocabulary's declared styles — a
    style admitted to the data without renderer support must refuse, never
    silently render default-style bytes."""
    demos = ({"question": "1+1?", "answer": "2"},)
    history = ({"question": "hi", "answer": "hello"},)
    for kind, styles in VOCABULARY["aggregate_styles"].items():
        for style in styles:
            render(
                f"{{{kind}(style='{style}')}}",
                mode="user_values",
                values={"question": "hi"},
                demos=demos,
                history=history,
            )


def test_unknown_aggregate_style_refuses_at_render_naming_the_valid_set():
    from dspy.adapters._engine.template.parser import AggregateSlot

    for kind in VOCABULARY["aggregate_styles"]:
        node = AggregateSlot(kind=kind, style="martian")
        with pytest.raises(TemplateRenderError) as excinfo:
            render_nodes((node,), ctx(mode="user_values", values={}, demos=(), history=()))
        message = str(excinfo.value)
        assert "'martian'" in message
        assert spell_out(VOCABULARY["aggregate_styles"][kind]) in message


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
    assert not capacity.hosts_role_textually("history")
    with pytest.raises(ValueError, match="valid roles"):
        capacity.hosts_role_textually("vibes")


def test_media_and_tools_capacity_is_per_field():
    """A field lands textually only where an inputs iteration or its own
    slot places it — an unrelated slot must not claim hosting for a field
    that provably reaches no message."""
    template = parse_message_template([{"role": "user", "content": "{question}"}])
    capacity = declared_capacity(template)
    assert capacity.hosts_role_textually("media", field="question")
    assert not capacity.hosts_role_textually("media", field="photo")
    assert not capacity.hosts_role_textually("tools", field="tools")
    with pytest.raises(ValueError, match="per-field"):
        capacity.hosts_role_textually("media")

    iterating = declared_capacity(
        parse_message_template([{"role": "user", "content": "{% for f in inputs %}{f.value}{% endfor %}"}])
    )
    assert iterating.hosts_role_textually("media", field="photo")


def test_output_only_slots_claim_no_media_hosting():
    """An assistant-prefill template whose only slot is an output field has
    nowhere for any input to land."""
    template = parse_message_template(
        [
            {"role": "system", "content": "Answer."},
            {"role": "assistant", "content": "{answer}"},
        ]
    )
    capacity = declared_capacity(template)
    assert not capacity.hosts_role_textually("media", field="photo")
    assert capacity.field_slots == {"answer"}


def test_declared_capacity_descends_into_directive_patterns():
    template = parse_message_template(
        [
            {"role": "system", "content": "Be brief."},
            {
                "role": "demos",
                "user": "{% for f in inputs %}{f.marker}\n{f.value}{% endfor %}\n{secret_reasoning}",
                "assistant": "{% for f in outputs %}{f.marker}\n{f.value}{% endfor %}",
            },
            {"role": "user", "content": "{question}"},
        ]
    )
    capacity = declared_capacity(template)
    # The example lane is visible as its own data...
    assert capacity.directive_iterates_inputs and capacity.directive_iterates_outputs
    assert capacity.directive_field_slots == {"secret_reasoning"}
    # ...and never counts as live-lane hosting: a demo pattern cannot
    # place the live call's fields.
    assert not capacity.iterates_inputs and not capacity.iterates_outputs
    assert capacity.field_slots == {"question"}
    assert not capacity.hosts_role_textually("reasoning")


def test_bare_directives_declare_their_default_pattern_capacity():
    template = parse_message_template(
        [
            {"role": "system", "content": "S"},
            {"role": "demos"},
            {"role": "user", "content": "{question}"},
        ]
    )
    capacity = declared_capacity(template)
    assert capacity.hosts_demos
    assert capacity.directive_iterates_inputs and capacity.directive_iterates_outputs


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


# ---------------------------------------------------------------------------
# Bare directives: default turn patterns, zero items render as nothing
# ---------------------------------------------------------------------------


def test_bare_demos_directive_expands_through_the_default_marker_patterns():
    template = [
        {"role": "system", "content": "S"},
        {"role": "demos"},
        {"role": "user", "content": "{question}"},
    ]
    messages = preview(template, QA, demos=[{"question": "1+1?", "answer": "2"}], inputs={"question": "hi"})
    assert messages == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "[[ ## question ## ]]\n1+1?"},
        {"role": "assistant", "content": "[[ ## answer ## ]]\n2"},
        {"role": "user", "content": "hi"},
    ]


def test_bare_directives_with_zero_demos_and_turns_render_as_no_op():
    template = [
        {"role": "system", "content": "S"},
        {"role": "demos"},
        {"role": "history"},
        {"role": "user", "content": "{question}"},
    ]
    assert preview(template, QA, inputs={"question": "hi"}) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "hi"},
    ]


def test_bare_directive_defaults_key_on_the_parser_binding():
    """Rendering through a preset context (parser= given), a bare directive
    demonstrates the shape that parser reads back (spec section 3); chat
    markers remain the no-preset fallback."""
    template = [
        {"role": "system", "content": "S"},
        {"role": "demos"},
        {"role": "user", "content": "{question}"},
    ]
    demos = [{"question": "1+1?", "answer": "2"}]

    json_messages = preview(template, QA, demos=demos, inputs={"question": "hi"}, parser="json")
    assert json_messages[1] == {"role": "user", "content": "[[ ## question ## ]]\n1+1?"}
    assert json_messages[2] == {"role": "assistant", "content": '{\n  "answer": "2"\n}'}

    xml_messages = preview(template, QA, demos=demos, inputs={"question": "hi"}, parser="xml")
    assert xml_messages[1] == {"role": "user", "content": "<question>\n1+1?\n</question>"}
    assert xml_messages[2] == {"role": "assistant", "content": "<answer>\n2\n</answer>"}

    full_text_messages = preview(template, QA, demos=demos, inputs={"question": "hi"}, parser="full_text")
    assert full_text_messages[2] == {"role": "assistant", "content": "2"}

    for parser in (None, "chat"):
        messages = preview(template, QA, demos=demos, inputs={"question": "hi"}, parser=parser)
        assert messages[1] == {"role": "user", "content": "[[ ## question ## ]]\n1+1?"}
        assert messages[2] == {"role": "assistant", "content": "[[ ## answer ## ]]\n2"}


def test_authored_directive_patterns_ignore_the_parser_key():
    template = [
        {"role": "system", "content": "S"},
        {"role": "demos", "user": "U {question}", "assistant": "A {answer}"},
        {"role": "user", "content": "{question}"},
    ]
    messages = preview(template, QA, demos=[{"question": "q", "answer": "a"}], inputs={"question": "hi"}, parser="xml")
    assert messages[1] == {"role": "user", "content": "U q"}
    assert messages[2] == {"role": "assistant", "content": "A a"}


def test_authored_demo_patterns_get_no_injected_preamble():
    """Persona repro (few-shot user, bug 2): an authored user= pattern gets
    exactly what the author wrote — no incomplete-demo prose injected (D-δ:
    the preamble is directive data, absent unless declared)."""
    template = [
        {"role": "system", "content": "S"},
        {"role": "demos", "user": "Q: {question} [{context}]", "assistant": "A: {answer}"},
        {"role": "user", "content": "{question}"},
    ]
    sig = dspy.Signature("question, context -> answer")
    incomplete = [{"question": "Sky color?", "answer": "blue"}]  # no context: incomplete
    messages = preview(template, sig, demos=incomplete, inputs={"question": "x", "context": "y"})
    assert messages[1]["content"] == "Q: Sky color? []"
    assert "This is an example of the task" not in str(messages)


def test_demos_directive_preamble_is_data():
    template = [
        {"role": "system", "content": "S"},
        {"role": "demos", "user": "Q: {question}", "assistant": "A: {answer}", "preamble": "EXAMPLE (partial):"},
        {"role": "user", "content": "{question}"},
    ]
    sig = dspy.Signature("question, context -> answer")
    incomplete = [{"question": "Sky?", "answer": "blue"}]
    complete = [{"question": "1+1?", "context": "math", "answer": "2"}]
    messages = preview(template, sig, demos=incomplete + complete, inputs={"question": "x", "context": "y"})
    assert messages[1]["content"] == "EXAMPLE (partial):\n\nQ: Sky?"  # incomplete demo gets the preamble
    assert messages[3]["content"] == "Q: 1+1?"  # complete demo does not


def test_demos_directive_preamble_key_validates():
    with pytest.raises(TemplateError, match="preamble"):
        parse_message_template([{"role": "demos", "preamble": 42}, {"role": "user", "content": "x"}])
    with pytest.raises(TemplateError, match="valid keys: role, user, assistant"):
        parse_message_template([{"role": "history", "preamble": "x"}, {"role": "user", "content": "x"}])


def test_bare_history_directive_no_longer_inherits_demo_patterns():
    """Persona repro (few-shot user): patterned demos must not leak into a
    bare history directive — each falls back to its own parser-keyed
    default (D-δ; the inheritance rule is retired)."""

    class Chatty(dspy.Signature):
        """Chat."""

        history: dspy.History = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    template = [
        {"role": "system", "content": "S"},
        {"role": "demos", "user": "DEMO-Q: {question}", "assistant": "DEMO-A: {answer}"},
        {"role": "history"},
        {"role": "user", "content": "{question}"},
    ]
    messages = preview(
        template,
        Chatty,
        inputs={"question": "Now?", "history": dspy.History(messages=[{"question": "past q", "answer": "past a"}])},
    )
    assert messages[1] == {"role": "user", "content": "[[ ## question ## ]]\npast q"}
    assert messages[2] == {"role": "assistant", "content": "[[ ## answer ## ]]\npast a"}
    assert "DEMO-Q" not in str(messages)


def test_f_role_loop_variable_renders_the_semantic_role():
    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        thoughts: str = dspy.OutputField(role="reasoning")
        answer: str = dspy.OutputField()

    out = preview(
        [{"role": "user", "content": "{% for f in outputs separator='\\n' %}{f.name}={f.role}{% endfor %}"}],
        Sig,
        inputs={"question": "q"},
    )
    assert out[0]["content"] == "thoughts=reasoning\nanswer=plain"


def test_duplicate_loop_options_refuse_with_a_teaching_error():
    """Matching call-kwargs behavior (spec section 3): each loop option
    appears once; last-wins lexing is gone."""
    with pytest.raises(TemplateError, match="duplicate loop option 'separator'"):
        parse_content("{% for f in inputs separator=',' separator=';' %}{f.name}{% endfor %}")
    with pytest.raises(TemplateError, match="duplicate loop option 'strip'"):
        parse_content("{% for f in inputs strip strip %}{f.name}{% endfor %}")


def test_orphan_history_directive_expands_turns_through_the_default_patterns():
    class Chatty(dspy.Signature):
        """Chat."""

        history: dspy.History = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    template = [
        {"role": "system", "content": "S"},
        {"role": "history"},
        {"role": "user", "content": "{question}"},
    ]
    messages = preview(
        template,
        Chatty,
        inputs={"question": "Now?", "history": dspy.History(messages=[{"question": "Q1", "answer": "A1"}])},
    )
    assert messages == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "[[ ## question ## ]]\nQ1"},
        {"role": "assistant", "content": "[[ ## answer ## ]]\nA1"},
        {"role": "user", "content": "Now?"},
    ]


# ---------------------------------------------------------------------------
# Walker message-role asymmetry: declared, pinned
# ---------------------------------------------------------------------------


def test_walker_user_messages_join_then_strip_and_drop_when_empty():
    """User turns assemble through the historical join-then-strip and are
    omitted when empty; system/assistant emit verbatim, always (spec
    section 3, user-turn assembly)."""
    assert preview([{"role": "user", "content": "  {question}  "}], QA, inputs={"question": "hi"}) == [
        {"role": "user", "content": "hi"}
    ]
    messages = preview(
        [{"role": "system", "content": "  S  "}, {"role": "user", "content": "{question}"}], QA, inputs={}
    )
    assert messages == [{"role": "system", "content": "  S  "}]
    messages = preview(
        [{"role": "assistant", "content": "  {question}  "}, {"role": "user", "content": "x"}],
        QA,
        inputs={"question": "hi"},
    )
    assert messages == [{"role": "assistant", "content": "  hi  "}, {"role": "user", "content": "x"}]


# ---------------------------------------------------------------------------
# Preview == engine: schema positions render without call values
# ---------------------------------------------------------------------------


def test_preview_refuses_loop_value_in_schema_position_like_the_engine():
    """{f.value} in a system message refuses in preview exactly as the
    engine delegation context (schema mode, values=None) does."""
    hostile = [
        {"role": "system", "content": "Context:\n{% for f in inputs %}{f.value}{% endfor %}"},
        {"role": "user", "content": "{question}"},
    ]
    with pytest.raises(TemplateRenderError, match="schema positions render without"):
        preview(hostile, QA, inputs={"question": "CALL_VALUE"})
    parsed = parse_message_template(hostile)
    with pytest.raises(TemplateRenderError, match="schema positions render without"):
        render_nodes(parsed.messages[0].nodes, ctx())


def test_loop_value_refuses_in_schema_mode_even_with_values_present():
    with pytest.raises(TemplateRenderError, match="schema positions render without"):
        render("{% for f in inputs %}{f.value}{% endfor %}", values={"question": "leak"})


def test_bare_value_slot_in_schema_position_refuses_like_f_value():
    """Persona repro (exact-control + generic-template authors): `{question}`
    in a system message must refuse exactly as `{f.value}` does — never
    silently render empty with the value in hand (D-δ). Preview and the
    engine schema context refuse identically."""
    template = [
        {"role": "system", "content": "Context: {question}"},
        {"role": "user", "content": "{question}"},
    ]
    parsed = parse_message_template(template)
    with pytest.raises(TemplateRenderError, match="schema positions .*render without"):
        preview(parsed, QA, inputs={"question": "SECRET-INPUT"})
    with pytest.raises(TemplateRenderError, match="schema positions .*render without"):
        render_nodes(parsed.messages[0].nodes, ctx())


def test_unknown_slot_in_schema_position_does_not_claim_availability():
    """The unknown-slot error is mode-aware (D-δ): in a schema position it
    must not claim the field is 'available here'."""
    parsed = parse_message_template([{"role": "system", "content": "{nosuch}"}, {"role": "user", "content": "x"}])
    with pytest.raises(TemplateRenderError) as excinfo:
        preview(parsed, QA, inputs={})
    message = str(excinfo.value)
    assert "declared fields" in message
    assert "available here" not in message


def test_aggregates_in_schema_position_render_without_call_values():
    """Aggregates stay schema-side in schema positions, on both surfaces."""
    template = [
        {"role": "system", "content": "{inputs()}"},
        {"role": "user", "content": "{question}"},
    ]
    parsed = parse_message_template(template)
    engine_side = render_nodes(parsed.messages[0].nodes, ctx())
    previewed = preview(parsed, QA, inputs={"question": "SECRET-INPUT"})
    assert previewed[0]["content"] == engine_side
    assert "SECRET-INPUT" not in previewed[0]["content"]
    assert previewed[1]["content"] == "SECRET-INPUT"
