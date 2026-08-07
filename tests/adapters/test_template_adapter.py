"""TemplateAdapter: the template authoring surface (Epic D, PR D-7).

Your messages are the prompt: the adapter renders exactly the authored
message list through the engine's pure walker, parses through the bound
parse mode, serializes as a component-4 entry, and is strategy-aware from
birth — natively-served fields leave the token stream and the bake-time
capacity check refuses calls whose data has no lane in the template.
"""

import pytest
from golden.harness import Recorder, StubLM

import dspy
from dspy.adapters._engine.builder import build_plan
from dspy.utils.dummies import DummyLM


class Summarize(dspy.Signature):
    """Summarize input text concisely."""

    text: str = dspy.InputField()
    summary: str = dspy.OutputField()


def _summarizer():
    return dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "You are a concise assistant. {instruction}"},
            {"role": "user", "content": "Summarize:\n\n{text}"},
        ],
        parse_mode="full_text",
    )


# ---------------------------------------------------------------------------
# Exact control: the messages are the prompt
# ---------------------------------------------------------------------------


def test_messages_are_the_prompt_nothing_added():
    messages = _summarizer().format(Summarize, [], {"text": "DSPy is a framework."})
    assert messages == [
        {"role": "system", "content": "You are a concise assistant. Summarize input text concisely."},
        {"role": "user", "content": "Summarize:\n\nDSPy is a framework."},
    ]


def test_full_text_parse_maps_the_whole_completion():
    adapter = _summarizer()
    assert adapter.parse(Summarize, "A concise summary.") == {"summary": "A concise summary."}


def test_loop_blocks_and_aggregates_render():
    adapter = dspy.TemplateAdapter(
        messages=[
            {
                "role": "system",
                "content": "Fields:\n{% for f in outputs strip %}\n{f.i}. `{f.name}` ({f.type})\n{% endfor %}",
            },
            {"role": "user", "content": "{% for f in inputs separator='\\n' %}{f.name}={f.value}{% endfor %}"},
        ],
        parse_mode="json",
    )
    signature = dspy.Signature("question, hint -> answer, score: int")
    messages = adapter.format(signature, [], {"question": "Q?", "hint": "H"})
    assert messages[0]["content"] == "Fields:\n1. `answer` (str)\n2. `score` (int)"
    assert messages[1]["content"] == "question=Q?\nhint=H"


@pytest.mark.parametrize(
    "parse_mode,completion,expected",
    [
        ("chat", "[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]", "Paris"),
        ("json", '{"answer": "Paris"}', "Paris"),
        ("xml", "<answer>Paris</answer>", "Paris"),
        ("full_text", "Paris", "Paris"),
    ],
)
def test_parse_modes_bind_the_matching_parser(parse_mode, completion, expected):
    adapter = dspy.TemplateAdapter(
        messages=[{"role": "user", "content": "{question}"}],
        parse_mode=parse_mode,
    )
    assert adapter.parse(dspy.Signature("question -> answer"), completion) == {"answer": expected}


def test_end_to_end_predict_with_chat_parse_mode():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "{instruction}"},
            {"role": "demos"},
            {"role": "user", "content": "Question: {question}"},
        ],
        parse_mode="chat",
    )
    lm = DummyLM([{"answer": "42"}])  # chat-marker completions
    with dspy.context(lm=lm, adapter=adapter):
        result = dspy.Predict("question -> answer")(question="Q?")
    assert result.answer == "42"
    sent = lm.history[-1]["messages"]
    assert sent[-1] == {"role": "user", "content": "Question: Q?"}


def test_demos_directive_expands_pairs():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "Classify."},
            {"role": "demos", "user": "In: {text}", "assistant": "Out: {label}"},
            {"role": "user", "content": "In: {text}"},
        ],
        parse_mode="full_text",
    )
    signature = dspy.Signature("text -> label")
    messages = adapter.format(signature, [{"text": "a", "label": "x"}], {"text": "b"})
    assert messages[1] == {"role": "user", "content": "In: a"}
    assert messages[2] == {"role": "assistant", "content": "Out: x"}
    assert messages[-1] == {"role": "user", "content": "In: b"}


def test_image_slot_splits_into_content_blocks():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "Describe the image."},
            {"role": "user", "content": "What is in this image? {image}"},
        ],
        parse_mode="full_text",
    )

    class Describe(dspy.Signature):
        image: dspy.Image = dspy.InputField()
        description: str = dspy.OutputField()

    messages = adapter.format(Describe, [], {"image": dspy.Image("https://example.com/x.png")})
    blocks = messages[-1]["content"]
    assert isinstance(blocks, list)
    assert any(isinstance(block, dict) and block.get("type") == "image_url" for block in blocks)


# ---------------------------------------------------------------------------
# preview(): the learn-by-looking loop
# ---------------------------------------------------------------------------


def test_preview_matches_format_bytes():
    adapter = _summarizer()
    inputs = {"text": "DSPy is a framework."}
    assert adapter.preview(Summarize, inputs=inputs) == adapter.format(Summarize, [], dict(inputs))


def test_preview_accepts_string_signatures():
    adapter = dspy.TemplateAdapter(
        messages=[{"role": "user", "content": "{question}"}],
        parse_mode="full_text",
    )
    assert adapter.preview("question -> answer", inputs={"question": "Q?"}) == [
        {"role": "user", "content": "Q?"}
    ]


def test_format_accepts_string_signatures_like_preview():
    """Persona repro (portability): format() crashed with a raw
    AttributeError on the string signatures preview() accepts (D-δ)."""
    adapter = dspy.TemplateAdapter(
        messages=[{"role": "user", "content": "{question}"}],
        parse_mode="full_text",
    )
    assert adapter.format("question -> answer", [], {"question": "Q?"}) == [{"role": "user", "content": "Q?"}]
    assert dspy.ChatAdapter().format("question -> answer", [], {"question": "Q?"}) == dspy.ChatAdapter().format(
        dspy.Signature("question -> answer"), [], {"question": "Q?"}
    )


def test_preview_with_lm_shows_the_planned_bytes():
    """Persona repro (strategies user, finding 1): with lm=, preview renders
    what a live call would actually send — natively-served fields leave the
    prompt — while preview without lm keeps the pre-plan view (D-δ)."""
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "{instruction}"},
            {"role": "user", "content": "{inputs(style='chat')}\n\n{outputs(style='chat')}"},
        ],
        parse_mode="chat",
    )
    thinker = StubLM(Recorder(), supports_reasoning=True)
    without_plan = adapter.preview(Thoughtful, inputs={"question": "Q?"})
    assert "reasoning" in str(without_plan)
    planned = adapter.preview(Thoughtful, inputs={"question": "Q?"}, lm=thinker)
    assert "reasoning" not in str(planned)

    # And the planned bytes ARE the live-call bytes.
    recorder = Recorder()
    lm = StubLM(recorder, supports_reasoning=True)
    try:
        adapter(lm, {}, Thoughtful, [], {"question": "Q?"})
    except BaseException:
        pass
    sent = recorder.calls[0]["messages"]
    assert [m["content"] for m in sent] == [m["content"] for m in adapter.preview(Thoughtful, inputs={"question": "Q?"}, lm=lm)]


def test_preview_with_lm_surfaces_bake_refusals():
    """Persona repro (strategies user, ex11): preview(lm=) refuses the exact
    triple a live call refuses — pure inspection no longer lies."""
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "Answer."},
            {"role": "user", "content": "{question}"},
        ],
        parse_mode="full_text",
    )
    assert adapter.preview(Thoughtful, inputs={"question": "Q?"})  # pre-plan view renders
    with pytest.raises(ValueError, match="ADP-006"):
        adapter.preview(Thoughtful, inputs={"question": "Q?"}, lm=StubLM(Recorder(), supports_reasoning=False))


def test_explain_plan_returns_the_recorded_resolution():
    """The auto decision is public data now (D-δ): {role, binding, served}
    per field, hidden fields, and the trace."""
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "{instruction}"},
            {"role": "user", "content": "{inputs(style='chat')}\n\n{outputs(style='chat')}"},
        ],
        parse_mode="chat",
    )
    plan = adapter.explain_plan(Thoughtful, lm=StubLM(Recorder(), supports_reasoning=True))
    assert plan["strategy_resolution"]["reasoning"] == {
        "role": "reasoning",
        "binding": "auto",
        "served": "native",
    }
    assert plan["hidden_fields"] == ["reasoning"]
    assert any(t["strategy"] == "reasoning.native" for t in plan["trace"])
    import json as json_module

    json_module.dumps(plan)  # plain serializable data

    textual = adapter.explain_plan(Thoughtful, lm=StubLM(Recorder(), supports_reasoning=False))
    assert textual["strategy_resolution"]["reasoning"]["served"] == "textual"
    assert textual["hidden_fields"] == []


def test_full_text_arity_refuses_before_the_lm_call():
    """Persona repro (three personas): full_text with two output fields is
    statically invalid — it must never cost a completion (D-δ). The
    refusal fires at preview/format (no strategy could hide the surplus)
    and at bake inside a live call, with zero LM calls made."""
    adapter = dspy.TemplateAdapter(
        messages=[{"role": "user", "content": "{q}"}],
        parse_mode="full_text",
    )
    two_out = dspy.Signature("q -> a, b")
    with pytest.raises(ValueError, match="exactly one output field"):
        adapter.preview(two_out, inputs={"q": "hi"})
    with pytest.raises(ValueError, match="exactly one output field"):
        adapter.format(two_out, [], {"q": "hi"})

    lm = DummyLM([{"a": "x", "b": "y"}])
    with dspy.context(lm=lm, adapter=adapter):
        with pytest.raises(ValueError, match="exactly one output field"):
            dspy.Predict(two_out)(q="hi")
    assert len(lm.history) == 0  # refused before any LM call


def test_full_text_arity_allows_signatures_a_strategy_can_thin():
    """A two-output signature is viable when one field can leave the token
    stream: preview stays permissive pre-plan, bake decides."""
    adapter = dspy.TemplateAdapter(
        messages=[{"role": "user", "content": "{question}"}],
        parse_mode="full_text",
    )
    # Pre-plan: renders (reasoning could hide under a capable LM).
    assert adapter.preview(Thoughtful, inputs={"question": "Q?"})
    # Baked against a capable LM: reasoning leaves the stream, one output remains.
    assert adapter.preview(Thoughtful, inputs={"question": "Q?"}, lm=StubLM(Recorder(), supports_reasoning=True))


def test_second_history_field_refuses_loudly():
    """Persona repro (few-shot user, bug 1): a second History field's turns
    must never silently vanish (D-δ)."""

    class TwoHistories(dspy.Signature):
        h1: dspy.History = dspy.InputField()
        h2: dspy.History = dspy.InputField()
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    adapter = dspy.TemplateAdapter(
        messages=[{"role": "history"}, {"role": "user", "content": "{question}"}],
        parse_mode="chat",
    )
    with pytest.raises(ValueError, match="h1, h2"):
        adapter.preview(
            TwoHistories,
            inputs={
                "h1": dspy.History(messages=[{"question": "FROM-H1", "answer": "x"}]),
                "h2": dspy.History(messages=[{"question": "FROM-H2", "answer": "y"}]),
                "question": "now",
            },
        )


def test_parse_mode_is_readable_back():
    adapter = dspy.TemplateAdapter(messages=[{"role": "user", "content": "{q}"}], parse_mode="xml")
    assert adapter.parse_mode == "xml"
    assert adapter.dump_entry()["parser"] == "xml"  # the entry-key spelling of the same binding


# ---------------------------------------------------------------------------
# Serialization: the adapter is data
# ---------------------------------------------------------------------------


def test_dump_load_round_trip_renders_identical_bytes():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "You are a helper. {instruction}"},
            {"role": "demos"},
            {"role": "user", "content": "{% for f in inputs separator='\\n' %}{f.name}: {f.value}{% endfor %}"},
        ],
        parse_mode="json",
        name="support_triage",
    )
    entry = adapter.dump_entry()
    assert entry["name"] == "support_triage"
    assert entry["parser"] == "json"

    loaded = dspy.adapters.load_entry(entry)
    signature = dspy.Signature("ticket -> category")
    payload = ([{"ticket": "t1", "category": "billing"}], {"ticket": "t2"})
    assert loaded.format(signature, *payload) == adapter.format(signature, *payload)
    assert loaded.dump_entry() == entry


def test_literal_table_derives_from_the_authored_template():
    table = _summarizer().literal_table()
    assert isinstance(table, dict)


def test_strategies_kwarg_rides_the_entry_and_binds():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "{instruction}"},
            {"role": "user", "content": "{inputs(style='chat')}\n{outputs(style='chat')}"},
        ],
        parse_mode="chat",
        strategies={"reasoning": "textual_field"},
    )
    assert adapter.dump_entry()["strategies"]["reasoning"] == "textual_field"
    assert adapter.strategies["reasoning"] == "textual_field"


# ---------------------------------------------------------------------------
# Eager refusals: teaching errors at construction
# ---------------------------------------------------------------------------


def test_unknown_template_construct_refuses_at_construction():
    with pytest.raises(Exception, match="unknown slot function"):
        dspy.TemplateAdapter(messages=[{"role": "user", "content": "{outpts()}"}])


def test_unknown_parse_mode_refuses_naming_the_valid_set():
    with pytest.raises(Exception, match="valid parsers: chat, json, xml, full_text"):
        dspy.TemplateAdapter(messages=[{"role": "user", "content": "hi"}], parse_mode="yaml")


def test_callable_parse_mode_refuses_pointing_at_registration():
    with pytest.raises(ValueError, match="registration API"):
        dspy.TemplateAdapter(
            messages=[{"role": "user", "content": "hi"}],
            parse_mode=lambda signature, completion: {},
        )


def test_unknown_codec_direction_refuses():
    with pytest.raises(ValueError, match="'input' and 'output'"):
        dspy.TemplateAdapter(
            messages=[{"role": "user", "content": "hi"}],
            codecs={"per_field": {}},
        )


def test_unknown_strategy_binding_refuses_eagerly():
    with pytest.raises(ValueError, match="bindable roles"):
        dspy.TemplateAdapter(messages=[{"role": "user", "content": "hi"}], strategies={"vibes": "auto"})


# ---------------------------------------------------------------------------
# Hosting refusals: call data with no lane never drops silently
# ---------------------------------------------------------------------------


def test_demos_with_no_hosting_refuse_loudly():
    adapter = _summarizer()
    with pytest.raises(ValueError, match="hosts none"):
        adapter.format(Summarize, [{"text": "a", "summary": "s"}], {"text": "b"})


def test_history_with_no_hosting_refuses_loudly():
    adapter = dspy.TemplateAdapter(
        messages=[{"role": "user", "content": "{question}"}],
        parse_mode="full_text",
    )

    class Chat(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    turns = dspy.History(messages=[{"question": "1+1?", "answer": "2"}])
    with pytest.raises(ValueError, match="history"):
        adapter.format(Chat, [], {"question": "2+2?", "history": turns})


def test_history_directive_hosts_turns():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "history", "user": "{question}", "assistant": "{answer}"},
            {"role": "user", "content": "{question}"},
        ],
        parse_mode="full_text",
    )

    class Chat(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    turns = dspy.History(messages=[{"question": "1+1?", "answer": "2"}])
    messages = adapter.format(Chat, [], {"question": "2+2?", "history": turns})
    assert messages == [
        {"role": "user", "content": "1+1?"},
        {"role": "assistant", "content": "2"},
        {"role": "user", "content": "2+2?"},
    ]


def test_zero_demos_and_zero_history_are_not_refused():
    messages = _summarizer().format(Summarize, [], {"text": "fine"})
    assert len(messages) == 2


# ---------------------------------------------------------------------------
# Strategy awareness from birth: the bake-time triple check
# ---------------------------------------------------------------------------


class Thoughtful(dspy.Signature):
    """Answer the question."""

    question: str = dspy.InputField()
    reasoning: dspy.Reasoning = dspy.OutputField()
    answer: str = dspy.OutputField()


def test_natively_served_field_leaves_the_authored_prompt():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "{instruction}"},
            {"role": "user", "content": "{inputs(style='chat')}\n\n{outputs(style='chat')}"},
        ],
        parse_mode="chat",
    )
    lm_kwargs = {}
    built = build_plan(adapter, StubLM(Recorder(), supports_reasoning=True), lm_kwargs, Thoughtful, {"question": "Q?"})
    assert lm_kwargs == {"reasoning_effort": "low"}
    rendered = adapter.format(built.render_signature, [], {"question": "Q?"})
    assert "reasoning" not in str(rendered)


def test_explicit_slot_on_natively_hidden_field_refuses_adp007():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "Think in {reasoning}."},
            {"role": "user", "content": "{question}"},
        ],
        parse_mode="chat",
    )
    with pytest.raises(ValueError, match="ADP-007"):
        build_plan(adapter, StubLM(Recorder(), supports_reasoning=True), {}, Thoughtful, {"question": "Q?"})


def test_textually_served_role_with_no_lane_refuses_adp006():
    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "Answer."},
            {"role": "user", "content": "{question}"},
        ],
        parse_mode="full_text",
    )
    with pytest.raises(ValueError, match="ADP-006"):
        build_plan(adapter, StubLM(Recorder(), supports_reasoning=False), {}, Thoughtful, {"question": "Q?"})


def test_strategy_fragments_with_no_slot_refuse_adp006():
    from dspy.adapters._engine.patch import AdapterPatch
    from dspy.adapters._engine.strategies import register_field_strategy, unregister_field_strategy

    class _NoteType(dspy.Type):
        content: str = ""

        @classmethod
        def description(cls):
            return "note"

    class _FragStrategy:
        name = "test.fragments"
        priority = 500
        exclusive_group = None

        def applies(self, ctx):
            return True

        def contribute(self, ctx):
            return AdapterPatch(fragments={"system": ["FRAGMENT LINE."]})

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        note: _NoteType = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "Answer.\n{outputs(style='chat')}"},
            {"role": "user", "content": "{question}"},
        ],
        parse_mode="chat",
        native_response_types=[_NoteType],
    )
    register_field_strategy(_NoteType, _FragStrategy())
    try:
        with pytest.raises(ValueError, match="fragments"):
            build_plan(adapter, StubLM(Recorder()), {}, Sig, {"question": "Q?"})
    finally:
        unregister_field_strategy(_NoteType)


def test_template_lane_runs_strategy_contributed_parser_hooks():
    """Persona repro (strategy author, bug 3): a strategy that hides a field
    carries the parser that fills it back — through the TemplateAdapter
    path a hidden field must never silently come back None."""
    from dspy.adapters._engine.patch import AdapterPatch
    from dspy.adapters._engine.transforms import HideOutputField

    class _FillingStrategy:
        name = "citations_filler"
        priority = 500
        exclusive_group = None
        capability_requirements = ()

        class _Parser:
            def parse(self, response_view, ctx):
                return {"citations": "FILLED-BY-STRATEGY"}

        parser = _Parser()

        def applies(self, ctx):
            return True

        def contribute(self, ctx):
            return AdapterPatch(
                field_transforms=[HideOutputField(ctx.field.name, reason="citations_filler")],
                parsers=[self.parser],
            )

    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "{instruction}"},
            {"role": "user", "content": "Question: {question}"},
        ],
        parse_mode="chat",
    )
    dspy.adapters.register_strategy(_FillingStrategy(), role="citations")
    try:
        lm = DummyLM([{"answer": "42"}])
        with dspy.context(lm=lm, adapter=adapter):
            result = dspy.Predict("question -> citations @citations, answer")(question="Q?")
        assert result.answer == "42"
        assert result.citations == "FILLED-BY-STRATEGY"  # not silently None
    finally:
        dspy.adapters.unregister_strategy(role="citations")


def test_template_parse_errors_self_identify():
    """Template-lane parse failures name the template adapter, not the
    borrowed parser's historical class (D-δ)."""
    from dspy.utils.exceptions import AdapterParseError

    adapter = dspy.TemplateAdapter(
        messages=[{"role": "user", "content": "{question}"}],
        parse_mode="chat",
        name="my_generic",
    )
    with pytest.raises(AdapterParseError, match=r"TemplateAdapter\('my_generic'\)"):
        adapter.parse(dspy.Signature("question -> answer"), "no markers at all")


def test_fragment_slot_hosts_strategy_fragments():
    from dspy.adapters._engine.patch import AdapterPatch
    from dspy.adapters._engine.render import plan_scope
    from dspy.adapters._engine.strategies import register_field_strategy, unregister_field_strategy

    class _NoteType2(dspy.Type):
        content: str = ""

        @classmethod
        def description(cls):
            return "note"

    class _FragStrategy:
        name = "test.fragments2"
        priority = 500
        exclusive_group = None

        def applies(self, ctx):
            return True

        def contribute(self, ctx):
            return AdapterPatch(fragments={"system": ["FRAGMENT LINE."]})

    class Sig(dspy.Signature):
        question: str = dspy.InputField()
        note: _NoteType2 = dspy.OutputField()
        answer: str = dspy.OutputField()

    adapter = dspy.TemplateAdapter(
        messages=[
            {"role": "system", "content": "Answer.\n{outputs(style='chat')}\n{fragments('system')}"},
            {"role": "user", "content": "{question}"},
        ],
        parse_mode="chat",
        native_response_types=[_NoteType2],
    )
    register_field_strategy(_NoteType2, _FragStrategy())
    try:
        built = build_plan(adapter, StubLM(Recorder()), {}, Sig, {"question": "Q?"})
        with plan_scope(built.plan):
            messages = adapter.format(built.render_signature, [], {"question": "Q?"})
        assert "FRAGMENT LINE." in messages[0]["content"]
    finally:
        unregister_field_strategy(_NoteType2)
