"""Stage A1 tests: the LM layer — capabilities, DummyLM, bindings."""

import pytest

import dspy
from dspy.lm import BINDINGS, resolve
from dspy.lm.lm import LMCapabilities


@pytest.fixture(autouse=True)
def clean_bindings():
    saved = dict(BINDINGS)
    BINDINGS.clear()
    yield
    BINDINGS.clear()
    BINDINGS.update(saved)


# ---------------------------------------------------------------------------
# Capability facts are declared constructor data
# ---------------------------------------------------------------------------


class TestCapabilities:
    def test_defaults(self):
        lm = dspy.LM("openai/gpt-4o-mini")
        assert lm.capabilities == LMCapabilities(
            instruct=True, native_reasoning=False, native_fc=False, native_citations=False
        )

    def test_declared_facts(self):
        lm = dspy.LM("anthropic/claude-x", native_reasoning=True, native_fc=True, native_citations=True)
        assert lm.capabilities.native_reasoning
        assert lm.capabilities.native_fc
        assert lm.capabilities.native_citations
        assert lm.capabilities.to_dict() == {
            "instruct": True,
            "native_reasoning": True,
            "native_fc": True,
            "native_citations": True,
            "image_input": False,
        }

    def test_facts_are_frozen(self):
        lm = dspy.LM("m")
        with pytest.raises(AttributeError):
            lm.capabilities.native_fc = True

    def test_base_model_fact(self):
        lm = dspy.LM("base-model", instruct=False)
        assert not lm.capabilities.instruct

    def test_default_request_kwargs(self):
        lm = dspy.LM("m", temperature=0.7, max_tokens=100, top_p=0.9)
        assert lm.kwargs == {"temperature": 0.7, "max_tokens": 100, "top_p": 0.9}

    def test_no_baked_in_generation_defaults(self):
        # Unset temperature/max_tokens stay out of the request; lm15
        # and the provider decide.
        lm = dspy.LM("m")
        assert lm.kwargs == {}


# ---------------------------------------------------------------------------
# DummyLM: scripted answers, recorded calls
# ---------------------------------------------------------------------------


class TestDummyLM:
    def test_scripted_answers_in_order(self):
        lm = dspy.DummyLM(["Paris", "Berlin"])
        assert lm(prompt="Capital of France?") == ["Paris"]
        assert lm(prompt="Capital of Germany?") == ["Berlin"]

    def test_records_calls(self):
        lm = dspy.DummyLM(["Paris"])
        lm(prompt="Capital of France?", temperature=0.3)
        assert len(lm.calls) == 1
        call = lm.calls[0]
        assert call["messages"] == [{"role": "user", "content": "Capital of France?"}]
        assert call["kwargs"]["temperature"] == 0.3  # per-call kwargs override defaults

    def test_history_recorded(self):
        lm = dspy.DummyLM(["Paris"])
        lm(prompt="Q?")
        assert len(lm.history) == 1
        assert lm.history[0]["outputs"] == ["Paris"]

    def test_messages_form(self):
        lm = dspy.DummyLM(["ok"])
        messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
        assert lm(messages) == ["ok"]
        assert lm.calls[0]["messages"] == messages

    def test_exhaustion_refuses_loudly(self):
        lm = dspy.DummyLM(["only one"])
        lm(prompt="a")
        with pytest.raises(dspy.LMError, match="exhausted"):
            lm(prompt="b")

    def test_callable_script(self):
        lm = dspy.DummyLM(lambda messages: messages[-1]["content"].upper())
        assert lm(prompt="echo me") == ["ECHO ME"]
        assert lm(prompt="twice") == ["TWICE"]  # callables never exhaust

    def test_prompt_and_messages_exclusive(self):
        lm = dspy.DummyLM(["x"])
        with pytest.raises(ValueError, match="exactly one"):
            lm([{"role": "user", "content": "hi"}], prompt="hi")
        with pytest.raises(ValueError, match="exactly one"):
            lm()

    def test_capability_facts_pass_through(self):
        lm = dspy.DummyLM(["x"], native_fc=True)
        assert lm.capabilities.native_fc


# ---------------------------------------------------------------------------
# Bindings: default table, per-predictor overrides, loud refusal
# ---------------------------------------------------------------------------


class TestBindings:
    def test_missing_binding_refuses(self):
        with pytest.raises(dspy.BindingError, match="No 'lm' binding"):
            resolve("lm")

    def test_configure_writes_the_table(self):
        lm = dspy.DummyLM(["x"])
        dspy.configure(lm=lm)
        assert resolve("lm") is lm
        assert BINDINGS == {"lm": lm}  # a plain dict, fully inspectable

    def test_overrides_win(self):
        default = dspy.DummyLM(["default"])
        local = dspy.DummyLM(["local"])
        dspy.configure(lm=default)
        assert resolve("lm", overrides={"lm": local}) is local

    def test_none_override_falls_through(self):
        default = dspy.DummyLM(["default"])
        dspy.configure(lm=default)
        assert resolve("lm", overrides={"lm": None}) is default
        assert resolve("lm", overrides={}) is default

    def test_override_without_default(self):
        local = dspy.DummyLM(["local"])
        assert resolve("lm", overrides={"lm": local}) is local

    def test_configure_none_unbinds(self):
        dspy.configure(lm=dspy.DummyLM(["x"]))
        dspy.configure(lm=None)
        assert BINDINGS == {}
        with pytest.raises(dspy.BindingError):
            resolve("lm")

    def test_arbitrary_binding_names(self):
        dspy.configure(adapter="chat-v2")
        assert resolve("adapter") == "chat-v2"

    def test_binding_error_is_not_catchable_program_error(self):
        assert not issubclass(dspy.BindingError, dspy.CatchableError)
        assert not issubclass(dspy.BindingError, dspy.PirError)


# ---------------------------------------------------------------------------
# The lm15-routed transport (faked at the canonical level; no network)
# ---------------------------------------------------------------------------


def _tool_call_response():
    from lm15 import Message, Response, ToolCallPart, Usage

    return Response(
        id="r1",
        model="fake",
        message=Message(
            role="assistant",
            parts=(ToolCallPart(id="c1", name="f", input={}),),
        ),
        finish_reason="tool_call",
        usage=Usage(),
    )


class TestLMTransport:
    def test_completion_call_and_history(self):
        from lm15.testing import FakeLM

        fake = FakeLM(["hello"])
        lm = dspy.LM("openai/gpt-4o-mini", temperature=0.2, router=fake)
        outputs = lm(prompt="hi", max_tokens=5)
        assert outputs == ["hello"]
        request = fake.requests[0]
        assert request.model == "openai/gpt-4o-mini"
        assert request.config.temperature == 0.2
        assert request.config.max_tokens == 5  # per-call override
        assert request.messages[0].role == "user"
        assert lm.history[0]["outputs"] == ["hello"]

    def test_provider_failure_maps_to_typed_lm_error(self):
        from lm15.errors import RateLimitError
        from lm15.testing import FakeLM

        fake = FakeLM([RateLimitError("boom from provider")])
        lm = dspy.LM("m", router=fake)
        with pytest.raises(dspy.LMError, match="boom from provider") as err:
            lm(prompt="hi")
        assert "rate_limit" in str(err.value)  # canonical code travels along
        assert isinstance(err.value.__cause__, RateLimitError)

    def test_contentless_response_refuses(self):
        from lm15.testing import FakeLM

        lm = dspy.LM("m", router=FakeLM([_tool_call_response()]))
        with pytest.raises(dspy.LMError, match="no text content"):
            lm(prompt="hi")

    def test_system_messages_fold_into_system_prompt(self):
        from lm15.testing import FakeLM

        fake = FakeLM(["ok"])
        lm = dspy.LM("m", router=fake)
        lm(
            messages=[
                {"role": "system", "content": "Be terse."},
                {"role": "user", "content": "hi"},
            ]
        )
        request = fake.requests[0]
        assert "Be terse." in str(request.system)
        assert [m.role for m in request.messages] == ["user"]

    def test_unknown_kwargs_travel_as_extensions(self):
        from lm15.testing import FakeLM

        fake = FakeLM(["ok"])
        lm = dspy.LM("m", router=fake, seed=7)
        lm(prompt="hi")
        assert fake.requests[0].config.extensions == {"seed": 7}

    def test_non_text_content_refuses_with_pointer(self):
        from lm15.testing import FakeLM

        lm = dspy.LM("m", router=FakeLM(["ok"]))
        with pytest.raises(ValueError, match="lm.complete"):
            lm(messages=[{"role": "user", "content": [{"type": "image"}]}])

    def test_unknown_model_maps_to_typed_lm_error(self):
        lm = dspy.LM("no-such-model-xyz")  # real shared router
        with pytest.raises(dspy.LMError, match="unknown_model"):
            lm(prompt="hi")

    def test_endpoint_kwargs_never_enter_the_request_body(self):
        from lm15.testing import FakeLM

        fake = FakeLM(["ok"])
        # An explicit router wins over endpoint kwargs; they are still
        # stripped from the body.
        lm = dspy.LM("m", router=fake, api_key="sk-secret", api_base="http://example.invalid/v1")
        lm(prompt="hi")
        assert fake.requests[0].config.extensions is None

    def test_canonical_escape_hatch(self):
        from lm15 import Message, Request
        from lm15.testing import FakeLM

        lm = dspy.LM("m", router=FakeLM(["canon"]))
        response = lm.complete(Request(model="m", messages=(Message.user("hi"),)))
        assert response.text == "canon"
