import dspy
from tests.programir.corpus_programs import MiniReAct, MiniRLM, NestedAnswerer, _remote_lm, _terse_template


def _configured(program):
    program.set_lm(_remote_lm(local=True))
    program.set_adapter(dspy.XMLAdapter())
    return program


def test_nested_and_react_corpus_programs_compile_from_live_forwards():
    nested = dspy.programir.compile(_configured(NestedAnswerer())).manifest["components"]
    react = dspy.programir.compile(_configured(MiniReAct())).manifest["components"]

    assert list(nested["5_forward"]) == ["drafter.classifier", "drafter", "self"]
    assert nested["5_forward"]["drafter"]["body"][1]["node"] == "If"
    assert react["5_forward"]["self"]["body"][2]["node"] == "For"
    assert react["5_forward"]["self"]["body"][2]["body"][2]["node"] == "Try"
    assert set(react["6_tools"]) == {"calculator", "lookup"}


def test_terse_template_reproduces_legacy_messages_and_prefill():
    signature = dspy.Signature("text -> sentiment").with_instructions(
        "Classify the sentiment of the text as positive or negative."
    )

    messages = _terse_template().format(signature, [], {"text": "A delightful surprise"})

    assert messages == [
        {
            "role": "system",
            "content": (
                "Classify the sentiment of the text as positive or negative. "
                "Given text, reply with one line `sentiment: <sentiment>` and nothing else."
            ),
        },
        {"role": "user", "content": "text: A delightful surprise"},
        {"role": "assistant", "content": "</think>\nsentiment:"},
    ]
    assert _terse_template().parse(signature, "positive") == {"sentiment": "positive"}


def test_rlm_corpus_forward_uses_the_ratified_interpreter_ref():
    # The weight-owning LM path is covered separately; this test compiles the
    # same forward with a declared LM so it stays small in the test matrix.
    components = dspy.programir.compile(_configured(MiniRLM())).manifest["components"]
    body = components["5_forward"]["self"]["body"]

    assert body[2]["node"] == "While"
    assert body[2]["body"][2]["body"][0]["value"]["leaf"] == {
        "kind": "interpreter",
        "ref": "interpreter",
    }
