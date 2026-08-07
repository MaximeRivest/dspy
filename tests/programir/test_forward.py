import pytest

from dspy.programir.errors import ProgramIRRefusal
from dspy.programir.forward import LeafRef, compile_forward


def answerer_forward(self, question):
    draft = self.drafter(question=question)
    final = self.polish(question=question, draft_answer=draft.draft_answer)
    return final


def branching_forward(self, question):
    cat = self.classifier(question=question)
    if cat.category == "creative":
        draft = self.draft_creative(question=question)
    else:
        draft = self.draft_factual(question=question)
    return draft


def control_flow_forward(self, question):
    answer = ""
    for index in range(3):
        try:
            answer = self.generate(question=question)
            if answer.done != False:
                break
        except LMError:
            continue
    while answer == "":
        raise InterpreterError("empty")
    return answer


def proposed_forward(self, question):
    values = {"question": question}
    return values


def unresolved_forward(self, question):
    return self.helper(question=question)


def splat_forward(self, kwargs):
    return self.predict(**kwargs)


def with_forward(self, question):
    with context():
        return question


def yield_forward(self, question):
    return (yield question)


def lambda_forward(self, question):
    transform = lambda value: value
    return question


def import_forward(self, question):
    import example  # noqa: F401

    return question


def nested_def_forward(self, question):
    def helper(value):
        return value

    return question


def test_compile_forward_matches_nested_module_fixture_shape():
    compiled = compile_forward(
        answerer_forward,
        {
            "drafter": LeafRef("module", "drafter"),
            "polish": LeafRef("predict", "polish"),
        },
    )

    assert compiled == {
        "language": "restricted-python-ast",
        "args": ["question"],
        "body": [
            {
                "node": "Assign",
                "target": "draft",
                "value": {
                    "node": "Call",
                    "leaf": {"kind": "module", "ref": "drafter"},
                    "kwargs": {"question": {"node": "Var", "name": "question"}},
                },
            },
            {
                "node": "Assign",
                "target": "final",
                "value": {
                    "node": "Call",
                    "leaf": {"kind": "predict", "ref": "polish"},
                    "kwargs": {
                        "question": {"node": "Var", "name": "question"},
                        "draft_answer": {"node": "Attr", "obj": "draft", "attr": "draft_answer"},
                    },
                },
            },
            {"node": "Return", "value": {"node": "Var", "name": "final"}},
        ],
    }


def test_compile_forward_emits_if_and_comparison():
    compiled = compile_forward(
        branching_forward,
        {
            "classifier": LeafRef("module", "classifier"),
            "draft_creative": LeafRef("predict", "draft_creative"),
            "draft_factual": LeafRef("predict", "draft_factual"),
        },
    )

    branch = compiled["body"][1]
    assert branch["node"] == "If"
    assert branch["test"] == {
        "node": "Compare",
        "op": "eq",
        "left": {"node": "Attr", "obj": "cat", "attr": "category"},
        "right": {"node": "Const", "value": "creative"},
    }


def test_compile_forward_covers_v01_control_flow():
    compiled = compile_forward(control_flow_forward, {"generate": LeafRef("predict", "generate")})

    assert [statement["node"] for statement in compiled["body"]] == ["Assign", "For", "While", "Return"]
    assert compiled["body"][1]["range"] == 3
    assert compiled["body"][1]["body"][0]["node"] == "Try"
    assert compiled["body"][2]["body"] == [
        {"node": "Raise", "exc": "InterpreterError", "message": "empty"}
    ]


def test_compile_forward_names_v02_proposal_and_source_line():
    with pytest.raises(ProgramIRRefusal) as caught:
        compile_forward(proposed_forward, {})

    assert caught.value.code == "PIR-E-NODE-001"
    assert caught.value.detail["node"] == "Dict"
    assert caught.value.detail["proposed"] == "node-set v0.2"
    assert "test_forward.py:" in str(caught.value)
    assert "not ratified" in str(caught.value)


def test_compile_forward_refuses_unresolved_call_by_name():
    with pytest.raises(ProgramIRRefusal) as caught:
        compile_forward(unresolved_forward, {})

    assert caught.value.code == "PIR-E-NODE-002"
    assert caught.value.detail["leaf"] == "self.helper"


def test_compile_forward_refuses_kwargs_splat():
    with pytest.raises(ProgramIRRefusal, match="keyword splats"):
        compile_forward(splat_forward, {"predict": LeafRef("predict", "predict")})


@pytest.mark.parametrize(
    ("function", "node"),
    [
        (with_forward, "With"),
        (yield_forward, "Yield"),
        (lambda_forward, "Lambda"),
        (import_forward, "Import"),
        (nested_def_forward, "FunctionDef"),
    ],
)
def test_compile_forward_refuses_named_constructs(function, node):
    with pytest.raises(ProgramIRRefusal) as caught:
        compile_forward(function, {})

    assert caught.value.code == "PIR-E-NODE-001"
    assert caught.value.detail["node"] == node
