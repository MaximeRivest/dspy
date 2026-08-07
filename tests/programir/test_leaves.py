import pytest

import dspy
from dspy.programir import compile, read, write
from dspy.programir.leaves import extract_tool, parse_deps

GLOBAL_PREFIX = "unsafe"


def lookup(query: str, k: int = 3) -> list[str]:
    """Look up matching passages."""
    # deps: httpx, beautifulsoup4
    import httpx

    return [httpx.URL(query).path] * k


def global_reader(query: str) -> str:
    return GLOBAL_PREFIX + query


def quality(example, prediction) -> float:
    """Score whether the answer is present."""
    return 1.0


class ToolProgram(dspy.Module):
    def __init__(self):
        self.lookup = lookup
        self.answer = dspy.Predict("passages -> answer")

    def forward(self, query):
        passages = self.lookup(query=query, k=3)
        return self.answer(passages=passages)


class DynamicToolProgram(dspy.Module):
    def __init__(self):
        self.tools = {"search": lookup}

    def forward(self, prediction):
        return self.tools[prediction.tool_name](query=prediction.query, k=3)


def test_extract_tool_reuses_dspy_tool_schema_and_deps_comment():
    extracted = extract_tool(lookup, name="lookup")

    assert extracted.entry["parameters"] == {
        "query": {"type": "string"},
        "k": {"type": "integer", "default": 3},
    }
    assert extracted.entry["return_schema"] == {"items": {"type": "string"}, "type": "array"}
    assert extracted.entry["deps"] == ["httpx", "beautifulsoup4"]
    assert extracted.source_path == "tools/lookup.py"
    assert b"# deps: httpx, beautifulsoup4" in extracted.source


def test_parse_deps_uses_first_matching_comment():
    source = "def f():\n    # unrelated\n    # deps: first, second_pkg\n    # deps: ignored\n    pass\n"

    assert parse_deps(source) == ["first", "second_pkg"]


def test_extract_tool_refuses_global_reads():
    with pytest.raises(ValueError, match="GLOBAL_PREFIX"):
        extract_tool(global_reader, name="bad")


def test_compile_static_tool_and_metric_sidecars(tmp_path):
    program = ToolProgram()
    program.set_lm(dspy.LM("openai/model"))
    devset = [dspy.Example(query="DSPy", answer="yes").with_inputs("query")]

    ir = compile(program, metric=quality, devset=devset)
    components = ir.manifest["components"]

    assert list(components["6_tools"]) == ["lookup"]
    assert components["1_module_tree"]["tools"] == ["lookup"]
    assert components["5_forward"]["self"]["body"][0]["value"]["leaf"] == {
        "kind": "tool",
        "ref": "lookup",
    }
    assert components["12_metric"]["devset"] == [
        {"query": "DSPy", "answer": "yes", "input_keys": ["query"]}
    ]
    assert list(components["12_metric"]["metrics"]) == ["quality"]
    assert set(ir.sidecars) == {"tools/lookup.py", "metric/quality.py"}

    destination = tmp_path / "tools.ir"
    write(ir, destination)
    assert read(destination) == ir


def test_compile_dynamic_tool_table_uses_dispatch_keys_as_identity():
    components = compile(DynamicToolProgram()).to_manifest()["components"]

    assert list(components["6_tools"]) == ["search"]
    call = components["5_forward"]["self"]["body"][0]["value"]
    assert call["leaf"] == {"kind": "tool"}
    assert call["name"] == {"node": "Attr", "obj": "prediction", "attr": "tool_name"}


def test_metric_and_devset_are_independently_optional():
    predictor = dspy.Predict("question -> answer")
    predictor.set_lm(dspy.LM("openai/model"))

    metric_only = compile(predictor, metric=quality).manifest["components"]["12_metric"]
    devset_only = compile(
        predictor,
        devset=[dspy.Example(question="q").with_inputs("question")],
    ).manifest["components"]["12_metric"]

    assert metric_only["devset"] == []
    assert devset_only["metrics"] == {}
