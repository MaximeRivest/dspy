import importlib.util
import sys
from pathlib import Path

import pytest

from dspy.programir.contract_validate import forward_refusal
from dspy.programir.errors import ProgramIRRefusal
from dspy.programir.forward import LeafRef, compile_forward
from dspy.programir.versions import IMPLEMENTED_VERSIONS, supported_versions

_CONTRACT_REFERENCE = Path("/home/maxime/Projects/programir-contract/reference")


def _load_reference(module_name):
    path = _CONTRACT_REFERENCE / f"{module_name}.py"
    if not path.exists():
        return None
    qualified = f"pir_reference_{module_name}"
    if qualified in sys.modules:
        return sys.modules[qualified]
    spec = importlib.util.spec_from_file_location(qualified, path)
    module = importlib.util.module_from_spec(spec)
    if module_name != "errors":
        sys.modules["errors"] = _load_reference("errors")
    sys.modules[qualified] = module
    spec.loader.exec_module(module)
    return module


def _assert_valid(compiled):
    """Every compiled forward must round-trip the node-set schema."""
    assert forward_refusal(compiled) is None


def _run(compiled, inputs):
    """Execute a compiled forward on the contract reference interpreter."""
    interp = _load_reference("interp")
    if interp is None:
        pytest.skip("programir-contract reference interpreter is not available")

    class _NoLeaves:
        def predict(self, path, kwargs):
            raise AssertionError("no predict leaves in this fixture")

        def tool(self, name, kwargs):
            raise AssertionError("no tool leaves in this fixture")

        def interpreter(self, ref, code):
            raise AssertionError("no interpreter leaves in this fixture")

    return interp.run_forward({"self": compiled}, "", inputs, _NoLeaves())


# --- v0.1 fixtures (unchanged behavior) -------------------------------------


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


def dynamic_tool_forward(self, prediction):
    return self.tools[prediction.tool_name](query=prediction.query)


def interpreter_forward(self, code):
    return self.interpreter(code=code)


def unresolved_forward(self, question):
    return self.helper(question=question)


def splat_forward(self, kwargs):
    return self.predict(**kwargs)


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
    _assert_valid(compiled)


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
    _assert_valid(compiled)


def test_compile_forward_covers_v01_control_flow():
    compiled = compile_forward(control_flow_forward, {"generate": LeafRef("predict", "generate")})

    assert [statement["node"] for statement in compiled["body"]] == ["Assign", "For", "While", "Return"]
    assert compiled["body"][1]["range"] == 3
    assert compiled["body"][1]["body"][0]["node"] == "Try"
    assert compiled["body"][2]["body"] == [
        {"node": "Raise", "exc": "InterpreterError", "message": "empty"}
    ]
    _assert_valid(compiled)


def test_compile_forward_emits_dynamic_tool_and_interpreter_leaves():
    dynamic = compile_forward(dynamic_tool_forward, {"tools": LeafRef("tool")})
    interpreter = compile_forward(
        interpreter_forward,
        {"interpreter": LeafRef("interpreter", "main")},
    )

    assert dynamic["body"][0]["value"] == {
        "node": "Call",
        "leaf": {"kind": "tool"},
        "name": {"node": "Attr", "obj": "prediction", "attr": "tool_name"},
        "kwargs": {"query": {"node": "Attr", "obj": "prediction", "attr": "query"}},
    }
    assert interpreter["body"][0]["value"]["leaf"] == {"kind": "interpreter", "ref": "main"}
    _assert_valid(dynamic)
    _assert_valid(interpreter)


def test_compile_forward_refuses_unresolved_call_by_name():
    with pytest.raises(ProgramIRRefusal) as caught:
        compile_forward(unresolved_forward, {})

    assert caught.value.code == "PIR-E-NODE-002"
    assert caught.value.detail["leaf"] == "self.helper"


def test_compile_forward_refuses_kwargs_splat():
    with pytest.raises(ProgramIRRefusal, match="keyword splats"):
        compile_forward(splat_forward, {"predict": LeafRef("predict", "predict")})


# --- v0.2 nodes -------------------------------------------------------------


def setindex_forward(self, question):
    trajectory = {}
    trajectory["thought"] = question
    steps = [question]
    steps[0] = question
    return trajectory


def test_compile_setindex():
    compiled = compile_forward(setindex_forward, {})
    assert compiled["body"][1] == {
        "node": "SetIndex",
        "target": "trajectory",
        "key": {"node": "Const", "value": "thought"},
        "value": {"node": "Var", "name": "question"},
    }
    assert compiled["body"][3]["key"] == {"node": "Const", "value": 0}
    _assert_valid(compiled)
    result = _run(compiled, {"question": "why"})
    assert result == {"thought": "why"}


def assigntuple_forward(self, pair):
    first, second = pair
    return {"first": first, "second": second}


def test_compile_assigntuple_and_dict_literal():
    compiled = compile_forward(assigntuple_forward, {})
    assert compiled["body"][0] == {
        "node": "AssignTuple",
        "targets": ["first", "second"],
        "value": {"node": "Var", "name": "pair"},
    }
    _assert_valid(compiled)
    assert _run(compiled, {"pair": ["a", "b"]}) == {"first": "a", "second": "b"}


def value_ops_forward(self, flag, count):
    label = "big" if count > 2 else "small"
    ok = flag and not (count < 0)
    total = count * 2 - 1
    ratio = count / 2
    items = [label, "x"]
    has = "x" in items
    tag = f"n={count} {label}"
    return {"label": label, "ok": ok, "total": total, "ratio": ratio, "has": has, "tag": tag, "neg": -count}


def test_compile_v02_value_expressions():
    compiled = compile_forward(value_ops_forward, {})
    nodes = {statement["target"]: statement["value"]["node"] for statement in compiled["body"][:-1]}
    assert nodes == {
        "label": "IfExp",
        "ok": "BoolOp",
        "total": "BinOp",
        "ratio": "BinOp",
        "items": "List",
        "has": "Compare",
        "tag": "Format",
    }
    _assert_valid(compiled)
    result = _run(compiled, {"flag": True, "count": 3})
    assert result == {
        "label": "big",
        "ok": True,
        "total": 5,
        "ratio": 1.5,
        "has": True,
        "tag": "n=3 big",
        "neg": -3,
    }


def for_list_forward(self, items):
    joined = ""
    for item in items:
        joined = joined + item
    return {"joined": joined}


def test_compile_for_over_list():
    compiled = compile_forward(for_list_forward, {})
    loop = compiled["body"][1]
    assert loop["node"] == "For"
    assert loop["iter"] == {"node": "Var", "name": "items"}
    assert "range" not in loop
    _assert_valid(compiled)
    assert _run(compiled, {"items": ["a", "b", "c"]}) == {"joined": "abc"}


def index_forward(self, record, items):
    value = record["key"]
    head = items[0]
    return {"value": value, "head": head}


def test_compile_index_reads():
    compiled = compile_forward(index_forward, {})
    assert compiled["body"][0]["value"] == {
        "node": "Index",
        "obj": {"node": "Var", "name": "record"},
        "key": {"node": "Const", "value": "key"},
    }
    _assert_valid(compiled)
    assert _run(compiled, {"record": {"key": "v"}, "items": [7]}) == {"value": "v", "head": 7}


def orderings_forward(self, a, b):
    return {"lt": a < b, "ge": a >= b, "ni": a not in [b]}


def test_compile_widened_compare():
    compiled = compile_forward(orderings_forward, {})
    values = compiled["body"][0]["value"]
    ops = [entry["value"]["op"] for entry in values["entries"]]
    assert ops == ["lt", "ge", "not_in"]
    _assert_valid(compiled)
    assert _run(compiled, {"a": 1, "b": 2}) == {"lt": True, "ge": False, "ni": True}


# --- v0.3 nodes -------------------------------------------------------------


def slice_forward(self, passages, text):
    top = passages[:2]
    tail = passages[-2:]
    reversed_text = text[::-1]
    clipped = text[1:3]
    return {"top": top, "tail": tail, "rev": reversed_text, "clip": clipped}


def test_compile_slice_reads():
    compiled = compile_forward(slice_forward, {})
    first = compiled["body"][0]["value"]
    assert first == {
        "node": "Slice",
        "obj": {"node": "Var", "name": "passages"},
        "stop": {"node": "Const", "value": 2},
    }
    third = compiled["body"][2]["value"]
    assert third == {"node": "Slice", "obj": {"node": "Var", "name": "text"}, "step": -1}
    _assert_valid(compiled)
    result = _run(compiled, {"passages": [1, 2, 3], "text": "abcd"})
    assert result == {"top": [1, 2], "tail": [2, 3], "rev": "dcba", "clip": "bc"}


def slice_step_forward(self, items):
    return items[::2]


def test_compile_refuses_non_unit_slice_step():
    with pytest.raises(ProgramIRRefusal, match="literal 1 or -1"):
        compile_forward(slice_step_forward, {})


def pass_forward(self, question):
    if question == "":
        pass
    return {"question": question}


def test_compile_pass():
    compiled = compile_forward(pass_forward, {})
    assert compiled["body"][0]["body"] == [{"node": "Pass"}]
    _assert_valid(compiled)


def log_forward(self, question):
    print("asking", question)
    return {"question": question}


def test_compile_print_desugars_to_log():
    compiled = compile_forward(log_forward, {})
    assert compiled["body"][0] == {
        "node": "Log",
        "parts": [
            {"node": "Const", "value": "asking"},
            {"node": "Const", "value": " "},
            {"node": "Var", "name": "question"},
        ],
    }
    _assert_valid(compiled)


def builtins_forward(self, items, record):
    n = len(items)
    rendered = str(n)
    biggest = max(items)
    ordered = sorted(items)
    every = all([True, n > 0])
    blob = json.dumps(record)
    parsed = json.loads(blob)
    return {"n": n, "rendered": rendered, "biggest": biggest, "ordered": ordered, "every": every, "parsed": parsed}


def test_compile_builtin_table():
    compiled = compile_forward(builtins_forward, {})
    names = [statement["value"]["builtin"] for statement in compiled["body"][:4]]
    assert names == ["len", "str", "max", "sorted"]
    assert compiled["body"][5]["value"]["builtin"] == "json_dumps"
    assert compiled["body"][6]["value"]["builtin"] == "json_parse"
    _assert_valid(compiled)
    result = _run(compiled, {"items": [3, 1, 2], "record": {"k": 1}})
    assert result == {
        "n": 3,
        "rendered": "3",
        "biggest": 3,
        "ordered": [1, 2, 3],
        "every": True,
        "parsed": {"k": 1},
    }


def methods_forward(self, text, record):
    clean = text.strip().lower()
    words = clean.split(" ")
    joined = "-".join(words)
    swapped = joined.replace("-", "_")
    starts = swapped.startswith("a")
    value = record.get("missing", "fallback")
    names = record.keys()
    steps = []
    steps.append(clean)
    steps.extend(words)
    return {"joined": joined, "swapped": swapped, "starts": starts, "value": value, "names": names, "steps": steps}


def test_compile_method_table():
    compiled = compile_forward(methods_forward, {})
    first = compiled["body"][0]["value"]
    assert first["method"] == "lower"
    assert first["obj"]["method"] == "strip"
    append = compiled["body"][8]
    assert append == {
        "node": "Assign",
        "target": "_",
        "value": {
            "node": "Call",
            "method": "append",
            "obj": {"node": "Var", "name": "steps"},
            "args": [{"node": "Var", "name": "clean"}],
        },
    }
    _assert_valid(compiled)
    result = _run(compiled, {"text": "  A B ", "record": {"k": 1}})
    assert result == {
        "joined": "a-b",
        "swapped": "a_b",
        "starts": True,
        "value": "fallback",
        "names": ["k"],
        "steps": ["a b", "a", "b"],
    }


def unknown_method_forward(self, text):
    return {"out": text.title()}


def test_compile_refuses_unknown_method_by_name():
    with pytest.raises(ProgramIRRefusal, match=r"\.title\(\) is outside the closed 18-name"):
        compile_forward(unknown_method_forward, {})


def sorted_key_forward(self, items):
    return {"out": sorted(items, key=len)}


def test_compile_refuses_builtin_keywords():
    with pytest.raises(ProgramIRRefusal, match="positional arguments only"):
        compile_forward(sorted_key_forward, {})


def unknown_builtin_forward(self, items):
    return {"out": zip(items, items)}


def test_compile_refuses_unknown_builtin_as_undeclared_leaf():
    with pytest.raises(ProgramIRRefusal) as caught:
        compile_forward(unknown_builtin_forward, {})
    assert caught.value.code == "PIR-E-NODE-002"
    assert caught.value.detail["leaf"] == "zip"


# --- desugars ---------------------------------------------------------------


def comprehension_forward(self, items):
    doubled = [item + item for item in items]
    longs = [item for item in items if len(item) > 1]
    lengths = {item: len(item) for item in items}
    return {"doubled": doubled, "longs": longs, "lengths": lengths}


def test_compile_comprehensions_desugar_to_for_accumulation():
    compiled = compile_forward(comprehension_forward, {})
    kinds = [statement["node"] for statement in compiled["body"]]
    assert kinds == ["Assign", "For", "Assign", "Assign", "For", "Assign", "Assign", "For", "Assign", "Return"]
    conditional = compiled["body"][4]["body"][0]
    assert conditional["node"] == "If"
    _assert_valid(compiled)
    result = _run(compiled, {"items": ["ab", "c"]})
    assert result == {
        "doubled": ["abab", "cc"],
        "longs": ["ab"],
        "lengths": {"ab": 2, "c": 1},
    }


def is_none_forward(self, value):
    if value is None:
        return {"missing": True}
    if value is not None:
        return {"missing": False}
    return {"missing": False}


def test_compile_is_none_desugar():
    compiled = compile_forward(is_none_forward, {})
    assert compiled["body"][0]["test"] == {
        "node": "Compare",
        "op": "eq",
        "left": {"node": "Var", "name": "value"},
        "right": {"node": "Const", "value": None},
    }
    assert compiled["body"][1]["test"]["op"] == "ne"
    _assert_valid(compiled)
    assert _run(compiled, {"value": None}) == {"missing": True}


def is_identity_forward(self, a, b):
    return {"same": a is b}


def test_compile_refuses_general_is():
    with pytest.raises(ProgramIRRefusal, match="only `is None`"):
        compile_forward(is_identity_forward, {})


def augassign_forward(self, count, record):
    count += 2
    count -= 1
    record["hits"] += 1
    return {"count": count, "record": record}


def test_compile_augassign_desugar():
    compiled = compile_forward(augassign_forward, {})
    assert compiled["body"][0] == {
        "node": "Assign",
        "target": "count",
        "value": {
            "node": "BinOp",
            "op": "add",
            "left": {"node": "Var", "name": "count"},
            "right": {"node": "Const", "value": 2},
        },
    }
    assert compiled["body"][2]["node"] == "SetIndex"
    _assert_valid(compiled)
    result = _run(compiled, {"count": 1, "record": {"hits": 0}})
    assert result == {"count": 2, "record": {"hits": 1}}


def walrus_forward(self, text):
    if (clean := text.strip()) == "":
        return {"clean": "empty"}
    return {"clean": clean}


def test_compile_walrus_hoists_assign():
    compiled = compile_forward(walrus_forward, {})
    assert compiled["body"][0] == {
        "node": "Assign",
        "target": "clean",
        "value": {
            "node": "Call",
            "method": "strip",
            "obj": {"node": "Var", "name": "text"},
            "args": [],
        },
    }
    assert compiled["body"][1]["node"] == "If"
    _assert_valid(compiled)
    assert _run(compiled, {"text": " hi "}) == {"clean": "hi"}


def walrus_in_while_forward(self, text):
    while (chunk := text.strip()) != "":
        text = chunk
    return {"text": text}


def test_compile_refuses_walrus_in_while_test():
    with pytest.raises(ProgramIRRefusal, match="While test re-evaluates"):
        compile_forward(walrus_in_while_forward, {})


def percent_format_forward(self, name, count):
    return {"out": "hello %s, %d times (100%%)" % (name, count)}


def test_compile_percent_format_desugar():
    compiled = compile_forward(percent_format_forward, {})
    value = compiled["body"][0]["value"]["entries"][0]["value"]
    assert value == {
        "node": "Format",
        "parts": [
            {"node": "Const", "value": "hello "},
            {"node": "Var", "name": "name"},
            {"node": "Const", "value": ", "},
            {"node": "Var", "name": "count"},
            {"node": "Const", "value": " times (100%)"},
        ],
    }
    _assert_valid(compiled)
    assert _run(compiled, {"name": "ada", "count": 2}) == {"out": "hello ada, 2 times (100%)"}


def str_format_forward(self, name):
    return {"out": f"hi {name}!"}


def test_compile_str_format_desugar():
    compiled = compile_forward(str_format_forward, {})
    value = compiled["body"][0]["value"]["entries"][0]["value"]
    assert value["node"] == "Format"
    _assert_valid(compiled)
    assert _run(compiled, {"name": "bo"}) == {"out": "hi bo!"}


def enumerate_forward(self, items):
    lines = []
    for index, item in enumerate(items):
        lines.append(f"{index}:{item}")
    return {"lines": lines}


def test_compile_enumerate_for_desugar():
    compiled = compile_forward(enumerate_forward, {})
    assert compiled["body"][1] == {
        "node": "Assign",
        "target": "index",
        "value": {"node": "Const", "value": -1},
    }
    loop = compiled["body"][2]
    assert loop["node"] == "For"
    assert loop["target"] == "item"
    assert loop["body"][0]["target"] == "index"
    _assert_valid(compiled)
    assert _run(compiled, {"items": ["a", "b"]}) == {"lines": ["0:a", "1:b"]}


def dict_merge_forward(self, defaults, overrides):
    merged = defaults | overrides
    return {"merged": merged}


def test_compile_dict_merge_desugar():
    compiled = compile_forward(dict_merge_forward, {})
    kinds = [statement["node"] for statement in compiled["body"]]
    assert kinds == ["Assign", "Assign", "Assign", "Assign", "Return"]
    assert compiled["body"][0]["value"] == {"node": "Dict", "entries": []}
    assert compiled["body"][1]["value"]["method"] == "update"
    assert compiled["body"][2]["value"]["method"] == "update"
    _assert_valid(compiled)
    result = _run(compiled, {"defaults": {"a": 1, "b": 2}, "overrides": {"b": 3}})
    assert result == {"merged": {"a": 1, "b": 3}}


def assert_forward(self, count):
    assert count > 0, "count must be positive"
    return {"count": count}


def test_compile_assert_desugar():
    compiled = compile_forward(assert_forward, {})
    assert compiled["body"][0] == {
        "node": "If",
        "test": {
            "node": "UnaryOp",
            "op": "not",
            "value": {
                "node": "Compare",
                "op": "gt",
                "left": {"node": "Var", "name": "count"},
                "right": {"node": "Const", "value": 0},
            },
        },
        "body": [{"node": "Raise", "exc": "InterpreterError", "message": "count must be positive"}],
        "orelse": [],
    }
    _assert_valid(compiled)


def chained_compare_forward(self, low, value, high):
    return {"ok": low < value < high}


def test_compile_chained_compare_with_var_interior():
    compiled = compile_forward(chained_compare_forward, {})
    test = compiled["body"][0]["value"]["entries"][0]["value"]
    assert test["node"] == "BoolOp"
    assert test["op"] == "and"
    assert [pair["op"] for pair in test["values"]] == ["lt", "lt"]
    _assert_valid(compiled)
    assert _run(compiled, {"low": 1, "value": 2, "high": 3}) == {"ok": True}


def chained_compare_computed_forward(self, items, high):
    return {"ok": 0 < len(items) < high}


def test_compile_refuses_chained_compare_with_computed_interior():
    with pytest.raises(ProgramIRRefusal, match="evaluate it twice"):
        compile_forward(chained_compare_computed_forward, {})


def annassign_forward(self, question):
    label: str = question
    return {"label": label}


def test_compile_annassign_desugar():
    compiled = compile_forward(annassign_forward, {})
    assert compiled["body"][0] == {
        "node": "Assign",
        "target": "label",
        "value": {"node": "Var", "name": "question"},
    }
    _assert_valid(compiled)


def tuple_for_forward(self, pairs):
    total = 0
    for key, value in pairs:
        total = total + value
    return {"total": total}


def test_compile_tuple_for_target_desugar():
    compiled = compile_forward(tuple_for_forward, {})
    loop = compiled["body"][1]
    assert loop["body"][0]["node"] == "AssignTuple"
    assert loop["body"][0]["targets"] == ["key", "value"]
    _assert_valid(compiled)
    assert _run(compiled, {"pairs": [["a", 1], ["b", 2]]}) == {"total": 3}


# --- the aforward rule ------------------------------------------------------


def sync_twin_forward(self, question):
    draft = self.drafter(question=question)
    return draft


async def async_twin_forward(self, question):
    draft = await self.drafter.acall(question=question)
    return draft


def test_aforward_compiles_to_the_identical_graph():
    leaves = {"drafter": LeafRef("module", "drafter")}
    assert compile_forward(async_twin_forward, leaves) == compile_forward(sync_twin_forward, leaves)


async def gather_forward(self, question):
    results = await asyncio.gather(self.a.acall(q=question), self.b.acall(q=question))
    return results


def test_aforward_refuses_explicit_concurrency_by_name():
    with pytest.raises(ProgramIRRefusal, match="asyncio.gather is explicit concurrency"):
        compile_forward(gather_forward, {"a": LeafRef("predict", "a"), "b": LeafRef("predict", "b")})


async def await_non_leaf_forward(self, question):
    value = await question
    return value


def test_aforward_refuses_await_of_non_call():
    with pytest.raises(ProgramIRRefusal, match="await wraps only a declared leaf"):
        compile_forward(await_non_leaf_forward, {})


# --- loud refusals ----------------------------------------------------------


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


def set_forward(self, items):
    return {"unique": {1, 2}}


def starred_forward(self, items):
    return {"out": [*items]}


def bitand_forward(self, a, b):
    return {"out": a & b}


def floordiv_forward(self, a, b):
    return {"out": a // b}


def mod_forward(self, a, b):
    return {"out": a % b}


def self_state_forward(self, question):
    return {"limit": self.max_iters}


def test_compile_refuses_module_state_reads():
    with pytest.raises(ProgramIRRefusal, match="reads module state"):
        compile_forward(self_state_forward, {})


@pytest.mark.parametrize(
    ("function", "node", "fragment"),
    [
        (with_forward, "With", "dspy.context"),
        (yield_forward, "Yield", "node set"),
        (lambda_forward, "Lambda", "hidden state"),
        (import_forward, "Import", "undeclared leaves"),
        (nested_def_forward, "FunctionDef", "undeclared leaf"),
        (set_forward, "Set", "no set value"),
        (starred_forward, "Starred", "splat"),
        (bitand_forward, "BinOp", "not admitted"),
        (floordiv_forward, "BinOp", "not admitted"),
        (mod_forward, "BinOp", "refused"),
    ],
)
def test_compile_forward_refuses_named_constructs(function, node, fragment):
    with pytest.raises(ProgramIRRefusal) as caught:
        compile_forward(function, {})

    assert caught.value.code == "PIR-E-NODE-001"
    assert caught.value.detail["node"] == node
    assert caught.value.detail["line"] is not None
    assert "test_forward.py:" in str(caught.value)
    assert fragment in str(caught.value)


# --- version stamp ----------------------------------------------------------


def test_node_set_version_stamp_is_v03():
    assert IMPLEMENTED_VERSIONS["node_set"] == "0.3"
    assert supported_versions("node_set") == ("0.1", "0.2", "0.3")
    assert supported_versions("ir_version") == ("0.1",)


def test_contract_reference_validate_accepts_compiled_v03_nodes():
    validate = _load_reference("validate")
    if validate is None:
        pytest.skip("programir-contract reference validator is not available")
    for function in (
        setindex_forward,
        value_ops_forward,
        slice_forward,
        builtins_forward,
        methods_forward,
        comprehension_forward,
        dict_merge_forward,
        enumerate_forward,
    ):
        compiled = compile_forward(function, {})
        violations = validate.check(compiled, validate.node_set_schema())
        assert violations == [], f"{function.__name__}: {violations}"
