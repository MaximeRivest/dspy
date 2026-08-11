"""The builder, the shared validator, and the printer's round-trip law.

Three claims, tested hard:

1. BYTE EQUALITY — a tree built with `programir.build` constructors is
   byte-identical (JSON text, key order included) to the compiler's
   output for the equivalent source.
2. THE ROUND-TRIP LAW — for every printable tree,
   `compile_forward(to_function(tree), leaves) == tree`, exact JSON
   equality: a synthetic tree covering EVERY node kind, the corpus
   programs the suite uses, and the three migrated modules.
3. ONE ADMISSION — hostile names refuse at BUILD time with teaching
   errors, and a deliberately-bad built tree gets the same refusal
   (code + named offender) as its parsed twin: the validator is shared,
   the two frontends cannot drift.
"""

import json

import pytest

import dspy
from dspy.lm import BINDINGS
from dspy.programir import build, printer
from dspy.programir.errors import ProgramIRRefusal
from dspy.programir.forward import LeafRef, admit_forward, compile_forward


@pytest.fixture(autouse=True)
def clean_bindings():
    saved = dict(BINDINGS)
    BINDINGS.clear()
    yield
    BINDINGS.clear()
    BINDINGS.update(saved)


def leaves_of(tree: dict) -> dict[str, LeafRef]:
    """Derive the declared-leaf table a forward tree calls against."""
    found: dict[str, LeafRef] = {}

    def walk(value):
        if isinstance(value, dict):
            if value.get("node") == "Call" and isinstance(value.get("leaf"), dict):
                leaf = value["leaf"]
                if leaf.get("ref") is not None:
                    found[leaf["ref"]] = LeafRef(leaf["kind"], leaf["ref"])
                else:
                    found["tools"] = LeafRef("tool")
            for sub in value.values():
                walk(sub)
        elif isinstance(value, list):
            for sub in value:
                walk(sub)

    walk(tree)
    return found


def assert_roundtrip(tree: dict):
    """THE LAW: compile(print(tree)) == tree, byte-for-byte JSON."""
    function = printer.to_function(tree, tag="roundtrip")
    back = compile_forward(function, leaves_of(tree))
    assert json.dumps(back) == json.dumps(tree)


def lookup(country: str) -> str:
    """Return the capital of a country from a fixed table."""
    capitals = {"france": "Paris", "canada": "Ottawa"}
    return capitals.get(country.lower(), "unknown")


# ---------------------------------------------------------------------------
# 1. Builder-vs-parser byte equality
# ---------------------------------------------------------------------------


class TestBuilderParserByteEquality:
    def test_representative_forward(self):
        def forward(self, question, hints):
            trajectory = {}
            count = 0
            for idx in range(3):
                pred = self.step(question=question, notes=trajectory.get("notes", ""))
                trajectory["thought_" + str(idx)] = pred.thought
                if pred.thought == "done" or count > 2:
                    break
                try:
                    observation = self.tools[pred.tool](args=pred.args)
                except ToolError:  # noqa: F821 — resolved by the engine's error table
                    observation = "failed"
                count = count + 1
                trajectory["observation"] = observation
            summary = self.finalize(question=question, journal=json.dumps(trajectory), first=hints[0])
            return summary

        leaves = {
            "step": LeafRef("predict", "step"),
            "finalize": LeafRef("predict", "finalize"),
            "tools": LeafRef("tool"),
        }
        parsed = compile_forward(forward, leaves)

        b = build
        built = b.Forward(
            ["question", "hints"],
            [
                b.Assign("trajectory", b.Dict()),
                b.Assign("count", b.Const(0)),
                b.For(
                    "idx",
                    [
                        b.Assign(
                            "pred",
                            b.CallPredict(
                                "step",
                                question=b.Var("question"),
                                notes=b.Method(b.Var("trajectory"), "get", b.Const("notes"), b.Const("")),
                            ),
                        ),
                        b.SetIndex(
                            "trajectory",
                            b.BinOp("add", b.Const("thought_"), b.Builtin("str", b.Var("idx"))),
                            b.Attr("pred", "thought"),
                        ),
                        b.If(
                            b.BoolOp(
                                "or",
                                b.Compare("eq", b.Attr("pred", "thought"), b.Const("done")),
                                b.Compare("gt", b.Var("count"), b.Const(2)),
                            ),
                            [b.Break()],
                        ),
                        b.Try(
                            [
                                b.Assign(
                                    "observation",
                                    b.CallToolDynamic(b.Attr("pred", "tool"), args=b.Attr("pred", "args")),
                                )
                            ],
                            [b.Except("ToolError", [b.Assign("observation", b.Const("failed"))])],
                        ),
                        b.Assign("count", b.BinOp("add", b.Var("count"), b.Const(1))),
                        b.SetIndex("trajectory", b.Const("observation"), b.Var("observation")),
                    ],
                    range=3,
                ),
                b.Assign(
                    "summary",
                    b.CallPredict(
                        "finalize",
                        question=b.Var("question"),
                        journal=b.Builtin("json_dumps", b.Var("trajectory")),
                        first=b.Index(b.Var("hints"), b.Const(0)),
                    ),
                ),
                b.Return(b.Var("summary")),
            ],
        )
        assert json.dumps(built) == json.dumps(parsed)

    def test_literal_coercion_matches_spelled_nodes(self):
        # Python literals in expression slots lower exactly as Const/List/Dict.
        sugar = build.Assign("x", {"kind": "note", "tags": ["a", 1, None], "on": True})
        spelled = build.Assign(
            "x",
            build.Dict(
                [
                    (build.Const("kind"), build.Const("note")),
                    (build.Const("tags"), build.List(build.Const("a"), build.Const(1), build.Const(None))),
                    (build.Const("on"), build.Const(True)),
                ]
            ),
        )
        assert json.dumps(sugar) == json.dumps(spelled)


# ---------------------------------------------------------------------------
# 2. THE ROUND-TRIP LAW
# ---------------------------------------------------------------------------


def all_statements_tree() -> dict:
    """Every statement kind (both For forms, multi-handler Try, Log shapes)."""
    b = build
    return b.Forward(
        ["question", "flags"],
        [
            b.Assign("total", b.Const(0)),
            b.AssignTuple(["first", "second"], b.List(b.Const(1), b.Const(2))),
            b.Assign("journal", b.Dict()),
            b.SetIndex("journal", b.Const("first"), b.Var("first")),
            b.For(
                "idx",
                [
                    b.Assign("total", b.BinOp("add", b.Var("total"), b.Var("idx"))),
                    b.If(b.Compare("eq", b.Var("idx"), b.Const(1)), [b.Continue()]),
                ],
                range=4,
            ),
            b.For("flag", [b.If(b.Var("flag"), [b.Break()], [b.Pass()])], iter=b.Var("flags")),
            b.While(
                b.Compare("lt", b.Var("total"), b.Const(100)),
                [b.Assign("total", b.BinOp("mult", b.Var("total"), b.Const(2))), b.Break()],
            ),
            b.Try(
                [
                    b.Assign("pred", b.CallPredict("step", question=b.Var("question"))),
                    b.Raise("ToolError", "unreachable"),
                ],
                [
                    b.Except("AdapterParseError", [b.Assign("pred", b.Const(None))]),
                    b.Except("Exception", [b.Pass()]),
                ],
            ),
            b.Log(),
            b.Log(b.Const("total:"), b.Var("total"), b.Attr("pred", "answer")),
            b.Return(b.Var("total")),
        ],
    )


def all_expressions_tree() -> dict:
    """Every expression kind, precedence traps included."""
    b = build
    rows = [
        b.Assign("scalars", b.List(b.Const("s"), b.Const(1), b.Const(2.5), b.Const(True), b.Const(None), b.Const(-3))),
        b.Assign("nested_add", b.BinOp("add", b.Var("a"), b.BinOp("add", b.Var("b"), b.Var("c")))),
        b.Assign("mixed", b.BinOp("mult", b.BinOp("add", b.Var("a"), b.Var("b")), b.Const(2))),
        b.Assign("division", b.BinOp("div", b.Var("a"), b.BinOp("sub", b.Var("b"), b.Const(-5)))),
        b.Assign("compared", b.Compare("eq", b.Compare("lt", b.Var("a"), b.Var("b")), b.Const(True))),
        b.Assign(
            "membership",
            b.BoolOp(
                "and", b.Compare("in", b.Var("a"), b.Var("scalars")), b.Compare("not_in", b.Var("b"), b.Var("scalars"))
            ),
        ),
        b.Assign(
            "orderings",
            b.BoolOp(
                "or",
                b.Compare("le", b.Var("a"), b.Var("b")),
                b.Compare("ge", b.Var("a"), b.Var("b")),
                b.Compare("ne", b.Var("a"), b.Var("b")),
                b.Compare("gt", b.Var("a"), b.Var("b")),
            ),
        ),
        b.Assign("boolean_nest", b.BoolOp("or", b.BoolOp("and", b.Var("p"), b.Var("q")), b.UnaryOp("not", b.Var("p")))),
        b.Assign("negated", b.UnaryOp("neg", b.BinOp("add", b.Var("a"), b.Var("b")))),
        b.Assign("chosen", b.IfExp(b.Var("p"), b.Var("a"), b.IfExp(b.Var("q"), b.Var("b"), b.Var("c")))),
        b.Assign("keyed", b.Index(b.Var("record"), b.Const("answer"))),
        b.Assign("head", b.Slice(b.Var("scalars"), stop=b.Const(2))),
        b.Assign("tail", b.Slice(b.Var("scalars"), start=b.Const(1))),
        b.Assign("window", b.Slice(b.Var("scalars"), start=b.Const(1), stop=b.Const(-1))),
        b.Assign("reversed_copy", b.Slice(b.Var("scalars"), step=-1)),
        b.Assign("described", b.Format(b.Const("total "), b.Var("a"), b.Const(" of "), b.Var("b"))),
        b.Assign("adjacent", b.Format(b.Const("a"), b.Const("b"), b.Const(""), b.Var("a"))),
        b.Assign("numbered", b.Format(b.Const(5), b.Const(" items"))),
        b.Assign("table", b.Dict([(b.Var("key"), b.Var("a")), (b.Const("fixed"), b.Const(1))])),
        b.Return(b.Var("described")),
    ]
    return build.Forward(["a", "b", "c", "p", "q", "key", "record"], rows)


def all_calls_tree() -> dict:
    """Every builtin, every method, every leaf kind, dynamic dispatch."""
    b = build
    rows = [
        b.Assign("pred", b.CallPredict("step", question=b.Var("question"))),
        b.Assign("routed", b.CallModule("inner", question=b.Var("question"))),
        b.Assign("fetched", b.CallTool("fetch", url=b.Var("question"))),
        b.Assign("observed", b.CallToolDynamic(b.Attr("pred", "tool"), args=b.Attr("pred", "args"))),
        b.Assign("ran", b.CallInterpreter("interp", code=b.Attr("pred", "code"))),
        b.Assign(
            "sizes",
            b.List(
                b.Builtin("len", b.Var("question")), b.Builtin("int", b.Const("3")), b.Builtin("float", b.Const("2.5"))
            ),
        ),
        b.Assign(
            "stats",
            b.List(
                b.Builtin("max", b.Var("sizes")), b.Builtin("min", b.Var("sizes")), b.Builtin("sum", b.Var("sizes"))
            ),
        ),
        b.Assign("orderly", b.Builtin("sorted", b.Var("sizes"))),
        b.Assign("truthy", b.List(b.Builtin("any", b.Var("flags")), b.Builtin("all", b.Var("flags")))),
        b.Assign("rendered", b.Builtin("str", b.Var("stats"))),
        b.Assign("packed", b.Builtin("json_dumps", b.Var("stats"))),
        b.Assign("unpacked", b.Builtin("json_parse", b.Var("packed"))),
        b.Assign(
            "trimmed",
            b.Method(b.Method(b.Method(b.Var("rendered"), "strip"), "lstrip", b.Const("[")), "rstrip", b.Const("]")),
        ),
        b.Assign("cased", b.Method(b.Method(b.Var("trimmed"), "lower"), "upper")),
        b.Assign("pieces", b.Method(b.Var("cased"), "split", b.Const(","))),
        b.Assign("joined", b.Method(b.Const(", "), "join", b.Var("pieces"))),
        b.Assign("swapped", b.Method(b.Var("joined"), "replace", b.Const("A"), b.Const("B"))),
        b.Assign(
            "edges",
            b.List(
                b.Method(b.Var("swapped"), "startswith", b.Const("B")),
                b.Method(b.Var("swapped"), "endswith", b.Const("Z")),
            ),
        ),
        b.Do(b.Method(b.Var("pieces"), "append", b.Const("tail"))),
        b.Do(b.Method(b.Var("pieces"), "extend", b.List(b.Const("x"), b.Const("y")))),
        b.Assign("last", b.Method(b.Var("pieces"), "pop")),
        b.Assign("journal", b.Dict([(b.Const("last"), b.Var("last"))])),
        b.Assign("names", b.Method(b.Var("journal"), "keys")),
        b.Assign("stored", b.Method(b.Var("journal"), "values")),
        b.Assign("both", b.Method(b.Var("journal"), "items")),
        b.Do(b.Method(b.Var("journal"), "update", b.Dict([(b.Const("extra"), b.Const(1))]))),
        b.Assign("looked", b.Method(b.Var("journal"), "get", b.Const("missing"), b.Const(""))),
        b.Return(b.Var("looked")),
    ]
    return build.Forward(["question", "flags"], rows)


class TestRoundTripLaw:
    def test_all_statement_kinds(self):
        assert_roundtrip(all_statements_tree())

    def test_all_expression_kinds(self):
        assert_roundtrip(all_expressions_tree())

    def test_all_call_kinds(self):
        assert_roundtrip(all_calls_tree())

    def test_the_three_plain_forward_modules(self):
        # A9: the three signature-polymorphic modules are PLAIN forwards
        # (the v0.4 record envelope, D-041) — their COMPILED forward, splat
        # and all, obeys the round-trip law. The compiled tree comes from
        # the manifest now (no _forward_tree builder).
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        modules = [
            dspy.ReAct("question -> answer", tools=[lookup], max_iters=5),
            dspy.ReActV2("question -> answer", tools=[lookup], max_iters=4),
            dspy.ReActV2("question -> answer", max_iters=3),  # toolless submit agent
            dspy.RLM("question -> answer", max_iters=3),
        ]
        for module in modules:
            forward = module.to_manifest()["components"]["5_forward"]["self"]
            # Each carries the record envelope; the round trip reproduces it.
            assert forward["args"] == [{"name": "inputs", "record": "self"}]
            assert_roundtrip(forward)

    def test_every_corpus_forward(self):
        # The suite's own programs: branchy composite, desugar-heavy
        # source, bare Predict, ChainOfThought — every compiled forward
        # obeys the law.
        class Branchy(dspy.Module):
            def __init__(self):
                self.classify = dspy.Predict("question -> category")
                self.solve = dspy.Predict("question, hint -> answer")

            def forward(self, question):
                route = self.classify(question=question)
                if route.category == "math":
                    answer = self.solve(question=question, hint="use arithmetic")
                else:
                    answer = self.solve(question=question, hint="answer plainly")
                return answer

        class Desugary(dspy.Module):
            """Comprehension, walrus, dict merge, enumerate — gensym names."""

            def __init__(self):
                self.solve = dspy.Predict("question -> answer")

            def forward(self, question, options):
                lengths = [len(option) for option in options]
                merged = {"count": len(lengths)} | {"question": question}
                for position, option in enumerate(options):
                    merged["option_" + str(position)] = option
                if (biggest := max(lengths)) > 3:
                    merged["biggest"] = biggest
                answer = self.solve(question=json.dumps(merged))
                return answer

        dspy.configure(lm=dspy.DummyLM(["unused"]))
        programs = [
            Branchy(),
            Desugary(),
            dspy.Predict("question -> answer"),
            dspy.ChainOfThought("question -> answer"),
        ]
        checked = 0
        for program in programs:
            for tree in program.to_manifest()["components"]["5_forward"].values():
                assert_roundtrip(tree)
                checked += 1
        assert checked == len(programs)

    def test_v03_forward_compiles_byte_identical(self):
        # The v0.4 inputs-bag admission (D-041) must not perturb v0.3
        # lowering: a plain-named forward compiles to the SAME tree — plain
        # string args, no record binding, no splat — that it did at v0.3.
        # Compiled here against the CURRENT (v0.4) compiler and pinned.
        def forward(self, question, hint):
            answer = self.solve(question=question, hint=hint)
            return answer

        leaves = {"solve": LeafRef("predict", "solve")}
        tree = compile_forward(forward, leaves)
        assert tree == {
            "language": "restricted-python-ast",
            "args": ["question", "hint"],  # plain strings — NOT a record binding
            "body": [
                {
                    "node": "Assign",
                    "target": "answer",
                    "value": {
                        "node": "Call",
                        "leaf": {"kind": "predict", "ref": "solve"},
                        "kwargs": {
                            "question": {"node": "Var", "name": "question"},
                            "hint": {"node": "Var", "name": "hint"},
                        },
                    },
                },
                {"node": "Return", "value": {"node": "Var", "name": "answer"}},
            ],
        }
        # No splat field anywhere in a v0.3 forward.
        assert "splat" not in json.dumps(tree)

    def test_raise_with_negative_numeric_message(self):
        # `raise LMError(-3)` prints from an admitted tree, and `-3` parses
        # as USub(Constant(3)); the parser must fold it back or the law
        # breaks on every admitted negative message.
        from dspy.modules._generate import load_generated

        b = build
        for message in (-3, -3.5):
            assert_roundtrip(b.Forward(["q"], [b.Raise("LMError", message)]))
        # Negative zero stays out of the value model on BOTH frontends.
        assert repr(b.Raise("ToolError", -0.0)["message"]) == "0.0"
        source = "def forward(self, q):\n    raise ToolError(-0.0)\n"
        parsed = compile_forward(load_generated(source, tag="negzero", name="forward"), {})
        assert repr(parsed["body"][0]["message"]) == "0.0"

    def test_record_splat_round_trips(self):
        # THE SPLAT LAW: the v0.4 record envelope + `**inputs` splat prints
        # and reparses to the SAME tree (D-041). A synthetic tree drives
        # both call forms — static leaf splat and dynamic-tool splat — plus
        # the splat-first-then-kwargs merge order.
        b = build
        tree = b.Forward(
            [],
            record="inputs",
            body=[
                b.Assign("pred", b.CallPredict("step", splat="inputs", trajectory=b.Var("t"))),
                b.Assign("more", b.CallModule("inner", splat="inputs")),
                b.Assign("obs", b.CallToolDynamic(b.Attr("pred", "tool"), splat="inputs", args=b.Var("a"))),
                b.Return(b.Var("pred")),
            ],
        )
        assert tree["args"] == [{"name": "inputs", "record": "self"}]
        assert tree["body"][0]["value"]["splat"] == "inputs"
        assert_roundtrip(tree)

    def test_record_splat_rule_is_enforced_in_the_shared_gate(self):
        # THE D-041 LINE THAT MUST HOLD (PIR-E-NODE-002): a leaf-call splat
        # is lawful ONLY when it names the forward's record. The parse
        # compiler checked this, but the builder frontend and the
        # printer-reparse path bypassed it — the shared gate (admit_forward)
        # never re-checked. It does now, so both frontends agree.
        b = build

        # (1) The builder bypass: splat='pred' names an LM-produced value,
        # not the record. This is exactly the LM-decided-dict form the
        # source compiler refuses; the builder must refuse it too.
        with pytest.raises(ProgramIRRefusal, match="PIR-E-NODE-002|not the forward's record"):
            b.Forward(
                [],
                record="inputs",
                body=[
                    b.Assign("pred", b.CallPredict("step", splat="inputs")),
                    b.Assign("obs", b.CallToolDynamic(b.Attr("pred", "tool"), splat="pred")),
                ],
            )

        # (2) A splat in a plain-form forward (no record envelope) names no
        # declared record — unlawful.
        with pytest.raises(ProgramIRRefusal, match="PIR-E-NODE-002|no declared record"):
            b.Forward(
                ["question"],
                body=[b.Assign("pred", b.CallPredict("step", splat="question"))],
            )

        # (3) Round-trip drift closed: a lawful record-splat tree prints and
        # reparses (already covered above); the shared gate refuses the
        # unlawful shape at admission, so compile(print(tree)) can never
        # disagree with the builder on it — both raise the same code.
        function = printer.to_function(
            b.Forward([], record="inputs", body=[b.Assign("pred", b.CallPredict("step", splat="inputs"))]),
            tag="splatlaw",
        )
        # A hand-built tree whose splat names a non-record — the exact shape
        # the round-trip claim guards — refuses at admission (via the shared
        # gate directly).
        with pytest.raises(ProgramIRRefusal) as excinfo:
            admit_forward(
                {
                    "language": "restricted-python-ast",
                    "args": [{"name": "inputs", "record": "self"}],
                    "body": [
                        {
                            "node": "Assign",
                            "target": "obs",
                            "value": {"node": "Call", "leaf": {"kind": "tool", "ref": "t"}, "kwargs": {}, "splat": "pred"},
                        }
                    ],
                }
            )
        assert excinfo.value.code == "PIR-E-NODE-002"
        assert function  # the lawful print produced source

    def test_literal_cap_bakes_into_the_compiled_forward(self):
        # The literal loop cap survives the plain-forward move: max_iters
        # bakes as the For range in the compiled artifact, and an edit
        # recompiles it (ir_literals; the mechanism is orthogonal to the
        # record envelope, D-041).
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        react = dspy.ReAct("question -> answer", tools=[lookup], max_iters=5)
        assert react.to_manifest()["components"]["5_forward"]["self"]["body"][1]["range"] == 5
        react.max_iters = 2
        assert react.to_manifest()["components"]["5_forward"]["self"]["body"][1]["range"] == 2


# ---------------------------------------------------------------------------
# 3a. Hostile names refuse at BUILD time
# ---------------------------------------------------------------------------


class TestBuildTimeHygiene:
    @pytest.mark.parametrize(
        ("hostile", "problem"),
        [
            ("foo (x)", "is not a Python identifier"),
            ("foo bar", "is not a Python identifier"),
            ("class", "is a Python keyword"),
            ("__dict__", "is a dunder"),
            ("self", "is the module receiver"),
        ],
    )
    def test_hostile_names_refuse_everywhere(self, hostile, problem):
        with pytest.raises(ProgramIRRefusal, match=problem):
            build.Var(hostile)
        with pytest.raises(ProgramIRRefusal, match=problem):
            build.Assign(hostile, build.Const(1))
        with pytest.raises(ProgramIRRefusal, match=problem):
            build.CallTool(hostile)
        with pytest.raises(ProgramIRRefusal, match=problem):
            build.Attr("pred", hostile)
        with pytest.raises(ProgramIRRefusal, match=problem):
            build.Forward([hostile], [build.Return(build.Const(1))])

    def test_hostile_kwarg_name_refuses(self):
        with pytest.raises(ProgramIRRefusal, match="is a Python keyword"):
            build.CallPredict("step", **{"class": build.Var("x")})

    def test_closed_tables_refuse_at_construction(self):
        with pytest.raises(ProgramIRRefusal, match="unknown builtin 'reversed'"):
            build.Builtin("reversed", build.Var("x"))
        with pytest.raises(ProgramIRRefusal, match="outside the closed 18-name"):
            build.Method(build.Var("x"), "shuffle")
        with pytest.raises(ProgramIRRefusal, match="outside the typed error table"):
            build.Raise("ValueError", "boom")
        with pytest.raises(ProgramIRRefusal, match="outside the typed error table"):
            build.Except("KeyError", [build.Pass()])
        with pytest.raises(ProgramIRRefusal, match="outside"):
            build.Compare("xor", build.Var("a"), build.Var("b"))

    def test_shape_rules_refuse_at_construction(self):
        with pytest.raises(ProgramIRRefusal, match="exactly one of range="):
            build.For("i", [build.Pass()])
        with pytest.raises(ProgramIRRefusal, match="non-negative int literal"):
            build.For("i", [build.Pass()], range=-1)
        with pytest.raises(ProgramIRRefusal, match="non-negative int literal"):
            build.For("i", [build.Pass()], range=True)
        with pytest.raises(ProgramIRRefusal, match="two or more names"):
            build.AssignTuple(["only"], build.Var("x"))
        with pytest.raises(ProgramIRRefusal, match="distinct names"):
            build.AssignTuple(["a", "a"], build.Var("x"))
        with pytest.raises(ProgramIRRefusal, match="JSON scalar"):
            build.Const([1, 2])
        with pytest.raises(ProgramIRRefusal, match="non-finite"):
            build.Const(float("inf"))
        with pytest.raises(ProgramIRRefusal, match="exactly one positional argument"):
            build.Builtin("json_dumps", build.Var("a"), build.Var("b"))
        with pytest.raises(ProgramIRRefusal, match="literal 1 or -1"):
            build.Slice(build.Var("xs"), step=2)
        with pytest.raises(ProgramIRRefusal, match="must be a statement node"):
            build.If(build.Var("p"), [build.Var("x")])
        with pytest.raises(ProgramIRRefusal, match="statements live in bodies"):
            build.Assign("x", build.Pass())

    def test_neg_of_literal_folds_like_the_parser(self):
        assert build.UnaryOp("neg", build.Const(5)) == {"node": "Const", "value": -5}
        assert build.UnaryOp("neg", build.Const(0.0)) == {"node": "Const", "value": 0.0}


# ---------------------------------------------------------------------------
# 3b. Bad-tree refusal parity: built twin and parsed twin, one law
# ---------------------------------------------------------------------------


def _parse_refusal(source: str, leaves: dict | None = None) -> ProgramIRRefusal:
    from dspy.modules._generate import load_generated

    function = load_generated(source, tag="parity", name="forward")
    with pytest.raises(ProgramIRRefusal) as caught:
        compile_forward(function, leaves or {})
    return caught.value


def _admit_refusal(tree: dict, leaves: dict | None = None) -> ProgramIRRefusal:
    with pytest.raises(ProgramIRRefusal) as caught:
        admit_forward(tree, leaves)
    return caught.value


def _wrap(statement: dict) -> dict:
    return {"language": "restricted-python-ast", "args": ["x"], "body": [statement]}


class TestBadTreeRefusalParity:
    def test_unknown_method_same_refusal(self):
        built = _wrap(
            {
                "node": "Assign",
                "target": "y",
                "value": {"node": "Call", "method": "shuffle", "obj": {"node": "Var", "name": "x"}, "args": []},
            }
        )
        tree_side = _admit_refusal(built)
        source_side = _parse_refusal("def forward(self, x):\n    y = x.shuffle()\n    return y\n")
        assert tree_side.code == source_side.code == "PIR-E-NODE-002"
        assert tree_side.detail["target"] == source_side.detail["target"] == "shuffle"

    def test_unknown_builtin_same_refusal(self):
        built = _wrap(
            {
                "node": "Assign",
                "target": "y",
                "value": {"node": "Call", "builtin": "reversed", "args": [{"node": "Var", "name": "x"}]},
            }
        )
        tree_side = _admit_refusal(built)
        source_side = _parse_refusal("def forward(self, x):\n    y = reversed(x)\n    return y\n")
        assert tree_side.code == source_side.code == "PIR-E-NODE-002"
        assert "reversed" in str(tree_side) and "reversed" in str(source_side)

    def test_untyped_raise_same_refusal(self):
        built = _wrap({"node": "Raise", "exc": "ValueError", "message": "boom"})
        tree_side = _admit_refusal(built)
        source_side = _parse_refusal("def forward(self, x):\n    raise ValueError('boom')\n")
        assert tree_side.code == source_side.code == "PIR-E-NODE-001"
        assert tree_side.detail["node"] == source_side.detail["node"] == "Raise"

    def test_json_arity_same_refusal(self):
        built = _wrap(
            {
                "node": "Assign",
                "target": "y",
                "value": {
                    "node": "Call",
                    "builtin": "json_dumps",
                    "args": [{"node": "Var", "name": "x"}, {"node": "Var", "name": "x"}],
                },
            }
        )
        tree_side = _admit_refusal(built)
        source_side = _parse_refusal("def forward(self, x):\n    y = json.dumps(x, x)\n    return y\n")
        assert tree_side.code == source_side.code == "PIR-E-NODE-001"
        assert "exactly one positional argument" in str(tree_side)
        assert "exactly one positional argument" in str(source_side)

    def test_undeclared_leaf_same_refusal(self):
        leaves = {"step": LeafRef("predict", "step")}
        built = _wrap(
            {
                "node": "Assign",
                "target": "y",
                "value": {"node": "Call", "leaf": {"kind": "predict", "ref": "nope"}, "kwargs": {}},
            }
        )
        tree_side = _admit_refusal(built, leaves)
        source_side = _parse_refusal("def forward(self, x):\n    y = self.nope(question=x)\n    return y\n", leaves)
        assert tree_side.code == source_side.code == "PIR-E-NODE-002"
        assert "nope" in str(tree_side) and "nope" in str(source_side)

    def test_finalize_catches_malformed_nodes_the_constructors_missed(self):
        # A statement dict with the right kind but a missing field slips
        # past construction; the outer schema check refuses it.
        with pytest.raises(ProgramIRRefusal, match="required"):
            build.Forward(["x"], [{"node": "Assign", "target": "y"}, build.Return(build.Var("y"))])

    def test_unknown_node_kind_refuses(self):
        refusal = _admit_refusal(_wrap({"node": "Goto", "target": "y"}))
        assert refusal.code == "PIR-E-NODE-001"
        assert "Goto" in str(refusal)


class TestSharedNameHygiene:
    """Findings of the A7 adversarial review: name hygiene is ONE law.

    The schema's identifier pattern admits dunders, keywords, and
    'self'; only the builder refused them, so the two frontends drifted
    (parse admitted `rec.__class__`) and foreign trees printed to
    unloadable or unreparsable source. `admit_forward` now carries the
    builder's rule over every identifier position.
    """

    def test_dunder_attr_same_refusal_on_both_frontends(self):
        source_side = _parse_refusal("def forward(self, rec):\n    y = rec.__class__\n    return y\n")
        tree_side = _admit_refusal(
            _wrap({"node": "Assign", "target": "y", "value": {"node": "Attr", "obj": "rec", "attr": "__class__"}})
        )
        with pytest.raises(ProgramIRRefusal, match="is a dunder") as built:
            build.Attr("rec", "__class__")
        assert source_side.code == tree_side.code == built.value.code == "PIR-E-NODE-001"
        assert "__class__" in str(source_side) and "is a dunder" in str(source_side)
        assert tree_side.detail["name"] == built.value.detail["name"] == "__class__"

    def test_dunder_variable_names_refuse_at_parse(self):
        refusal = _parse_refusal("def forward(self, x):\n    __weird__ = x\n    return __weird__\n")
        assert refusal.code == "PIR-E-NODE-001"
        assert "is a dunder" in str(refusal)

    @pytest.mark.parametrize(
        ("statement", "offender", "problem"),
        [
            ({"node": "Return", "value": {"node": "Var", "name": "class"}}, "class", "Python keyword"),
            ({"node": "Assign", "target": "pass", "value": {"node": "Const", "value": 1}}, "pass", "Python keyword"),
            ({"node": "Return", "value": {"node": "Attr", "obj": "self", "attr": "x"}}, "self", "module receiver"),
            ({"node": "Return", "value": {"node": "Var", "name": "__dict__"}}, "__dict__", "is a dunder"),
            (
                {"node": "For", "target": "in", "iter": {"node": "Var", "name": "x"}, "body": [{"node": "Pass"}]},
                "in",
                "Python keyword",
            ),
        ],
    )
    def test_schema_legal_hostile_names_refuse_at_admission(self, statement, offender, problem):
        # Every one of these trees passes the raw schema (the pattern
        # matches) but would print to source that cannot load or cannot
        # reparse; the shared admission refuses it by name.
        refusal = _admit_refusal(_wrap(statement))
        assert refusal.code == "PIR-E-NODE-001"
        assert refusal.detail["name"] == offender
        assert problem in str(refusal)

    def test_keyword_forward_argument_refuses_at_admission(self):
        tree = {
            "language": "restricted-python-ast",
            "args": ["class"],
            "body": [{"node": "Return", "value": {"node": "Const", "value": 1}}],
        }
        refusal = _admit_refusal(tree)
        assert "Python keyword" in str(refusal)
        assert refusal.detail["role"] == "forward argument"

    def test_hostile_leaf_ref_and_kwarg_key_refuse_at_admission(self):
        bad_ref = _wrap(
            {
                "node": "Assign",
                "target": "y",
                "value": {"node": "Call", "leaf": {"kind": "predict", "ref": "__init__"}, "kwargs": {}},
            }
        )
        assert "is a dunder" in str(_admit_refusal(bad_ref))
        bad_key = _wrap(
            {
                "node": "Assign",
                "target": "y",
                "value": {
                    "node": "Call",
                    "leaf": {"kind": "predict", "ref": "step"},
                    "kwargs": {"__proto__": {"node": "Var", "name": "x"}},
                },
            }
        )
        assert "is a dunder" in str(_admit_refusal(bad_key))

    def test_the_self_leaf_convention_stays_admitted(self):
        # A bare Predict IS its own leaf: ref 'self' is the compiler's
        # convention, prints as `self.self(...)`, and round-trips.
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        tree = dspy.Predict("question -> answer").to_manifest()["components"]["5_forward"]["self"]
        assert tree["body"][0]["value"]["leaf"]["ref"] == "self"
        assert_roundtrip(tree)


# ---------------------------------------------------------------------------
# The printer refuses shapes no source lowers to
# ---------------------------------------------------------------------------


class TestPrinterHonesty:
    def test_log_outside_print_convention_refuses(self):
        tree = _wrap({"node": "Log", "parts": [{"node": "Var", "name": "x"}, {"node": "Var", "name": "x"}]})
        with pytest.raises(ProgramIRRefusal, match="separator convention"):
            printer.render_forward(tree)

    def test_neg_on_literal_refuses(self):
        tree = _wrap(
            {
                "node": "Assign",
                "target": "y",
                "value": {"node": "UnaryOp", "op": "neg", "value": {"node": "Const", "value": 5}},
            }
        )
        with pytest.raises(ProgramIRRefusal, match="folded Const"):
            printer.render_forward(tree)

    def test_slice_with_explicit_step_1_refuses(self):
        # The schema enum admits step 1, but source `[a:b:1]` reparses to
        # a step-less Slice — printing would silently MUTATE the tree.
        slice_node = {
            "node": "Slice",
            "obj": {"node": "Var", "name": "x"},
            "start": {"node": "Const", "value": 1},
            "stop": {"node": "Const", "value": 5},
            "step": 1,
        }
        tree = _wrap({"node": "Return", "value": slice_node})
        admit_forward(tree)  # schema-legal: the refusal is the PRINTER's
        with pytest.raises(ProgramIRRefusal, match="explicit step of 1"):
            printer.render_forward(tree)
        # The step-less spelling of the same slice still round-trips.
        without_step = {key: value for key, value in slice_node.items() if key != "step"}
        assert_roundtrip(_wrap({"node": "Return", "value": without_step}))
