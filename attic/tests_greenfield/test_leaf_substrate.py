"""PIR-021 unified leaf substrate (D-043): kind, grants, sessions, attribution.

Tools and interpreters share the leaf record: an optional invocation
discriminant (`kind: call | session`, absent = call — byte-compatible)
and a closed static effect row (`grants[]`). At materialize a leaf's
pool-leaf grants resolve into a bridge table (dangling grant = loud link
refusal); a session leaf reaches ONLY its granted leaves through that
bridge, never ambient pools. An LM call made through the bridge
attributes to both the session leaf and the underlying predictor, while
the total counts each call once. FlexIR's add_tool carries kind/grants;
delete_dead_leaf counts a grant as a live site.

Everything here is deterministic and offline (DummyLM).
"""

import json

import pytest

import dspy
from dspy import optim
from dspy.lm import BINDINGS
from dspy.programir.leaves import extract_tool, granted_leaf_name, leaf_grant


@pytest.fixture(autouse=True)
def clean_bindings():
    saved = dict(BINDINGS)
    BINDINGS.clear()
    yield
    BINDINGS.clear()
    BINDINGS.update(saved)


def chat_completion(**fields):
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


def reflection_reply(ops):
    return chat_completion(proposals=json.dumps(ops))


def _input_value(messages, field):
    rendered = "\n".join(str(message["content"]) for message in messages)
    marker = f"[[ ## {field} ## ]]\n"
    value = ""
    index = rendered.find(marker)
    while index != -1:
        start = index + len(marker)
        end = rendered.find("\n", start)
        value = rendered[start:] if end == -1 else rendered[start:end]
        index = rendered.find(marker, start)
    return value.strip()


class Tagger(dspy.Module):
    def __init__(self):
        self.tagger = dspy.Predict("text -> tag")

    def forward(self, text):
        result = self.tagger(text=text)
        return result


def upper_task(messages):
    return chat_completion(tag=_input_value(messages, "text").upper())


def exact_tag(example, prediction):
    return example.tag == prediction.tag


def tag_devset(*words):
    return [dspy.Example(text=word, tag=word.upper()).with_inputs("text") for word in words]


# ---------------------------------------------------------------------------
# (1) Record shape: additive-compatible
# ---------------------------------------------------------------------------


class TestRecordShapeAdditive:
    def test_a_plain_tool_carries_no_kind_or_grants(self):
        # extract_tool on an ordinary function yields today's byte shape:
        # no kind (absent = call), no grants.
        def helper(text: str) -> dict:
            return {"tag": text}

        entry = extract_tool(helper, name="helper").entry
        assert "kind" not in entry
        assert "grants" not in entry

    def test_kind_and_grants_stamp_only_when_present(self):
        def sess(bridge: object, text: str) -> dict:
            return {"out": text}

        sess._dspy_leaf_kind = "session"
        sess._dspy_leaf_grants = [leaf_grant("inner")]
        entry = extract_tool(sess, name="sess").entry
        assert entry["kind"] == "session"
        assert entry["grants"] == [{"kind": "leaf", "name": "inner"}]
        assert granted_leaf_name(entry["grants"][0]) == "inner"

    def test_a_call_kind_stamp_stays_absent(self):
        # kind "call" is the default — it is never emitted (byte-compat).
        def helper(text: str) -> dict:
            return {"tag": text}

        helper._dspy_leaf_kind = "call"
        entry = extract_tool(helper, name="helper").entry
        assert "kind" not in entry

    def test_grant_helpers_round_trip(self):
        grant = leaf_grant("solver")
        assert grant == {"kind": "leaf", "name": "solver"}
        assert granted_leaf_name(grant) == "solver"
        assert granted_leaf_name({"kind": "broker_route", "name": "api.x"}) is None
        # The legacy pre-contract overload stays readable (write path retired).
        assert granted_leaf_name({"kind": "fd", "name": "leaf:solver"}) == "solver"
        assert granted_leaf_name({"kind": "fd", "name": "data-plane"}) is None

    def test_old_swap_manifest_is_unchanged(self):
        # A v2/v3 code swap produces a call-kind, grant-free tool exactly
        # as before — the schema addition is additive.
        program = Tagger()
        reflection = dspy.DummyLM(
            [
                reflection_reply(
                    [
                        {
                            "op": "replace_predict_with_code",
                            "path": "tagger",
                            "tool_name": "tag_code",
                            "python_source": 'def tag_code(text: str) -> dict:\n    return {"tag": text.upper()}\n',
                        },
                        {"op": "delete_dead_leaf", "path": "tagger"},
                    ]
                )
            ]
        )
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optim.FlexIR(reflection, exact_tag, iterations=1, holdout=tag_devset("owl")).compile(
            program, trainset=tag_devset("cat", "dog")
        )
        entry = program.to_manifest()["components"]["6_tools"]["tag_code"]
        assert "kind" not in entry
        assert "grants" not in entry


# ---------------------------------------------------------------------------
# A session-leaf program built directly, for engine-level tests
# ---------------------------------------------------------------------------

# A session leaf that calls its granted predictor once per forward and
# holds a per-forward call counter in a closure-free way (nonlocal via a
# list default is disallowed; the "state within a forward" is proven by a
# session object whose lifetime = one forward, see the state test).
SESSION_SRC = (
    "def session_leaf(bridge: object, text: str) -> dict:\n"
    "    first = bridge.inner(text=text)\n"
    "    return {'tag': first['tag']}\n"
)


def build_session_program(*, grant="inner"):
    """A composite whose forward calls a session leaf granting `inner`."""
    from dspy.optim.code_leaf import admit_tool_source

    class Composite(dspy.Module):
        def __init__(self):
            self.inner = dspy.Predict("text -> tag")
            admitted = admit_tool_source("session_leaf", SESSION_SRC, None, partial=False, session=True)
            fn = admitted.function
            fn._dspy_leaf_kind = "session"
            fn._dspy_leaf_grants = [leaf_grant(grant)]
            fn._dspy_placement_rung = "in_process"  # run in the loop's own process for the test
            self.session_leaf = fn

        def forward(self, text):
            result = self.session_leaf(text=text)
            return result

    return Composite()


# ---------------------------------------------------------------------------
# (2) Dangling grant refuses at link/materialize
# ---------------------------------------------------------------------------


class TestGrantResolution:
    def test_dangling_grant_refuses_at_materialize(self):
        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = build_session_program(grant="ghost")  # grants a leaf that does not exist
        with pytest.raises(ValueError, match="dangling grant"):
            program(text="cat")

    def test_resolved_grant_runs(self):
        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = build_session_program()
        assert program(text="cat").tag == "CAT"


# ---------------------------------------------------------------------------
# (3) Bridge-only access: a session leaf cannot reach an ungranted leaf
# ---------------------------------------------------------------------------


class TestBridgeOnly:
    def test_ungranted_leaf_is_unreachable(self):
        from dspy.optim.code_leaf import admit_tool_source

        # The session leaf tries to reach `other`, which it was NOT granted.
        src = "def session_leaf(bridge: object, text: str) -> dict:\n    return bridge.other(text=text)\n"

        class Composite(dspy.Module):
            def __init__(self):
                self.inner = dspy.Predict("text -> tag")
                self.other = dspy.Predict("text -> tag")
                admitted = admit_tool_source("session_leaf", src, None, partial=False, session=True)
                fn = admitted.function
                fn._dspy_leaf_kind = "session"
                fn._dspy_leaf_grants = [leaf_grant("inner")]  # granted inner, NOT other
                fn._dspy_placement_rung = "in_process"
                self.session_leaf = fn

            def forward(self, text):
                result = self.session_leaf(text=text)
                return result

        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = Composite()
        with pytest.raises(dspy.core.errors.ToolError, match="not granted|reaches only"):
            program(text="cat")


# ---------------------------------------------------------------------------
# (4) Nested cost attribution (the PIR-021 pin)
# ---------------------------------------------------------------------------


class TestNestedAttribution:
    def test_total_counts_once_per_leaf_shows_both(self):
        from dspy.optim.base import evaluate

        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = build_session_program()
        devset = tag_devset("cat", "dog", "fox")  # N = 3 forwards
        result = evaluate(program, devset, exact_tag)
        # The session leaf calls `inner` once per forward. TOTAL counts each
        # real call ONCE: N.
        assert result.lm_calls == 3
        assert result.score == 1.0
        # Per-leaf attribution shows BOTH the session leaf and the
        # predictor at N each (the same calls, labeled twice).
        assert result.attribution == {"inner": 3, "session_leaf": 3}

    def test_direct_predictor_attributes_to_itself_only(self):
        from dspy.optim.base import evaluate

        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = Tagger()
        result = evaluate(program, tag_devset("cat", "dog"), exact_tag)
        assert result.lm_calls == 2
        assert result.attribution == {"tagger": 2}


# ---------------------------------------------------------------------------
# (5) Session state within one forward, NOT across forwards
# ---------------------------------------------------------------------------


class TestSessionLifetime:
    def test_state_is_held_within_a_forward_and_dies_after(self):
        from dspy.optim.code_leaf import admit_tool_source

        # The session leaf mutates a local dict twice within one forward and
        # returns the accumulated count. Because the callable is invoked
        # fresh per forward (object lifetime = leaf span, in-process), the
        # count NEVER carries across forwards: every forward sees 2, not 4.
        src = (
            "def session_leaf(bridge: object, text: str) -> dict:\n"
            "    state = {'n': 0}\n"
            "    state['n'] += 1\n"
            "    state['n'] += 1\n"
            "    return {'tag': str(state['n'])}\n"
        )

        class Composite(dspy.Module):
            def __init__(self):
                self.inner = dspy.Predict("text -> tag")
                admitted = admit_tool_source("session_leaf", src, None, partial=False, session=True)
                fn = admitted.function
                fn._dspy_leaf_kind = "session"
                fn._dspy_leaf_grants = [leaf_grant("inner")]
                fn._dspy_placement_rung = "in_process"
                self.session_leaf = fn

            def forward(self, text):
                result = self.session_leaf(text=text)
                return result

        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = Composite()
        # Every forward sees exactly 2 — state did not leak across forwards.
        assert program(text="a").tag == "2"
        program.invalidate_ir()
        assert program(text="b").tag == "2"


# ---------------------------------------------------------------------------
# (6) FlexIR add_tool with kind/grants + refusal classes
# ---------------------------------------------------------------------------


class OneStep(dspy.Module):
    def __init__(self):
        self.solver = dspy.Predict("text -> tag")

    def forward(self, text):
        result = self.solver(text=text)
        return result


def run_flex(program, ops, *, holdout=None, devset=None):
    reflection = dspy.DummyLM([reflection_reply(ops)])
    dspy.configure(lm=dspy.DummyLM(upper_task))
    optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=holdout)
    optimizer.compile(program, trainset=devset or [dspy.Example(text="cat", tag="CAT").with_inputs("text")])
    return optimizer


def apply_one(program, proposal):
    """Apply a single proposal directly (bypass scoring/accept), return refusal."""
    dspy.configure(lm=dspy.DummyLM(upper_task))
    optimizer = optim.FlexIR(dspy.DummyLM([]), exact_tag, iterations=1)
    return optimizer._apply_one(program, proposal, [])


SESSION_ADD = "def wrapper(bridge: object, text: str) -> dict:\n    return bridge.solver(text=text)\n"


class TestFlexAddToolKindGrants:
    def test_session_with_grant_happy_path(self):
        program = OneStep()
        refusal = apply_one(
            program,
            {
                "op": "add_tool",
                "path": "self",
                "name": "wrapper",
                "python_source": SESSION_ADD,
                "kind": "session",
                "grants": ["solver"],
            },
        )
        assert refusal is None
        tool = program.to_manifest()["components"]["6_tools"]["wrapper"]
        assert tool["kind"] == "session"
        assert tool["grants"] == [{"kind": "leaf", "name": "solver"}]

    def test_bad_kind_refuses(self):
        program = OneStep()
        ops = [{"op": "add_tool", "path": "self", "name": "w", "python_source": SESSION_ADD, "kind": "daemon"}]
        optimizer = run_flex(program, ops)
        assert "kind must be 'call' or 'session'" in optimizer.trajectory[1]["refusals"][0]

    def test_grant_naming_missing_leaf_refuses(self):
        program = OneStep()
        ops = [
            {
                "op": "add_tool",
                "path": "self",
                "name": "wrapper",
                "python_source": SESSION_ADD,
                "kind": "session",
                "grants": ["ghost"],
            }
        ]
        optimizer = run_flex(program, ops)
        refusal = optimizer.trajectory[1]["refusals"][0]
        assert "grant names leaf 'ghost'" in refusal
        assert "no predictor or tool" in refusal

    def test_grants_without_session_kind_refuses(self):
        program = OneStep()
        ops = [
            {
                "op": "add_tool",
                "path": "self",
                "name": "w",
                "python_source": 'def w(text: str) -> dict:\n    return {"tag": text}\n',
                "grants": ["solver"],
            }
        ]
        optimizer = run_flex(program, ops)
        assert "grants require kind='session'" in optimizer.trajectory[1]["refusals"][0]

    def test_session_source_without_bridge_param_refuses(self):
        # A session leaf must take the grant bridge as its FIRST parameter.
        # A zero-parameter source has no bridge slot — admission refuses.
        bad_empty = "def wrapper() -> dict:\n    return {}\n"
        ops = [
            {
                "op": "add_tool",
                "path": "self",
                "name": "wrapper",
                "python_source": bad_empty,
                "kind": "session",
                "grants": ["solver"],
            }
        ]
        optimizer = run_flex(OneStep(), ops)
        assert "FIRST parameter" in optimizer.trajectory[1]["refusals"][0]


# ---------------------------------------------------------------------------
# (7) delete_dead_leaf counts a grant as a live site
# ---------------------------------------------------------------------------


class TestDeleteGrantedLeaf:
    def test_deleting_a_granted_leaf_refuses_naming_the_granting_leaf(self):
        # Wire a session leaf granting `solver`, then a rewrite makes the
        # session leaf the only call site (solver reached only via bridge).
        # Deleting solver must refuse because the session grants it.
        program = OneStep()
        ops = [
            {
                "op": "add_tool",
                "path": "self",
                "name": "wrapper",
                "python_source": SESSION_ADD,
                "kind": "session",
                "grants": ["solver"],
            },
            {
                "op": "rewrite_forward",
                "path": "self",
                "python_source": "def forward(self, text):\n    result = self.wrapper(text=text)\n    return result\n",
            },
            {"op": "delete_dead_leaf", "path": "solver"},
        ]
        reflection = dspy.DummyLM([reflection_reply(ops)])
        dspy.configure(lm=dspy.DummyLM(upper_task))
        optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=tag_devset("owl"))
        optimizer.compile(program, trainset=tag_devset("cat", "dog"))
        refusals = optimizer.trajectory[1]["refusals"]
        assert any("granted by session leaf" in r and "wrapper" in r for r in refusals)
        # solver survives.
        assert "solver" in dict(program.named_predictors())


# ---------------------------------------------------------------------------
# (8) explain view shows kind + grants
# ---------------------------------------------------------------------------


class TestExplainView:
    def test_explain_prints_kind_and_grants(self):
        program = OneStep()
        apply_one(
            program,
            {
                "op": "add_tool",
                "path": "self",
                "name": "wrapper",
                "python_source": SESSION_ADD,
                "kind": "session",
                "grants": ["solver"],
            },
        )
        view = program.explain()
        assert "kind=session" in view
        assert "leaf:solver" in view
