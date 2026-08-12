"""FlexIR environment rungs: install-on-admit and artifact-mode scoring.

Rung 2 (`auto_install`): after admission accepts a dep-carrying leaf,
missing granted packages are installed into the current env via `uv pip
install` — default off, failure is a teaching refusal, never an abort.
Rung 3 (`eval_mode="artifact"`): candidates are exported and scored in a
subprocess under the artifact's own environment, with the acceptance
rule, holdout gate, ledger, and unwind unchanged.

Offline determinism: the uv seam (`env_prepare._run_uv`) is faked for
install paths, and artifact-mode tests use the documented TEST-ONLY
`_eval_same_env` hatch (child = current interpreter), so the export/
harness/protocol/caching logic is genuinely exercised without building
environments. The genuine uv paths (a real install; a real uv-built
child env) are gated behind DSPY_FLEX_UV_TESTS=1 because they need
network or a warm uv cache.
"""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import dspy
from dspy import optim
from dspy.lm import BINDINGS
from dspy.optim import env_prepare
from dspy.optim.flex import _env_var_holding, _is_secret_kwarg


@pytest.fixture(autouse=True)
def clean_bindings():
    saved = dict(BINDINGS)
    BINDINGS.clear()
    yield
    BINDINGS.clear()
    BINDINGS.update(saved)


def chat_completion(**fields: str) -> str:
    parts = [f"[[ ## {name} ## ]]\n{value}" for name, value in fields.items()]
    return "\n\n".join([*parts, "[[ ## completed ## ]]"])


def reflection_reply(ops: list) -> str:
    return chat_completion(proposals=json.dumps(ops))


class Tagger(dspy.Module):
    def __init__(self):
        self.tagger = dspy.Predict("text -> tag")

    def forward(self, text):
        result = self.tagger(text=text)
        return result


def sc_upper_task(messages):
    """A SELF-CONTAINED task LM function (its source travels to the child)."""
    rendered = "\n".join(str(message["content"]) for message in messages)
    marker = "[[ ## text ## ]]\n"
    value = ""
    index = rendered.find(marker)
    while index != -1:
        start = index + len(marker)
        end = rendered.find("\n", start)
        value = rendered[start:] if end == -1 else rendered[start:end]
        index = rendered.find(marker, start)
    return "[[ ## tag ## ]]\n" + value.strip().upper() + "\n\n[[ ## completed ## ]]"


def exact_tag(example, prediction):
    return example.tag == prediction.tag


def tag_devset(*words: str):
    return [dspy.Example(text=word, tag=word.upper()).with_inputs("text") for word in words]


DEP_TOOL = 'def tag_code(text: str) -> dict:\n    # deps: beautifulsoup4\n    return {"tag": text.upper()}\n'

DEP_OPS = [
    {"op": "replace_predict_with_code", "path": "tagger", "tool_name": "tag_code", "python_source": DEP_TOOL},
    {"op": "delete_dead_leaf", "path": "tagger"},
]

SWAP_OPS = [
    {
        "op": "replace_predict_with_code",
        "path": "tagger",
        "tool_name": "tag_code",
        "python_source": 'def tag_code(text: str) -> dict:\n    return {"tag": text.upper()}\n',
    },
    {"op": "delete_dead_leaf", "path": "tagger"},
]


def run_flex(program, ops, *, task=sc_upper_task, devset=None, holdout=None, checkpoint_dir=None, **flex_kwargs):
    reflection = dspy.DummyLM([reflection_reply(ops)])
    dspy.configure(lm=dspy.DummyLM(task))
    optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=holdout, **flex_kwargs)
    optimizer.compile(program, trainset=devset or tag_devset("cat", "dog"), checkpoint_dir=checkpoint_dir)
    return optimizer


# ---------------------------------------------------------------------------
# (1) Rung 2 — auto_install
# ---------------------------------------------------------------------------


class TestAutoInstall:
    def test_default_off_is_exactly_todays_behavior(self, monkeypatch):
        # auto_install=False: no dep check, no uv call — even for a leaf
        # that carries a granted dep. Pinned by a runner that would fail
        # the test if invoked.
        def forbidden(args):
            raise AssertionError(f"uv was invoked with {args} while auto_install=False")

        monkeypatch.setattr(env_prepare, "_run_uv", forbidden)
        program = Tagger()
        optimizer = run_flex(
            program,
            DEP_OPS,
            holdout=tag_devset("owl"),
            allowed_deps=frozenset({"beautifulsoup4"}),
        )
        entry = optimizer.trajectory[1]
        assert entry["refusals"] == []
        assert entry["accepted"] is True
        assert program.to_manifest()["components"]["6_tools"]["tag_code"]["deps"] == ["beautifulsoup4"]

    def test_install_failure_is_a_teaching_refusal_not_an_abort(self, monkeypatch):
        monkeypatch.setattr(env_prepare, "missing_deps", lambda deps: list(deps))
        monkeypatch.setattr(
            env_prepare,
            "_run_uv",
            lambda args: subprocess.CompletedProcess(args, 1, stdout="", stderr="error: No solution found\n"),
        )
        program = Tagger()
        optimizer = run_flex(
            program,
            DEP_OPS,
            holdout=tag_devset("owl"),
            allowed_deps=frozenset({"beautifulsoup4"}),
            auto_install=True,
        )
        entry = optimizer.trajectory[1]
        # The whole loop COMPLETED; the failure is a ledger refusal.
        refusal = entry["refusals"][0]
        assert "auto_install could not install dep(s) ['beautifulsoup4']" in refusal
        assert "No solution found" in refusal
        # Atomic: nothing applied, no tool leaked, the predict stands.
        assert entry["applied"] == []
        assert program.to_manifest()["components"]["6_tools"] == {}
        assert not hasattr(program, "tag_code")

    def test_install_success_via_the_seam(self, monkeypatch):
        calls = []
        state = {"installed": False}

        def fake_missing(deps):
            return [] if state["installed"] else list(deps)

        def fake_uv(args):
            calls.append(args)
            state["installed"] = True
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        monkeypatch.setattr(env_prepare, "missing_deps", fake_missing)
        monkeypatch.setattr(env_prepare, "_run_uv", fake_uv)
        program = Tagger()
        optimizer = run_flex(
            program,
            DEP_OPS,
            holdout=tag_devset("owl"),
            allowed_deps=frozenset({"beautifulsoup4"}),
            auto_install=True,
        )
        entry = optimizer.trajectory[1]
        assert entry["refusals"] == []
        assert entry["accepted"] is True
        # The exact invocation: uv pip install into THIS interpreter's env.
        assert calls == [["pip", "install", "--python", sys.executable, "beautifulsoup4"]]

    def test_already_installed_deps_never_invoke_uv(self, monkeypatch):
        def forbidden(args):
            raise AssertionError("uv invoked although nothing was missing")

        monkeypatch.setattr(env_prepare, "missing_deps", lambda deps: [])
        monkeypatch.setattr(env_prepare, "_run_uv", forbidden)
        program = Tagger()
        optimizer = run_flex(
            program,
            DEP_OPS,
            holdout=tag_devset("owl"),
            allowed_deps=frozenset({"beautifulsoup4"}),
            auto_install=True,
        )
        assert optimizer.trajectory[1]["accepted"] is True

    @pytest.mark.skipif(
        not os.environ.get("DSPY_FLEX_UV_TESTS"),
        reason="genuine uv install needs network or a warm uv cache; set DSPY_FLEX_UV_TESTS=1",
    )
    def test_genuine_install_of_a_tiny_wheel(self):
        import importlib.metadata as metadata

        dep_tool = 'def tag_code(text: str) -> dict:\n    # deps: six\n    return {"tag": text.upper()}\n'
        ops = [
            {"op": "replace_predict_with_code", "path": "tagger", "tool_name": "tag_code", "python_source": dep_tool},
            {"op": "delete_dead_leaf", "path": "tagger"},
        ]
        program = Tagger()
        optimizer = run_flex(
            program, ops, holdout=tag_devset("owl"), allowed_deps=frozenset({"six"}), auto_install=True
        )
        assert optimizer.trajectory[1]["accepted"] is True
        metadata.distribution("six")  # raises if the install did not land


# ---------------------------------------------------------------------------
# (2) Rung 3 — eval_mode="artifact"
# ---------------------------------------------------------------------------


def trajectory_shape(optimizer):
    return [
        (entry["label"], entry["score"], entry["lm_calls"], entry["holdout_score"], entry["accepted"])
        for entry in optimizer.trajectory
    ]


class TestArtifactMode:
    def test_results_identical_to_in_process_on_the_same_script(self):
        shapes = {}
        for mode, hatch in (("in_process", False), ("artifact", True)):
            program = Tagger()
            optimizer = run_flex(
                program,
                SWAP_OPS,
                devset=tag_devset("cat", "dog", "fox"),
                holdout=tag_devset("owl", "bee"),
                eval_mode=mode,
                _eval_same_env=hatch,
            )
            shapes[mode] = trajectory_shape(optimizer)
        assert shapes["artifact"] == shapes["in_process"]
        # And the artifact run genuinely accepted the cheapness swap.
        assert shapes["artifact"][1][1:] == (1.0, 0, 1.0, True)

    def test_candidate_errors_score_zero_across_the_boundary(self):
        # A raising tool is a CANDIDATE failure: the child's evaluate
        # catches it per example (ToolError is catchable), the child
        # exits 0, and the loop completes with a rejected candidate —
        # never an infra abort.
        boom_ops = [
            {
                "op": "add_tool",
                "path": "self",
                "name": "boom",
                "python_source": 'def boom(text: str) -> dict:\n    raise ValueError("boom")\n',
            },
            {
                "op": "rewrite_forward",
                "path": "self",
                "python_source": "def forward(self, text):\n    result = self.boom(text=text)\n    return result\n",
            },
        ]
        program = Tagger()
        optimizer = run_flex(
            program,
            boom_ops,
            holdout=tag_devset("owl"),
            eval_mode="artifact",
            _eval_same_env=True,
        )
        entry = optimizer.trajectory[1]
        assert [proposal["op"] for proposal in entry["applied"]] == ["add_tool", "rewrite_forward"]
        assert entry["score"] == 0.0
        assert entry["accepted"] is False
        # Unwound: the champion still answers.
        assert program(text="elk").tag == "ELK"

    def test_infra_failures_cross_as_raises_not_scores(self):
        # An empty dataset makes the child's evaluate raise ValueError —
        # exactly the class of error that must NEVER be scored against
        # the candidate. The exit-code/stderr protocol surfaces it as a
        # parent-side raise.
        program = Tagger()
        dspy.configure(lm=dspy.DummyLM(["unused"]))
        optimizer = optim.FlexIR(dspy.DummyLM([]), exact_tag, iterations=1, eval_mode="artifact", _eval_same_env=True)
        optimizer._prepare_artifact_mode(None)
        with pytest.raises(RuntimeError, match="infrastructure") as caught:
            optimizer._evaluate_as_artifact(program, [])
        assert "FLEXIR-INFRA" in str(caught.value)

    def test_env_cache_reuses_by_lock_hash(self, tmp_path):
        # Baseline dev + baseline holdout + candidate dev + candidate
        # holdout = four child runs, one dependency set -> ONE provision.
        # A fresh checkpoint dir keys the cache under tmp so a persisted
        # `~/.cache/dspy-flexir` env from another run cannot mask it.
        program = Tagger()
        optimizer = run_flex(
            program,
            SWAP_OPS,
            holdout=tag_devset("owl"),
            checkpoint_dir=tmp_path / "run",
            eval_mode="artifact",
            _eval_same_env=True,
        )
        assert optimizer.trajectory[1]["accepted"] is True
        assert optimizer._env_cache.provisions == 1

    def test_env_cache_provisions_again_only_when_the_lock_changes(self, tmp_path):
        from dspy.optim.env_prepare import EnvCache

        first = tmp_path / "a"
        second = tmp_path / "b"
        third = tmp_path / "c"
        for directory, lock in ((first, "lock-one"), (second, "lock-one"), (third, "lock-two")):
            directory.mkdir()
            (directory / "env_entry.py.lock").write_text(lock)
        cache = EnvCache(tmp_path / "envs", same_env=True)
        cache.interpreter_for(first)
        cache.interpreter_for(second)  # same bytes -> cached
        assert cache.provisions == 1
        cache.interpreter_for(third)  # different lock -> a new resolve
        assert cache.provisions == 2

    def test_non_self_contained_metric_refuses_at_compile(self):
        lookup = {"cat": "CAT"}

        def leaky_metric(example, prediction):
            return prediction.tag == lookup.get(example.text)

        program = Tagger()
        dspy.configure(lm=dspy.DummyLM(sc_upper_task))
        optimizer = optim.FlexIR(
            dspy.DummyLM([reflection_reply([])]),
            leaky_metric,
            iterations=1,
            eval_mode="artifact",
            _eval_same_env=True,
        )
        with pytest.raises(ValueError, match="self-contained"):
            optimizer.compile(program, trainset=tag_devset("cat"))

    def test_bad_eval_mode_refuses_at_construction(self):
        with pytest.raises(ValueError, match="eval_mode"):
            optim.FlexIR(dspy.DummyLM([]), exact_tag, eval_mode="remote")
        with pytest.raises(ValueError, match="auto_install"):
            optim.FlexIR(dspy.DummyLM([]), exact_tag, auto_install="yes")

    @pytest.mark.skipif(
        not os.environ.get("DSPY_FLEX_UV_TESTS"),
        reason="building a real uv env needs network or a warm uv cache; set DSPY_FLEX_UV_TESTS=1",
    )
    def test_genuine_uv_built_child_environment(self, tmp_path):
        program = Tagger()
        # checkpoint_dir pins the env cache under tmp_path — without it the
        # cache lands in ~/.cache/dspy-flexir, and a warm cache from a prior
        # run makes `provisions == 1` flake to 0 (a cache HIT, not a bug).
        optimizer = run_flex(
            program,
            SWAP_OPS,
            holdout=tag_devset("owl"),
            eval_mode="artifact",
            checkpoint_dir=tmp_path,
        )
        assert optimizer.trajectory[1]["accepted"] is True
        assert optimizer._env_cache.provisions == 1


# ---------------------------------------------------------------------------
# (3) Real-LM child binding — credentials ride env-var names, never disk
# ---------------------------------------------------------------------------


def make_optimizer(**flex_kwargs):
    return optim.FlexIR(
        dspy.DummyLM([]), exact_tag, iterations=1, eval_mode="artifact", _eval_same_env=True, **flex_kwargs
    )


class TestRealLMBinding:
    def serialize(self, lm):
        program = Tagger()
        dspy.configure(lm=lm)
        return make_optimizer()._serialize_lms(program)

    def test_env_var_name_recovery(self, monkeypatch):
        monkeypatch.setenv("FLEX_TEST_KEY", "sk-fake-unit-123")
        specs, extra_env = self.serialize(
            dspy.LM("openai/gpt-fake", api_key="sk-fake-unit-123", api_base="http://example.invalid/v1")
        )
        (spec,) = specs.values()
        # The binding reuses the LM's own constructor contract...
        assert spec["class"] == "LM"
        assert spec["model"] == "openai/gpt-fake"
        assert spec["capabilities"]["instruct"] is True
        assert spec["kwargs"]["api_base"] == "http://example.invalid/v1"
        assert spec["kwargs"]["temperature"] == 0.0
        # ...and the credential crosses as the env var's NAME only.
        assert spec["credentials"] == {"api_key": {"env": "FLEX_TEST_KEY"}}
        assert extra_env == {}
        assert "sk-fake-unit-123" not in json.dumps(specs)

    def test_fallback_env_injection_for_orphan_secrets(self):
        specs, extra_env = self.serialize(dspy.LM("openai/gpt-fake", api_key="sk-orphan-987"))
        (spec,) = specs.values()
        (env_name,) = spec["credentials"]["api_key"].values()
        assert env_name.startswith("DSPY_FLEX_LM_")
        # The secret rides the child's process env only — never the payload.
        assert extra_env == {env_name: "sk-orphan-987"}
        assert "sk-orphan-987" not in json.dumps(specs)

    def test_sibling_credentials_are_caught_by_the_markers(self):
        assert _is_secret_kwarg("azure_ad_token")
        assert _is_secret_kwarg("aws_secret_access_key")
        assert _is_secret_kwarg("api_key")
        assert not _is_secret_kwarg("api_base")
        specs, extra_env = self.serialize(dspy.LM("openai/gpt-fake", azure_ad_token="tok-555"))
        (spec,) = specs.values()
        assert "azure_ad_token" in spec["credentials"]
        assert "tok-555" not in json.dumps(specs)
        assert "tok-555" in extra_env.values()

    def test_opaque_header_bags_refuse_loudly(self):
        with pytest.raises(ValueError, match="opaque"):
            self.serialize(dspy.LM("openai/gpt-fake", extra_headers={"Authorization": "Bearer x"}))

    def test_non_json_kwarg_refuses_instead_of_silent_loss(self):
        with pytest.raises(ValueError, match="not\\s+JSON-serializable"):
            self.serialize(dspy.LM("openai/gpt-fake", api_key="k", weird=object()))

    def test_unbindable_subclass_keeps_a_narrow_teaching_refusal(self):
        class ExoticLM(dspy.LM):
            pass

        with pytest.raises(ValueError, match="ExoticLM.*construction contract"):
            self.serialize(ExoticLM("openai/gpt-fake", api_key="k"))

    def test_the_secret_scan_catches_a_leak(self):
        optimizer = make_optimizer()
        optimizer._secret_values = {"sk-planted-1"}
        optimizer._assert_secret_free(b"clean payload", "x")  # no raise
        with pytest.raises(RuntimeError, match="credential value leaked"):
            optimizer._assert_secret_free(b'{"api_key": "sk-planted-1"}', "the scoring job payload")

    def test_env_var_holding_finds_names(self, monkeypatch):
        monkeypatch.setenv("FLEX_HOLDING_TEST", "value-abc")
        assert _env_var_holding("value-abc") == "FLEX_HOLDING_TEST"
        assert _env_var_holding("value-that-is-nowhere") is None


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.server.auth_headers.append(self.headers.get("Authorization"))
        content = "[[ ## tag ## ]]\nCAT\n\n[[ ## completed ## ]]"
        body = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture()
def stub_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    server.auth_headers = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


class TestRealLMEndToEnd:
    def run(self, stub_server, api_key):
        port = stub_server.server_address[1]
        lm = dspy.LM("openai/stub-model", api_key=api_key, api_base=f"http://127.0.0.1:{port}/v1")
        dspy.configure(lm=lm)
        program = Tagger()
        optimizer = optim.FlexIR(
            dspy.DummyLM([reflection_reply([])]),
            exact_tag,
            iterations=1,
            eval_mode="artifact",
            _eval_same_env=True,
        )
        optimizer.compile(program, trainset=tag_devset("cat"))
        return optimizer

    def test_child_scores_through_the_stub_with_env_credential(self, stub_server, monkeypatch):
        # The recovered-name path: the key lives in a parent env var; the
        # binding carries the NAME; the child reads it and authenticates.
        monkeypatch.setenv("FLEX_STUB_KEY", "sk-stub-e2e-42")
        optimizer = self.run(stub_server, "sk-stub-e2e-42")
        baseline = optimizer.trajectory[0]
        assert (baseline["score"], baseline["lm_calls"]) == (1.0, 1)
        # The stub SAW the credential — proof the child read it from env.
        assert stub_server.auth_headers == ["Bearer sk-stub-e2e-42"]

    def test_child_scores_through_the_stub_with_fallback_credential(self, stub_server):
        # The fallback path: an orphan key with no env-var origin rides a
        # private var set on the child process only.
        optimizer = self.run(stub_server, "sk-stub-orphan-77")
        baseline = optimizer.trajectory[0]
        assert (baseline["score"], baseline["lm_calls"]) == (1.0, 1)
        assert stub_server.auth_headers == ["Bearer sk-stub-orphan-77"]
