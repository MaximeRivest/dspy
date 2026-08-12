"""The authoring surface: @dspy.tool, Session, Envelope, Broker, retrust, confined.

The human-facing sugar over the campaign's substrate. Each test pins that
a declaration lands WHERE claimed (floor stamp, broker_route grant bytes,
deps line, session bridge) and that declared-not-enforced fields ride the
entry as pure declarations with no behavior claim. Offline and
deterministic; the genuine-ratchet confined variant is gated.
"""

import json
import os

import pytest

import dspy
from dspy.lm import BINDINGS
from dspy.programir.leaves import extract_tool


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


def upper_task(messages):
    rendered = "\n".join(str(m["content"]) for m in messages)
    marker = "[[ ## text ## ]]\n"
    value = ""
    i = rendered.find(marker)
    while i != -1:
        s = i + len(marker)
        e = rendered.find("\n", s)
        value = rendered[s:] if e == -1 else rendered[s:e]
        i = rendered.find(marker, s)
    return chat_completion(tag=value.strip().upper())


# ---------------------------------------------------------------------------
# (1) @dspy.tool — bare parity + each kwarg lands where claimed
# ---------------------------------------------------------------------------


class TestToolDecorator:
    def test_bare_decorator_is_todays_plain_tool(self):
        @dspy.tool
        def plain(x: str) -> dict:
            return {"y": x}

        entry = extract_tool(plain, name="plain").entry
        assert "kind" not in entry
        assert "grants" not in entry
        assert "isolation_floor" not in entry["placement"]
        # Still a normal callable.
        assert plain("hi") == {"y": "hi"}

    def test_isolation_stamps_the_floor(self):
        @dspy.tool(isolation="sandbox")
        def t(x: str) -> dict:
            return {"y": x}

        entry = extract_tool(t, name="t").entry
        assert entry["placement"]["isolation_floor"] == "sandbox"

    def test_net_becomes_broker_route_grants(self):
        @dspy.tool(net=["api.anthropic.com", "pypi.org"])
        def t(x: str) -> dict:
            return {"y": x}

        entry = extract_tool(t, name="t").entry
        assert entry["grants"] == [
            {"kind": "broker_route", "name": "api.anthropic.com"},
            {"kind": "broker_route", "name": "pypi.org"},
        ]

    def test_files_memory_cpus_gpu_are_declared_not_enforced(self):
        # DECLARATIONS only: they ride the function as intent; nothing in
        # the pool entry claims enforcement (no Landlock/cgroup byte shape).
        @dspy.tool(files={"/data": "ro"}, memory="2G", cpus=2, gpu=True)
        def t(x: str) -> dict:
            return {"y": x}

        assert t._dspy_declared_resources == {
            "files": {"/data": "ro"},
            "memory": "2G",
            "cpus": 2,
            "gpu": True,
        }
        # The tool entry itself carries no resource enforcement fields.
        entry = extract_tool(t, name="t").entry
        assert "memory" not in entry and "files" not in entry

    def test_bad_files_mode_refuses(self):
        with pytest.raises(ValueError, match="must be 'ro' or 'rw'"):

            @dspy.tool(files={"/x": "exec"})
            def t(x: str) -> dict:
                return {"y": x}

    def test_deps_injects_and_conflict_refuses(self):
        @dspy.tool(deps=["httpx"])
        def t(x: str) -> dict:
            return {"y": x}

        assert t._dspy_declared_deps == ["httpx"]

        # A source whose `# deps:` disagrees with the decorator refuses.
        with pytest.raises(ValueError, match="conflicts with the source"):

            @dspy.tool(deps=["httpx"])
            def u(x: str) -> dict:
                # deps: requests
                return {"y": x}

    def test_grants_without_session_refuses(self):
        with pytest.raises(ValueError, match="require session=True"):

            @dspy.tool(grants=["solver"])
            def t(bridge: object, x: str) -> dict:
                return bridge.solver(x=x)

    def test_session_stamps_kind_and_grants(self):
        @dspy.tool(session=True, grants=["solver"])
        def t(bridge: object, x: str) -> dict:
            return bridge.solver(text=x)

        entry = extract_tool(t, name="t").entry
        assert entry["kind"] == "session"
        assert entry["grants"] == [{"kind": "leaf", "name": "solver"}]

    def test_decorated_tool_is_a_normal_tool_in_react(self):
        # A decorated function is a normal tool everywhere a tool is taken:
        # ReAct accepts it and it COMPILES into the artifact (the decorator
        # marker is stripped from the embedded source), floor stamp intact.
        @dspy.tool(isolation="fork_ratchet")
        def look(query: str) -> str:
            """Look something up."""
            return query

        dspy.configure(
            lm=dspy.DummyLM([chat_completion(next_thought="t", next_tool_name="finish", next_tool_args="{}")])
        )
        react = dspy.ReAct("question -> answer", tools=[look], max_iters=2)
        manifest = react.to_manifest()  # compiles clean — the decorator did not break extraction
        assert "look" in manifest["components"]["6_tools"]
        # NOTE: ReAct's dynamic-dispatch tools are re-embedded through a
        # generated wrapper that does not carry the original function's
        # `_dspy_*` stamps, so the floor does not survive that path today
        # (a known limitation — see the report's declared-vs-enforced list).
        # A tool bound DIRECTLY as a module attribute keeps its stamp; the
        # dispatch-wrapper path is the gap.

    def test_direct_bound_decorated_tool_keeps_floor_and_grants(self):
        # The primary path: a decorated tool bound as a plain module
        # attribute carries its floor + grants into the manifest.
        dspy.configure(lm=dspy.DummyLM(upper_task))

        class M(dspy.Module):
            def __init__(self):
                self.inner = dspy.Predict("text -> tag")

                @dspy.tool(isolation="sandbox", net=["api.x"])
                def helper(x: str) -> dict:
                    return {"y": x}

                self.helper = helper

            def forward(self, text):
                return self.inner(text=text)

        entry = M().to_manifest()["components"]["6_tools"]["helper"]
        assert entry["placement"]["isolation_floor"] == "sandbox"
        assert entry["grants"] == [{"kind": "broker_route", "name": "api.x"}]


# ---------------------------------------------------------------------------
# (2) dspy.Session — sugar over the session substrate, end to end
# ---------------------------------------------------------------------------


class TestSession:
    def build(self, *, grant="inner"):
        def session_leaf(bridge: object, text: str) -> dict:
            first = bridge.inner(text=text)
            return {"tag": first["tag"]}

        wrapped = dspy.Session(session_leaf, grants=[grant])
        wrapped._dspy_placement_rung = "in_process"  # run in the loop's own process for the test

        class Composite(dspy.Module):
            def __init__(self, leaf):
                self.inner = dspy.Predict("text -> tag")
                self.session_leaf = leaf

            def forward(self, text):
                result = self.session_leaf(text=text)
                return result

        return Composite(wrapped)

    def test_session_runs_through_the_real_bridge(self):
        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = self.build()
        assert program(text="cat").tag == "CAT"
        entry = program.to_manifest()["components"]["6_tools"]["session_leaf"]
        assert entry["kind"] == "session"
        assert entry["grants"] == [{"kind": "leaf", "name": "inner"}]

    def test_session_refuses_bad_lifetime(self):
        def leaf(bridge: object, text: str) -> dict:
            return {"tag": text}

        with pytest.raises(ValueError, match="lifetime must be 'forward'"):
            dspy.Session(leaf, grants=["inner"], lifetime="global")

    def test_session_dangling_grant_refuses_at_load(self):
        dspy.configure(lm=dspy.DummyLM(upper_task))
        program = self.build(grant="ghost")
        with pytest.raises(ValueError, match="dangling grant"):
            program(text="cat")


# ---------------------------------------------------------------------------
# (3) dspy.Envelope through dspy.load satisfies an isolation_required leaf
# ---------------------------------------------------------------------------


class Tagger(dspy.Module):
    def __init__(self):
        self.tagger = dspy.Predict("text -> tag")

    def forward(self, text):
        result = self.tagger(text=text)
        return result


def _authored_artifact(tmp_path):
    """A program with an optimizer-authored (isolation_required) tool leaf."""
    from dspy import optim

    program = Tagger()
    ops = [
        {
            "op": "replace_predict_with_code",
            "path": "tagger",
            "tool_name": "tag_code",
            "python_source": 'def tag_code(text: str) -> dict:\n    return {"tag": text.upper()}\n',
        },
        {"op": "delete_dead_leaf", "path": "tagger"},
    ]
    reflection = dspy.DummyLM([chat_completion(proposals=json.dumps(ops))])
    dspy.configure(lm=dspy.DummyLM(upper_task))
    optim.FlexIR(
        reflection,
        lambda e, p: e.tag == p.tag,
        iterations=1,
        holdout=[dspy.Example(text="owl", tag="OWL").with_inputs("text")],
    ).compile(program, trainset=[dspy.Example(text="cat", tag="CAT").with_inputs("text")])
    artifact = tmp_path / "art"
    program.save(artifact)
    return artifact


class TestEnvelope:
    def test_envelope_at_floor_satisfies_the_leaf(self, tmp_path):
        artifact = _authored_artifact(tmp_path)
        loaded = dspy.load(artifact, bindings={"lm": {}}, envelope=dspy.Envelope("fork_ratchet"))
        assert loaded(text="world").tag == "WORLD"

    def test_envelope_below_floor_still_fails_closed(self, tmp_path):
        artifact = _authored_artifact(tmp_path)
        with pytest.raises(ValueError, match="requires isolation"):
            dspy.load(artifact, bindings={"lm": {}}, envelope=dspy.Envelope("fork"))

    def test_no_envelope_keeps_the_grant_gate(self, tmp_path):
        artifact = _authored_artifact(tmp_path)
        with pytest.raises(ValueError, match="requires isolation"):
            dspy.load(artifact, bindings={"lm": {}})

    def test_envelope_level_name(self):
        assert dspy.Envelope("sandbox").level_name == "sandbox"


# ---------------------------------------------------------------------------
# (4) dspy.Broker / SecretFromEnv
# ---------------------------------------------------------------------------


class TestBroker:
    def test_secret_from_env_reads_at_serve_time_never_stores(self, monkeypatch):
        secret = dspy.SecretFromEnv("MY_KEY")
        assert "value" not in repr(secret)  # nothing held
        assert secret.var in repr(secret)
        monkeypatch.setenv("MY_KEY", "sk-live-1")
        assert secret.resolve() == "sk-live-1"

    def test_secret_from_env_unset_refuses(self, monkeypatch):
        monkeypatch.delenv("MISSING_KEY", raising=False)
        with pytest.raises(ValueError, match="not set in the environment"):
            dspy.SecretFromEnv("MISSING_KEY").resolve()

    def test_broker_materializes_injection_from_secret(self, monkeypatch):
        monkeypatch.setenv("PROV_KEY", "sk-prov-9")
        broker = dspy.Broker(allow=["api.x"], inject={"api.x": dspy.SecretFromEnv("PROV_KEY")})
        resolved = broker._materialize_inject()
        assert resolved == {"api.x": {"header": "Authorization", "value": "Bearer sk-prov-9"}}

    def test_broker_start_allowlist_and_log(self, tmp_path, monkeypatch):
        import http.client
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from urllib.parse import urlsplit

        class Stub(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)
                self.server.seen.append(self.headers.get("Authorization"))
                body = b'{"ok": true}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        server = HTTPServer(("127.0.0.1", 0), Stub)
        server.seen = []
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        host = "127.0.0.1"
        monkeypatch.setenv("BK", "sk-broker-e2e")
        log = tmp_path / "reqs.json"
        broker = dspy.Broker(allow=[host], inject={host: dspy.SecretFromEnv("BK")}, log=str(log)).start()
        try:
            proxy = urlsplit(broker.proxy_url)
            conn = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=10)
            conn.request("POST", f"http://{host}:{port}/v1", body=b"{}", headers={"Content-Length": "2"})
            assert conn.getresponse().status == 200
            # A non-allowlisted host is refused.
            conn2 = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=10)
            conn2.request("POST", "http://elsewhere.invalid/v1", body=b"{}", headers={"Content-Length": "2"})
            assert conn2.getresponse().status == 403
        finally:
            broker.stop()
            server.shutdown()
        # The credential was injected on the allowlisted host.
        assert server.seen == ["Bearer sk-broker-e2e"]
        # The request log was written on stop.
        records = json.loads(log.read_text())
        assert any(r["allowed"] and r["injected"] for r in records)
        assert any(not r["allowed"] for r in records)


# ---------------------------------------------------------------------------
# (5) dspy.retrust — lower with provenance, refuse raising
# ---------------------------------------------------------------------------


def _program_with_floor(floor="fork_ratchet"):
    dspy.configure(lm=dspy.DummyLM(upper_task))

    def make():
        class M(dspy.Module):
            def __init__(self):
                self.inner = dspy.Predict("text -> tag")

                @dspy.tool(isolation=floor)
                def helper(x: str) -> dict:
                    return {"y": x}

                self.helper = helper

            def forward(self, text):
                return self.inner(text=text)

        return M()

    return make()


class TestRetrust:
    def test_lowering_records_provenance_bytes(self):
        program = _program_with_floor("fork_ratchet")
        dspy.retrust(program, "helper", floor="none", reason="I read the code")
        placement = program.to_manifest()["components"]["6_tools"]["helper"]["placement"]
        assert placement["isolation_floor"] == "none"
        prov = placement["floor_lowered_by"]
        assert prov["old"] == "fork_ratchet"
        assert prov["new"] == "none"
        assert prov["reason"] == "I read the code"
        assert "by" in prov and "date" in prov

    def test_raising_is_refused_pointing_at_envelope(self):
        program = _program_with_floor("fork")
        with pytest.raises(ValueError, match="refuses to RAISE") as caught:
            dspy.retrust(program, "helper", floor="sandbox")
        assert "Envelope" in str(caught.value)

    def test_same_floor_refuses(self):
        program = _program_with_floor("fork_ratchet")
        with pytest.raises(ValueError, match="already fork_ratchet"):
            dspy.retrust(program, "helper", floor="fork_ratchet")

    def test_missing_leaf_refuses(self):
        program = _program_with_floor()
        with pytest.raises(ValueError, match="no tool leaf"):
            dspy.retrust(program, "ghost", floor="none")

    def test_there_is_no_ignore_floors_flag(self):
        # The API surface deliberately offers no runtime override.
        import inspect

        sig = inspect.signature(dspy.retrust)
        assert "ignore_floors" not in sig.parameters
        assert "force" not in sig.parameters


# ---------------------------------------------------------------------------
# (6) dspy.confined — run a callable once in a child
# ---------------------------------------------------------------------------


class TestConfined:
    def test_runs_a_real_child_at_fork(self):
        def add(a, b):
            return a + b

        assert dspy.confined(add, 2, 3, isolation="fork") == 5

    def test_child_error_is_reraised(self):
        def boom():
            raise ValueError("kaboom")

        with pytest.raises(RuntimeError, match="raised in the child"):
            dspy.confined(boom, isolation="fork")

    def test_non_serializable_args_refuse_before_launch(self):
        def f(x):
            return x

        with pytest.raises(ValueError, match="JSON-serializable"):
            dspy.confined(f, object(), isolation="fork")

    def test_under_floor_host_refuses_loudly(self):
        # A backend whose ceiling is below the request refuses (fail closed).
        from dspy.programir.engine.isolation import IsolationDowngrade, LinuxIsolationBackend

        class Poor:
            def has_cgroup_v2(self):
                return False

            def cgroupfs_writable(self, root="/sys/fs/cgroup"):
                return False

            def has_unshare(self):
                return False

            def has_bwrap(self):
                return False

        def add(a, b):
            return a + b

        backend = LinuxIsolationBackend(Poor())
        with pytest.raises(IsolationDowngrade):
            dspy.confined(add, 1, 2, isolation="sandbox", _backend=backend)

    @pytest.mark.skipif(
        not os.environ.get("DSPY_FLEX_UV_TESTS"),
        reason="genuine fork_ratchet needs unprivileged user namespaces; set DSPY_FLEX_UV_TESTS=1",
    )
    def test_genuine_fork_ratchet_confined(self):
        from dspy.programir.engine.isolation import IsolationDowngrade, IsolationLevel, LinuxIsolationBackend

        backend = LinuxIsolationBackend()
        if backend.reachable_level() < IsolationLevel.fork_ratchet:
            pytest.skip("host does not reach fork_ratchet")

        def add(a, b):
            return a + b

        try:
            assert dspy.confined(add, 4, 5, isolation="fork_ratchet") == 9
        except IsolationDowngrade:
            pytest.skip("host refused fork_ratchet at run time")
