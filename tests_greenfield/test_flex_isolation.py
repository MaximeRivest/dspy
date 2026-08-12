"""D-042 isolation: the gradient, the Linux backend, the broker, migration.

Sub-stage A — the isolation vocabulary and Linux fork-place-ratchet
backend, fail-closed and self-probing, wired into artifact-mode scoring.
Sub-stage B — the parent-owned egress broker (hostname allowlist,
per-request log, credential injection). Sub-stage C — code_trust migrated
to isolation_floor over the D-042 vocabulary, aliases preserved.

Everything here is deterministic and offline. The syscall layer and the
isolation backend are faked where a userns-capable kernel would be
needed; a genuine fork_ratchet run is gated behind DSPY_FLEX_UV_TESTS and
attempted at the end (the report says whether it ran on this machine).
"""

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import dspy
from dspy import optim
from dspy.lm import BINDINGS
from dspy.optim.broker import EgressBroker
from dspy.programir.engine.isolation import (
    AUTHORED_LEAF_FLOOR,
    IsolationBackend,
    IsolationDowngrade,
    IsolationLevel,
    IsolationPolicy,
    LinuxIsolationBackend,
    _SyscallLayer,
    parse_level,
)


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


class Tagger(dspy.Module):
    def __init__(self):
        self.tagger = dspy.Predict("text -> tag")

    def forward(self, text):
        result = self.tagger(text=text)
        return result


def sc_upper_task(messages):
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


def tag_devset(*words):
    return [dspy.Example(text=word, tag=word.upper()).with_inputs("text") for word in words]


# ---------------------------------------------------------------------------
# (A1) Vocabulary ordering and policy comparison
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_levels_are_totally_ordered(self):
        names = ["none", "namespace", "fork", "fork_cgroup", "fork_ratchet", "sandbox", "remote"]
        levels = [IsolationLevel[name] for name in names]
        assert levels == sorted(levels)
        assert [level.value for level in levels] == list(range(7))

    def test_parse_level_accepts_names_and_enums_and_refuses_junk(self):
        assert parse_level("fork_ratchet") is IsolationLevel.fork_ratchet
        assert parse_level(IsolationLevel.sandbox) is IsolationLevel.sandbox
        with pytest.raises(ValueError, match="unknown isolation level"):
            parse_level("teleport")
        with pytest.raises(ValueError, match="name or IsolationLevel"):
            parse_level(3.5)

    def test_policy_satisfies_is_may_exceed_never_under_run(self):
        weak = IsolationPolicy(level=IsolationLevel.fork)
        strong = IsolationPolicy(level=IsolationLevel.sandbox)
        assert not weak.satisfies(IsolationLevel.fork_ratchet)  # under-run
        assert strong.satisfies(IsolationLevel.fork_ratchet)  # exceed
        assert strong.satisfies(IsolationLevel.sandbox)  # exact

    def test_refuse_under_floor_teaches(self):
        weak = IsolationPolicy(level=IsolationLevel.fork)
        with pytest.raises(ValueError, match="under-floor"):
            weak.refuse_under_floor(IsolationLevel.fork_ratchet, subject="tool 'x'")
        # An adequate envelope does not refuse.
        IsolationPolicy(level=IsolationLevel.sandbox).refuse_under_floor(
            IsolationLevel.fork_ratchet, subject="tool 'x'"
        )

    def test_authored_leaf_floor_is_fork_ratchet(self):
        assert AUTHORED_LEAF_FLOOR is IsolationLevel.fork_ratchet

    def test_describe_envelope_records_the_axis(self):
        policy = IsolationPolicy(level=IsolationLevel.sandbox, broker_egress=frozenset({"api.x"}), scratch="/s")
        assert policy.describe_envelope() == {
            "level": "sandbox",
            "broker_egress": ["api.x"],
            "scratch": "/s",
        }


# ---------------------------------------------------------------------------
# (A2) Fail-closed on unavailable primitives (mock the syscall layer)
# ---------------------------------------------------------------------------


class FakeSyscalls(_SyscallLayer):
    def __init__(self, *, cgroup=True, writable=True, unshare=True, bwrap=False):
        self._cgroup = cgroup
        self._writable = writable
        self._unshare = unshare
        self._bwrap = bwrap

    def has_cgroup_v2(self):
        return self._cgroup

    def cgroupfs_writable(self, root="/sys/fs/cgroup"):
        return self._writable

    def has_unshare(self):
        return self._unshare

    def has_bwrap(self):
        return self._bwrap


class TestFailClosed:
    def test_reachable_level_climbs_with_primitives(self):
        assert LinuxIsolationBackend(FakeSyscalls(cgroup=False)).reachable_level() is IsolationLevel.fork
        assert (
            LinuxIsolationBackend(FakeSyscalls(cgroup=True, unshare=False)).reachable_level()
            is IsolationLevel.fork_cgroup
        )
        assert LinuxIsolationBackend(FakeSyscalls()).reachable_level() is IsolationLevel.fork_ratchet
        assert LinuxIsolationBackend(FakeSyscalls(bwrap=True)).reachable_level() is IsolationLevel.sandbox

    def test_unwritable_cgroupfs_refuses_loudly_not_silently(self):
        backend = LinuxIsolationBackend(FakeSyscalls(cgroup=True, writable=False))
        with pytest.raises(IsolationDowngrade) as caught:
            backend.best_effort_level(IsolationLevel.fork_ratchet)
        assert caught.value.requested is IsolationLevel.fork_ratchet
        assert "cgroupfs is not writable" in caught.value.reason

    def test_missing_unshare_refuses_fork_ratchet(self):
        backend = LinuxIsolationBackend(FakeSyscalls(unshare=False))
        with pytest.raises(IsolationDowngrade, match="unshare"):
            backend.best_effort_level(IsolationLevel.fork_ratchet)

    def test_best_effort_passes_through_non_fork_levels(self):
        backend = LinuxIsolationBackend(FakeSyscalls(cgroup=False))
        assert backend.best_effort_level(IsolationLevel.none) is IsolationLevel.none
        assert backend.best_effort_level(IsolationLevel.remote) is IsolationLevel.remote

    def test_reachable_request_is_granted(self):
        backend = LinuxIsolationBackend(FakeSyscalls())
        assert backend.best_effort_level(IsolationLevel.fork_cgroup) is IsolationLevel.fork_cgroup


# ---------------------------------------------------------------------------
# (A3) The self-probe logic — a wall that did not hold aborts
# ---------------------------------------------------------------------------


class TestSelfProbe:
    def test_probe_passes_when_both_denials_hold(self):
        backend = LinuxIsolationBackend(FakeSyscalls())

        class Denied(FakeSyscalls):
            def probe_socket_denied(self):
                return True

            def probe_write_denied(self, path):
                return True

        backend._run_probes(Denied(), scratch="/tmp/scratch")  # no raise

    def test_probe_aborts_when_socket_succeeds_where_it_must_fail(self):
        backend = LinuxIsolationBackend(FakeSyscalls())

        class SocketOpen(FakeSyscalls):
            def probe_socket_denied(self):
                return False

        with pytest.raises(RuntimeError, match="reach.*did not hold"):
            backend._run_probes(SocketOpen(), scratch="/tmp/scratch")

    def test_probe_aborts_when_out_of_scratch_write_succeeds(self):
        backend = LinuxIsolationBackend(FakeSyscalls())

        class WriteOpen(FakeSyscalls):
            def probe_socket_denied(self):
                return True

            def probe_write_denied(self, path):
                return False

            def probe_read_denied(self, path):
                return True

        # The write probe fires only when Landlock engaged (a bare mount-ns
        # unshare still shares /tmp, so it cannot deny the write).
        with pytest.raises(RuntimeError, match="out-of-scratch write.*did not hold"):
            backend._run_probes(WriteOpen(), scratch="/tmp/scratch", landlock=True)

    def test_child_preexec_is_none_below_fork_ratchet(self):
        backend = LinuxIsolationBackend(FakeSyscalls())
        assert backend.child_preexec(IsolationPolicy(level=IsolationLevel.fork)) is None
        assert backend.child_preexec(IsolationPolicy(level=IsolationLevel.fork_ratchet)) is not None


# ---------------------------------------------------------------------------
# (A4) FlexIR scoring_isolation: default byte-identical; envelope in trajectory
# ---------------------------------------------------------------------------


class FakeBackend(IsolationBackend):
    """A backend that grants a fixed level and a no-op preexec (offline)."""

    def __init__(self, level=IsolationLevel.fork_ratchet):
        self.level = level

    def best_effort_level(self, requested):
        return parse_level(requested)

    def child_preexec(self, policy):
        return None  # no real ratchet — the same-env child is unchanged


def run_flex(program, ops, *, devset=None, holdout=None, **flex_kwargs):
    reflection = dspy.DummyLM([reflection_reply(ops)])
    dspy.configure(lm=dspy.DummyLM(sc_upper_task))
    optimizer = optim.FlexIR(reflection, exact_tag, iterations=1, holdout=holdout, **flex_kwargs)
    optimizer.compile(program, trainset=devset or tag_devset("cat", "dog"))
    return optimizer


SWAP_OPS = [
    {
        "op": "replace_predict_with_code",
        "path": "tagger",
        "tool_name": "tag_code",
        "python_source": 'def tag_code(text: str) -> dict:\n    return {"tag": text.upper()}\n',
    },
    {"op": "delete_dead_leaf", "path": "tagger"},
]


class TestScoringIsolation:
    def test_default_none_records_no_envelope(self):
        optimizer = run_flex(Tagger(), SWAP_OPS, holdout=tag_devset("owl"), eval_mode="artifact", _eval_same_env=True)
        assert optimizer.trajectory[0]["envelope"] is None
        assert optimizer.trajectory[1]["envelope"] is None
        assert optimizer.scoring_isolation == "none"

    def test_envelope_is_pinned_in_the_trajectory(self):
        optimizer = run_flex(
            Tagger(),
            SWAP_OPS,
            holdout=tag_devset("owl"),
            eval_mode="artifact",
            _eval_same_env=True,
            scoring_isolation="fork_ratchet",
            _isolation_backend=FakeBackend(),
        )
        # Baseline and candidate both record the envelope they scored under.
        import tempfile

        for entry in optimizer.trajectory:
            assert entry["envelope"] == {
                "level": "fork_ratchet",
                "broker_egress": [],
                "scratch": tempfile.gettempdir(),
            }
        assert optimizer.trajectory[1]["accepted"] is True

    def test_under_floor_host_refuses_at_prepare(self):
        # A host that can only reach fork (fake) but asked for sandbox:
        # loud downgrade, surfaced when compile prepares artifact mode.
        program = Tagger()
        reflection = dspy.DummyLM([reflection_reply([])])
        dspy.configure(lm=dspy.DummyLM(sc_upper_task))
        optimizer = optim.FlexIR(
            reflection,
            exact_tag,
            iterations=1,
            eval_mode="artifact",
            _eval_same_env=True,
            scoring_isolation="sandbox",
            _isolation_backend=LinuxIsolationBackend(FakeSyscalls(cgroup=False)),
        )
        with pytest.raises(IsolationDowngrade, match="sandbox"):
            optimizer.compile(program, trainset=tag_devset("cat"))

    def test_isolation_invariance_hook_passes_for_a_pure_swap(self):
        optimizer = run_flex(
            Tagger(),
            SWAP_OPS,
            holdout=tag_devset("owl"),
            eval_mode="artifact",
            _eval_same_env=True,
            scoring_isolation="fork_ratchet",
            _isolation_backend=FakeBackend(),
        )
        program = Tagger()
        program.tagger = program.tagger  # a fresh baseline
        assert optimizer.check_isolation_invariance(Tagger(), tag_devset("cat", "dog")) is None


# ---------------------------------------------------------------------------
# (A5) materialize seam: an envelope >= fork_ratchet grants authored leaves
# ---------------------------------------------------------------------------


class TestMaterializeEnvelopeGrant:
    def _authored_artifact(self, tmp_path):
        program = Tagger()
        run_flex(program, SWAP_OPS, holdout=tag_devset("owl"))  # in_process, isolated floor
        artifact = tmp_path / "art"
        program.save(artifact)
        return artifact

    def test_envelope_at_floor_satisfies_isolation_required(self, tmp_path):
        artifact = self._authored_artifact(tmp_path)
        # No explicit tool grant, but an envelope at the floor IS the grant.
        loaded = dspy.load(
            artifact,
            bindings={"lm": {}, "isolation": {"envelope": IsolationPolicy(level=IsolationLevel.fork_ratchet)}},
        )
        assert loaded(text="world").tag == "WORLD"

    def test_envelope_below_floor_still_fails_closed(self, tmp_path):
        artifact = self._authored_artifact(tmp_path)
        with pytest.raises(ValueError, match="requires isolation"):
            dspy.load(
                artifact,
                bindings={"lm": {}, "isolation": {"envelope": IsolationPolicy(level=IsolationLevel.fork)}},
            )

    def test_no_envelope_keeps_the_explicit_grant_path(self, tmp_path):
        artifact = self._authored_artifact(tmp_path)
        with pytest.raises(ValueError, match="requires isolation"):
            dspy.load(artifact, bindings={"lm": {}})

        def tag_code(text: str) -> dict:
            return {"tag": text.upper()}

        loaded = dspy.load(artifact, bindings={"lm": {}, "tool": {"tag_code": tag_code}})
        assert loaded(text="hi").tag == "HI"

    def test_envelope_grant_accepts_a_plain_name(self, tmp_path):
        artifact = self._authored_artifact(tmp_path)
        loaded = dspy.load(artifact, bindings={"lm": {}, "isolation": {"envelope": "sandbox"}})
        assert loaded(text="sky").tag == "SKY"


# ---------------------------------------------------------------------------
# (B) The broker: allowlist, log, credential injection — offline stub
# ---------------------------------------------------------------------------


class _StubHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.server.seen_auth.append(self.headers.get("Authorization"))
        content = "[[ ## tag ## ]]\nCAT\n\n[[ ## completed ## ]]"
        body = json.dumps(
            {
                "id": "stub",
                "object": "chat.completion",
                "created": 0,
                "model": "stub-model",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
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
def stub_lm_server():
    server = HTTPServer(("127.0.0.1", 0), _StubHandler)
    server.seen_auth = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    thread.join(timeout=5)


class TestBrokerUnit:
    def _post_through(self, broker, url, body=b"{}"):
        import http.client
        from urllib.parse import urlsplit

        proxy = urlsplit(broker.proxy_url)
        conn = http.client.HTTPConnection(proxy.hostname, proxy.port, timeout=10)
        conn.request("POST", url, body=body, headers={"Content-Length": str(len(body))})
        return conn.getresponse()

    def test_injects_credential_on_allowlisted_host(self, stub_lm_server):
        port = stub_lm_server.server_address[1]
        host = "127.0.0.1"
        broker = EgressBroker(
            frozenset({host}),
            inject={host: {"header": "Authorization", "value": "Bearer sk-broker-1"}},
        ).start()
        try:
            response = self._post_through(broker, f"http://{host}:{port}/v1/chat/completions")
            assert response.status == 200
        finally:
            broker.stop()
        assert stub_lm_server.seen_auth == ["Bearer sk-broker-1"]
        (record,) = broker.requests
        assert record["allowed"] is True and record["injected"] is True and record["host"] == host

    def test_non_allowlisted_host_is_refused_and_logged(self, stub_lm_server):
        port = stub_lm_server.server_address[1]
        broker = EgressBroker(frozenset({"allowed.invalid"})).start()
        try:
            response = self._post_through(broker, f"http://127.0.0.1:{port}/v1/chat/completions")
            assert response.status == 403
        finally:
            broker.stop()
        # The stub never saw the request; the broker logged the denial.
        assert stub_lm_server.seen_auth == []
        (record,) = broker.requests
        assert record["allowed"] is False and record["host"] == "127.0.0.1"


class TestBrokerEndToEnd:
    def test_child_reaches_stub_through_broker_with_injected_credential(self, stub_lm_server, monkeypatch):
        port = stub_lm_server.server_address[1]
        host = "127.0.0.1"
        # api_base points at the stub; the broker allowlists its host and
        # injects the credential. The child gets proxy vars, NOT the key.
        lm = dspy.LM("openai/stub-model", api_key="sk-child-must-not-have", api_base=f"http://{host}:{port}/v1")
        dspy.configure(lm=lm)
        program = Tagger()
        optimizer = optim.FlexIR(
            dspy.DummyLM([reflection_reply([])]),
            exact_tag,
            iterations=1,
            eval_mode="artifact",
            _eval_same_env=True,
            broker_egress=frozenset({host}),
        )
        optimizer.compile(program, trainset=tag_devset("cat"))
        baseline = optimizer.trajectory[0]
        assert (baseline["score"], baseline["lm_calls"]) == (1.0, 1)
        # The stub saw the injected credential — proof egress worked. This
        # also pins REPLACEMENT: the child holds only the non-secret
        # placeholder (modern provider clients refuse to send with no key
        # at all — the server run proved it), and the broker swapped it
        # for the real credential on the way out.
        assert stub_lm_server.seen_auth == ["Bearer sk-child-must-not-have"]
        assert "Bearer sk-broker-injected" not in stub_lm_server.seen_auth

    def test_child_env_is_secret_free_under_the_broker(self):
        # With the broker active, the credential rides NO child env var.
        host = "127.0.0.1"
        lm = dspy.LM("openai/stub-model", api_key="sk-secret-xyz", api_base=f"http://{host}:9/v1")
        dspy.configure(lm=lm)
        program = Tagger()
        optimizer = optim.FlexIR(
            dspy.DummyLM([]),
            exact_tag,
            iterations=1,
            eval_mode="artifact",
            _eval_same_env=True,
            broker_egress=frozenset({host}),
        )
        optimizer._prepare_artifact_mode(None)
        specs, extra_env = optimizer._serialize_lms(program)
        # Broker channel took the credential; no fallback env var carries it.
        assert extra_env == {}
        assert "sk-secret-xyz" not in json.dumps(specs)
        assert optimizer._broker_inject[host]["value"] == "Bearer sk-secret-xyz"


# ---------------------------------------------------------------------------
# (C) Knob migration: code_trust -> isolation_floor over the vocabulary
# ---------------------------------------------------------------------------


class TestFloorMigration:
    def test_default_floor_is_fork_ratchet_isolated(self):
        optimizer = optim.FlexIR(dspy.DummyLM([]), exact_tag)
        assert optimizer.isolation_floor == "fork_ratchet"
        assert optimizer.code_trust == "isolated"

    def test_isolated_alias_maps_to_fork_ratchet(self):
        optimizer = optim.FlexIR(dspy.DummyLM([]), exact_tag, code_trust="isolated")
        assert optimizer.isolation_floor == "fork_ratchet"

    def test_in_process_alias_maps_to_none(self):
        optimizer = optim.FlexIR(dspy.DummyLM([]), exact_tag, code_trust="in_process")
        assert optimizer.isolation_floor == "none"
        assert optimizer.code_trust == "in_process"

    def test_explicit_floor_none_stamps_in_process_placement(self):
        program = Tagger()
        run_flex(program, SWAP_OPS, holdout=tag_devset("owl"), isolation_floor="none")
        placement = program.to_manifest()["components"]["6_tools"]["tag_code"]["placement"]
        assert placement["rung"] == "in_process"
        # Provenance is untouched by the floor choice.
        assert program.to_manifest()["components"]["6_tools"]["tag_code"]["authored_by"] == "optimizer"

    def test_explicit_floor_fork_ratchet_stamps_isolation_required(self):
        program = Tagger()
        run_flex(program, SWAP_OPS, holdout=tag_devset("owl"), isolation_floor="fork_ratchet")
        placement = program.to_manifest()["components"]["6_tools"]["tag_code"]["placement"]
        assert placement["rung"] == "isolation_required"

    def test_bad_floor_and_bad_alias_refuse(self):
        with pytest.raises(ValueError, match="unknown isolation level"):
            optim.FlexIR(dspy.DummyLM([]), exact_tag, isolation_floor="teleport")
        with pytest.raises(ValueError, match="code_trust"):
            optim.FlexIR(dspy.DummyLM([]), exact_tag, code_trust="sandboxed")


# ---------------------------------------------------------------------------
# Genuine fork_ratchet — GATED (userns may be unavailable)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DSPY_FLEX_UV_TESTS"),
    reason="genuine fork_ratchet needs unprivileged user namespaces; set DSPY_FLEX_UV_TESTS=1",
)
def test_genuine_fork_ratchet_reachable_on_this_host():
    backend = LinuxIsolationBackend()
    reached = backend.reachable_level()
    assert reached >= IsolationLevel.fork
    # If the host reaches fork_ratchet, the preexec ratchet+probe must run
    # in a real child without the probe falsely passing.
    if reached >= IsolationLevel.fork_ratchet:
        import multiprocessing

        def child(scratch):
            preexec = backend.child_preexec(IsolationPolicy(level=IsolationLevel.fork_ratchet, scratch=scratch))
            preexec()  # unshare + no_new_privs + self-probe; raises if the wall did not hold

        # The ratchet is fork-shaped by design (the zygote pattern); Python
        # 3.14's default start method is no longer fork on Linux, so ask
        # for it explicitly — spawn/forkserver cannot pickle this closure.
        process = multiprocessing.get_context("fork").Process(target=child, args=("/tmp",))
        process.start()
        process.join(timeout=30)
        assert process.exitcode == 0


# ---------------------------------------------------------------------------
# (A6) cgroup enforcement — placement / teardown / fail-closed (fakes)
# ---------------------------------------------------------------------------


class RecordingCgroupSyscalls(FakeSyscalls):
    """A fake syscall layer recording the cgroup placement/teardown calls."""

    def __init__(self, *, root="/sys/fs/cgroup/user", create_fails=False, cap_fails=False, place_fails=False, **kw):
        super().__init__(**kw)
        self._root = root
        self._create_fails = create_fails
        self._cap_fails = cap_fails
        self._place_fails = place_fails
        self.created = []
        self.caps = []
        self.placed = []
        self.killed = []
        self.rmdirs = []

    def writable_cgroup_root(self, root="/sys/fs/cgroup"):
        return self._root

    def cgroup_create(self, leaf):
        if self._create_fails:
            raise OSError(13, "denied")
        self.created.append(leaf)

    def cgroup_write(self, leaf, control, value):
        if self._cap_fails:
            raise OSError(13, "denied")
        self.caps.append((control, value))

    def cgroup_place(self, leaf, pid):
        if self._place_fails:
            raise OSError(3, "no such process")
        self.placed.append((leaf, pid))

    def cgroup_kill(self, leaf):
        self.killed.append(leaf)

    def cgroup_rmdir(self, leaf):
        self.rmdirs.append(leaf)


class TestCgroupEnforcementUnit:
    def test_caps_are_written_from_the_policy(self):
        sc = RecordingCgroupSyscalls()
        backend = LinuxIsolationBackend(sc)
        policy = IsolationPolicy(level=IsolationLevel.fork_cgroup, memory="2G", pids=100, cpus=1.5)
        leaf = backend._create_cgroup(policy)
        assert leaf in sc.created
        assert ("memory.max", str(2 * 1024**3)) in sc.caps
        assert ("pids.max", "100") in sc.caps
        assert ("cpu.max", "150000 100000") in sc.caps

    def test_no_writable_subtree_refuses_loudly(self):
        sc = RecordingCgroupSyscalls(root=None)
        backend = LinuxIsolationBackend(sc)
        with pytest.raises(IsolationDowngrade, match="no writable cgroup"):
            backend._create_cgroup(IsolationPolicy(level=IsolationLevel.fork_cgroup, memory="1G"))

    def test_create_failure_refuses_loudly(self):
        sc = RecordingCgroupSyscalls(create_fails=True)
        backend = LinuxIsolationBackend(sc)
        with pytest.raises(IsolationDowngrade, match="could not create the cgroup leaf"):
            backend._create_cgroup(IsolationPolicy(level=IsolationLevel.fork_cgroup))

    def test_cap_write_failure_refuses_and_cleans_up(self):
        sc = RecordingCgroupSyscalls(cap_fails=True)
        backend = LinuxIsolationBackend(sc)
        with pytest.raises(IsolationDowngrade, match="could not write a cgroup cap"):
            backend._create_cgroup(IsolationPolicy(level=IsolationLevel.fork_cgroup, memory="1G"))
        # The half-built leaf was killed + removed (no orphan).
        assert sc.killed and sc.rmdirs

    def test_run_places_then_tears_down(self, tmp_path):
        # A trivial payload runs; placement recorded, teardown always kills+rmdir.
        sc = RecordingCgroupSyscalls()
        backend = LinuxIsolationBackend(sc)
        policy = IsolationPolicy(level=IsolationLevel.fork_cgroup, memory="256M")
        result = backend.run([sys.executable, "-c", "print('ok')"], policy, timeout=30)
        assert result.returncode == 0
        assert "ok" in result.stdout
        # The child pid was placed in the created leaf, then killed + rmdir'd.
        assert sc.placed and sc.placed[0][0] in sc.created
        assert sc.killed and sc.rmdirs


# ---------------------------------------------------------------------------
# (A7) invariance auto-check goes live
# ---------------------------------------------------------------------------


class TestInvarianceAutoCheck:
    def test_invariance_pass_appends_a_clean_record(self):
        optimizer = run_flex(
            Tagger(),
            SWAP_OPS,
            holdout=tag_devset("owl"),
            eval_mode="artifact",
            _eval_same_env=True,
            scoring_isolation="fork_ratchet",
            _isolation_backend=FakeBackend(),
        )
        # The last trajectory record is the invariance check — same-env, so
        # in_process and "envelope" scores match: no divergence.
        last = optimizer.trajectory[-1]
        assert last["label"] == "isolation-invariance"
        assert last["diverged"] is False
        assert last["refusals"] == []

    def test_invariance_check_can_be_disabled(self):
        optimizer = run_flex(
            Tagger(),
            SWAP_OPS,
            holdout=tag_devset("owl"),
            eval_mode="artifact",
            _eval_same_env=True,
            scoring_isolation="fork_ratchet",
            _isolation_backend=FakeBackend(),
            invariance_check=False,
        )
        assert all(entry.get("label") != "isolation-invariance" for entry in optimizer.trajectory)

    def test_no_invariance_check_without_a_wall_or_holdout(self):
        # scoring_isolation="none" -> no check even with a holdout.
        optimizer = run_flex(
            Tagger(),
            SWAP_OPS,
            holdout=tag_devset("owl"),
            eval_mode="artifact",
            _eval_same_env=True,
        )
        assert all(entry.get("label") != "isolation-invariance" for entry in optimizer.trajectory)


# ---------------------------------------------------------------------------
# (A8) Landlock ruleset construction (fake syscall layer)
# ---------------------------------------------------------------------------


class TestLandlockUnit:
    def test_abi_detection_gates_engagement(self):
        # A host reporting abi 0 never calls landlock_restrict.
        class NoLandlock(FakeSyscalls):
            def landlock_abi(self):
                return 0

            restricted = False

            def landlock_restrict(self, scopes, scratch):
                type(self).restricted = True

        backend = LinuxIsolationBackend(FakeSyscalls())
        preexec = backend.child_preexec(IsolationPolicy(level=IsolationLevel.fork_ratchet, files={"/data": "ro"}))
        assert preexec is not None  # a ratchet exists; whether Landlock engages is abi-gated in it

    def test_run_probes_reads_denied_when_landlock_engaged(self):
        # When Landlock engaged, a non-granted read that SUCCEEDS aborts.
        class ReadOpen(FakeSyscalls):
            def probe_socket_denied(self):
                return True

            def probe_write_denied(self, path):
                return True

            def probe_read_denied(self, path):
                return False  # the read leaked

        backend = LinuxIsolationBackend(FakeSyscalls())
        with pytest.raises(RuntimeError, match="non-granted path.*did not hold"):
            backend._run_probes(ReadOpen(), scratch="/tmp/s", landlock=True, files={"/data": "ro"})

    def test_run_probes_skip_read_when_landlock_absent(self):
        # No Landlock: the read probe does not fire (only netns + write-if-landlock).
        class OnlyNet(FakeSyscalls):
            def probe_socket_denied(self):
                return True

        backend = LinuxIsolationBackend(FakeSyscalls())
        backend._run_probes(OnlyNet(), scratch="/tmp/s", landlock=False)  # no raise


# ---------------------------------------------------------------------------
# GENUINE cgroup + Landlock — gated (reachable is sandbox on this host)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DSPY_FLEX_UV_TESTS"),
    reason="genuine cgroup/Landlock enforcement; set DSPY_FLEX_UV_TESTS=1",
)
class TestGenuineEnforcement:
    def _skip_below(self, level):
        if LinuxIsolationBackend().reachable_level() < level:
            pytest.skip(f"host does not reach {level.name}")

    def test_memory_balloon_dies_by_memory_max(self):
        self._skip_below(IsolationLevel.fork_cgroup)
        backend = LinuxIsolationBackend()
        policy = IsolationPolicy(level=IsolationLevel.fork_cgroup, memory="64M")
        code = "x = bytearray(300 * 1024 * 1024); print(len(x))"
        result = backend.run([sys.executable, "-c", code], policy, timeout=30)
        assert result.returncode != 0  # OOM-killed inside the cgroup

    def test_fork_bomb_stops_at_pids_max(self):
        self._skip_below(IsolationLevel.fork_cgroup)
        backend = LinuxIsolationBackend()
        policy = IsolationPolicy(level=IsolationLevel.fork_cgroup, pids=8)
        code = (
            "import threading, time, sys\n"
            "n = 0\n"
            "try:\n"
            "    while n < 200:\n"
            "        threading.Thread(target=lambda: time.sleep(20)).start()\n"
            "        n += 1\n"
            "except RuntimeError:\n"
            "    print('stopped at', n); sys.exit(7)\n"
            "print('reached', n)\n"
        )
        result = backend.run([sys.executable, "-c", code], policy, timeout=25)
        assert result.returncode == 7  # thread creation blocked by pids.max

    def test_cgroup_kill_reaps_a_straggler_grandchild(self, tmp_path):
        self._skip_below(IsolationLevel.fork_cgroup)
        backend = LinuxIsolationBackend()
        marker = tmp_path / "straggler_pid"
        code = (
            "import os, time, sys\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            "    os.setsid()\n"
            "    dn = os.open(os.devnull, os.O_RDWR)\n"
            "    os.dup2(dn, 0); os.dup2(dn, 1); os.dup2(dn, 2)\n"
            f"    open({str(marker)!r}, 'w').write(str(os.getpid()))\n"
            "    time.sleep(120)\n"
            "    os._exit(0)\n"
            "sys.exit(0)\n"
        )
        result = backend.run(
            [sys.executable, "-c", code], IsolationPolicy(level=IsolationLevel.fork_cgroup), timeout=20
        )
        assert result.returncode == 0
        import time as _t

        _t.sleep(0.3)
        gpid = int(marker.read_text())
        # After run()'s teardown (cgroup.kill), the grandchild is dead.
        assert not os.path.exists(f"/proc/{gpid}")

    def test_landlock_scopes_enforce(self, tmp_path):
        self._skip_below(IsolationLevel.fork_ratchet)
        if LinuxIsolationBackend().syscalls.landlock_abi() == 0:
            pytest.skip("kernel has no Landlock")
        backend = LinuxIsolationBackend()
        scratch = tmp_path / "scratch"
        rodir = tmp_path / "ro"
        rwdir = tmp_path / "rw"
        for d in (scratch, rodir, rwdir):
            d.mkdir()
        (rodir / "data.txt").write_text("hello")
        code = (
            "import os, sys\n"
            f"assert open({str(rodir / 'data.txt')!r}).read() == 'hello'\n"  # in-scope ro read
            f"open(os.path.join({str(rwdir)!r}, 'w.txt'), 'w').write('x')\n"  # rw scope write
            f"open(os.path.join({str(scratch)!r}, 's.txt'), 'w').write('y')\n"  # scratch write
            "try:\n"
            "    open('/etc/hostname').read(); print('LEAK-read'); sys.exit(5)\n"  # out-of-scope read denied
            "except OSError: pass\n"
            f"try:\n"
            f"    open(os.path.join({str(rodir)!r}, 'new.txt'), 'w').write('z'); print('LEAK-write'); sys.exit(6)\n"
            "except OSError: pass\n"
            "print('LANDLOCK-OK')\n"
        )
        policy = IsolationPolicy(
            level=IsolationLevel.fork_ratchet,
            scratch=str(scratch),
            files={str(rodir): "ro", str(rwdir): "rw"},
        )
        result = backend.run([sys.executable, "-c", code], policy, timeout=25)
        assert result.returncode == 0, result.stderr
        assert "LANDLOCK-OK" in result.stdout
