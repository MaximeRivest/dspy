"""The authoring surface: human-facing sugar over the leaf substrate.

Everything this campaign built — the isolation gradient (D-042), the
egress broker (Q7), the unified leaf substrate (PIR-021), floor-lowering
provenance (PIR-017) — is reachable through the optimizer's knobs and the
engine's bindings. This module is the ERGONOMIC face of it, so a human
author writes intent as decorators and small objects instead of stamping
function attributes by hand.

The through-line: a decorated function stays a NORMAL callable and a
NORMAL tool everywhere a tool is accepted (`ReAct(tools=[...])`, Flex
tools, `extract_tool`). The decorator only ATTACHES declared metadata
(`_dspy_*` stamps) that `extract_tool` already reads; it never wraps or
changes the callable's behavior.

DECLARATION vs ENFORCEMENT — stated once, honestly, here:

- ENFORCED today: `isolation=` floor (materialize/dspy.load fail closed
  below it — the isolation stage), `session=`/`grants=` (the real bridge
  in the engine), `net=` broker routes (the broker injects and allowlists
  when FlexIR runs under it), `deps=` (validated + injected into the
  `# deps:` line, unioned into the artifact environment).
- ENFORCED WHEN RUN UNDER AN ENVELOPE AT THE RIGHT LEVEL: `memory=`/
  `cpus=` become real cgroup v2 caps at `fork_cgroup`+; `files=` becomes
  a real Landlock filesystem wall at `fork_ratchet`+ (when the kernel has
  Landlock). `dspy.confined(memory=/cpus=/files=)` and `dspy.Session`
  flow these into the run's policy; `FlexIR(scoring_memory=/scoring_cpus=)`
  caps the scoring child. On a `@dspy.tool` these still ride the pool
  entry as the LEAF's declared intent — a receiver's envelope is what
  turns them into enforcement.
- DECLARED-ONLY (recorded as intent, no behavior claim; the mechanism is
  owed): `gpu=` (device placement).
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "Broker",
    "Envelope",
    "SecretFromEnv",
    "Session",
    "confined",
    "retrust",
    "tool",
]


# ---------------------------------------------------------------------------
# @dspy.tool — attach declared metadata to a plain tool function
# ---------------------------------------------------------------------------


def tool(
    func: Callable[..., Any] | None = None,
    *,
    isolation: str | None = None,
    net: list[str] | None = None,
    files: dict[str, str] | None = None,
    deps: list[str] | None = None,
    memory: str | None = None,
    cpus: float | int | None = None,
    gpu: bool | str | None = None,
    session: bool = False,
    grants: list[Any] | None = None,
) -> Callable[..., Any]:
    """Declare a tool's capability surface, keeping it a normal callable.

    Bare `@dspy.tool` is a no-op marker: today's plain tool, zero change.
    Every kwarg attaches a `_dspy_*` stamp `extract_tool` reads at compile,
    so the declaration survives into the pool entry. The function itself is
    untouched — it is still callable, still a tool everywhere a tool is
    accepted.

    Args:
        isolation: A D-042 level name (`"fork_ratchet"`, `"sandbox"`, ...)
            stamped as the leaf's isolation FLOOR — the minimum a receiver
            envelope must meet (PIR-016). ENFORCED: load fails closed below
            it.
        net: Hostnames the leaf may reach, declared as `broker_route`
            grants (contract-valid `grants[]`). ENFORCED when the run goes
            through an egress broker (the broker allowlists + injects);
            otherwise a recorded declaration.
        files: `{path: "ro"|"rw"}` filesystem scopes — the leaf's declared
            filesystem intent. ENFORCED as a real Landlock wall when the
            leaf runs under a `fork_ratchet`+ envelope on a Landlock
            kernel (via `confined`/`Session`/an envelope); otherwise a
            recorded declaration.
        deps: Third-party packages. Validated against and injected into the
            source's `# deps:` line (conflict refuses); ENFORCED via the
            artifact environment union.
        memory: A cgroup memory cap (e.g. `"2G"`) — the leaf's declared
            intent. ENFORCED as `memory.max` under a `fork_cgroup`+
            envelope; otherwise recorded.
        cpus: A cgroup CPU budget — enforced as `cpu.max` under a
            `fork_cgroup`+ envelope; otherwise recorded (as `memory`).
        gpu: GPU access intent. DECLARED-ONLY (device placement is owed).
        session: `True` marks a PIR-021 session-kind leaf (the function
            takes the grant bridge as its FIRST parameter). ENFORCED by the
            engine bridge.
        grants: Leaf names (or tools) a session leaf may call back into,
            as pool-leaf bridge grants. ENFORCED: the bridge exposes only
            these; a dangling grant refuses at load.

    Returns:
        The same function, with declared metadata attached.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        if isolation is not None:
            from dspy.programir.engine.isolation import parse_level

            fn._dspy_isolation_floor = parse_level(isolation).name  # validates the level name
        grant_entries: list[dict[str, str]] = []
        for host in net or []:
            if not isinstance(host, str) or not host:
                raise ValueError(f"@dspy.tool net entries must be non-empty hostname strings, got {host!r}")
            grant_entries.append({"kind": "broker_route", "name": host})
        for grant in grants or []:
            grant_entries.append(_leaf_grant_of(grant))
        if grant_entries:
            fn._dspy_leaf_grants = grant_entries
        if session:
            fn._dspy_leaf_kind = "session"
        elif grants:
            raise ValueError("@dspy.tool grants require session=True (only a session leaf holds a grant bridge)")
        if deps is not None:
            _apply_deps(fn, deps)
        declared = _declared_resources(files=files, memory=memory, cpus=cpus, gpu=gpu)
        if declared:
            # DECLARED-NOT-ENFORCED intent, kept on the function so a
            # receiver/auditor can read it. No engine behavior consumes it.
            fn._dspy_declared_resources = declared
        return fn

    if func is not None:
        return decorate(func)
    return decorate


def _leaf_grant_of(grant: Any) -> dict[str, str]:
    from dspy.adapters.types.tool import Tool
    from dspy.programir.leaves import leaf_grant

    if isinstance(grant, str):
        return leaf_grant(grant)
    if isinstance(grant, Tool):
        return leaf_grant(grant.name)
    if callable(grant):
        return leaf_grant(getattr(grant, "__name__", None) or repr(grant))
    if isinstance(grant, dict) and "kind" in grant and "name" in grant:
        return dict(grant)
    raise ValueError(f"@dspy.tool grant must be a leaf name, a tool, or a callable, got {grant!r}")


def _declared_resources(*, files, memory, cpus, gpu) -> dict[str, Any]:
    declared: dict[str, Any] = {}
    if files is not None:
        for path, mode in files.items():
            if mode not in ("ro", "rw"):
                raise ValueError(f"@dspy.tool files mode for {path!r} must be 'ro' or 'rw', got {mode!r}")
        declared["files"] = dict(files)
    if memory is not None:
        declared["memory"] = memory
    if cpus is not None:
        declared["cpus"] = cpus
    if gpu is not None:
        declared["gpu"] = gpu
    return declared


_DEPS_LINE = re.compile(r"^\s*#\s*deps:\s*(.*?)\s*$", re.MULTILINE)


def _apply_deps(fn: Callable[..., Any], deps: list[str]) -> None:
    """Validate `deps` against any existing `# deps:` line; refuse conflict.

    The stamp `_dspy_declared_deps` records the intended set; `extract_tool`
    reads the SOURCE's `# deps:` line, so the two must agree. We refuse a
    conflict loudly here (the author asked for one set, the source declares
    another) rather than silently pick one.
    """
    import inspect
    import textwrap

    wanted = sorted({str(dep) for dep in deps})
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        source = ""
    found = _DEPS_LINE.search(source)
    if found:
        existing = sorted({item.strip() for item in found.group(1).split(",") if item.strip()})
        if existing and existing != wanted:
            raise ValueError(
                f"@dspy.tool deps={wanted} conflicts with the source's `# deps: {', '.join(existing)}` line; "
                "make them agree (or drop one) — a tool declares one dependency set, not two"
            )
    fn._dspy_declared_deps = wanted


# ---------------------------------------------------------------------------
# dspy.Session — sugar for a session-kind tool leaf with grants
# ---------------------------------------------------------------------------


def Session(  # noqa: N802 — a constructor-shaped factory, capitalized like a class
    func: Callable[..., Any],
    *,
    grants: list[Any],
    isolation: str | None = None,
    memory: str | None = None,
    lifetime: str = "forward",
) -> Callable[..., Any]:
    """Construct a session-kind tool leaf granting the given tools/names.

    A thin wrapper over `@dspy.tool(session=True, grants=..., ...)`: the
    substrate does the work, so this refuses exactly what the substrate
    refuses (a dangling grant refuses at load; a source without a bridge
    first parameter refuses at admission).

    Args:
        func: The session function (takes the grant bridge FIRST).
        grants: Leaf names/tools the session may call back into.
        isolation: Optional D-042 floor for the session leaf.
        memory: Optional declared cgroup memory cap (declared-not-enforced).
        lifetime: Only `"forward"` is supported — the session's lifetime is
            one forward span (object lifetime in-process, fork-scoped at
            fork_ratchet+). Any other value refuses.
    """
    if lifetime != "forward":
        raise ValueError(
            f"dspy.Session lifetime must be 'forward' (the only span the substrate holds), got {lifetime!r}"
        )
    return tool(func, session=True, grants=grants, isolation=isolation, memory=memory)


# ---------------------------------------------------------------------------
# dspy.Envelope — the receiver-side isolation envelope
# ---------------------------------------------------------------------------


class Envelope:
    """A receiver's isolation envelope — wraps an `IsolationPolicy` (D-042).

    Pass it to `dspy.load(..., envelope=...)` to satisfy an
    `isolation_required` leaf at load (an envelope meeting the leaf's floor
    IS the grant), or to `FlexIR(scoring_isolation=...)` as the level name.

    Args:
        level: A D-042 level name (`"fork_ratchet"`, `"sandbox"`, ...).
        broker: An optional `dspy.Broker` — its allowlist rides the
            envelope record so a run shows the network reality.
        scratch: The one writable directory the payload may use.
    """

    def __init__(self, level: str, *, broker: Broker | None = None, scratch: str | None = None):
        from dspy.programir.engine.isolation import IsolationPolicy, parse_level

        egress = frozenset(broker.allow) if broker is not None else frozenset()
        self.policy = IsolationPolicy(level=parse_level(level), broker_egress=egress, scratch=scratch)
        self.broker = broker

    @property
    def level_name(self) -> str:
        return self.policy.level.name

    def as_binding(self) -> dict[str, Any]:
        """The `bindings['isolation']` value `materialize` reads."""
        return {"envelope": self.policy}


# ---------------------------------------------------------------------------
# dspy.Broker / dspy.SecretFromEnv — the public face of the egress broker
# ---------------------------------------------------------------------------


class SecretFromEnv:
    """A credential read from an environment variable at SERVE time.

    The value is NEVER stored on the object — `resolve()` reads the env var
    each time the broker needs it, so the secret lives only transiently in
    the broker process and never on disk, in the artifact, or in a repr.
    """

    def __init__(self, var: str):
        self.var = var

    def resolve(self) -> str:
        import os

        value = os.environ.get(self.var)
        if not value:
            raise ValueError(f"dspy.SecretFromEnv({self.var!r}) is not set in the environment")
        return value

    def __repr__(self) -> str:  # never leak a value — there is none held
        return f"SecretFromEnv({self.var!r})"


class Broker:
    """The public face of the egress broker (parent-owned, localhost proxy).

    Deny-by-default egress: only `allow` hosts are reachable, each request
    is logged, and a credential named via `SecretFromEnv` is injected on
    egress so the child never holds the key.

    LIMIT, stated prominently: credential injection works on the PLAIN-HTTP
    forward path only. An HTTPS request goes through `CONNECT` as an opaque
    TLS tunnel — the broker cannot see or add a header inside end-to-end
    TLS. So a real `https://` provider endpoint reached by CONNECT gets NO
    injected credential; injection is for localhost/plain-HTTP endpoints
    (and the reason the scoring-child stub tests speak plain HTTP).

    Args:
        allow: Hostnames the child may reach.
        inject: `{host: SecretFromEnv | {"header", "value"}}` — a header
            attached on egress to that host (plain-HTTP path only).
        log: Optional path; the request log is written there on `stop()`.
    """

    def __init__(
        self,
        allow: list[str],
        inject: dict[str, Any] | None = None,
        log: str | None = None,
    ):
        self.allow = frozenset(allow)
        self.inject = inject or {}
        self.log = log
        self._broker: Any = None

    def _materialize_inject(self) -> dict[str, dict[str, str]]:
        resolved: dict[str, dict[str, str]] = {}
        for host, secret in self.inject.items():
            if isinstance(secret, SecretFromEnv):
                resolved[host] = {"header": "Authorization", "value": f"Bearer {secret.resolve()}"}
            elif isinstance(secret, dict) and "header" in secret and "value" in secret:
                resolved[host] = {"header": secret["header"], "value": secret["value"]}
            else:
                raise ValueError(
                    f"dspy.Broker inject for {host!r} must be a SecretFromEnv or {{'header','value'}}, got {secret!r}"
                )
        return resolved

    def start(self) -> Broker:
        from dspy.optim.broker import EgressBroker

        self._broker = EgressBroker(self.allow, inject=self._materialize_inject()).start()
        return self

    @property
    def proxy_url(self) -> str:
        if self._broker is None:
            raise ValueError("dspy.Broker is not started; call .start() first")
        return self._broker.proxy_url

    @property
    def requests(self) -> list[dict[str, object]]:
        return self._broker.requests if self._broker is not None else []

    def stop(self) -> None:
        if self._broker is not None:
            if self.log is not None:
                Path(self.log).write_text(json.dumps(self._broker.requests, indent=2))
            self._broker.stop()
            self._broker = None


# ---------------------------------------------------------------------------
# dspy.retrust — lower a leaf's floor as a recorded artifact edit
# ---------------------------------------------------------------------------


def retrust(program: Any, leaf_path: str, *, floor: str, reason: str | None = None, by: str = "receiver") -> Any:
    """Return a program whose named leaf's isolation floor is LOWERED.

    Lowering a floor is an authorship act, not a runtime flag (PIR-017):
    it edits the baked placement and records `floor_lowered_by {by, date,
    old, new, reason}` in provenance — the artifact stays truthful about
    how it runs. There is deliberately NO ignore-floors flag.

    RAISING a floor through retrust is REFUSED: raising isolation is a
    receiver's ENVELOPE choice (a binding, `dspy.Envelope`), not an edit —
    it must not fork the artifact. The refusal points there.

    Args:
        program: The `dspy.Module` to edit (mutated: the named leaf's stamp
            changes and the IR is invalidated).
        leaf_path: The tool leaf attribute to retrust (on the root or a
            submodule, dotted).
        floor: The new, LOWER D-042 floor level name.
        reason: Optional human note recorded in the provenance.
        by: Who is lowering it (recorded).

    Returns:
        The same program, with the leaf's floor lowered and provenance
        recorded. Compile/save it to get the edited artifact.
    """
    import datetime

    from dspy.modules.module import Module
    from dspy.programir.engine.isolation import parse_level
    from dspy.programir.leaves import extract_tool

    if not isinstance(program, Module):
        raise ValueError(f"dspy.retrust takes a dspy.Module, got {type(program).__name__}")
    owner, attribute = _owner_of(program, leaf_path)
    fn = owner.__dict__.get(attribute) if owner is not None else None
    if fn is None or not callable(fn) or isinstance(fn, Module):
        raise ValueError(f"dspy.retrust found no tool leaf at {leaf_path!r}")

    new_level = parse_level(floor)
    # The current effective floor: the stamp, else the entry's placement.
    current_name = getattr(fn, "_dspy_isolation_floor", None)
    if current_name is None:
        entry = extract_tool(fn, name=attribute).entry
        current_name = entry["placement"].get("isolation_floor")
        if current_name is None:
            # No explicit floor: an isolation-required leaf's implicit floor
            # is the authored-leaf floor; a plain leaf's is `none`.
            from dspy.programir.engine.isolation import AUTHORED_LEAF_FLOOR

            current_name = (
                AUTHORED_LEAF_FLOOR.name if entry["placement"].get("rung") == "isolation_required" else "none"
            )
    current_level = parse_level(current_name)

    if new_level > current_level:
        raise ValueError(
            f"dspy.retrust refuses to RAISE the floor ({current_level.name} -> {new_level.name}): raising "
            "isolation is a receiver's ENVELOPE choice, not an artifact edit — pass dspy.Envelope("
            f"{new_level.name!r}) to dspy.load or FlexIR(scoring_isolation=...) instead. retrust only lowers."
        )
    if new_level == current_level:
        raise ValueError(f"dspy.retrust: the floor is already {current_level.name}; nothing to lower")

    fn._dspy_isolation_floor = new_level.name
    fn._dspy_placement_rung = "in_process" if new_level.name == "none" else getattr(fn, "_dspy_placement_rung", None)
    fn._dspy_floor_lowered_by = {
        "by": by,
        "date": datetime.date.today().isoformat(),
        "old": current_level.name,
        "new": new_level.name,
        **({"reason": reason} if reason else {}),
    }
    program.invalidate_ir()
    return program


def _owner_of(program: Any, path: str):
    parts = path.split(".")
    attribute = parts[-1]
    parent = program
    for part in parts[:-1]:
        parent = parent.__dict__.get(part)
        if parent is None:
            return None, attribute
    return parent, attribute


# ---------------------------------------------------------------------------
# dspy.confined — run one callable under the envelope in a child
# ---------------------------------------------------------------------------


def confined(
    fn: Callable[..., Any],
    *args: Any,
    isolation: str = "fork_ratchet",
    memory: str | None = None,
    cpus: float | int | None = None,
    files: dict[str, str] | None = None,
    _backend: Any = None,
    **kwargs: Any,
) -> Any:
    """Run `fn(*args, **kwargs)` ONCE in a child under the envelope.

    Reuses the isolation backend (fork-place-gate-ratchet) and a minimal
    JSON-in/JSON-out child runner. Returns the callable's result, or
    re-raises the child's error.

    ENFORCEMENT by level: `memory`/`cpus` graduate to real cgroup caps at
    `fork_cgroup`+; `files` graduates to a real Landlock filesystem wall at
    `fork_ratchet`+ when the kernel supports it (otherwise the mount-ns
    wall alone stands, recorded honestly).

    FORK-CONTEXT REQUIREMENT: the ratchet half (`fork_ratchet`+) needs a
    single-threaded fork so `unshare(CLONE_NEWUSER)` is legal. The launcher
    runs the ratchet post-exec (single-threaded); on Python 3.14 the
    default multiprocessing start method changes, but the backend uses a
    subprocess exec, not multiprocessing, so this is unaffected. The
    backend refuses loudly (`IsolationDowngrade`) if the host cannot build
    the requested level.

    Args (fn/args/kwargs): the callable and its arguments — must be
    JSON-round-trippable (args, kwargs, and the return value cross as
    JSON). A non-serializable value refuses before the child launches.
    """
    from dspy.programir.engine.isolation import IsolationPolicy, LinuxIsolationBackend, parse_level

    level = parse_level(isolation)
    backend = _backend or LinuxIsolationBackend()
    reached = backend.best_effort_level(level)  # raises IsolationDowngrade if under-floor

    import inspect
    import textwrap

    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError) as error:
        raise ValueError(f"dspy.confined needs an introspectable function; {fn!r} is not: {error}") from error
    try:
        payload = json.dumps({"args": list(args), "kwargs": kwargs})
    except (TypeError, ValueError) as error:
        raise ValueError(f"dspy.confined args/kwargs must be JSON-serializable: {error}") from error

    with tempfile.TemporaryDirectory(prefix="dspy-confined-") as tmp:
        # The scratch dir the child may write is the confined-run tmp; the
        # caller's files= scopes are added to it.
        policy = IsolationPolicy(level=reached, scratch=tmp, memory=memory, cpus=cpus, files=dict(files or {}))
        job = Path(tmp) / "job.json"
        job.write_text(json.dumps({"source": source, "name": fn.__name__, "payload": payload}))
        argv = [sys.executable, "-m", "dspy.optim._confined_runner", str(job)]
        child = backend.run(argv, policy)
    if child.returncode != 0:
        tail = "\n".join((child.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(f"dspy.confined child failed (exit {child.returncode}):\n{tail}")
    result = json.loads(child.stdout.strip().splitlines()[-1])
    if result.get("error"):
        raise RuntimeError(f"dspy.confined callable raised in the child: {result['error']}")
    return result["value"]
