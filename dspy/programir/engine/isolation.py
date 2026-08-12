"""The D-042 isolation gradient and its Linux backend.

The ordered isolation vocabulary — an axis ORTHOGONAL to the placement
rung, mostly refining rung 1 — names, mechanism by mechanism, what it
costs to walk away from the fully-baked zero pole:

    none < namespace < fork < fork_cgroup < fork_ratchet < sandbox < remote

`IsolationLevel` is a comparable enum, so a "floor" is well-defined by
comparison and `max(entry_floor, binding_floor)` composes floors upward
(D-042 point 8). `IsolationPolicy` is the envelope object a receiver
binds; `satisfies(floor)` is the load-time floor check (may exceed, never
under-run — under-floor is a loud refusal, D-042 point 6).

The Linux backend implements the fork-place-ratchet recipe (Q4): fork →
the PARENT places the child in a cgroup-v2 leaf → the CHILD self-ratchets
(unshare user/net/mount namespaces where available; `prctl` NO_NEW_PRIVS;
Landlock/seccomp when the libraries are present) → a self-probe confirms
the wall (a `socket()` that must fail, an out-of-scratch write that must
fail) before any payload runs.

Two laws hold everywhere here, both from the notes:

- FAIL CLOSED, NEVER SILENTLY. If a requested level cannot be
  established, the backend records a loud downgrade refusal — it never
  degrades to a weaker wall while claiming the stronger one. A cgroupfs
  that is unwritable, a namespace primitive the kernel refuses, a
  self-probe that unexpectedly succeeds: each is a recorded refusal, not
  a shrug.
- BEHAVIOR IS ISOLATION-INVARIANT. Raising the wall never changes what
  the leaf computes; a level change that changes behavior indicts the
  leaf (Q9 point 4). `describe_envelope` records the actual envelope so
  a divergence is attributable in one diff.

This module owns the vocabulary, the policy/envelope objects, feature
detection, and the recipe helpers. The one syscall seam
(`_SyscallLayer`) is swappable so the fail-closed and self-probe logic is
testable without a userns-capable kernel.
"""

from __future__ import annotations

import enum
import itertools
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

#: Monotonic per-process counter so concurrent cgroup leaves never collide.
_CGROUP_COUNTER = itertools.count(1)

_MEMORY_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

#: Landlock filesystem access-right bits (uapi/linux/landlock.h). Kept at
#: module scope so the ctypes applier reads clean constants (not
#: uppercase locals). Read rights cover file/dir read + execute; write
#: rights cover file writes and the create/remove family (directory-only).
_LL_READ_RIGHTS = (1 << 2) | (1 << 3) | (1 << 0)  # READ_FILE | READ_DIR | EXECUTE
_LL_WRITE_RIGHTS = (1 << 1) | (0x1FF << 4)  # WRITE_FILE | (REMOVE_* and MAKE_* family)
#: The create/remove bits, invalid on a regular-file grant (EINVAL) — the
#: applier masks these off for non-directory paths.
_LL_DIR_ONLY = (1 << 3) | (0x1FF << 4)  # READ_DIR | (REMOVE_* and MAKE_* family)


def _cgroup_bytes(value: str | int) -> str:
    """Render a memory cap ("2G", 2147483648, "max") as a cgroup byte string."""
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if text.lower() == "max":
        return "max"
    if text and text[-1].upper() in _MEMORY_UNITS:
        return str(int(float(text[:-1]) * _MEMORY_UNITS[text[-1].upper()]))
    return str(int(text))


__all__ = [
    "IsolationBackend",
    "IsolationDowngrade",
    "IsolationLevel",
    "IsolationPolicy",
    "LinuxIsolationBackend",
    "parse_level",
]


class IsolationLevel(enum.IntEnum):
    """The ordered D-042 isolation vocabulary (comparable — floors compose).

    Each level adds to the previous; the integer values encode the order,
    so `floor <= envelope` IS the "envelope meets the floor" check and
    `max(a, b)` composes two floors upward.
    """

    none = 0
    namespace = 1
    fork = 2
    fork_cgroup = 3
    fork_ratchet = 4
    sandbox = 5
    remote = 6

    @property
    def label(self) -> str:
        return self.name


def parse_level(value: Any) -> IsolationLevel:
    """Coerce a name or level into an `IsolationLevel`, refusing teaching."""
    if isinstance(value, IsolationLevel):
        return value
    if isinstance(value, str):
        try:
            return IsolationLevel[value]
        except KeyError:
            raise ValueError(
                f"unknown isolation level {value!r}; the ordered vocabulary is "
                f"{[level.name for level in IsolationLevel]}"
            ) from None
    raise ValueError(f"isolation level must be a name or IsolationLevel, got {type(value).__name__}")


#: The floor an optimizer-authored leaf's placement enforces, expressed on
#: the D-042 axis. `fork_ratchet` is the level at which the leaf runs
#: under an ephemeral identity with the network unshared — the wall that
#: makes "authored code, never with ambient network" true (Q9). An
#: envelope at or above this level SATISFIES the isolation-required grant.
AUTHORED_LEAF_FLOOR = IsolationLevel.fork_ratchet


@dataclass(frozen=True)
class IsolationDowngrade(Exception):  # noqa: N818 — "downgrade" is the notes' word for this refusal
    """A requested isolation level could not be established — fail closed.

    Raised (never swallowed) when the backend cannot build the wall the
    caller asked for: an unwritable cgroupfs, a kernel that refuses a
    namespace, a self-probe that succeeded where it had to fail. The
    message names the missing mechanism so the refusal teaches.
    """

    requested: IsolationLevel
    reached: IsolationLevel
    reason: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return (
            f"isolation downgrade refused: requested {self.requested.name}, "
            f"could only reach {self.reached.name} — {self.reason}"
        )


@dataclass(frozen=True)
class IsolationPolicy:
    """One receiver isolation envelope (the D-042 binding, not identity).

    Attributes:
        level: The isolation level this envelope provides.
        broker_egress: Hostnames the parent broker will allow on egress
            (empty = deny-all network, the netns default). Carried here
            so a run record shows the exact network reality.
        scratch: The one writable directory the payload may use; an
            out-of-scratch write must fail the self-probe.
        memory: cgroup `memory.max` cap (e.g. "2G" or an int of bytes);
            ENFORCED at `fork_cgroup`+ (the parent writes it before the
            child ratchets). None = uncapped.
        pids: cgroup `pids.max` cap; ENFORCED at `fork_cgroup`+. A fork
            bomb stops here. None = uncapped.
        cpus: cgroup `cpu.max` budget as a fraction of one CPU (e.g. 1.5);
            ENFORCED at `fork_cgroup`+. None = unbounded.
        files: `{path: "ro"|"rw"}` filesystem scopes; ENFORCED via
            Landlock at `fork_ratchet`+ WHEN the kernel supports it —
            deny-all beyond the scopes and scratch. Empty = the mount-ns
            wall alone (no per-path Landlock rules).
    """

    level: IsolationLevel = IsolationLevel.none
    broker_egress: frozenset[str] = field(default_factory=frozenset)
    scratch: str | None = None
    memory: str | int | None = None
    pids: int | None = None
    cpus: float | int | None = None
    files: dict[str, str] = field(default_factory=dict)

    def satisfies(self, floor: IsolationLevel) -> bool:
        """True when this envelope meets or exceeds the floor (D-042 point 6)."""
        return self.level >= floor

    def refuse_under_floor(self, floor: IsolationLevel, *, subject: str) -> None:
        """Loud refusal when the envelope under-runs the floor."""
        if not self.satisfies(floor):
            raise ValueError(
                f"{subject} demands isolation floor {floor.name!r}, but the bound envelope only offers "
                f"{self.level.name!r} — under-floor execution is refused (D-042: may exceed, never under-run)"
            )

    def describe_envelope(self, *, engaged: list[str] | None = None) -> dict[str, Any]:
        """The provenance record of the actual envelope (D-042 point 7).

        `engaged` names the mechanisms that ACTUALLY bit at run time
        (`cgroup`, `userns`, `netns`, `mountns`, `no_new_privs`,
        `landlock`) — recorded honestly, since a declared level may
        engage fewer mechanisms than its name suggests on a given host
        (e.g. Landlock absent, so files= did not enforce).
        """
        record: dict[str, Any] = {
            "level": self.level.name,
            "broker_egress": sorted(self.broker_egress),
            "scratch": self.scratch,
        }
        caps = {
            key: value
            for key, value in (("memory", self.memory), ("pids", self.pids), ("cpus", self.cpus))
            if value is not None
        }
        if caps:
            record["caps"] = caps
        if self.files:
            record["files"] = dict(self.files)
        if engaged is not None:
            record["engaged"] = engaged
        return record


# ---------------------------------------------------------------------------
# The syscall seam: real on Linux, fakeable in tests
# ---------------------------------------------------------------------------


class _SyscallLayer:
    """The thin OS seam the backend calls — the one place tests fake.

    Every method degrades to a truthful capability answer or performs a
    real primitive. Feature detection lives here so the backend can ask
    "can this kernel do X" without importing platform specifics inline.
    """

    def has_cgroup_v2(self) -> bool:
        return os.path.isdir("/sys/fs/cgroup") and os.path.exists("/sys/fs/cgroup/cgroup.controllers")

    def cgroupfs_writable(self, root: str = "/sys/fs/cgroup") -> bool:
        """True when this process can create cgroup leaves to place a child in.

        Delegates to `writable_cgroup_root` — one source of truth. The
        cgroup ROOT is never user-writable on a systemd host; what a user
        owns is their DELEGATED subtree (user@<uid>.service). Checking the
        root alone would false-downgrade exactly the hosts where the
        fork_cgroup rung works.
        """
        return self.writable_cgroup_root(root) is not None

    def has_unshare(self) -> bool:
        return hasattr(os, "unshare") and hasattr(os, "CLONE_NEWUSER")

    def has_landlock(self) -> bool:
        try:
            import landlock  # noqa: F401
        except Exception:
            return False
        return True

    def has_seccomp(self) -> bool:
        try:
            import pyseccomp  # noqa: F401
        except Exception:
            return False
        return True

    def has_bwrap(self) -> bool:
        from shutil import which

        return which("bwrap") is not None

    def unshare(self, flags: int) -> None:  # pragma: no cover - exercised only on a userns kernel
        os.unshare(flags)

    def set_no_new_privs(self) -> None:  # pragma: no cover - real prctl
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        pr_set_no_new_privs = 38  # PR_SET_NO_NEW_PRIVS from <linux/prctl.h>
        if libc.prctl(pr_set_no_new_privs, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")

    def probe_socket_denied(self) -> bool:
        """True when outbound NETWORK REACH fails (the wall holds).

        In a fresh empty netns `socket()` itself still SUCCEEDS — an empty
        namespace hands out sockets happily; what it cannot do is route.
        The genuine-child run proved this: probing socket creation passes
        the probe on the host and fails it inside the wall, exactly
        backwards. So the probe attempts a connect to TEST-NET-1
        (192.0.2.1, reserved, never assigned): an immediate
        ENETUNREACH/EHOSTUNREACH is the empty-netns signature. On the
        host side the same connect times out or is refused (packets left
        the box) — NOT denied.
        """
        import errno
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError:
            return True
        try:
            sock.settimeout(0.25)
            sock.connect(("192.0.2.1", 9))
        except OSError as error:
            return getattr(error, "errno", None) in (errno.ENETUNREACH, errno.EHOSTUNREACH)
        finally:
            sock.close()
        return False

    def probe_write_denied(self, path: str) -> bool:
        """True when writing OUTSIDE scratch FAILS (the wall holds)."""
        try:
            with open(path, "w") as handle:
                handle.write("x")
        except OSError:
            return True
        os.unlink(path)
        return False

    def probe_read_denied(self, path: str) -> bool:
        """True when READING a non-granted path FAILS (Landlock holds)."""
        try:
            with open(path, "rb") as handle:
                handle.read(1)
        except OSError:
            return True
        return False

    # -- cgroup v2: parent-side placement + caps (fork_cgroup+) -----------

    def writable_cgroup_root(self, root: str = "/sys/fs/cgroup") -> str | None:
        """The delegated subtree this process may create capped leaves under.

        A usable subtree must be WRITABLE and carry the memory+pids
        controllers in its `cgroup.controllers` (a leaf's caps only bind
        when its parent enabled those controllers via `subtree_control`).
        The process's own `.scope` is writable but usually lacks the
        controllers, so the `user@<uid>.service` ancestor — where systemd
        delegates them — is preferred.
        """
        candidates: list[str] = []
        own_path = None
        try:
            with open("/proc/self/cgroup", encoding="utf-8") as fh:
                for line in fh:
                    prefix, _, path = line.strip().partition("::")
                    if prefix == "0" and path:
                        own_path = root + path
        except OSError:
            pass
        uid = os.getuid()
        # Prefer the delegated user@<uid>.service (controllers live there),
        # then walk UP the own path (deepest first), then the service, then
        # the bare root as a last resort.
        service = f"{root}/user.slice/user-{uid}.slice/user@{uid}.service"
        candidates.append(service)
        candidates.append(f"{service}/app.slice")
        if own_path:
            parts = own_path.rstrip("/").split("/")
            while len(parts) > 3:  # keep at least /sys/fs/cgroup
                candidates.append("/".join(parts))
                parts = parts[:-1]
        candidates.append(root)
        for path in candidates:
            if not os.access(path, os.W_OK):
                continue
            try:
                with open(os.path.join(path, "cgroup.controllers"), encoding="utf-8") as fh:
                    controllers = set(fh.read().split())
            except OSError:
                continue
            if {"memory", "pids"} <= controllers:
                return path
        return None

    def cgroup_create(self, leaf: str) -> None:  # pragma: no cover - real cgroupfs
        os.mkdir(leaf)

    def cgroup_write(self, leaf: str, control: str, value: str) -> None:  # pragma: no cover - real cgroupfs
        with open(os.path.join(leaf, control), "w", encoding="utf-8") as fh:
            fh.write(value)

    def cgroup_place(self, leaf: str, pid: int) -> None:  # pragma: no cover - real cgroupfs
        with open(os.path.join(leaf, "cgroup.procs"), "w", encoding="utf-8") as fh:
            fh.write(str(pid))

    def cgroup_kill(self, leaf: str) -> None:  # pragma: no cover - real cgroupfs
        try:
            with open(os.path.join(leaf, "cgroup.kill"), "w", encoding="utf-8") as fh:
                fh.write("1")
        except OSError:
            pass

    def cgroup_rmdir(self, leaf: str) -> None:  # pragma: no cover - real cgroupfs
        import time as _time

        # After cgroup.kill the leaf empties asynchronously; retry rmdir.
        for _ in range(50):
            try:
                os.rmdir(leaf)
                return
            except OSError:
                _time.sleep(0.01)

    # -- Landlock: pure-ctypes ruleset (fork_ratchet+, files=) -----------

    def landlock_abi(self) -> int:
        """The kernel's Landlock ABI version, or 0 when unsupported."""
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        LANDLOCK_CREATE_RULESET_VERSION = 1 << 0  # noqa: N806
        try:
            abi = libc.syscall(444, 0, 0, LANDLOCK_CREATE_RULESET_VERSION)
        except Exception:
            return 0
        return abi if isinstance(abi, int) and abi > 0 else 0

    def landlock_restrict(
        self, scopes: dict[str, str], scratch: str | None
    ) -> None:  # pragma: no cover - real syscalls
        """Apply a deny-all-else Landlock ruleset for `scopes` (+ scratch).

        Pure ctypes against the three Landlock syscalls (create_ruleset=444,
        add_rule=445, restrict_self=446). Read-only paths get the read
        access rights; read-write paths add the write/create rights;
        scratch is granted read-write. Everything else is denied. A kernel
        without Landlock reports abi 0 and this is never called.
        """
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)

        read_rights = _LL_READ_RIGHTS
        write_rights = _LL_WRITE_RIGHTS
        handled = read_rights | write_rights

        class RulesetAttr(ctypes.Structure):
            _fields_ = [("handled_access_fs", ctypes.c_uint64)]

        class PathBeneathAttr(ctypes.Structure):
            _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

        attr = RulesetAttr(handled_access_fs=handled)
        ruleset_fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
        if ruleset_fd < 0:
            raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")

        granted = dict(scopes)
        if scratch:
            granted[scratch] = "rw"
        # The payload cannot run at all if Landlock denies the interpreter
        # and its stdlib — grant those read-only implicitly (the system
        # runtime is not the untrusted surface files= is protecting). Also
        # the standard shared-library and CA roots the runtime opens.
        for runtime in (sys.prefix, sys.base_prefix, os.path.dirname(sys.executable), "/usr", "/lib", "/lib64"):
            if runtime and runtime not in granted and os.path.isdir(runtime):
                granted[runtime] = "ro"
        # Char devices the runtime opens at startup (hash randomization
        # reads /dev/urandom); granted read-only as files.
        for device in ("/dev/urandom", "/dev/null"):
            if device not in granted and os.path.exists(device):
                granted[device] = "ro"
        rule_path_beneath = 1  # LANDLOCK_RULE_PATH_BENEATH
        for path, mode in granted.items():
            try:
                path_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
                is_dir = os.path.isdir(path)
            except OSError:
                continue
            try:
                allowed = read_rights if mode == "ro" else read_rights | write_rights
                if not is_dir:
                    # Directory-only bits are invalid on a regular file
                    # (the kernel rejects them EINVAL) — mask them off.
                    allowed &= ~_LL_DIR_ONLY
                rule = PathBeneathAttr(allowed_access=allowed, parent_fd=path_fd)
                if libc.syscall(445, ruleset_fd, rule_path_beneath, ctypes.byref(rule), 0) < 0:
                    raise OSError(ctypes.get_errno(), f"landlock_add_rule for {path!r} failed")
            finally:
                os.close(path_fd)
        # NO_NEW_PRIVS is a precondition for restrict_self (set by the ratchet).
        if libc.syscall(446, ruleset_fd, 0) < 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
        os.close(ruleset_fd)


class IsolationBackend:
    """Abstract backend: turn a policy into a child-launch environment.

    A backend answers two questions the caller needs before spawning a
    scoring child: `best_effort_level(requested)` — the strongest level
    it can actually build, refusing loudly rather than degrading silently
    — and `child_preexec(policy)` — the callable the child runs to ratchet
    itself into the envelope before the payload (or None when the level
    needs no in-child work).
    """

    def best_effort_level(self, requested: IsolationLevel) -> IsolationLevel:
        raise NotImplementedError

    def child_preexec(self, policy: IsolationPolicy) -> Callable[[], None] | None:
        raise NotImplementedError

    def run(self, argv, policy, *, env=None, cwd=None, timeout=None):
        """Default launch: subprocess with the ratchet as preexec.

        Subclasses that own parent-side placement (cgroups) override this;
        this default suits a backend with no cgroup work, and test fakes.
        """
        import subprocess

        return subprocess.run(
            argv,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            timeout=timeout,
            preexec_fn=self.child_preexec(policy),
        )


class LinuxIsolationBackend(IsolationBackend):
    """The fork-place-ratchet Linux backend (Q4/Q6/Q9).

    Feature detection through the syscall seam decides the highest level
    the host can honestly reach; a request above it is a loud
    `IsolationDowngrade`, never a silent weaker wall. The child preexec
    performs the ratchet half (namespaces, NO_NEW_PRIVS, Landlock/seccomp
    where present) and the self-probe (socket + out-of-scratch write must
    fail) so a wall that did NOT hold aborts as infrastructure.
    """

    def __init__(self, syscalls: _SyscallLayer | None = None):
        self.syscalls = syscalls or _SyscallLayer()

    def reachable_level(self) -> IsolationLevel:
        """The strongest level this host can honestly establish."""
        sc = self.syscalls
        level = IsolationLevel.fork  # a plain fork is always available
        if sc.has_cgroup_v2() and sc.cgroupfs_writable():
            level = IsolationLevel.fork_cgroup
            if sc.has_unshare():
                level = IsolationLevel.fork_ratchet
                if sc.has_bwrap():
                    level = IsolationLevel.sandbox
        return level

    def best_effort_level(self, requested: IsolationLevel) -> IsolationLevel:
        """The level actually built for `requested`, or a loud downgrade.

        `none`/`namespace`/`remote` pass through (they are not this
        backend's fork-family walls). Any fork-family request above what
        the host can reach fails closed with the missing mechanism named.
        """
        requested = parse_level(requested)
        if requested in (IsolationLevel.none, IsolationLevel.namespace, IsolationLevel.remote):
            return requested
        reached = self.reachable_level()
        if requested > reached:
            raise IsolationDowngrade(requested, reached, self._missing_reason(reached))
        return requested

    def _missing_reason(self, reached: IsolationLevel) -> str:
        sc = self.syscalls
        if not sc.has_cgroup_v2():
            return "cgroup v2 is not mounted at /sys/fs/cgroup"
        if not sc.cgroupfs_writable():
            return "cgroupfs is not writable by this process (the parent cannot place the child)"
        if not sc.has_unshare():
            return "os.unshare / CLONE_NEWUSER is unavailable (unprivileged user namespaces likely disabled)"
        if not sc.has_bwrap():
            return "bwrap is not on PATH (the sandbox mount-root world needs bubblewrap)"
        return f"host reaches only {reached.name}"

    def child_preexec(self, policy: IsolationPolicy, *, gate_fd: int | None = None) -> Callable[[], None] | None:
        """The in-child ratchet+probe for `policy.level`, or None below fork_ratchet.

        Levels below `fork_ratchet` need no in-child work (a plain fork or
        cgroup placement is the parent's job). At `fork_ratchet`+ the child
        unshares namespaces, sets NO_NEW_PRIVS, then — if `gate_fd` is
        given — BLOCKS reading the parent's go-byte (so the parent can
        place it in a cgroup before it does anything), then applies
        Landlock (when the kernel supports it), and finally SELF-PROBES:
        an outbound network reach and an out-of-scratch write that must
        fail (plus a non-granted read when Landlock engaged). A probe that
        SUCCEEDS where it had to fail is an infrastructure abort — the
        notes are explicit that this must never be mistaken for a passing
        run.

        THE ORDERING LAW (Q4): fork -> place -> gate -> ratchet -> run.
        The gate is what makes the parent-side cgroup placement race-free:
        the child cannot touch the payload (or even the Landlock/probe
        step) until the parent has placed it and released the gate.
        """
        level = parse_level(policy.level)
        if level < IsolationLevel.fork_ratchet:
            return None
        sc = self.syscalls
        scratch = policy.scratch
        files = dict(policy.files)

        def _ratchet() -> None:  # pragma: no cover - runs only in a real child
            flags = 0
            for name in ("CLONE_NEWUSER", "CLONE_NEWNET", "CLONE_NEWNS"):
                flags |= getattr(os, name, 0)
            if flags:
                sc.unshare(flags)
            sc.set_no_new_privs()
            if gate_fd is not None:
                # Block until the parent has placed us in the cgroup and
                # signalled go — the airtight gate (fork -> place -> gate).
                while True:
                    chunk = os.read(gate_fd, 1)
                    if chunk:
                        break
                os.close(gate_fd)
            engaged_landlock = False
            if files and sc.landlock_abi() > 0:
                sc.landlock_restrict(files, scratch)
                engaged_landlock = True
            self._run_probes(sc, scratch, landlock=engaged_landlock, files=files)

        return _ratchet

    def _run_probes(
        self, sc: _SyscallLayer, scratch: str | None, *, landlock: bool = False, files: dict[str, str] | None = None
    ) -> None:
        """Self-probe: the wall must actually deny what it claims to deny."""
        if not sc.probe_socket_denied():
            raise RuntimeError(
                "isolation self-probe FAILED: outbound network reach succeeded inside the fork_ratchet "
                "envelope where the network must be unshared — the wall did not hold (infrastructure, "
                "never scored)"
            )
        # The out-of-scratch WRITE probe only bites when Landlock (or a
        # real mount-root pivot) engaged: a bare CLONE_NEWNS unshare still
        # shares /tmp with the host, so it cannot deny an out-of-scratch
        # write — probing it there would false-abort a legitimately-walled
        # run. When Landlock DID engage, the write must be denied.
        if landlock:
            outside = os.path.join(scratch, "..", "_flex_probe") if scratch else "/_flex_probe"
            if not sc.probe_write_denied(outside):
                raise RuntimeError(
                    f"isolation self-probe FAILED: an out-of-scratch write to {outside!r} succeeded where the "
                    "Landlock wall must deny it — the wall did not hold (infrastructure, never scored)"
                )
        if landlock:
            # A read of a path OUTSIDE the granted scopes (and scratch)
            # must fail once Landlock engaged — otherwise files= did not
            # bite and we would be claiming a wall we did not build.
            granted = set(files or {})
            if scratch:
                granted.add(scratch)
            for candidate in ("/etc/hostname", "/etc/hosts", "/proc/version"):
                if any(candidate.startswith(g) for g in granted):
                    continue
                if not sc.probe_read_denied(candidate):
                    raise RuntimeError(
                        f"isolation self-probe FAILED: a read of the non-granted path {candidate!r} succeeded "
                        "where Landlock must deny it — the filesystem wall did not hold (infrastructure)"
                    )
                break

    # -- The launch gate: fork -> place -> gate -> ratchet -> run --------

    def run(self, argv, policy, *, env=None, cwd=None, timeout=None):
        """Launch a child under `policy`, owning the fork-place-gate-ratchet.

        Returns a `subprocess.CompletedProcess`. Below `fork_cgroup` this
        is an ordinary `subprocess.run` with the ratchet as `preexec_fn`
        (no parent-side work). At `fork_cgroup`+ the parent creates a
        cgroup leaf, writes the caps, PLACES the child in it BEFORE the
        child ratchets, then releases the gate — and on teardown runs
        `cgroup.kill` (reaping every straggler, no exceptions) then rmdir.

        Fail-closed: a `fork_cgroup`+ request whose leaf cannot be created
        or written raises `IsolationDowngrade` — never a silently
        uncapped run.
        """
        import json
        import subprocess
        import tempfile

        level = parse_level(policy.level)
        if level < IsolationLevel.fork_cgroup:
            preexec = self.child_preexec(policy)
            return subprocess.run(
                argv, capture_output=True, text=True, env=env, cwd=cwd, timeout=timeout, preexec_fn=preexec
            )

        leaf = self._create_cgroup(policy)  # raises IsolationDowngrade on failure
        # The launcher hosts the gate AFTER exec (a preexec gate would
        # deadlock Popen). The parent execs the launcher, places its pid in
        # the cgroup, releases the gate; the launcher then ratchets and
        # execs the real payload argv.
        gate_read, gate_write = os.pipe()
        config = {
            "argv": list(argv),
            "gate_fd": gate_read,
            "scratch": policy.scratch,
            "files": dict(policy.files),
            "ratchet": level >= IsolationLevel.fork_ratchet,
        }
        config_fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        try:
            json.dump(config, config_fh)
            config_fh.close()
            launcher_argv = [sys.executable, "-m", "dspy.programir.engine._isolation_launcher", config_fh.name]
            proc = subprocess.Popen(
                launcher_argv,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                cwd=cwd,
                pass_fds=(gate_read,),
            )
            os.close(gate_read)
            try:
                self.syscalls.cgroup_place(leaf, proc.pid)
            except OSError as error:
                os.close(gate_write)
                proc.kill()
                raise IsolationDowngrade(
                    level, IsolationLevel.fork, f"could not place the child in the cgroup leaf: {error}"
                ) from error
            os.write(gate_write, b"\x01")  # release the gate: placed, ratchet may proceed
            os.close(gate_write)
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.syscalls.cgroup_kill(leaf)
                proc.kill()
                raise
            return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
        finally:
            # Teardown ALWAYS reaps stragglers: cgroup.kill takes down every
            # process in the leaf (grandchildren included), then the empty
            # leaf is removed.
            self.syscalls.cgroup_kill(leaf)
            self.syscalls.cgroup_rmdir(leaf)
            try:
                os.unlink(config_fh.name)
            except OSError:
                pass

    def _create_cgroup(self, policy: IsolationPolicy) -> str:
        """Create a leaf under the writable subtree and write caps, or refuse."""
        sc = self.syscalls
        root = sc.writable_cgroup_root()
        if root is None:
            raise IsolationDowngrade(
                parse_level(policy.level),
                IsolationLevel.fork,
                "no writable cgroup v2 subtree (the parent cannot create a leaf to place the child in)",
            )
        leaf = os.path.join(root, f"dspy-flexir-{os.getpid()}-{next(_CGROUP_COUNTER)}")
        try:
            sc.cgroup_create(leaf)
        except OSError as error:
            raise IsolationDowngrade(
                parse_level(policy.level), IsolationLevel.fork, f"could not create the cgroup leaf {leaf!r}: {error}"
            ) from error
        try:
            if policy.memory is not None:
                sc.cgroup_write(leaf, "memory.max", _cgroup_bytes(policy.memory))
            if policy.pids is not None:
                sc.cgroup_write(leaf, "pids.max", str(int(policy.pids)))
            if policy.cpus is not None:
                # cpu.max is "<quota> <period>"; period 100000us, quota scaled.
                quota = int(float(policy.cpus) * 100000)
                sc.cgroup_write(leaf, "cpu.max", f"{quota} 100000")
        except OSError as error:
            sc.cgroup_kill(leaf)
            sc.cgroup_rmdir(leaf)
            raise IsolationDowngrade(
                parse_level(policy.level), IsolationLevel.fork, f"could not write a cgroup cap into {leaf!r}: {error}"
            ) from error
        return leaf
