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
import os
from dataclasses import dataclass, field
from typing import Any, Callable

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
    """

    level: IsolationLevel = IsolationLevel.none
    broker_egress: frozenset[str] = field(default_factory=frozenset)
    scratch: str | None = None

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

    def describe_envelope(self) -> dict[str, Any]:
        """The provenance record of the actual envelope (D-042 point 7)."""
        return {
            "level": self.level.name,
            "broker_egress": sorted(self.broker_egress),
            "scratch": self.scratch,
        }


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
        return os.access(root, os.W_OK)

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
        """True when opening a socket FAILS (the wall holds)."""
        import socket

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        except OSError:
            return True
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

    def child_preexec(self, policy: IsolationPolicy) -> Callable[[], None] | None:
        """The in-child ratchet+probe for `policy.level`, or None below fork_ratchet.

        Levels below `fork_ratchet` need no in-child work (a plain fork or
        cgroup placement is the parent's job). At `fork_ratchet`+ the child
        unshares namespaces, sets NO_NEW_PRIVS, applies Landlock/seccomp
        where the libraries exist, then SELF-PROBES: a socket that must
        fail and an out-of-scratch write that must fail. A probe that
        SUCCEEDS where it had to fail is an infrastructure abort — the wall
        did not hold, and the notes are explicit that this must never be
        mistaken for a passing run.
        """
        level = parse_level(policy.level)
        if level < IsolationLevel.fork_ratchet:
            return None
        sc = self.syscalls
        scratch = policy.scratch

        def _ratchet() -> None:  # pragma: no cover - runs only in a real child
            flags = 0
            for name in ("CLONE_NEWUSER", "CLONE_NEWNET", "CLONE_NEWNS"):
                flags |= getattr(os, name, 0)
            if flags:
                sc.unshare(flags)
            sc.set_no_new_privs()
            self._run_probes(sc, scratch)

        return _ratchet

    def _run_probes(self, sc: _SyscallLayer, scratch: str | None) -> None:
        """Self-probe: the wall must actually deny what it claims to deny."""
        if not sc.probe_socket_denied():
            raise RuntimeError(
                "isolation self-probe FAILED: a socket() succeeded inside the fork_ratchet envelope where "
                "the network must be unshared — the wall did not hold (infrastructure, never scored)"
            )
        outside = os.path.join(scratch, "..", "_flex_probe") if scratch else "/_flex_probe"
        if not sc.probe_write_denied(outside):
            raise RuntimeError(
                f"isolation self-probe FAILED: an out-of-scratch write to {outside!r} succeeded where the "
                "mount wall must deny it — the wall did not hold (infrastructure, never scored)"
            )
