"""In-child launcher for the gated fork-place-ratchet path (D-042, Q4).

The parent-side cgroup placement must happen AFTER the child exists (its
PID is known) but BEFORE the child ratchets or touches the payload — the
ordering law `fork -> place -> gate -> ratchet -> run`. A `preexec_fn`
cannot host the gate: if it blocked waiting for the parent, `Popen` would
never return (it waits for the child to finish exec), and the parent
could not write the go-byte. So the gate lives HERE, after exec.

The parent execs `python -m ..._isolation_launcher <config.json>` with the
gate-read fd inherited. This launcher, already a live exec'd process the
parent has placed in the cgroup:

  1. BLOCKS reading one go-byte from the gate fd (the parent releases it
     only after placing us in the cgroup);
  2. ratchets: unshare namespaces, NO_NEW_PRIVS, Landlock (when present);
  3. self-probes the wall (network / out-of-scratch write / non-granted
     read), aborting non-zero if a probe passes where it must fail;
  4. `execv`s the real payload argv — so the payload inherits every wall.

Namespaces are unshared HERE (post-exec, single-threaded) so
`unshare(CLONE_NEWUSER)` is legal.
"""

from __future__ import annotations

import json
import os
import sys


def _fail(message: str) -> None:
    print(f"ISOLATION-INFRA {message}", file=sys.stderr)
    sys.exit(3)


def main(config_path: str) -> None:
    config = json.loads(open(config_path, encoding="utf-8").read())
    gate_fd = config["gate_fd"]
    scratch = config.get("scratch")
    files = config.get("files") or {}
    do_ratchet = config["ratchet"]

    # 1. gate: wait for the parent's placement to complete.
    if gate_fd is not None:
        while not os.read(gate_fd, 1):
            pass
        os.close(gate_fd)

    if do_ratchet:
        from dspy.programir.engine.isolation import _SyscallLayer

        sc = _SyscallLayer()
        # 2. ratchet.
        flags = 0
        for name in ("CLONE_NEWUSER", "CLONE_NEWNET", "CLONE_NEWNS"):
            flags |= getattr(os, name, 0)
        try:
            if flags:
                sc.unshare(flags)
            sc.set_no_new_privs()
        except OSError as error:
            _fail(f"ratchet failed: {error}")
        engaged_landlock = False
        if files and sc.landlock_abi() > 0:
            try:
                sc.landlock_restrict(files, scratch)
                engaged_landlock = True
            except OSError as error:
                _fail(f"landlock failed: {error}")
        # 3. self-probe.
        from dspy.programir.engine.isolation import LinuxIsolationBackend

        try:
            LinuxIsolationBackend(sc)._run_probes(sc, scratch, landlock=engaged_landlock, files=files)
        except RuntimeError as error:
            _fail(str(error))

    # 4. exec the real payload — it inherits every wall.
    argv = config["argv"]
    try:
        os.execv(argv[0], argv)
    except OSError as error:
        _fail(f"exec of payload failed: {error}")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as error:  # pragma: no cover - infra
        _fail(f"{type(error).__name__}: {error}")
