# Linux isolation — conversation notes (2026-08-12)

These notes capture a design conversation about sandboxing untrusted code on a plain Linux host.
The arc starts with what a process can do to lock itself down (seccomp, Landlock, rlimits, the
fork-ratchet pattern) versus what needs an outside supervisor (cgroups, ephemeral users, namespaces).
It compares plain-OS isolation with heavier substrates: Wasm/WASI, gVisor, Firecracker, and Pyodide,
and where each sits on the assurance-versus-friction gradient. Multi-tenancy risks (shared kernel,
side channels, /tmp and abstract-socket leaks) shape the recommendations. Network control lands on a
broker design: deny-all in the sandbox, with a supervised proxy making the actual connections.
Tooling covers bubblewrap, nsjail, systemd-run, and Anthropic's sandbox-runtime (srt). The
conversation ends by applying the discussion to ProgramIR: the §e0-isolation spec section and
decision D-042, which define an isolation gradient for executor embedders.


## Q1: Here's the comparison across the five, then the footnote question — which has a…

> Here's the comparison across the five, then the footnote question — which has a genuinely interesting answer.
> 
> | | **Plain Linux** (namespaces + seccomp + Landlock) | **Wasm runtimes** (Wasmtime/Wasmer) | **gVisor** (runsc) | **Firecracker** | **Pyodide** (+ Deno/browser) |
> |---|---|---|---|---|---|
> | **Size** | 0 — it's the kernel | ~30–100 MB (or a pip/cargo library) | ~30–50 MB binary, plus a container runtime to be useful | Binary is a few MB, but needs a guest kernel + rootfs image you build/ship (tens–hundreds of MB) | ~100–200 MB Pyodide payload + ~100 MB Deno |
> | **Install complexity** | None — syscalls, callable via ctypes. The one gotcha is a *config* flag: unprivileged user namespaces disabled on some hardened distros and inside many Docker containers | Low — a normal package/library dependency, easily auto-installed | High — must be wired in as a container runtime; operator-grade setup | Very high — KVM access, kvm group/root, guest image plumbing | Moderate — auto-downloadable, but large, network-on-first-run, version-pinning across the pair |
> | **Portability (cross-platform)** | Linux only. (macOS/Windows have *equivalents* — Seatbelt, AppContainer — but they're different code, not the same mechanism) | Excellent — everywhere | Linux only | Linux only, and needs hardware virt — often unavailable in CI runners and nested-virt clouds | Excellent — anywhere JS runs |
> | **Portability (cross-user / weird setups)** | Mostly fine; fails closed where userns is disabled — detect and refuse | Basically always works — it's just a process | Needs privileged setup | Needs KVM — frequently absent | Basically always works |
> | **Python realness** | **Perfect.** Real CPython, any version you have, every library including native extensions (numpy, torch), threads, subprocess-if-allowed. It *is* normal Python — the OS constrains reach, not semantics | Poor-to-partial: you need a CPython-compiled-to-Wasm build; ecosystem support is young and patchy | **Near-perfect.** Real CPython on a userspace kernel; rare syscall-coverage gaps can break exotic libraries, but mainstream stacks work | **Perfect.** A real kernel in a microVM — full fidelity, at the cost of everything above | **Partial by design.** CPython-in-Wasm: no real threads, no subprocess, restricted sockets; only pure-Python wheels or the specially-built Pyodide package set; Python version lags upstream |
> 
> The pattern: Linux-native and Firecracker are the two with *perfect* Python fidelity — one costs nothing and one costs the most. Wasm/Pyodide buy portability by selling Python completeness, which for your use case (optimizer-authored code importing sklearn/torft/etc.) is exactly the wrong trade.
> 
> **The footnote — can I lock down my *existing* process/PID mid-session?**
> 
> This is where Linux is genuinely special, with one big caveat:
> 
> - **Yes, self-lockdown is a designed-for feature.** `seccomp` and **Landlock** exist precisely for this: a running process (even a specific *thread*) can call `prctl(PR_SET_NO_NEW_PRIVS)` + install a seccomp filter or a Landlock ruleset on *itself*, and from that instant it can no longer open sockets, write files outside a set, etc. It's a **one-way ratchet** — you can tighten, never loosen. This is how Chrome, OpenSSH, and systemd services self-confine. So "I have my Python session, now I want to run something with fewer permissions" is literally expressible: fork (or start a thread), ratchet down, run the code.
> 
> - **The caveat that kills the thread version: shared memory.** A "less-privileged thread" still lives in your address space. Seccomp will genuinely stop it from making syscalls (no network — the cheapness guarantee holds), but it can still read and scribble over every object in your interpreter, including your API keys sitting in memory. So per-thread lockdown is real for *effect* containment but worthless for *confidentiality/integrity*. The honest pattern is **fork-then-ratchet**: `fork()`, and the child self-applies seccomp/Landlock/unshare before `exec`-or-run. That's milliseconds, and it's exactly the shape the rung-3 scoring subprocess already has. (One wrinkle: `unshare(CLONE_NEWUSER)` refuses in a multithreaded process, so namespaces want to be set up at fork/spawn time, not mid-thread-soup.)
> 
> - **The others: no.** gVisor and Firecracker wrap a process *from outside, from birth* — you cannot retrofit them onto a running PID. Wasm/Pyodide invert the question: they can't contain your existing process, but they *can* host a sandboxed guest **inside** it (that's their party trick — an in-process memory-safe boundary). macOS Seatbelt historically allowed runtime self-sandboxing via `sandbox_init`, same ratchet idea, less documented; Windows wants the restricted token set up at process creation.
> 
> So the direct answer to your scenario: on Linux you don't need to "turn your PID into a sandbox" — you fork a child that ratchets itself down before touching the untrusted code, which gives you everything the mid-process lockdown promises without the shared-memory hole, and it's the same fork you already pay for artifact-mode scoring.
> 
> I am exploring um sandboxing in Python for running LNM generated code and um now here's a bit of the context of like some back and ports I had with a coding agent and uh now I wanna know more about this like self-locked down can I have a Python session that has sub threads or sub process and that these are locked down button at the parent session and also can I do that in all operating system if I'm using the plain version of it the plain way of doing it um so I understand that I will need three different code implementation for all three of them but do all three of them give me that feature and are all three like fully real python and do I have control over granularity like I'm thinking of things like if I want to run byTorch with CUDA and like and and this the GPUs or VN would they still run if I sandbox?

Good questions. Here are the answers, one by one.

## 1. Do all three operating systems support "fork, then lock down the child"?

Yes, all three support the pattern. But the mechanisms and the quality differ.

| | Linux | macOS | Windows |
|---|---|---|---|
| Mechanism | seccomp + Landlock + namespaces | Seatbelt (`sandbox_init`) | Restricted tokens + Job Objects + AppContainer |
| Child self-lockdown after fork/spawn | Yes, first-class | Yes, but the API is deprecated-ish and poorly documented | No. The parent must create the child with a restricted token. The child cannot ratchet itself down well |
| Thread-level lockdown | Yes (seccomp per thread), but the shared-memory hole applies | No | No |
| Real Python in the child | Yes, full CPython | Yes, full CPython | Yes, full CPython |

Key differences:

- **Linux**: the child locks itself down. Fork, ratchet, run. One-way, reliable, designed for this.
- **macOS**: Seatbelt allows a process to sandbox itself at runtime with a profile string. Chrome uses it. Apple marks the public API as deprecated, but it still works and Apple's own code uses it. `fork()` + `sandbox_init()` in the child gives you the same shape as Linux.
- **Windows**: the model is inverted. The *parent* builds a restricted token and creates the child process with it (`CreateProcessAsUser` or the AppContainer path). The child does not lock itself down after start. So you do not fork-then-ratchet. You spawn-restricted. Python's `multiprocessing` uses spawn on Windows anyway, so this fits, but you must write Win32 plumbing (via `pywin32` or ctypes).

So: same *shape* everywhere (parent stays free, child is confined), three different implementations, and Windows forces the restriction to happen at process creation, not after.

## 2. Are all three "fully real Python"?

Yes. This is the strong point of the plain-OS approach on every platform. The child runs your normal CPython binary, your normal site-packages, native extensions included. The OS limits what syscalls or resources the process can reach. It does not change Python semantics. numpy, torch, threads inside the child — all normal, up to the limits you set.

## 3. Threads vs subprocesses

Same answer on all platforms: use subprocesses, not threads.

- A locked-down thread still shares your memory. It can read your API keys. Only Linux even lets you try per-thread seccomp, and the shared-memory hole makes it worthless for secrets.
- A locked-down subprocess has its own address space. The parent keeps its secrets. This is the honest boundary on all three OSes.

## 4. Granularity — can you allow CUDA/GPU and block the rest?

Yes, and this is where Linux shines.

- **Linux**: GPU access is file access. CUDA talks to `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm` (and `/dev/dri/*` for others). With Landlock or bind-mount namespaces you write a rule: allow read/write on those device nodes, allow read on the CUDA libraries and the model weights, deny network, deny everything else. PyTorch + CUDA then runs at full speed inside the sandbox. There is no performance cost — the filter runs at syscall entry, and GPU kernels do not go through syscalls per operation.
- One trap: the NVIDIA driver also wants `/proc/devices`, `/sys/bus/pci/...`, and it creates `/dev/nvidia-uvm` on first use. You find these with `strace` once, then add them to the allowlist. It is a finite, stable list.
- **macOS**: Seatbelt profiles can allow IOKit access (which is how Metal/GPU works) while denying network and files. Granular, but you write it in the Seatbelt profile language.
- **Windows**: coarser. AppContainer capabilities are broad categories, not per-device-node rules. GPU access generally works inside an AppContainer (UWP apps use the GPU), but "allow GPU, deny this specific directory" is harder to express than on Linux.

Same story for a virtual env: it is just files. Allow read on the venv path, and imports work normally.

## Summary

- The pattern "parent free, child confined" works on all three OSes with plain OS mechanisms.
- You need three implementations, as you expected. Linux and macOS: fork/spawn then self-ratchet. Windows: parent spawns with a restricted token.
- All three run real, full CPython in the child.
- Granularity is best on Linux (per-file, per-device, per-syscall), good on macOS, coarse on Windows.
- CUDA works inside the sandbox on Linux. Allow the NVIDIA device nodes and driver paths. Full speed, no semantic change.

If Linux is your main target (it usually is, for GPU training), build the Landlock+seccomp fork-child first. Add a Seatbelt profile for macOS dev machines. Treat Windows as "spawn restricted or refuse".


## Q2: Okay, and what are the risks, uh particularly in multi-tenancy situations?

> Okay, and what are the risks, uh particularly in multi-tenancy situations?

Good question. Multi-tenancy changes the threat model a lot. Here are the risks, ranked.

## 1. Kernel attack surface — the big one

- A seccomp/Landlock sandbox still talks to your real, shared kernel. Every allowed syscall is attack surface.
- A kernel exploit in one tenant's sandbox owns the host. Then it owns every tenant.
- This is the core reason gVisor and Firecracker exist. gVisor puts a userspace kernel between the code and the real kernel. Firecracker gives each tenant its own kernel. Plain namespaces do not.
- History supports the worry: `io_uring`, `waitid`, `keyctl`, and many namespace-related CVEs gave container escapes. A tight seccomp allowlist blocks many of these. It cannot block bugs in the syscalls you must allow (`read`, `write`, `mmap`, `futex`).

Rule of thumb from the big providers: plain-OS sandboxing is fine for *your own* semi-trusted code. For *hostile* code from many customers, they all add a second boundary (microVM or userspace kernel).

## 2. GPU sharing — worse than the kernel

You asked about CUDA earlier. In multi-tenancy the GPU is the weakest point.

- The NVIDIA driver is a huge, closed, privileged kernel module. Giving a tenant `/dev/nvidia*` gives them that whole attack surface. It has a steady CVE stream.
- GPU memory is not zeroed between processes by default on all paths. Tenant B can sometimes read tenant A's leftover VRAM (model weights, activations, data).
- The GPU has no per-tenant isolation model in plain mode. No namespaces, no seccomp equivalent. Real isolation needs MIG (A100/H100 class), vGPU/SR-IOV, or one-GPU-per-tenant.
- Practical rule: never share one GPU between mutually distrusting tenants without MIG. If you must share, wipe and reset the device between tenants (`nvidia-smi --gpu-reset` or a scrub kernel), and accept residual risk.

## 3. Side channels

- Tenants share CPU caches, branch predictors, and memory buses. Spectre-class attacks let one tenant read another's secrets slowly.
- Namespaces do nothing here. Firecracker mitigates partly (separate address spaces + can pin to separate cores). Full defense needs core pinning or physical separation.
- Also mundane channels: `/proc` leaks host-wide data (load, interrupts, timing) unless you mask it. Landlock does not cover `/proc` semantics by itself.

## 4. Resource exhaustion (denial of service)

- Seccomp and Landlock control *what* code touches, not *how much*. One tenant can eat all CPU, RAM, disk, PIDs, or file descriptors and starve the others.
- You must add cgroups v2 (cpu, memory, pids, io controllers) per tenant. This is a separate mechanism with its own setup cost.
- Classic traps: fork bombs (pids controller), memory balloons that trigger the host OOM killer (memory controller + `memory.max`), disk fill in shared tmp (per-tenant tmpfs with size limit), and GPU VRAM exhaustion (nothing stops it without MIG).

## 5. Configuration fragility

- The plain-OS approach is a stack of five mechanisms you compose by hand: userns, mountns, netns, seccomp, Landlock, cgroups. Each has defaults that fail open, not closed.
- Common real-world mistakes: forgot to close inherited file descriptors before the ratchet; left the abstract Unix socket namespace shared (netns covers it, but only if you actually unshare netns); allowed `ptrace` in the seccomp filter so tenants attach to each other; mounted `/proc` without hiding other tenants' PIDs.
- One mistake anywhere is a cross-tenant hole. Firecracker's value is that the boundary is one mechanism, small and audited, instead of five you assembled.

## 6. Shared-fate operational risks

- All tenants share one kernel's uptime. One tenant triggers a kernel panic or soft lockup, everyone dies.
- Kernel updates for a CVE force host reboots for all tenants at once.

## What this means in practice

Match the boundary to the trust level:

| Situation | Adequate boundary |
|---|---|
| Your own optimizer-generated code, your infra, one tenant | Plain namespaces + seccomp + Landlock + cgroups. Fine. |
| Multiple internal teams, low mutual hostility | Same, plus strict cgroups and masked `/proc`. Acceptable. |
| Arbitrary customer code, real multi-tenancy | Firecracker or gVisor *on top of* the plain-OS layer. Non-negotiable in industry practice. |
| Multi-tenant **GPU** work | MIG or one GPU per tenant. Software isolation alone is not credible today. |

The layering point matters: the plain-OS mechanisms are not wasted in the multi-tenant case. Firecracker-based products still run seccomp *inside* the VM. Defense in depth, cheap layer first.

For your actual use case — running LM-generated code from your own optimization loop on your own machine — you are in row 1. The multi-tenant risks above are the reason not to *sell* that sandbox as a service, not a reason to avoid it locally.


## Q3: The C groups you're talking about are these like uh the Linux Linux groups, Linu…

> The C groups you're talking about are these like uh the Linux Linux groups, Linux users?

No — different thing, easy to confuse.

- **Users and groups** (`/etc/passwd`, `/etc/group`, the thing `chmod`/`chown` use) answer: **who can access what**. Identity and permissions.
- **cgroups** ("control groups") answer: **how much can this set of processes consume**. CPU time, RAM, number of PIDs, disk I/O bandwidth.

They share the word "group" but are unrelated mechanisms from different eras. cgroups arrived around 2008 (Google contributed them for exactly the containment problem we discussed) and are the engine under Docker's `--memory=2g` and `--cpus=1.5` flags, and under systemd resource limits.

Concretely, a cgroup is a directory in a virtual filesystem:

```bash
# Create a cgroup, cap it at 2 GB RAM and 100 processes
mkdir /sys/fs/cgroup/tenant-a
echo "2G"  > /sys/fs/cgroup/tenant-a/memory.max
echo "100" > /sys/fs/cgroup/tenant-a/pids.max

# Put a process in it — from now on the kernel enforces the caps
echo <pid> > /sys/fs/cgroup/tenant-a/cgroup.procs
```

Every process the tenant forks stays in the cgroup and shares those caps. A fork bomb hits `pids.max` and stops. A memory balloon hits `memory.max` and the OOM killer kills *that cgroup*, not your host.

The clean mental model for the full sandbox stack:

| Mechanism | Question it answers |
|---|---|
| Users/groups, file permissions | Who may access this file? |
| Namespaces | What can this process *see*? |
| seccomp / Landlock | What can this process *do*? |
| **cgroups** | How much can this process *consume*? |

You still use plain users too, by the way — running each tenant as a different unprivileged UID is a cheap, old, effective extra layer. But it limits *access*, not *consumption*. That's the gap cgroups fill.


## Q4: Is it possible to change the C group specs while a PID is in that C group? Is it…

> Is it possible to change the C group specs while a PID is in that C group? Is it possible to remove or change the PID from a C group? And uh if if if I'm starting a PID a Python process and I'm loading some libraries in RAM and then I'm locking it is it i is it gonna work like because say I have control over the session and then at some point I delegate control because I have plugins that I'm running tools for others.

Yes to all three, and the third one has a subtle trap. In order:

## 1. Can you change a cgroup's limits while PIDs live in it?

Yes, at any time. The limit files are live knobs:

```bash
echo "1G" > /sys/fs/cgroup/tenant-a/memory.max   # takes effect immediately
```

- Tightening below current usage works too: the kernel tries to reclaim (swap/drop caches), and OOM-kills within the cgroup if it can't get under the new cap.
- This is a real difference from seccomp/Landlock: **cgroups are not a ratchet**. You can loosen them again. Which leads to the security consequence in question 2.

## 2. Can you move a PID out of a cgroup, or between cgroups?

Yes — anyone with **write access to the cgroupfs files** can:

```bash
echo <pid> > /sys/fs/cgroup/other-group/cgroup.procs   # moves it
```

So the containment is not a property of the process; it's a property of who can write those files. That means for your delegation scenario:

- The **parent/broker** keeps write access and can re-limit, move, or release the child at will. Full dynamic control, exactly what you want for "I run tools on behalf of plugins."
- The **sandboxed child must not have write access** to `/sys/fs/cgroup`, or it can simply move itself out. Two guards, use both:
  1. Ordinary file permissions — the cgroup directories stay owned by the parent's UID/root; the child runs as another UID or gets Landlock-denied on `/sys/fs/cgroup`.
  2. `cgroup_ns` — unshare the cgroup namespace at lockdown so the child can't even *see* the sibling groups, only its own subtree.

One kernel-imposed rule to know: with cgroups v2 you put PIDs only in **leaf** groups (no processes in a group that has child groups). Plan the tree as `session/ → session/plugin-a/, session/plugin-b/`, processes only in the leaves.

## 3. Load libraries first, lock afterwards — does the memory accounting work?

This is the trap. **Mostly yes, with one accounting quirk you should know.**

How cgroup v2 charges memory: a page is charged to the cgroup of the process that **first faults it in** — and the charge *stays with that cgroup* even if the process later moves. Moving a PID does not re-charge its existing pages.

Consequences for your exact pattern (start Python, `import torch`, then delegate):

- **Pages you touched before entering the cgroup are charged to the old (parent/root) cgroup.** So your preloaded 2 GB of libraries does *not* count against the child cgroup's `memory.max`. Usually this is what you want: the plugin's budget covers *its* allocations, not your preloaded runtime.
- Everything the process allocates **after** the move is charged to the new cgroup. So the cap correctly bounds the untrusted phase.
- The read-only library pages (code segments of `libtorch.so` etc.) are shared page cache anyway — fork'd children reuse them for free under copy-on-write.

The clean recipe for your delegation flow, combining everything:

```
1. Parent Python session: load the heavy libraries once. Unconfined.
2. Per tool-call: fork() a child.
     - CoW: the child sees the loaded libraries at zero copy cost.
3. Parent writes the child's PID into session/plugin-a/cgroup.procs
     (parent does this, not the child — parent keeps the keys).
4. Child self-ratchets: unshare(netns, cgroupns, ...) → Landlock → seccomp.
5. Child runs the plugin/LM code. Its new allocations hit memory.max;
     fork bombs hit pids.max; parent can tighten limits live mid-run.
6. Parent kills the cgroup wholesale when done:
     echo 1 > session/plugin-a/cgroup.kill   (kills every process in it, no stragglers)
```

Note the ordering in steps 3–4: cgroup placement must happen **before** the child drops privileges — after the ratchet, the child couldn't be trusted to place itself, and the parent placing it needs the child to still be in a known state. Fork → place → ratchet → run.

The one residual risk in the pattern: the child shares the *loaded* library pages read-only, but if the untrusted code writes to any inherited writable memory (module globals, cached tensors), CoW duplicates those pages and charges them to the child's cgroup — correct accounting, but it means a hostile plugin can't corrupt the *parent's* copy. The fork boundary protects you exactly where the thread boundary (from our earlier discussion) failed.

So: dynamic control, yes; revocation and re-delegation, yes — as long as the parent alone holds write access to the cgroup tree.


## Q5: Okay, and what does uh users do like you know i the the worry now is uh to have…

> Okay, and what does uh users do like you know i the the worry now is uh to have uh an AI write code in the computer and test code and like automatically create code and the worry is that this AI will start could write like stupid code, we could write code that uh could break the systems or has some risks. Um if we spin up a separate user, like a new separate Linux user for that optimizer, would that help too?

Yes, it helps, and it's one of the oldest and cheapest layers you can add. But know exactly what it buys and what it doesn't.

## What a separate user gives you

The classic Unix permission model answers "who may touch what," and it's enforced everywhere, always, with zero setup fragility:

- **Your files are safe from writes.** The optimizer user can't modify your home directory, your dotfiles, your SSH keys, your git repos — anything not world-writable. An `rm -rf ~` bug in generated code destroys *its* home, not yours.
- **Your processes are safe from interference.** It can't `kill` your processes, can't `ptrace`/debug them, can't read `/proc/<your-pid>/` memory maps.
- **Your secrets are safe if permissions are right.** `~/.ssh`, `~/.aws`, keychains are mode 700/600 — unreadable to another UID.
- **Blast radius is legible.** `ls -l` tells you what the optimizer could touch. Auditing is trivial compared to reasoning about a seccomp filter.
- **It's durable.** No namespaces to configure, nothing that fails open after a kernel update. UID checks have been enforced correctly for 50 years.

And it composes perfectly with everything we discussed: run the optimizer's processes as `optuser` *inside* the cgroup, *behind* the seccomp filter. Layers, not alternatives.

## What it does not give you

- **Read access to world-readable files.** Most of the filesystem is `o+r`. The optimizer user can read `/etc`, installed code, and anything in your home you left world-readable (common: `~/Projects` is often 755). Check with `find ~ -perm -o+r` if reads matter, or `chmod 750 ~`.
- **Network — nothing at all.** Any user can open sockets, exfiltrate data, call APIs, download malware. This is the biggest gap for AI-generated code. Only netns (or firewall rules keyed on UID — `iptables -m owner --uid-owner optuser -j DROP`, which is a nice cheap trick) closes it.
- **Resource limits — weak.** One UID can still fork-bomb or eat all RAM and take the host down. `ulimit`/`/etc/security/limits.conf` exist but are coarse; cgroups do this properly.
- **Shared writable places.** `/tmp` and `/var/tmp` are world-writable. Sloppy-but-not-malicious code can fill the disk or collide with your files there (sticky bit stops overwrites, not filling).
- **Sudo/setuid mistakes.** If `optuser` lands in the `sudo` group by habit, or a setuid binary is exploitable, the wall is gone. Create it with no groups: `useradd --no-user-group -M -s /usr/sbin/nologin` style, then add only what's needed.
- **Same kernel.** A kernel exploit still ends the game, as in the multi-tenant discussion. For "AI writes dumb code," this is a negligible worry; dumb code doesn't write kernel exploits.

## The realistic threat model for your case

For an optimizer writing sloppy-but-not-malicious code, the failure modes are: deleting/overwriting the wrong files, runaway loops eating CPU/RAM, filling the disk, and accidentally hammering some network API. Map those to layers:

| Failure mode | Separate user | + cgroups | + netns/Landlock |
|---|---|---|---|
| Overwrites your files | ✅ stopped | — | — |
| Reads your secrets | ✅ (if perms are tight) | — | — |
| Runaway CPU/RAM/fork bomb | ❌ | ✅ | — |
| Fills the disk | ❌ (mostly) | ✅ (io/quota) | — |
| Hammers the network / exfiltrates | ❌ | ❌ | ✅ |

So the pragmatic stack for "AI develops code on my machine," in order of effort:

1. **Separate user** — one command, big win, do it regardless.
2. **cgroup around its session** — a few lines, kills the runaway-resource class.
3. **UID-based firewall rule or netns** — one rule for "no network except the LM API endpoint."
4. Landlock/seccomp ratchet — when you want the per-tool-call granularity from our earlier discussion.

Step 1 alone already converts "could break my system" into "could break its own sandbox account," which is most of the fear. Steps 2–3 close the loudest remaining holes for under an hour of work.


## Q6: Is there some cost to create user per PID temporarily so that they kind of have…

> Is there some cost to create user per PID temporarily so that they kind of have access to only their own their file system world?

Yes, that's a real pattern, and the cost depends heavily on *how* you mint the identity. There are three tiers, and the interesting answer is that you don't actually need `useradd` at all.

## Tier 1: `useradd` per process — works, but clunky

- **Time cost:** ~10–50 ms per `useradd` (it locks and rewrites `/etc/passwd`, `/etc/shadow`, `/etc/group`, makes a home dir). Fine per session, sluggish per tool-call.
- **Privilege cost:** needs root for both create and delete.
- **The real cost is state and cleanup:** every crash that skips `userdel` leaves a zombie account. Concurrent `useradd`s serialize on the passwd lock. Any system daemon watching accounts (NSS caches, sssd) sees churn. It's system-wide mutable state for what you wanted to be an ephemeral, per-PID thing.

## Tier 2: ephemeral UIDs without accounts — the good trick

Here's the key insight: **a UID doesn't need an entry in `/etc/passwd` to work.** The kernel checks numbers, not names. A process can just *become* UID 61234:

```c
setgroups(0, NULL);              /* drop supplementary groups first! */
setresgid(61234, 61234, 61234);
setresuid(61234, 61234, 61234);  /* one-way: needs root/CAP_SETUID before, unprivileged after */
```

- **Time cost: microseconds.** No files touched, no locks, nothing to clean up. The "user" ceases to exist when the last process with that UID exits.
- Pick UIDs from a reserved range (say 60000–65000), one per concurrent sandbox, reuse freely.
- This is exactly what **systemd's `DynamicUser=yes`** does in production — allocates a UID from 61184–65519 at service start, releases it at stop. Prior art, battle-tested.
- Two gotchas: (a) `ls` shows raw numbers instead of names — cosmetic; (b) **UID reuse**: if sandbox A left files on disk and sandbox B later gets the same UID, B owns A's leftovers. So wipe the sandbox's writable directory on release, or never reuse a UID within a session.

## Tier 3: user namespaces — no root at all

`unshare(CLONE_NEWUSER)` gives the process its own UID mapping — it can be "root" inside while being your unprivileged UID (or an ephemeral one) outside. Zero cost, no privileges needed, composes with the mount namespace below. This is how rootless Podman/Docker mint identities.

## "Their own filesystem world" — the UID is only half

A fresh UID stops *writes*, but the process still **sees and reads** the whole world-readable filesystem (the gap from last time). If you want each sandbox to genuinely live in its own world, add a mount namespace on top — and this pairs beautifully with the ephemeral UID:

```
fork →
  setresuid(ephemeral)                    # identity: microseconds
  unshare(CLONE_NEWNS | CLONE_NEWUSER)    # own mount table
  mount tmpfs on a private root, bind-mount in ONLY:
     /usr, /lib (read-only), the venv, the task directory
  pivot_root                              # now "/" IS their world
```

Now the sandbox's `/` contains exactly what you granted. There is no "rest of the filesystem" to read. Cost of the whole ceremony: ~1–5 ms, dominated by the mounts. That's cheap enough for per-tool-call use, and it's the same fork-place-ratchet slot from our cgroup recipe — identity and mount setup happen at step 4, before the seccomp ratchet.

## Summary

| Approach | Cost per sandbox | Needs root | Cleanup debt |
|---|---|---|---|
| `useradd`/`userdel` | 10–50 ms | yes | high (system state) |
| **Ephemeral UID (`setresuid`)** | ~µs | CAP_SETUID at fork | zero (wipe dir on reuse) |
| User namespace | ~µs | no | zero |
| + mount ns private root | +1–5 ms | no (with userns) | zero (tmpfs evaporates) |

So: yes it's possible, and done right it's nearly free. Skip real `useradd` users; use ephemeral UIDs (or a userns) plus a private mount root, and each PID gets a disposable identity in a disposable world that both evaporate on exit.


## Q7: Okay, this is all very very good. Um the subject of network um what type of thin…

> Okay, this is all very very good. Um the subject of network um what type of things we should do and what types of controls do we get if we want to give them access to some endpoints. It may be we want to have them be able to have some um you know inter-process communication we want the sub thing to call to create other processes that have the same permission of themselves but then they have more CPUs to to play with and they can kind of network um we may want also to have them to like I said like call internet on some endpoints or some IPs we may also want um to be able have them to be able to declare that they want to have a service to be open like maybe they don't have a lot of permissions but they can have uh a hook or like they may say like I have this uh fast API on this because I want this other agent to send me data or whatever or um also I'm thinking about uh inter-process communication with RPC error and uh things like arrow and stuff but that's that can be that can be after

Excellent — networking is where the design gets interesting, because everything you listed maps onto one clean architectural idea. Let me build it up.

## The core insight: a network namespace is an empty network *world*, and you add capabilities by plugging things in

`unshare(CLONE_NEWNET)` gives the sandbox a network stack with only a dead loopback. From there, everything is opt-in. That inverts the problem nicely: instead of "block the bad," you construct exactly the network reality you want them to inhabit.

## 1. Sibling processes + "more CPUs to play with"

This one is nearly free. A namespace is joinable:

- The sandbox forks children → they inherit the netns, mountns, UID, cgroup automatically. Same permissions, more processes. Your cgroup's `cpuset`/`cpu.max` bounds the *total*, so "spawn 16 workers" is safe by construction — they share the tenant's budget, not the host's.
- All processes in one netns share a private `127.0.0.1`. They can bind ports, connect to each other, run a full multiprocess app with sockets — and none of it is visible outside the namespace. Port 8000 in sandbox A and port 8000 in sandbox B don't collide.
- So multiprocessing, `torch.distributed` on localhost, Arrow Flight between workers — all work unmodified inside the bubble.

## 2. Selected outbound endpoints — the honest answer is a proxy

You *can* do pure packet filtering: give the sandbox a veth pair into the host, then nftables rules allowing only certain destination IPs. But IP allowlists rot: `api.anthropic.com` is anycast, IPs rotate, CDNs share IPs with the whole internet. Everyone who builds this seriously converges on the same design:

**No route to the internet at all. Instead, a broker proxy owned by the parent.**

```
sandbox netns:  [loopback only] + one Unix socket (or veth to parent only)
                       │
parent process: CONNECT proxy / forward-proxy listening on that socket
                • allowlist by *hostname*: api.anthropic.com ✅, *.pypi.org ✅, else ❌
                • logs every request
                • can inject credentials so the API key NEVER enters the sandbox
```

That last bullet is the killer feature: the sandbox says "POST to the LM API," the broker attaches the `Authorization` header on the way out. Generated code can't leak a key it never had. Hostname-level control, per-request audit log, revocable live, and standard tooling honors it (`HTTPS_PROXY=...` — requests, httpx, pip, curl all obey).

Granularity you get: per-hostname, per-port, per-HTTP-method, per-path if you want (it's your proxy), rate limits, byte quotas. Far richer than any packet filter.

## 3. "I want to expose a service" — inbound as a declared capability

Same broker idea, reversed. The sandbox can't accept outside connections (its netns has no inbound route), but it can *ask*:

```
sandbox → broker:  "I'm serving on my 127.0.0.1:8000, please expose me as 'scorer-a'"
parent:  policy check → starts a forwarder: host socket → sandbox's port
         (socat into the netns, or pass a pre-bound listening FD through
          the Unix socket — SCM_RIGHTS — so the parent controls WHERE it binds)
```

- The FD-passing variant is elegant: the parent binds `host:9100` (or a Unix socket in a shared directory), passes the listening FD in; the sandbox `accept()`s on a socket it could never have created. The capability *is* the file descriptor — unforgeable, revocable by closing it.
- Other agents then reach `scorer-a` through the parent's registry, never by raw IP. You get naming, discovery, and a choke point for "who may talk to whom" for free.

## 4. Sandbox-to-sandbox IPC (RPC, Arrow, later)

Two good shapes, pick per trust level:

- **Unix sockets in a shared bind-mounted directory.** Mount `/run/mesh/` into both sandboxes' private roots. Filesystem permissions (their ephemeral UIDs + a shared group) say who may connect. Landlock can grant/deny per-sandbox. Arrow Flight, gRPC, anything socket-shaped runs over Unix sockets unmodified — and Arrow over a Unix socket on the same host is extremely fast.
- **A shared netns per *group*.** If sandboxes A and B should talk freely, put them in the *same* netns (two cgroups, two UIDs, one network world). They use localhost TCP; outsiders don't exist. Zero-config for the code inside.

For Arrow specifically, later: the shared-directory trick also carries memory-mapped Arrow files and shared memory segments, so you can do zero-copy handoff between sandboxes under the same UID/group discipline. It slots in without changing the architecture.

## The composed picture

```
parent (broker): holds keys, proxy, registry, cgroup tree, FD-granting
 ├─ sandbox A netns ── workers a1,a2,a3 (localhost mesh, shared cpu budget)
 │      │ unix socket → broker: egress proxy, service declarations
 ├─ sandbox B netns ── (same)
 └─ /run/mesh/: unix sockets for A↔B where policy allows
```

Everything you asked for is one mechanism family: **the sandbox starts with nothing, and every network capability is an FD or a route the parent explicitly handed over.** That's capability-style security, and it's why the design stays auditable as it grows: the answer to "what can sandbox A reach?" is always "read the broker's grant table," never "simulate iptables."

Build order that matches value: netns + proxy for egress allowlisting first (that's the exfiltration hole from earlier, now closed properly), sibling spawning is free immediately, FD-passed inbound services second, the Arrow mesh when you need it.


## Q8: Okay, brilliant. I feel uh I feel close to ready to to get it all going. Um mayb…

> Okay, brilliant. I feel uh I feel close to ready to to get it all going. Um maybe one more question before we get going. Any any good tooling that is uh doing all that at this level with this amount of you know kind of Linux simplicity and um and that makes it so that you know normal Python can run

Yes — you're in luck, because this exact niche ("plain kernel primitives, no daemon, real binaries run unmodified") has several mature tools. None does *everything* from our design, but they cover the layers well, and one recent entry is almost precisely your architecture.

## The exec-wrapper class — "run this command in a namespace bottle"

**bubblewrap (`bwrap`)** — the closest to our design's spirit. It's the sandbox under Flatpak, ~a few thousand lines of setuid-free C, and it is literally "compose namespaces from the command line":

```bash
bwrap --unshare-all \
      --ro-bind /usr /usr --ro-bind ~/venv ~/venv \
      --bind ~/task ~/task --tmpfs /tmp \
      --uid 61234 --die-with-parent \
      python solution.py
```

That's the private mount root, ephemeral UID, netns, pidns — our tiers 2–3 — in one flag list. Real CPython, native extensions, no image format, no daemon. What it *doesn't* do: seccomp policy authoring (it accepts a precompiled filter FD), cgroups, or any network brokering. It builds the world; you still add limits and the proxy.

**nsjail** (Google) — bubblewrap's stricter sibling: adds first-class seccomp policy language, resource limits, and time limits. It's what CTF infrastructure and Google's code-execution services use. Slightly more config-file-shaped; excellent when you want the seccomp layer handled for you.

**minijail** (ChromeOS) — same family, library + CLI, very fine-grained (it's how ChromeOS confines every system service). More of a building block than a turnkey tool.

## The underrated one: systemd-run

You already have it installed. One line gets you ephemeral UID + cgroup + basic network policy:

```bash
systemd-run --uid=... -p DynamicUser=yes -p MemoryMax=2G -p TasksMax=100 \
  -p IPAddressAllow=10.0.0.5 -p IPAddressDeny=any \
  -p ProtectHome=yes -p PrivateTmp=yes -p NoNewPrivileges=yes \
  python solution.py
```

`DynamicUser` is the ephemeral-UID trick from earlier, `MemoryMax`/`TasksMax` are the cgroup knobs, and cleanup is automatic. Weakness: IP-level (not hostname-level) network control, so it doesn't replace the broker proxy. But as the "step 1–2 of the stack in one command" tool, it's unbeatable for effort.

## The one that matches your *whole* architecture

**Anthropic's `sandbox-runtime` (`srt`)** — open-sourced from Claude Code's sandboxing in late 2025. It is almost exactly the design we converged on: bubblewrap-style namespace isolation on Linux (Seatbelt on macOS — your cross-platform question, answered in one codebase) **plus a broker proxy for egress with domain-level allowlists**, so "this process may reach `api.anthropic.com` and nothing else" is a config entry, not an nftables project. Its threat model is literally "an agent writes and runs code on your machine." Worth reading even if you don't adopt it, because it's a reference implementation of the netns + proxy pattern.

## What none of them give you

The **in-process, per-tool-call fork-then-ratchet** — fork from your warm Python session with torch loaded, then Landlock/seccomp down *that child*. The wrappers all exec a fresh process, which costs you the CoW-shared preloaded libraries. For the warm-fork path you write ~200 lines of Python against:

- `landlock` (PyPI) — clean ctypes bindings, pure Python
- `pyseccomp` — seccomp filter authoring from Python
- raw `os.unshare()` — in the stdlib since Python 3.12(!), so namespaces need no C at all

## How I'd assemble it

| Layer | Tool |
|---|---|
| Cold isolated runs (scoring, artifact mode) | **bubblewrap** or `systemd-run`, one-liners |
| Warm fork-ratchet from the loaded session | ~200 lines: `os.unshare` + `landlock` + `pyseccomp` |
| Egress broker with hostname allowlist | crib the design from **srt** (or adopt it outright) |
| Resource caps + cleanup | cgroup v2 filesystem directly (it's 10 lines), or systemd-run's `-p` flags |

Notably absent from my list: Docker/Podman (daemon + image ceremony you don't need), Firejail (checkered CVE history, setuid-root design), and everything from our heavyweight column (gVisor/Firecracker) per the earlier trust-level table.

The encouraging summary: the primitives are so close to the surface on Linux that the "tooling" is thin by nature — bubblewrap is a flag list, the cgroup API is `echo` into files, and the stdlib now ships `os.unshare`. Your architecture doesn't need a framework; it needs about a weekend.


## Q9: @roadmap/IR-program-spec.md consider that this is for leaves in in forward (betw…

> @roadmap/IR-program-spec.md consider that this is for leaves in in forward (between the ops) it would be nice to have those leaves being subprocess with which we share the ram from inputs outputs data but within the body of the tool / submodule we can isolate in all the ways we discuss above with a very nice gradient from, absolutely no isalation (not even subprocess) to complete isoluation. also, would be nice if the some pid could travel in and out of the sandbox as the leaves boundaries.

The spec already has the hook for this — let me check the §e0 placement section before I answer.

This fits the spec well. The placement ladder already reserves the slot: rung 1 is "same-sandbox subprocess," and the `placement.isolation` field exists. Your proposal upgrades that field from a 3-value enum to a real gradient, and adds a shared-RAM data plane. Here is my assessment, plus the two design decisions it forces.

## 1. The gradient — split it into two kinds of knobs

Our earlier discussion gives the key structural fact: some isolation mechanisms are **revocable** (cgroup membership, netns joining, broker grants — the parent can change them live) and some are **one-way ratchets** (seccomp, Landlock, `no_new_privs`, UID drop — set once, never lifted). This split decides *where* each knob can sit relative to leaf boundaries:

- **Revocable knobs can move at leaf boundaries.** The parent moves a PID into a leaf's cgroup, grants a broker route, passes an FD — and takes it all back when the leaf returns. So "a PID travels in and out of the sandbox" is *literally true* for this half.
- **Ratchet knobs cannot be lifted.** A PID that entered a seccomp/Landlock envelope never leaves it. So for this half, "travel out" must mean **process lifetime = leaf span**: fork at leaf entry, ratchet, run the body, exit at leaf return. The PID doesn't exit the sandbox; the sandbox dies with the PID.

The clean gradient, as one ordered vocabulary (each level adds to the previous):

| level | mechanism | leaf-boundary cost |
|---|---|---|
| `none` | same process, same namespace — today's rung 0 | 0 |
| `namespace` | same process, isolated Python namespace (rung-0 interpreter today) | ~0 |
| `fork` | subprocess via fork, CoW shared RAM, no lockdown | ~1 ms |
| `fork+cgroup` | + resource caps, parent-controlled, revocable | ~1 ms |
| `fork+ratchet` | + ephemeral UID, netns, Landlock, seccomp | ~2–5 ms |
| `sandbox` | separate mount-root world (bwrap-style), broker-only network | ~5–20 ms |
| `remote` | today's `remote_sandbox` — rung 2/3 territory | RPC latency |

Note the gradient is mostly a dial *within* rung 1. It composes with the rung axis instead of replacing it: rung says *where the backend lives*, isolation says *how hard the wall is*.

## 2. Shared RAM for inputs/outputs — two mechanisms, both compatible with full isolation

- **CoW from fork.** The warm parent holds the loaded libraries, the module tree, the pools. A forked leaf sees all of it read-only for free. This is the zygote pattern: fork per leaf call from a preloaded template. Nothing to design; it falls out of `fork`.
- **Explicit zero-copy for values: sealed memfds + FD passing.** The parent writes the leaf's inputs as Arrow IPC (or raw tensors) into a `memfd`, applies `F_SEAL_WRITE`, and hands the FD to the child. The child maps it read-only, zero-copy. Outputs come back the same way, child-to-parent. Three properties matter:
  - The seal makes inputs immutable. Even a hostile leaf body cannot corrupt what the parent shares.
  - An FD crosses every wall on the gradient. Landlock and netns do not block inherited or SCM_RIGHTS-passed FDs. So data sharing costs nothing in isolation strength — the FD *is* the declared grant, capability-style, consistent with the broker design.
  - This directly answers the D-027 open surface: the interpreter rung's `vars` marshaling. "Sealed Arrow-over-memfd at local rungs, JSON over the wire at remote rungs" is one contract with two transports.

## 3. What the spec should say — identity vs binding, again

The gradient must not create a new identity axis. The discipline the spec already has (D-033) extends cleanly:

- **The actual isolation envelope is the receiver's binding.** Which level a leaf runs at does not change `execute(code, vars)→result` semantics, so scores stay warranted across re-placement. Same rule as endpoint bindings.
- **The artifact bakes a floor, not an envelope.** The interpreter profile already carries `isolation_floor`; tools (component 6) should gain the same field. An optimizer-authored tool body can demand `fork+ratchet` as its floor ("this code came from an optimizer; never run it with network"). The receiver may exceed the floor, never go under it. Under-floor execution is a loud refusal, like every declared capability.
- **The `placement.isolation` enum becomes this vocabulary**, versioned like the other closed vocabularies. Ordered, so "floor" is well-defined by comparison.

## 4. One caveat: fork purity at the boundary

The fork-per-leaf pattern has one semantic edge the spec should name. A forked leaf's writes to inherited state (module globals, caches) die with the child. That is a *feature* for isolation — but it means a leaf at `fork`+ can only affect the program through its declared outputs. Today's in-process tools *can* mutate ambient state as a side effect. So the gradient enforces, mechanically, what the IR already wants philosophically: leaves communicate through typed edges only. A leaf that breaks under `fork` was relying on undeclared side channels — that is a defect the gradient *detects*, not a compatibility problem it creates. Worth stating in the spec as an explicit law: **any leaf must produce identical behavior at every isolation level; a level change that changes behavior indicts the leaf.** That is the isolation analogue of the structural-equivalence criterion, and it is what keeps the whole gradient a binding rather than an identity.

## Summary of proposed spec deltas

1. Replace the `isolation` enum with the ordered 7-level vocabulary, versioned.
2. Add `isolation_floor` to tool placement (component 6), mirroring the interpreter profile.
3. Record the data-plane contract: sealed-memfd Arrow at local rungs, JSON at remote rungs — closing the D-027 `vars` marshaling hole.
4. State the leaf-purity law: revocable knobs may change at leaf boundaries; ratchets bind PIDs for life, so ratcheted leaves are fork-scoped; behavior must be isolation-invariant.

The nicest property of the whole design: the gradient's zero end is *exactly* today's rung 0 — no subprocess, one RAM, the reference frame the thesis privileges. Nothing about the fully-baked pole changes. The gradient only names what it costs, mechanism by mechanism, to walk away from it.

Want me to draft these as a D-numbered decision entry, or fold them into §e0 directly?


## Q10: yes go

> yes go

I will add a decision entry and fold the design into §e0. First I check the decisions file format.

Now the decision entry:

Done. Two files changed, nothing committed (per your rule).

**`roadmap/IR-program-spec.md`** — new subsection **§e0-isolation**, placed between the placement ladder and §e0-lang, with the `placement` JSON block's `isolation` comment updated to point at it. It contains:

1. The ordered 7-level vocabulary (`none → namespace → fork → fork_cgroup → fork_ratchet → sandbox → remote`) as a versioned, comparable enum — framed as an axis orthogonal to the rung, mostly refining rung 1.
2. The revocable-vs-ratchet split, with the engine rule it forces: revocable knobs (cgroups, netns, FD grants) may change at leaf boundaries — the "PID travels" half; ratchets bind for life, so at `fork_ratchet`+ the leaf *is* the process (zygote fork → place → ratchet → run → exit).
3. The data plane: sealed `memfd` + Arrow, FD-as-capability — and it explicitly closes the D-027 `vars`-marshaling hole as one contract over two transports.
4. Floor-bakes/envelope-binds: `isolation_floor` extends from the D-033 interpreter profile to component-6 tools; under-floor is a loud refusal; composes with D-040 trust.
5. The isolation-invariance law: identical behavior at every level, or the leaf is indicted — which is what keeps the gradient a binding, not an identity axis.

**`roadmap/05-decisions.md`** — **D-042**, marked `DRAFT 2026-08-12 — pending Maxime` (matching the D-038..040 convention), compressing the five points above with the *why*.

I checked for stale mentions of the old `os_sandbox | remote_sandbox` enum — none remain in the spec. Review D-042 and ratify when ready; if you want the frontend-contract or the programir-contract repo's `manifest.md` to pick up the `isolation_floor` field on tools, that propagation is a separate wave I left untouched.


## Q11: what does this mean: may exceed the floor, never under-runs it — under-floor is…

> what does this mean: may exceed the floor, never under-runs it — under-floor is a loud refusal
> 
> explain simply with example, as if i had no special domain knowledge of computers and dspy.

It is a rule about minimum safety levels. Here is the idea in plain words.

**The setup.** A program file can contain small pieces of code that someone else (or an AI) wrote. When you receive the file and run it, your computer runs those pieces. The author of the file can attach a safety demand to each piece. That demand is the **floor**.

**The floor is a minimum, not an exact setting.** Think of a food label that says "keep refrigerated at 4°C or colder."

- You store it at 4°C → fine. You met the minimum.
- You store it in a deep freezer at −18°C → also fine. Colder than asked is allowed. That is "**may exceed the floor**."
- You leave it on the counter at 20°C → not allowed. That is "**under-running the floor**."

**"Loud refusal" = the program stops and tells you why, instead of quietly running anyway.** If your computer cannot (or will not) provide the minimum safety level, the program does not run that piece in a weaker way and hope for the best. It stops immediately with a clear message, something like:

> "The tool `fetch_data` requires safety level `fork_ratchet` (no network access). Your setup only offers `fork`. Refusing to run."

**A concrete example.** Suppose an AI optimizer wrote a small helper function inside the program. The author does not fully trust AI-written code, so they attach the demand: "run this in a locked box with no internet access."

- Receiver A runs it in a locked box with no internet. ✅ Meets the floor.
- Receiver B runs it in an even stronger box — no internet, no file access, strict memory limits. ✅ Exceeds the floor. Always allowed.
- Receiver C's machine cannot make locked boxes, so it would have to run the code with full internet access. ❌ Under the floor. The program refuses to start and names the exact piece and the missing protection.

**Why "loud" matters.** The alternative — running with less protection and saying nothing — is the dangerous case. You would believe the safety demand was honored when it was not. So the rule is: more protection than asked is always fine, less is never fine, and "less" produces an error you cannot miss rather than a silent downgrade.


## Q12: Okay, and what, they would have a mechanism to change the program IR so that the…

> Okay, and what, they would have a mechanism to change the program IR so that then the engine accepts it.say that they they want to further modify this program and um they want to give it more permissionSimilarly, maybe someone decides to trust the LM generated code in their context.

Yes — and the design already has the right shape for it. The key idea: **lowering a floor is allowed, but it must be an edit, never an override.**

## The two different moves

- **Running with *more* protection than the floor** — free. Nothing changes. That is the receiver's normal binding choice.
- **Running with *less* protection than the floor** — allowed too, but not as a runtime setting. You must **change the program file itself**: open the manifest, lower the floor field on that tool, save. Now you hold a *different artifact*, and the engine accepts it because the floor it checks is the one you wrote.

Why force the edit instead of offering an `--ignore-floors` flag? Three reasons:

1. **It makes trust a deliberate, visible act.** Editing the file and re-saving is a decision you can see in a diff, review, and audit later. A runtime flag is invisible after the fact — six months later nobody knows the guard was off.
2. **It keeps the record honest.** The artifact's provenance records the change: *who* lowered the floor, *when*, from what to what. The file now truthfully describes how it runs. With an override flag, the file would claim "locked box required" while reality was "ran open" — the file would lie.
3. **It scopes the trust.** You lower the floor on *that one tool* in *your copy*. The original artifact, and everyone else's copies, keep their protection. Trust does not leak.

## Your two examples, concretely

**"Give it more permission."** Say a tool has floor `fork_ratchet` (no network), but in your setup the tool genuinely needs to call an internal API you trust. You edit the manifest: floor drops to `fork_cgroup`, and — better — you pair it with a broker grant ("may reach `internal-api.corp` only"). The diff shows exactly what you opened and why. This is the same pattern as every other recorded change in the IR: a named, justified deviation, never a silent one.

**"I trust the LM-generated code in my context."** Same move, wholesale: you re-save the artifact with the optimizer-authored tools' floors lowered. Your copy's provenance now says `floor_lowered_by: <you>, date, old→new`. Scores and behavior are unaffected (the invariance law — isolation never changes what the code *computes*), so nothing else in the file needs to change. You have simply signed for the risk, in writing, inside the artifact.

## The asymmetry, stated as the rule

- **Raising isolation = a binding.** Receiver's runtime choice, free, unrecorded pressure on nothing.
- **Lowering a floor = an authorship act.** It edits a baked field, so it produces a new artifact version with the change in provenance — exactly like editing an instruction or a demo.

This mirrors the discipline the spec applies everywhere: the artifact is the truth; anything that would make the artifact untrue about how the program runs must go *through* the artifact, loudly, not around it.

One honest caveat: a receiver with full control of their machine can always bypass anything — they own the engine. The floor mechanism is not armor against a hostile receiver; it protects the *honest* receiver from accidental under-protection, and makes deliberate trust decisions leave a paper trail.

If you want, I can add this as point (6) of D-042: "floors are baked fields; lowering one is an artifact edit recorded in provenance (`floor_lowered_by`), never an engine override flag."


## Q13: I think both should be program modification because I think a uh er raising isol…

> I think both should be program modification because I think a uh er raising isolation has uh the risks of changing the program's performance, no? Or the b the code would not run which affects the performance, right?

You have put your finger on a real tension — but I think the fix is not "both are artifact edits." Let me separate the two risks you name, because the design answers them differently.

## Risk 1: "the code would not run" under stricter isolation

The design's claim is: this can only happen for two reasons, and both already have a lawful home.

- **The leaf had a declared need** — network to an endpoint, a credential, a file. Declared needs are grants, and grants are *orthogonal to the isolation level*: the receiver's envelope, at any level, must still supply everything the leaf declares. A `sandbox`-level envelope with a broker grant for `api.anthropic.com` satisfies a tool that declares that endpoint. An envelope that denies a declared grant is not "exceeding the floor" — it is failing a declared capability, and that is a loud refusal *at load*, before any run, exactly like a missing credential. So it never shows up as mysterious mid-run performance loss.
- **The leaf had an *undeclared* need** — it quietly reached for ambient network or a global variable. Stricter isolation breaks it. The invariance law says: this indicts the leaf, not the level. The breakage is a *detection* of a defect that was always there — the artifact claimed the leaf communicated only through typed edges, and that claim was false.

So under the design, "raising isolation changed behavior" is never a legitimate state. It is always either a load-time refusal (missing grant) or a surfaced defect (undeclared side channel). That is why raising can stay a binding: for a *lawful* artifact, it is behavior-invariant by construction.

## Risk 2: performance and scores — your stronger point

But you are right that in practice, the claim "behavior-invariant by construction" is only as good as the artifact's honesty, and scores were measured under *some* envelope. If a shipped score was measured at `none` and the receiver runs at `sandbox`, and a latent defect fires, the score is now unwarranted — and nothing recorded why.

The fix I'd propose is not to make raising an artifact edit — that would be very costly. Re-placement without re-authoring is one of the spec's core properties ("a dial, not a fork"): if every deployment's isolation choice forked the artifact, one program would splinter into per-environment copies, and the whole "same artifact, re-bound" story (cross-provider re-bind, rung-walk) collapses. Isolation would be the only binding that forks; that asymmetry is worse than the one you're questioning.

Instead: **record the envelope where scores and runs are recorded, not where identity lives.**

- Every *run* records the actual envelope used (per leaf), in the run's provenance — the View-2 channel.
- Every *score* carries the envelope it was measured under, in View-3 — the same way it already carries which LM binding served it.

Then the property you want holds: if performance shifts when someone raises isolation, the two run records differ in exactly one visible field, and the anomaly is attributable in one diff — without forking the artifact. And it gives the invariance law teeth: "measured at `none`, reproduced at `fork_ratchet`, same scores" becomes a *checkable claim* rather than an assumption, and a divergence is a filed defect against the leaf.

## The resulting rule, restated

- **Lowering a floor** → artifact edit, recorded in artifact provenance (trust decision — changes what the artifact *promises*).
- **Raising isolation** → binding, but recorded in *run/score* provenance (deployment decision — must not change what the program *does*, and the record is how you verify that).
- **Declared grants** are satisfied at every level, or load refuses — so "it stopped working" can never be a silent runtime surprise.

So: same observability for both directions, different homes — because one changes a promise and the other must not.

Want me to fold this into D-042 — point (6) for floor-lowering-as-edit, point (7) for envelope-in-run/score-provenance and grants-orthogonal-to-level?


## Q14: for risk one, what about piodyde might not work the same?

> for risk one, what about piodyde might not work the same?

Good catch — and the spec already has the concept that catches it, but this sharpens *why* the gradient must be defined the way it is.

Pyodide is not "more isolation." It is a **different runtime**. Recall from our earlier comparison: Pyodide is CPython compiled to WebAssembly — no real threads, no subprocesses, restricted sockets, only pure-Python or specially-built wheels, Python version lagging upstream. Running a leaf there can change *what the code computes*, not just what it can reach. `numpy` results may match; `torch` simply doesn't exist; a `threading` import dies.

So the answer is: **swapping to Pyodide is not a move on the isolation axis at all — it is a change of interpreter identity**, and the spec (D-033) already forces that to be a different, heavier kind of act:

- The artifact bakes an **interpreter identity profile**: language *and version*, exact runtime, packages, resource limits, result conventions. That profile is what the program's scores are claims about. It is identity, not binding.
- The isolation gradient's invariance claim ("same behavior at every level") holds only **within one identity profile**. Every level from `none` to `sandbox` runs *the same CPython, same packages, same machine* — the wall gets thicker, the runtime inside it is byte-identical (literally: it's a fork of the same loaded process). That is what makes the claim credible.
- The `remote` level does **not** mean "any remote runtime." It means: the same declared profile, satisfied elsewhere. A remote pool that runs real CPython 3.12 with the declared packages satisfies the profile — lawful binding. A Pyodide pool that *cannot* satisfy it ("wasm-CPython 3.11, no torch") fails profile verification at load — **loud refusal naming the mismatch**, the D-033 rule exactly.
- If the receiver decides "Pyodide is close enough for me," that is a **recorded deviation of the profile** — an identity change, so the shipped scores detach. Which is honest: the scores were measured on a runtime this one is not.

So risk 1's claim survives, but with its precise boundary now visible: *raising isolation is behavior-invariant because the gradient, by definition, never changes the runtime — only the wall around it.* Anything that changes the runtime (Pyodide, a different Python version, a different package set) is excluded from the gradient by construction and routed through the identity-profile machinery, where it is either verified, refused, or recorded as a score-detaching deviation.

This is worth one sentence in §e0-isolation, because someone will eventually read `remote` as "Pyodide pool" and treat it as an isolation raise: **"Every level of the gradient runs the leaf's declared interpreter profile unchanged; a level that cannot satisfy the profile (e.g. a wasm runtime lacking declared packages) is not a higher isolation level but an identity deviation, handled by D-033."** Want me to add it, along with the D-042 points (6) and (7) from before?


## Q15: does the specc have a way to express a leave - runtime/interpreter - isolation t…

> does the specc have a  way to express a leave - runtime/interpreter - isolation three way pairing?

Almost — two of the three legs exist, and the third is expressible only at the wrong granularity. Let me lay out what's there and where the gap is.

## What the spec can already say

- **Leaf → runtime.** Since D-029, interpreters are a named pool and every interpreter leaf carries a `ref` to an entry. Tool leaves reference component-6 entries. So "this leaf runs on that runtime" is a first-class binding, same machinery as `{adapter: "terse", lm: "baguettotron"}` on Predict leaves.
- **Runtime → identity + isolation.** The pool entry carries the D-033 identity profile (what the runtime *is* — the thing scores attach to, including `isolation_floor`) and the `placement` block (where it runs, and — post-D-042 — the ordered isolation level as the receiver's envelope).

So the chain *leaf → entry → (profile, isolation)* exists, and for the common case it is enough.

## The gap: the pairing is per-entry, not per-leaf

Isolation lives on the pool entry. So two leaves sharing one interpreter entry necessarily share one floor. But the motivating case for floors is exactly a *per-leaf* trust distinction: the same CPython profile, one leaf running the author's hand-written helper (trusted, `fork` is fine), another running optimizer-generated code (demand `fork_ratchet`). Today you can only express that by **forking the interpreter entry** — two entries with byte-identical profiles differing only in floor. That works, but it is a lie of the same species the pool shape was built to kill: it makes the artifact say "two runtimes" when the truth is "one runtime, two trust levels." Sharing an entry is supposed to be a *fact the artifact states* (same runtime identity, same semantics); floor-forking destroys that fact to smuggle in a per-use-site attribute.

## The fix the spec's own machinery suggests

The LM pool already solved this exact shape: entry-level truth plus **binding-level deltas** (base ⊕ delta — the binding carries the small per-use-site difference). Apply it verbatim:

- The pool entry keeps the *runtime's* floor: the minimum the runtime identity itself demands (a wasm host might inherently demand `remote`).
- The **leaf's binding gains an optional `isolation_floor` override**: `{interpreter: "main", isolation_floor: "fork_ratchet"}`.
- Effective floor at a leaf = **max(entry floor, binding floor)** — floors compose upward only, so a binding can tighten, never loosen, the entry's demand (the same one-way discipline as the ratchet itself).
- The receiver's envelope is still chosen per entry-or-leaf as they wish, verified against the *effective* floor; run/score provenance records the actual envelope per leaf (the point-7 machinery from earlier).

Then the three-way pairing reads directly off the artifact, per leaf: *this leaf* (tree position) runs on *this runtime* (entry ref → identity profile) at *at least this isolation* (effective floor), and actually ran at *this envelope* (run provenance). Trust is per-use-site, identity is shared and stated, and `authored_by: optimizer` on a tool body can even *default* its binding floor — the trust gradient (D-040) getting a mechanical hook.

This slots cleanly into D-042 as the missing granularity clause. Want me to add it — binding-level `isolation_floor` with max-composition — to the draft entry and §e0-isolation?


## Q16: yes!

> yes!

I will add all the pending pieces: the binding-level floor, the floor-lowering rule, the run/score provenance, and the identity-boundary sentence. First the spec.

Done. Both files updated, still uncommitted.

**§e0-isolation** gained three paragraphs after the floor/envelope one:

1. **Binding-level floors** — a leaf's binding may carry `isolation_floor`; effective floor = max(entry floor, binding floor); floors only tighten per binding. This completes the three-way pairing: leaf (tree position) → runtime (entry ref → D-033 profile) → minimum isolation (effective floor), readable per leaf without floor-forking a shared entry. `authored_by: optimizer` defaults the binding floor through the D-040 trust profile.
2. **Floor changes vs envelope choices** — lowering a floor is an artifact edit with `floor_lowered_by` provenance, never an engine flag; raising the envelope stays a binding but is recorded per leaf in run provenance (View-2) and on scores (View-3); declared grants are orthogonal to level, so under-supplied grants refuse at load rather than surprising at runtime.
3. **The gradient's boundary** — every level runs the same interpreter profile; Pyodide-class backends that cannot satisfy the profile are identity deviations under D-033, not isolation levels; `remote` means the same profile satisfied elsewhere.

**D-042** in the decisions file gained the matching points (6), (7), and (8), still marked DRAFT pending your ratification.

One propagation note for later, same as before: when D-042 ratifies, the programir-contract repo's manifest schema will need the binding-level `isolation_floor` field and the tool-placement floor — that byte-shape wave stays separate.
