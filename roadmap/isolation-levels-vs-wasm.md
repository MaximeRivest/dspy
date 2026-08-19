# The isolation levels: what each Linux level buys, and how Wasm and Pyodide compare

**Status:** companion note to `IR-program-spec.md` §e0-isolation and D-042
(ratified 2026-08-12; see also D-043, which grounds FlexIR's trust knobs in the
same gradient), and to `linux-isolation-notes.md` (the source conversation,
2026-08-12). This
document does two things: (1) it walks the seven-level gradient and states, per
level, exactly what is contained and what is not; (2) it places Wasm and Pyodide
against that gradient — and shows why they are not levels on it but a different
axis entirely (runtime identity, D-033).

---

## Part 1 — The Linux gradient, level by level

The gradient's design rule (§e0-isolation): every level runs the **same runtime**
— the leaf's declared CPython, same packages, same machine. Only the wall gets
thicker. Each level adds to the previous; nothing is ever swapped out.

The columns below track the five things a sandbox can contain:

- **memory** — can the leaf read/corrupt the parent's address space (API keys, program state)?
- **filesystem** — what can it read or write?
- **network** — what can it reach?
- **resources** — can it exhaust CPU/RAM/PIDs/disk?
- **kernel** — how much kernel attack surface does hostile code touch?

### `none` — same process, same namespace

Today's rung 0. The leaf is a function call. One RAM, one process — the
reference frame the ProgramIR thesis privileges.

- **Contains:** nothing. **Costs:** zero.
- Memory, filesystem, network, resources: all fully shared. A buggy leaf can
  scribble on program state; a hostile one owns the session.
- **Right for:** the author's own code, fully trusted. This is not a defect —
  it is the fully-baked pole, and most leaves live here.

### `namespace` — same process, isolated Python namespace

The rung-0 interpreter today: generated code runs with its own globals dict
(and optionally restricted builtins).

- **Contains:** *accidents only* — name collisions, accidental global
  mutation. A courtesy, not a sandbox (the spec's own words): `object.
  __subclasses__()`, `import ctypes`, or any C extension escapes it trivially.
- Everything else: shared, as `none`.
- **Right for:** sloppy-but-not-malicious generated code where the only worry
  is namespace pollution.

### `fork` — subprocess via fork, CoW-shared RAM, no lockdown

The first real boundary: a separate address space.

- **Memory: contained.** The child sees the parent's loaded libraries and data
  copy-on-write — reads are free and zero-copy, but its writes die with it. It
  cannot corrupt the parent, and (integrity aside) the parent's *secrets* are
  still readable in the CoW image — confidentiality arrives only at
  `fork_ratchet` (UID drop + ratchet closes /proc self-inspection routes) and
  is only really strong at `sandbox`.
- **Filesystem / network / resources: NOT contained.** The child runs as your
  UID with your ambient reach: it can delete your files, call the internet,
  fork-bomb the host.
- **Cost:** ~1 ms (fork from the warm zygote parent; preloaded torch/numpy are
  shared CoW — no re-import, no re-load).
- **Right for:** crash isolation and side-effect purity — the leaf-purity law
  ("leaves communicate through typed edges only") becomes mechanically true
  here, because inherited-state writes cannot survive the child.

### `fork_cgroup` — + cgroup v2 resource caps

Adds the consumption dimension: the parent places the child's PID in a cgroup
before it runs (`memory.max`, `pids.max`, `cpu.max`, io).

- **Resources: contained.** Fork bombs hit `pids.max`; memory balloons OOM the
  *cgroup*, not the host; `cgroup.kill` reaps every straggler at leaf return.
- **Crucially, revocable:** the parent can tighten/loosen limits live and move
  the PID between groups. This is the parent-controlled half of the gradient —
  a PID genuinely travels in and out of *this* sandbox at leaf boundaries.
- Filesystem and network: still open.
- **Right for:** optimizer loops running many candidate leaves concurrently
  under a shared budget — runaway code cannot starve the session.

### `fork_ratchet` — + ephemeral UID, netns, Landlock, seccomp

The one-way half. After fork and cgroup placement, the child **ratchets itself
down**: `setresuid` to an ephemeral UID (no `/etc/passwd` entry needed —
systemd `DynamicUser` prior art), `unshare` netns (dead loopback: no network at
all unless granted), a Landlock ruleset (filesystem allowlist: the venv
read-only, the task dir read-write, nothing else), a seccomp filter
(syscall allowlist), `no_new_privs`.

- **Filesystem: contained** — down to per-file/per-directory granularity.
  Secrets under other UIDs' 700-mode homes are unreadable.
- **Network: contained** — the netns has no route. Grants are explicit:
  a Unix-socket FD to the parent's broker proxy (hostname-level allowlist,
  credential injection so keys never enter the sandbox), or a passed
  listening FD for declared inbound services. The FD *is* the capability.
- **Memory, resources:** as before (separate address space + cgroup).
- **Kernel: partially contained.** The seccomp allowlist removes most exotic
  syscall surface (`io_uring`, `keyctl`, ptrace...), but the syscalls you must
  allow (`read`/`write`/`mmap`/`futex`) still hit the real shared kernel.
- **One-way:** none of this can be lifted for the life of the PID. Hence the
  gradient's rule: process lifetime = leaf span at this level and above —
  fork, place, ratchet, run, exit. The PID never leaves this sandbox; the
  sandbox dies with the PID.
- **Cost:** ~2–5 ms. Sealed-memfd Arrow I/O crosses the wall at zero cost
  (Landlock/netns do not block passed FDs).
- **Right for:** optimizer-authored and LM-generated code — the D-042 default
  floor for `authored_by: optimizer` under a cautious trust profile. This is
  the level where "generated code cannot exfiltrate, cannot touch my files,
  cannot outlive its call" all hold at once.

### `sandbox` — + private mount root, broker-only network

The bubblewrap-class envelope: mount namespace + `pivot_root` onto a tmpfs
containing *only* what was granted (`/usr` and the venv read-only, the task
directory, nothing else). GPU access, if granted, is explicit device-node
bind-mounts (`/dev/nvidia*`).

- **Filesystem: there is no "rest of the filesystem."** Where `fork_ratchet`
  denies paths, `sandbox` makes them not exist — world-readable leaks
  (`/etc`, other users' 755 homes, `/proc` details) are gone by construction.
  This is also where confidentiality becomes structural rather than
  configured.
- Everything else as `fork_ratchet`; the kernel caveat remains.
- **Cost:** ~5–20 ms (mount ceremony dominates).
- **Right for:** code you actively distrust, or multi-tenant-ish settings on
  one host — with the honest caveat that a *shared kernel* is still one
  kernel CVE away from cross-tenant compromise. True mutual-distrust
  multi-tenancy wants a second boundary (microVM/gVisor) below this one.

### `remote` — the same profile, satisfied elsewhere

The envelope is another machine or a managed pool. Isolation-wise it is the
strongest (separate kernel, separate hardware — side channels and kernel CVEs
stop mattering to *your* host) and the most expensive (RPC latency; the
sealed-memfd data plane gives way to the JSON-typed wire form).

- **The definitional constraint (D-042 point 8):** `remote` means the leaf's
  declared interpreter profile — same CPython version, same packages —
  *satisfied elsewhere*. It is a placement, not a runtime change. Which is
  exactly where Wasm and Pyodide enter, and fail to enter.

### The gradient at a glance

| level | memory | filesystem | network | resources | kernel surface | cost |
|---|---|---|---|---|---|---|
| `none` | shared | open | open | open | full | 0 |
| `namespace` | shared | open | open | open | full | ~0 |
| `fork` | **own space (CoW)** | open | open | open | full | ~1 ms |
| `fork_cgroup` | own space | open | open | **capped, revocable** | full | ~1 ms |
| `fork_ratchet` | own space | **allowlist (Landlock)** | **deny-all + broker grants** | capped | **seccomp-reduced** | ~2–5 ms |
| `sandbox` | own space | **private root — rest doesn't exist** | broker-only | capped | seccomp-reduced, still shared | ~5–20 ms |
| `remote` | other machine | other machine | other machine | other machine | **separate kernel** | RPC |

Two structural facts to keep in view (they shaped D-042):

1. **Revocable vs one-way.** cgroups, netns joins, broker grants, and passed
   FDs are parent-controlled and reversible — they may change at leaf
   boundaries. seccomp, Landlock, `no_new_privs`, and UID drops bind a PID for
   life — they apply only at process birth. The gradient's levels are ordered
   so the one-way half begins at `fork_ratchet`.
2. **Python stays perfect at every level.** Real CPython, native extensions,
   threads, CUDA-if-granted (`/dev/nvidia*` is just files to allowlist), at
   full speed. The OS constrains *reach*, never *semantics*. This is the
   invariance law's physical basis — and it is exactly the property Wasm and
   Pyodide give up.

---

## Part 2 — Where Wasm and Pyodide stand

### The category error to avoid

It is tempting to read Wasm/Pyodide as "levels above `sandbox`" — they feel
like *more* isolation. They are not on the axis. The gradient is defined by
**same runtime, thicker wall**; Wasm-based options achieve their isolation by
**changing the runtime**. In ProgramIR terms: moving a leaf from `sandbox` to
`remote` is a *binding* change (scores stay warranted); moving it onto Pyodide
is an *identity deviation* under D-033 (profile verification fails loudly, or
the receiver records a deviation and shipped scores detach). One is a dial;
the other is a different machine pretending to be the same one.

### What they actually are

- **Wasm runtimes (Wasmtime/Wasmer + a CPython-to-Wasm build).** An in-process,
  memory-safe guest: the sandbox boundary is the Wasm semantics itself, not the
  kernel. Capabilities (WASI) are explicit handles — philosophically identical
  to the gradient's FD-as-capability discipline, executed at a different layer.
- **Pyodide (+ Deno/browser host).** CPython compiled to Wasm with a curated
  package set. The most portable Python sandbox in existence — and the least
  *Python-complete*.

### Guarantee comparison

| dimension | Linux `fork_ratchet`/`sandbox` | Wasm/WASI | Pyodide |
|---|---|---|---|
| memory containment | separate address space | **in-process but memory-safe by construction** | same (inherits Wasm) |
| filesystem | Landlock allowlist / private root | WASI preopens (capability handles) | virtual FS; host reach only via granted APIs |
| network | deny-all + broker | deny-all + granted sockets | fetch-shaped, host-mediated |
| kernel attack surface | reduced (seccomp) but **shared kernel** | **near-zero** — guest never sees real syscalls | near-zero |
| resource caps | cgroups (strong) | engine-level fuel/memory limits | host-side (weaker, JS event loop) |
| **Python realness** | **perfect** — any CPython, any wheel, torch+CUDA, threads, subprocess-if-granted | poor-to-partial: wasm-CPython builds, young ecosystem | **partial by design**: no real threads, no subprocess, restricted sockets, pure-Python/curated wheels only, version lags |
| performance | native; GPU at full speed | interpreter-grade; no GPU | interpreter-grade; no GPU, no BLAS-native speed |
| portability | Linux only (macOS/Windows = different code) | everywhere | anywhere JS runs |
| cross-tenant credibility | needs a second boundary (shared kernel) | **strong** — the boundary is not the kernel | strong |
| retrofit onto a warm session | **yes** — fork-then-ratchet from the loaded parent | no — guest built from scratch, can't CoW-share your loaded torch | no |

### The trade, stated once

The Linux gradient and the Wasm family sit on opposite ends of one trade:

> **The gradient sells kernel-independence to keep Python perfect.
> Wasm/Pyodide sell Python completeness to buy kernel-independence.**

For the ProgramIR's actual workload — optimizer- and LM-authored code that
imports sklearn/torch, touches GPUs, and is scored under a declared CPython
profile — the gradient's side of the trade is the right one, which is why it
is the spec's axis. The kernel-surface residue it accepts is real but has a
known escalation path that *stays on the axis*: `remote` onto a
microVM-backed pool (Firecracker-class) buys a separate kernel **without**
changing the interpreter profile — full Python fidelity, true multi-tenant
credibility, paid in RPC latency and infrastructure. That, not Wasm, is the
lawful answer to "I need more than `sandbox`."

### Where Wasm/Pyodide DO fit the ProgramIR

Not nowhere — just not as isolation levels:

1. **As a declared interpreter profile in their own right.** A program *authored
   against* Pyodide's semantics (pure-Python leaves, its package set, its
   version) can bake that as its D-033 profile. Then a Pyodide pool at the
   `remote` level satisfies the profile honestly, scores attach to it, and
   browser-side execution becomes a lawful placement. The profile is the
   truth-teller: Pyodide is fine when *declared*, poisonous when *substituted*.
2. **As the cross-language rung-walk's far end.** For a Go/TS receiving engine
   (D-022), a wasm-compiled leaf is a candidate portable target — D-025 already
   marks WASM as the deliberately-deferred portable-compilation ruling for
   compiled authored languages.
3. **As a guest *inside* a level.** Wasm's party trick is an in-process
   memory-safe boundary — a host can embed one *within* a `fork_ratchet` child
   for defense in depth. That is an engine implementation detail below the
   spec's visibility; the manifest still records level + profile.

### Bottom line

- Want to run **real** generated Python safely on your own machine: walk the
  gradient — `fork` for purity, `fork_cgroup` for budgets, `fork_ratchet` for
  distrust, `sandbox` for structural confinement. Python stays perfect
  throughout; that is the gradient's entire point.
- Want **cross-tenant assurance** beyond a shared kernel: stay on the axis and
  go `remote` onto microVMs — do not reach for Wasm, which would silently
  change what the program *is*.
- Want **portability into browsers/edge** or a curated pure-Python world:
  Pyodide/Wasm — but as a *declared identity* the program was authored and
  scored against, never as a transparent swap. The D-033 machinery exists
  precisely to keep that boundary loud.
