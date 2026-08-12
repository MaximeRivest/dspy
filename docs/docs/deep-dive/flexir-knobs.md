# FlexIR: every knob, and when to turn it

FlexIR is the optimizer that rewrites a program's IR itself. A reflection
LM proposes edits from a closed vocabulary; every edit passes admission;
every candidate is scored on your metric and gated on a holdout split.
This page documents each knob, the trust postures they compose into, and
the solution space they open.

The one-line mental model: **FlexIR's power is fixed; your exposure is
tunable.** The vocabulary always allows any structural rewrite the node
set can express. The knobs decide what *generated code* may reach —
imports, packages, and process placement.

## The constructor

```python
optimizer = dspy.optim.FlexIR(
    reflection_lm,            # the LM that proposes edits
    metric,                   # metric(example, prediction) -> bool | float
    iterations=8,             # exact number of reflection rounds
    reward=None,              # enables wrap_best_of_n proposals
    holdout=None,             # the reward-hacking guard split
    eps=1e-9,                 # score tolerance for acceptance
    isolation_floor="fork_ratchet",  # the D-042 floor authored leaves demand
    extra_imports=None,       # widen the stdlib import allowlist
    allowed_deps=None,        # permit named pip packages in # deps:
    auto_install=False,       # uv pip install granted deps into THIS env
    eval_mode="in_process",   # "in_process" | "artifact" (score as artifact)
    scoring_isolation="none", # wall the artifact-mode scoring child runs behind
    broker_egress=None,       # hostnames the child may reach through the broker
)
compiled = optimizer.compile(program, trainset=trainset, checkpoint_dir="runs/flex")
```

### The loop knobs

| Knob | Default | What it does |
|---|---|---|
| `reflection_lm` | required | The proposer. A strong model here matters more than anywhere else: it reads the program report and writes the edits. A scripted `DummyLM` makes runs replayable in tests. |
| `metric` | required | Scores one example. Return a float or bool. A metric exception scores that example 0.0 and the run continues; typed LM/infra errors abort the step instead — infrastructure is never blamed on the candidate. |
| `iterations` | 8 | Exactly this many reflection calls. There is no early stop; budget is explicit. |
| `reward` | None | A plain `reward(outputs) -> float`. Required only for `wrap_best_of_n` proposals; without it those proposals are refused with a teaching error. |
| `holdout` | None | Examples the reflection LM **never sees**, disjoint from the trainset (overlap is refused at compile). Every accepted candidate must hold here — this is the reward-hacking guard. Run without it only for throwaway experiments: without a holdout, a memorizing edit can ship. |
| `eps` | 1e-9 | Tolerance in the two-channel acceptance rule: accept on strictly higher dev score, or equal score at strictly fewer LM calls. |
| `checkpoint_dir` | None | When set, the baseline and every accepted candidate are saved as loadable ProgramIR artifacts, with `scores.json` as the ordered trajectory. Rollback to any point is `dspy.load(<dir>)`. |

### The trust knobs

These three govern what optimizer-authored code may reach. They are
independent; compose them deliberately.

**`isolation_floor`** — the wall authored code demands (D-042).

The floor is the **minimum** isolation level an authored tool leaf will
accept, named on the ordered gradient
(`none < namespace < fork < fork_cgroup < fork_ratchet < sandbox <
remote`). It is baked into the leaf's placement; a receiver may exceed
it, never under-run it (under-floor is a loud refusal).

- `"fork_ratchet"` (default): authored tool leaves are stamped
  `isolation_required`. Loading the artifact fails closed unless the
  receiver grants the leaf — either explicitly (bind a reviewed
  callable) or by binding an isolation envelope at level ≥ `fork_ratchet`
  (**the envelope IS the grant**). Nothing runs in your process without a
  deliberate act.
- `"none"`: authored leaves get the ordinary in-process placement. Your
  own loop, and anyone who loads your artifact, runs the generated code
  with **full ambient authority** — your filesystem, your network, your
  credentials.

The `authored_by: "optimizer"` provenance stamp survives at every floor,
so a receiver can always audit or re-place the leaf before trusting it.

> **Migration.** `code_trust` is the old spelling: `code_trust="isolated"`
> is an alias for `isolation_floor="fork_ratchet"`, and
> `code_trust="in_process"` for `isolation_floor="none"`. Both still work;
> prefer the floor form for the full gradient.

**`extra_imports`** — what authored code may `import`.

The default allowlist is a closed pure-stdlib set (`re`, `json`, `math`,
`statistics`, `itertools`, ...). `extra_imports=frozenset({"numpy",
"sklearn"})` widens it for this optimizer instance only. The
builtin/dunder denylist (`eval`, `__import__`, `__globals__`, ...) stays
denied regardless.

Understand what this list is now: **defense in depth and teaching
ergonomics, not the wall.** When the floor is `fork_ratchet` or above,
the real boundary is the isolation envelope — a kernel wall, not an AST
filter. The import allowlist then serves two lesser purposes: it catches
an obviously-wrong import early with a teaching refusal (faster feedback
than a sandbox denial mid-run), and it is a second layer for the honest
case where the envelope is weaker than intended. At floor `none` it is
the *only* filter, and an AST filter is a speed bump, not armor — so
every module you add there is a door you opened.

**`allowed_deps`** — what packages authored code may declare.

Authored tools declare packages inline (`# deps: httpx, beautifulsoup4`),
the same mechanism human-authored tools use. By default, optimizer-
authored code may declare **none** — any `# deps:` line refuses. Passing
`allowed_deps=frozenset({"scikit-learn", "httpx"})` permits exactly those
names. Admitted deps flow into the leaf's `deps[]` and from there into
the artifact's single PEP 723 block and lockfile — the normal
union-and-lock path, nothing special for generated code.

Pairing rule: **a dep's import name must also be in `extra_imports`.**
`allowed_deps` speaks package names (`beautifulsoup4`); the import
allowlist speaks module names (`bs4`). They are not auto-unioned, on
purpose — each list is one explicit door.

### The environment knobs

| Knob | Default | What it does |
|---|---|---|
| `auto_install` | False | When a dep-carrying leaf passes admission, install its missing granted packages via `uv pip install` into the **current** interpreter's environment, before the candidate is scored. Install failure is a teaching refusal into the ledger (a bad package name is the proposal's fault), never an infra abort. Know the costs: installing a package executes third-party build code; a rejected candidate's installs are **not** unwound (your env drifts from the lock until the next export re-locks); it needs network or a warm uv cache. Only meaningful with `allowed_deps`. |
| `eval_mode` | "in_process" | Where candidates are scored — never what is accepted. `"artifact"` exports every candidate (baseline included) and scores it in a subprocess under the artifact's own environment: scoring semantics == deployment semantics, environment included. Requires a self-contained metric (its source travels to the child; refused at compile otherwise). LM bindings for the child are auto-derived from the live LMs — scripted `DummyLM` state as data, plain `dspy.LM` as a receiver binding with credentials as env-var **names** (see below). Child environments are cached by lockfile hash under `<checkpoint_dir>/.envs` (or `~/.cache/dspy-flexir`), so only candidates that change deps pay a resolve. |
| `eval_env_overrides` | `{"dspy": <this repo>}` | Artifact mode: distributions installed **editable** from a local path instead of the locked release. The default exists because the manifest pins `dspy==<greenfield version>`, which is not the PyPI dspy — the child must run the local tree. |

## Who prepares the environment

Three rungs, from hands-off to engine-owned:

**Rung 1 — user-managed (the default).** Deps admitted through
`allowed_deps` are declarations only: they flow into the artifact's PEP
723 block and lockfile, and the *receiver's* materialization installs
them. Your own loop needs the packages already present, or the leaf
fails at run time like any missing import.

**Rung 2 — install-on-admit (`auto_install=True`).** The optimizer
installs missing granted packages into your current environment the
moment an admitted leaf declares them. Frictionless for your own loop;
read the caveats in the table above — this rung deliberately trades env
hygiene for velocity, and it says so instead of pretending otherwise.

**Rung 3 — artifact-mode scoring (`eval_mode="artifact"`).** Every
candidate is scored **as the artifact it would ship as**, in a child
process under an environment built from its own lockfile. What you
accept is what deploys, environment included. The child re-runs the same
`evaluate` the in-process path runs, so the sacred error split survives
the process boundary: catchable program errors score 0.0 per example;
anything that escapes (an unloadable artifact, an engine guard) aborts
the step as infrastructure, never scored against the candidate.

### Real LMs in artifact mode

The child's LM bindings are derived from the live LMs your predictors
already hold — no parallel configuration. A scripted `DummyLM` crosses
as data (replayable tests). A plain `dspy.LM` crosses as a receiver
binding: model identity, capability facts, and the non-secret request
kwargs, straight from the LM's own constructor contract.

Credentials ride **environment-variable names, never disk**: the job
file, the exported artifact, the lockfile, and the argv carry no secret
material (a belt-and-suspenders scan refuses before writing if one
would). When a parent env var holds the key, the binding names that
var and the child reads it from its inherited environment. When the
live LM holds a raw key with no recoverable env origin, the fallback
path sets a private `DSPY_FLEX_LM_*` variable on the child process
only — process memory to process memory.

What still refuses, loudly and by name: an LM subclass with no
construction contract; a credential-bearing opaque header bag
(`extra_headers` and kin — refusing beats leaking); a non-JSON kwarg
(dropping it silently would change the child's sampling). The `dspy`
pin in the lock is overridden with the local tree via
`eval_env_overrides`.

## Isolation: the wall around scoring (D-042)

The isolation gradient is an axis *orthogonal* to where the backend runs.
It names, mechanism by mechanism, what it costs to walk away from the
zero pole (today's in-process scoring). Its zero end is byte-identical to
before.

| level | mechanism | leaf-boundary cost |
|---|---|---|
| `none` | same process, same namespace — today's default | 0 |
| `namespace` | same process, isolated Python namespace | ~0 |
| `fork` | subprocess via fork, CoW-shared RAM, no lockdown | ~1 ms |
| `fork_cgroup` | + resource caps (cpu/memory/pids), parent-controlled, revocable | ~1 ms |
| `fork_ratchet` | + ephemeral UID, netns, Landlock, seccomp — the one-way wall | ~2–5 ms |
| `sandbox` | + separate mount-root world (bubblewrap), broker-only network | ~5–20 ms |
| `remote` | the same declared profile satisfied elsewhere | RPC latency |

**`scoring_isolation`** puts the artifact-mode scoring child behind one of
these walls. The default `"none"` is exactly today's behavior. Any higher
level asks the Linux backend for that wall following the fork-place-ratchet
recipe: fork → the parent places the child in a cgroup leaf → the child
self-ratchets (unshare namespaces, `NO_NEW_PRIVS`, Landlock/seccomp where
present) → a self-probe confirms a socket and an out-of-scratch write both
fail before any payload runs. If the host cannot honestly build the
requested level — cgroupfs unwritable, user namespaces disabled — it
**refuses loudly** (`IsolationDowngrade`) rather than run behind a weaker
wall while claiming the stronger one. Fail closed, never silently.

**The envelope is recorded, per candidate.** Every trajectory entry (and
`scores.json`) carries the `envelope` it was scored under — the level, the
broker allowlist, the scratch dir. Because behavior must be
isolation-invariant (raising the wall never changes what the code
computes), a score that shifts when the wall changes shows up as one
differing field in one diff, and indicts the leaf's undeclared side
channel — not the wall.

**The envelope is also a grant.** At load time, binding an isolation
envelope at level ≥ `fork_ratchet`
(`bindings={"isolation": {"envelope": "fork_ratchet"}}`) satisfies an
`isolation_required` leaf directly — the wall the leaf demanded is
present, so the receiver need not hand-review and bind every callable.
The explicit per-leaf grant still works.

## The egress broker

At `sandbox` the child's network is deny-all by default. `broker_egress`
opens exactly the hosts you name, through a parent-owned localhost proxy
(the notes' Q7 design):

```python
FlexIR(..., scoring_isolation="sandbox",
       broker_egress=frozenset({"api.anthropic.com"}))
```

The broker gives three things: a **hostname allowlist** (anything else is
refused *and* logged — no IP-allowlist rot), a **per-request log** (one
list answers "what did the child reach"), and **credential injection** —
the child gets `HTTPS_PROXY`/`HTTP_PROXY` and *no* credential env vars;
the broker attaches the `Authorization` header on egress to an allowlisted
host. Generated code cannot leak a key it never held.

Which credential channel is live, stated plainly:

- **Broker active** (`broker_egress` names the LM's host): the secret
  never enters the child, not even as an env var. The broker injects it.
- **No broker** (the env-name channel, back-compat and the
  `scoring_isolation="none"` default): the secret rides an env-var *name*
  the child reads from its inherited environment, or a private
  `DSPY_FLEX_LM_*` fallback var — still never on disk or argv.

## One leaf substrate

Tools and interpreters were always two spellings of the same thing: a
leaf that runs a body under a profile. PIR-021 (D-043) makes that one
record. Two additive fields carry it, both absent by default so every
existing artifact is byte-identical.

**`kind`** — the invocation discriminant. `call` (the default, absent
from the manifest) is one request/response — exactly today's tool.
`session` is a held stateful context whose lifetime is one **leaf span**:
the callable is built once per forward and torn down after, so state is
held *within* a forward and never leaks across forwards (at `fork_ratchet`
and above the session is fork-scoped — the same ratchet rule).

**`grants`** — the closed static effect row. A leaf's `grants[]` lists
every capability it may be handed, readable from the manifest *without
executing anything*. A session leaf reaches its granted pool leaves
through a **bridge** it receives as its first parameter — never ambient
pool access. Reaching a leaf it was not granted refuses; a grant that
names no leaf refuses at load, the same species as a dangling leaf ref.

In FlexIR, `add_tool` takes optional `kind` and `grants`:

```python
{"op": "add_tool", "path": "self", "name": "wrapper",
 "kind": "session", "grants": ["solver"],
 "python_source": "def wrapper(bridge: object, text: str) -> dict:\n    return bridge.solver(text=text)\n"}
```

Grants are live sites: `delete_dead_leaf` refuses to delete a leaf a
session still grants, naming the granting leaf.

**Nested attribution.** An LM call made through a session leaf's grant
bridge attributes to **both** the session leaf and the predictor it
reached, transitively. The attribution is a *labeling*, not a
double-count: the total `lm_calls` still counts each real call once, and
the per-leaf measured counts show the same call under both names. A
session leaf granted one predictor, run over N forwards, reports total
`lm_calls` N with per-leaf `session:N` and `predictor:N`.

> Contract note: the ratified `$defs/grant` closes grant `kind` to `fd |
> broker_route`; a pool-leaf callback bridge has no contract byte shape
> yet (PIR-021 records nested attribution as "law, no byte shape yet"),
> so it rides as an `fd`-kind grant named `leaf:<pool_name>` — valid
> bytes today, a straight remap when the contract lands a `leaf` grant
> kind.

## The authoring surface

Everything above is reachable through the optimizer's knobs and the
engine's bindings. For a human author, the same intent reads as a small
set of decorators and objects — `dspy.tool`, `dspy.Session`,
`dspy.Envelope`, `dspy.Broker`, `dspy.SecretFromEnv`, `dspy.retrust`,
`dspy.confined`. A decorated function stays a normal callable and a
normal tool everywhere a tool is accepted; the decorator only attaches
declared metadata.

**Declaration vs enforcement — stated honestly.** Some declarations bite
today; some are recorded intent whose mechanism is owed. Do not read a
declaration as a guarantee:

| Declares | Enforced today? |
|---|---|
| `isolation=` (the leaf's floor) | **Yes** — load fails closed below it |
| `session=` / `grants=` | **Yes** — the real engine bridge |
| `net=` (broker routes) | **Yes when run under a broker** — allowlist + inject |
| `deps=` | **Yes** — validated + unioned into the artifact env |
| `files=` (`ro`/`rw` scopes) | **No** — declared only; no Landlock wiring yet |
| `memory=` / `cpus=` / `gpu=` | **No** — declared only; cgroup/device placement is detection-only |

A worked example, top to bottom:

```python
import dspy

# A tool that reaches the network — declared floor, declared route, real deps.
@dspy.tool(isolation="sandbox", net=["api.weather.example"], deps=["httpx"])
def weather(city: str) -> dict:
    # deps: httpx
    import httpx
    return httpx.get(f"https://api.weather.example/{city}").json()

# A GPU rerank tool — floor real, memory/gpu DECLARED (not enforced yet).
@dspy.tool(isolation="fork_ratchet", memory="8G", gpu=True)
def rerank(query: str, docs: list) -> dict:
    return {"order": sorted(range(len(docs)), key=lambda i: docs[i])}

# A session leaf that reaches its granted leaves through the bridge only.
def research(bridge: object, question: str) -> dict:
    hit = bridge.weather(city=question)   # only granted leaves are reachable
    return {"answer": str(hit)}
session = dspy.Session(research, grants=[weather], isolation="fork_ratchet")

# ... build and save a program using these as tools, then:

# A receiver raises isolation with an ENVELOPE (a binding — no artifact edit).
broker = dspy.Broker(
    allow=["api.weather.example"],
    inject={"api.weather.example": dspy.SecretFromEnv("WEATHER_KEY")},  # read at serve time, never stored
    log="egress.jsonl",
)
program = dspy.load("artifacts/agent", bindings={"lm": {...}},
                    envelope=dspy.Envelope("sandbox", broker=broker))

# A receiver who has read the code LOWERS a floor — a recorded edit.
program = dspy.retrust(program, "weather", floor="fork", reason="reviewed 2026-08")
# (retrust refuses to RAISE — raising is dspy.Envelope, not an edit.)

# Run one arbitrary callable once, confined, without an artifact at all.
total = dspy.confined(sum, [1, 2, 3], isolation="fork_ratchet")
```

Two limits worth stating in prose. **Broker credential injection is
plain-HTTP only**: an HTTPS request goes through `CONNECT` as an opaque
TLS tunnel the broker cannot rewrite, so a real `https://` endpoint
reached by CONNECT gets no injected header — injection is for the
localhost/plain-HTTP path. **`confined` needs a fork context** (the
ratchet's `unshare(CLONE_NEWUSER)` wants a single-threaded fork); on
Python 3.14 it uses an explicit `fork` start method, and the isolation
backend refuses loudly if the host cannot build the requested level.

## Three postures

**1. Paranoid (the default).** No knobs. Generated code is pure stdlib,
dep-free, and the artifact refuses to run it in-process without a grant.
Use this when you will ship the artifact to someone else, or when you
have not read the generated code.

```python
FlexIR(reflection_lm, metric, holdout=holdout)
```

**2. My own loop, my own machine.** You will read what ships; you want
the loop frictionless.

```python
FlexIR(reflection_lm, metric, holdout=holdout, isolation_floor="none")
```

**3. Full power in a disposable box.** You run the whole optimization
inside a container/VM with nothing to steal and nothing to break. Then
the AST filter is redundant belt-and-suspenders and the interesting
ceiling is the solution space, not the exposure. Open the doors:

```python
FlexIR(
    reflection_lm, metric, holdout=holdout,
    isolation_floor="none",
    extra_imports=frozenset({"numpy", "sklearn", "torch", "transformers", "xgboost"}),
    allowed_deps=frozenset({"numpy", "scikit-learn", "torch", "transformers", "xgboost"}),
)
```

This is the honest framing: isolation by *environment* (a real boundary)
beats isolation by *filter* (a speed bump). If you have the real
boundary, buy the power.

## What full power buys: LM calls become models

With deps and imports open, `replace_predict_with_code` is no longer
limited to string transforms. A predict leaf whose job is really
classification or extraction can be replaced by:

- **Feature engineering + classical ML**: a `sklearn` pipeline
  (TF-IDF + logistic regression) trained offline, its fitted state baked
  as data. Thousands of times cheaper per call than an LM, and often
  more accurate on narrow, well-labeled tasks.
- **A small fine-tuned model**: a `transformers` checkpoint specialized
  on the task, called in-process. The spec's weights slot already
  carries safetensors sidecars; a tool leaf can load and run one.
- **Plain deterministic code**: still the common case — parsing,
  lookup, arithmetic. The default posture already covers this.

The acceptance rule prices all of these the same way: dev score must
hold or rise, `lm_calls` measures the cheapness win, and the holdout
gate catches a replacement that memorized the devset. A swap from
"prompt an LM" to "run a specialized model" is just the unified-leaf law
doing its job: same signature, different implementation.

## Should Flex also write adapters, templates, and strategies?

Yes eventually, and the IR already points at the seam: adapters, custom
LM classes, and strategies are authored components in the manifest
exactly like tools — identity + source + deps. Nothing structural stops
`rewrite_adapter` or `set_strategy` ops from joining the vocabulary
under the same admission-and-holdout discipline.

But sequence matters. The layers form a hierarchy of blast radius:

1. **Instruction/demo edits** — cheap, local, reversible; move the score
   in small steps.
2. **Structural edits** (decompose, insert code, reroute) — reset the
   instruction optimum: after you split a step in two, the old
   instructions are stale.
3. **Adapter/template edits** — global: one adapter change shifts every
   predictor's rendering at once, invalidating instruction *and*
   structure conclusions below it.

A change at level N invalidates tuning at levels < N. That argues for a
coarse-to-fine schedule with re-tuning after every coarse move, not a
flat vocabulary where the proposer picks randomly among levels.

### What the literature says

The problem — allocating trials across edit families with different
costs and blast radii — is old and has good answers:

- **Alternating/coordinate optimization**: optimize one layer while
  freezing others, rotate. DSPy's own BetterTogether (prompts ↔ weights)
  showed the alternation beats either alone. The classic failure mode is
  ping-ponging; the fix is accepting a rotation only on joint
  improvement — which FlexIR's single scoreboard (score, lm_calls,
  holdout) already enforces.
- **Adaptive operator selection** (evolutionary computation): treat each
  edit family as an arm of a bandit; allocate proposals by *recent
  marginal gain per unit cost*, not uniformly. UCB-style credit
  assignment over operators is standard and directly applicable: the
  trajectory already records which op family each accepted candidate
  carried.
- **Neural architecture search**: the two-timescale lesson — structure
  moves on a slow clock, parameters on a fast clock, and evaluating a
  structure fairly requires re-tuning its parameters first. Translated:
  after a `rewrite_forward` accept, spend a few rounds on
  instruction/demo edits before judging further structural moves.
- **FunSearch / AlphaEvolve** (program evolution with LLM proposers):
  the strongest known signal for "LLM proposes code, evaluator gates" —
  and their consistent finding is that the *evaluator and the diversity
  of retained candidates* matter more than proposer cleverness. FlexIR's
  checkpoint trajectory is a lineage; keeping a small pool of diverse
  ancestors to seed proposals from (rather than a single champion) is
  the cheapest upgrade in that direction.
- **Hyperband / successive halving**: score cheap and shallow first,
  spend full evaluations only on survivors — relevant once devsets get
  big enough that full evaluation per candidate dominates cost.

### The design this implies

A **scheduler above the vocabulary**, not a bigger vocabulary alone:

1. Rounds are typed: a *fine* round offers only data ops; a *coarse*
   round offers structural ops; an *adapter* round (future) offers
   adapter ops. The reflection prompt shows only that round's catalog.
2. Rotation is earned: stay fine while fine edits keep paying; escalate
   to coarse when fine gains flatten (the plateau is measurable from the
   trajectory); after a coarse accept, drop back to fine to re-tune.
3. Credit is tracked: per op family, accepted-gain per reflection call —
   a bandit over rounds. "What climbs fastest" is then measured, not
   guessed.
4. Every level keeps the same gates: admission, apply-on-copy, holdout,
   cheapness. The scheduler changes *what is proposable when*, never
   what is acceptable.

That is the answer to "the right way at the right time": the acceptance
rule is already level-agnostic; the missing piece is a plateau-driven,
credit-tracking rotation — coordinate ascent with a bandit on top.
