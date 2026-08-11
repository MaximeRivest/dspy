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
    code_trust="isolated",    # "isolated" | "in_process"
    extra_imports=None,       # widen the stdlib import allowlist
    allowed_deps=None,        # permit named pip packages in # deps:
    auto_install=False,       # uv pip install granted deps into THIS env
    eval_mode="in_process",   # "in_process" | "artifact" (score as artifact)
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

**`code_trust`** — where authored code runs.

- `"isolated"` (default): authored tool leaves are stamped
  `isolation_required`. Loading the artifact fails closed unless the
  receiver explicitly grants the leaf. Nothing runs in your process
  without a deliberate act.
- `"in_process"`: authored leaves get the ordinary in-process placement.
  Your own loop, and anyone who loads your artifact, runs the generated
  code with **full ambient authority** — your filesystem, your network,
  your credentials. The `authored_by: "optimizer"` provenance stamp
  survives either way, so a receiver can always audit or re-place the
  leaf before trusting it.

**`extra_imports`** — what authored code may `import`.

The default allowlist is a closed pure-stdlib set (`re`, `json`, `math`,
`statistics`, `itertools`, ...). `extra_imports=frozenset({"numpy",
"sklearn"})` widens it for this optimizer instance only. The
builtin/dunder denylist (`eval`, `__import__`, `__globals__`, ...) stays
denied regardless — but understand what this is: an AST filter, not a
sandbox. Every module you add is a door you opened.

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
| `eval_mode` | "in_process" | Where candidates are scored — never what is accepted. `"artifact"` exports every candidate (baseline included) and scores it in a subprocess under the artifact's own environment: scoring semantics == deployment semantics, environment included. Requires a self-contained metric (its source travels to the child; refused at compile otherwise). Child environments are cached by lockfile hash under `<checkpoint_dir>/.envs` (or `~/.cache/dspy-flexir`), so only candidates that change deps pay a resolve. |
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

Current limits, stated plainly: artifact mode serializes scripted
`DummyLM` state to the child (replayable tests); binding a live provider
LM inside the child is not implemented yet — use `in_process` for live
providers. The `dspy` pin in the lock is overridden with the local tree
via `eval_env_overrides`.

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
FlexIR(reflection_lm, metric, holdout=holdout, code_trust="in_process")
```

**3. Full power in a disposable box.** You run the whole optimization
inside a container/VM with nothing to steal and nothing to break. Then
the AST filter is redundant belt-and-suspenders and the interesting
ceiling is the solution space, not the exposure. Open the doors:

```python
FlexIR(
    reflection_lm, metric, holdout=holdout,
    code_trust="in_process",
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
