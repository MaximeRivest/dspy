# /// script
# requires-python = ">=3.12"
# dependencies = ["dspy", "dpyr"]                       # component 9 — PEP 723; uv lock --script
# ///
"""Exemplar: every discussed part in one program (authoring surface only).

A support-ticket answering program that retrieves policy documents,
scores ticket priority with an R model, drafts and polishes answers with
two different LMs, is trained on three axes (prompts, LM weights,
embedder weights), scored by a baked metric + judge LM over a pinned
ETL'd devset, and exported as one artifact. Comment tags cite the
decision each line exercises. Not runnable today — this is the surface
the drafts D-045..D-049 aim at, over the ratified base.
"""

import dspy
from dspy import col  # the ETL expression proxy (D-047; dpyr chain)

# ----------------------------------------------------------------------
# 1 · DATA — dataset pool entries: pinned identity + ETL (D-046, D-047)
# ----------------------------------------------------------------------

tickets = (
    dspy.Dataset.hf("acme/support-tickets", revision="9c1ffe2")   # identity bakes: id + revision
    .filter(col.language == "en")                                 # closed predicate grammar (node-set expr)
    .mutate(question=col.subject + "\n" + col.body,               # dplyr names; restricted expressions
            answer=col.resolved_answer)
    .select("question", "answer", "product", "customer_tier")
    .slice_sample(n=800, seed=7)                                  # seeded — result_hash checkable
    .split(train=0.7, dev=0.3, seed=7)
    .with_inputs("question", "product", "customer_tier")          # designate_inputs → input_keys[]
)

# Referenced (not baked) warehouse dataset: location binds, identity pins.
# The same closed filter tree compiles to SQL at the binding — pushdown
# is placement, not syntax (D-047 outer rung; D-046 middle/outer rung).
embed_pairs = (
    dspy.Dataset.ref("DW_TICKET_PAIRS",                           # endpoint_ref — receiver binds
                     snapshot="2026-08-15T00:00Z")                # unpinned ⇒ unwarranted (D-046)
    .filter(col.label.is_in(["duplicate", "related"]))
    .select("text_a", "text_b", "label")
)

# ----------------------------------------------------------------------
# 2 · LEARNABLE LEAVES — LMs and model leaves (D-045, D-049)
# ----------------------------------------------------------------------

# In-process trainable LM: weights bake at rung 0 (engine: transformers).
drafter_lm = dspy.LM("hf:PleIAs/Baguettotron", device="cuda")     # 8b bake; programir_weight_spec

# Hosted LM, declared tier: identity bakes, endpoint/credential bind.
polisher_lm = dspy.LM("openai-chat:gpt-4o-mini", native_fc=True)  # §e0-binding; credential = name only

# Model entry (D-049 v2): exactly as dumb as an LM — weights + engine +
# contract + placement + the four training verbs. NO objective inside:
# identity is the pool entry's; training statements live in §6 (b-pools:
# what training means is bindings + tags, never entry-internal).
embedder_model = dspy.Model("hf:BAAI/bge-small-en-v1.5")          # weights_identity; safetensors bake

# Authored model entry, cross-language (D-049 + D-022/D-025/D-040/D-042):
# an R GAM. The analogue of an authored LM class (§e0-class): origin
# authored, language tag, renv.lock env block; a Python engine rung-walks
# it to a sidecar; rows cross as sealed Arrow. Trainable where R lives
# (the four verbs over the wire), and its BODY is `authored-code` —
# openable to D-013 seed regimes like any authored leaf.
priority_model = dspy.Model.authored(
    path="leaves/priority_gam.R",                                 # captured source, authored_by: human
    language="r",                                                 # D-025 language axis
    effects="pure",                                               # §d effects declaration
    isolation_floor="fork",                                       # D-042 floor bakes; envelope binds
)
# Pool entries end here. The LEAVES that bind them are module state —
# declared in __init__ with self.…, so each gets a named predictor path
# in the module tree (per-predictor state, bindings, View-2/3 addressing).

# ----------------------------------------------------------------------
# 3 · TOOLS + INTERPRETER (components 6, 7)
# ----------------------------------------------------------------------

def search_policies(query: str, k: int = 4) -> list[str]:
    # deps: httpx                                                 # inline deps → unioned PEP 723 block
    """Look up policy passages near the query embedding."""
    ...

# ----------------------------------------------------------------------
# 4 · THE PROGRAM — restricted forward over typed leaves (§d node-set)
# ----------------------------------------------------------------------

class Answerer(dspy.Module):
    def __init__(self):
        self.draft = dspy.Predict(
            "question, passages, priority -> answer: str, "
            "reasoning: str @reasoning",                          # role marker (D-011); intent, not exhaust
            lm=drafter_lm,                                        # explicit binding — no ambient settings
            adapter=dspy.adapters.preset("chat"),                 # adapter = preset entry (D-018)
        )
        self.polish = dspy.Predict(
            "question, draft_answer -> answer: str @citations",
            lm=polisher_lm,
            adapter=dspy.adapters.preset("json"),
        )
        self.search = dspy.Tool(search_policies)
        # Non-LM leaves: the SAME record — signature + adapter + model
        # binding. The embedder binds the IDENTITY adapter (degenerate
        # case); a promptable vision model (SAM: points/boxes/text ->
        # masks) would bind a spatial-prompt adapter whose prompt axis
        # opens to optimization exactly like an LM's — the model's
        # request contract decides which axes exist.
        self.embed = dspy.Predict("text -> embedding: list[float]",
                                  model=embedder_model,
                                  adapter=dspy.adapters.identity())
        self.priority = dspy.Predict(
            "question, customer_tier -> priority: float",
            model=priority_model,
            adapter=dspy.adapters.identity())

    def forward(self, inputs):                                    # inputs-bag envelope (D-041 upstream)
        pr = self.priority(question=inputs.question,
                           customer_tier=inputs.customer_tier)
        vec = self.embed(text=inputs.question)
        passages = self.search(query=inputs.question, k=4)
        draft = self.draft(question=inputs.question,
                           passages=passages, priority=pr.priority)
        if pr.priority > 0.8:                                     # data-dependent branch — visible AST node
            return self.polish(question=inputs.question,
                               draft_answer=draft.answer)
        return draft

program = Answerer()

# ----------------------------------------------------------------------
# 5 · METRIC — authored harness + judge LM, over the pinned devset (c.12)
# ----------------------------------------------------------------------

judge_lm = dspy.LM("claude-sonnet-4-5")                           # metric's judge: declared LM-ladder entry

def ticket_metric(example, prediction) -> float:
    # deps: rapidfuzz
    """0.6 exactness (fuzzy) + 0.4 groundedness (judge)."""
    ...

metric = dspy.Metric(ticket_metric, judge=judge_lm,
                     devset=tickets.dev)                          # score = claim about dataset_identity

# ----------------------------------------------------------------------
# 6 · OPTIMIZATION — three axes, one trajectory (checkpoint = save)
# ----------------------------------------------------------------------

# Axis 1: prompts/demos (existing optimizers).
program = dspy.optim.BootstrapFewShot(metric=metric).compile(
    program, trainset=tickets.train)

# Axis 2: LM weights — four-verb contract (D-045). The drafter trains
# in-process (rung 0); switching to a hosted trainer (Tinker-class) is a
# binding, not a code change:
#   dspy.configure(bindings={"TRAINER_ENDPOINT": ..., "TRAINER_API_KEY": ...})
program = dspy.optim.WeightTune(
    metric=metric, loss="cross_entropy", lr=1e-4, rank=32,        # rank>0 ⇒ LoRA delta (base ⊕ delta)
).compile(program, trainset=tickets.train)

# Axis 3: the embedder entry trains — the objective (dataset + loss) is
# stated HERE, in the training statement, referencing a dataset-pool
# entry; it is never a property of the model entry itself. Leaf-local
# means; the program metric stays the untouchable end (§e2, D-013 seeds).
program = dspy.optim.WeightTune(
    target=program.embed, trainset=embed_pairs, loss="contrastive",
).compile(program)

# ----------------------------------------------------------------------
# 7 · EXPORT — one artifact, everything above inside it (D-039)
# ----------------------------------------------------------------------

dspy.export(program, "artifacts/answerer.ir",
            metric=metric, devset=tickets.dev)
# The directory holds: manifest (tree + bindings + pools + versions),
# weights/ (drafter base + LoRA delta, embedder), leaves/priority_gam.R
# + renv.lock env block, tools/, ETL pipelines + dataset identities +
# result hashes, metric source + judge identity, PEP 723 + lock.
# Credentials and endpoints: names only. Receiver: uv sync → link →
# verify → run; deviations recorded, re-scored by the baked metric.

# ----------------------------------------------------------------------
# 8 · OUTER LOOP — objective evolution, separately owned (D-048)
# ----------------------------------------------------------------------

# A second program, run on a schedule — never inside the inner loop:
#   proposals = dspy.outer.propose(
#       runlog=dspy.RunLog.ref("PROD_TRAFFIC"),                   # outside the artifact (View-2 store)
#       feedback=dspy.Dataset.ref("SUPPORT_CSAT"),
#       objective=metric,
#       meta_metric=dspy.agreement_with_human_labels)             # human-owned fixed point
# Each accepted proposal = new devset hash / metric version = an
# objective-boundary node in the trajectory; historic scores detach and
# kept checkpoints re-score under the new objective.
