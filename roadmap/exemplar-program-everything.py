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
# 2 · LEARNABLE LEAVES — one entry shape, a FAMILY of declared faces
#     (D-045, D-049). Everything around the call — identity, weights,
#     placement rungs, endpoint/credential refs, engine, training — is
#     IDENTICAL for every entry. The `contract` field declares the typed
#     face; admission is structural against it (D-036). lm15/chat is one
#     member of the family — the richest.
# ----------------------------------------------------------------------

# In-process trainable LM: weights bake at rung 0 (engine: transformers).
drafter_lm = dspy.LM("hf:PleIAs/Baguettotron", device="cuda")     # 8b bake; programir_weight_spec

# Hosted LM, declared tier: identity bakes, endpoint/credential bind.
polisher_lm = dspy.LM("openai-chat:gpt-4o-mini", native_fc=True)  # §e0-binding; credential = name only

# Model entry (D-049 v2): exactly as dumb as an LM — weights + engine +
# contract + placement + the four training verbs. NO objective inside:
# identity is the pool entry's; training statements live in §6 (b-pools:
# what training means is bindings + tags, never entry-internal).
# Embed face: no messages, no sampling config, no finish_reason — a
# smaller honest contract, NOT chat with meaningless fields.
embedder_model = dspy.Model("hf:BAAI/bge-small-en-v1.5",
                            contract="embed-1")                   # embed(text|media) -> vector

# Authored R entry, LEARNABLE (D-049 + D-022/D-025/D-040/D-042): a
# fitted mgcv GAM. Origin authored, language tag, renv.lock env block; a
# Python engine rung-walks it to a sidecar; rows cross as sealed Arrow.
# The learnable state is the FITTED MODEL OBJECT; the training contract
# is the batch-fit face (`fit-1`: fit(dataset) -> state), not the
# gradient verbs — classic ML fits, it does not step. State serializes
# via a DECLARED format (RDS — outside the bit-for-bit safetensors
# story, and the manifest says so: the serialization floor, honest).
priority_model = dspy.Model.authored(
    source=r'''
    # deps: mgcv                                                  # inline deps -> renv.lock env block
    library(mgcv)

    train <- function(data) {                       # train/fit-1: fit(dataset) -> state
      gam(priority ~ s(question_len) + s(urgency_score) + customer_tier,
          data = data, family = betar())
    }

    predict_fn <- function(state, question, customer_tier) {  # predict-1 face
      newdata <- featurize(question, customer_tier)
      as.numeric(predict(state, newdata, type = "response"))
    }

    featurize <- function(question, customer_tier) {
      data.frame(
        question_len  = nchar(question),
        urgency_score = as.numeric(grepl("urgent|outage|down|asap",
                                         tolower(question))),
        customer_tier = customer_tier)
    }
    ''',                                                          # baked source, authored_by: human
    language="r",                                                 # D-025 language axis
    effects="pure",                                               # §d effects declaration
    isolation_floor="fork",                                       # D-042 floor bakes; envelope binds
    contract="predict-1",                                         # the plain typed-function face
    training=dspy.Training(contract="fit-1",
                           state_format="rds"),                   # declared, captured-or-refused
)

# Promptable vision model (the SAM case): the request contract is richer
# than tensor-in (point/box/text prompts -> masks), so it binds a REAL
# adapter — spatial prompts rendered from signature fields — and that
# adapter's prompt axis (prompt-point strategy, text-prompt literals)
# opens to optimization exactly like a chat adapter's. Promptability is
# a gradient: embedder (identity) < SAM (spatial) < LM (chat).
sam_model = dspy.Model("hf:facebook/sam3",
                       contract="segment-1")                      # segment(media, prompts) -> masks
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
        self.locate = dspy.Predict(
            "screenshot @media, ui_query -> region: dspy.Mask",   # media role (component 2)
            model=sam_model,
            adapter=dspy.adapters.preset("sam_points"))           # spatial-prompt adapter; optimizable
        # The migration valve (polyfill→native, the roles-governance
        # loop): the SAME intent can bind a chat-face VLM with a textual
        # strategy instead of the native segment face —
        #   dspy.Predict(..., lm=vlm, adapter=dspy.adapters.preset(
        #       "json", strategies={"media": "textual_mask_coords"}))
        # — one binding swap, a recorded View-3 choice diff. When
        # providers converge on a native mask channel in the chat
        # contract, lm15 grows the part, roles grow in lockstep, and
        # segment-1 retires for that capability — a vocabulary version
        # event, not a redesign.

    def forward(self, inputs):                                    # inputs-bag envelope (D-041 upstream)
        pr = self.priority(question=inputs.question,
                           customer_tier=inputs.customer_tier)
        vec = self.embed(text=inputs.question)
        query = inputs.question
        if inputs.screenshot is not None:                         # ticket carries a UI screenshot
            region = self.locate(screenshot=inputs.screenshot,    # SAM leaf: spatial prompts
                                 ui_query=inputs.question)
            query = query + " ui:" + region.label                 # refine retrieval with the region
        passages = self.search(query=query, k=4)
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

# Axis 4: the R GAM re-fits on labeled priorities — the batch-fit
# training face (`fit-1`), executed where R lives (sidecar). The fitted
# state blob re-hashes; everything else stays shared (§c1). The
# objective lives HERE, as always — never inside the entry.
priority_labels = (
    dspy.Dataset.ref("DW_PRIORITY_LABELS", snapshot="2026-08-15T00:00Z")
    .select("question", "customer_tier", "priority"))
program = dspy.optim.Fit(
    target=program.priority, trainset=priority_labels,
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

# ----------------------------------------------------------------------
# 9 · THE SAME PROGRAM, MINIMAL SYNTAX — deduction in the frontend,
#     resolution in the manifest. Rule: declare-don't-discover governs
#     the ARTIFACT, not the syntax. The compiler may deduce anything
#     that is (a) deterministic, (b) baked as a resolved decision with
#     provenance (`explain` shows "contract: embed-1, resolved_from:
#     hf:pipeline_tag"), (c) a loud refusal on ambiguity. The user's
#     irreducible statements are exactly the INTENT set: signatures,
#     roles, objectives, purity claims, trust tightenings.
# ----------------------------------------------------------------------

# Pool entries — every mechanical fact resolves from identity/shape:
#   contract faces from HF metadata (pipeline_tag/architectures);
#   authored-vs-packaged from the call shape; language from the file
#   extension; fit-1 vs sgd-1 structurally (which training verbs the
#   engine declares); state_format defaulted per language; isolation
#   floor defaulted from the D-040/D-043 trust profile (authored_by).
min_drafter  = dspy.LM("hf:PleIAs/Baguettotron", device="cuda")
min_polisher = dspy.LM("openai-chat:gpt-4o-mini", native_fc=True)
min_embedder = dspy.Model("hf:BAAI/bge-small-en-v1.5")   # ⇒ embed-1
min_sam      = dspy.Model("hf:facebook/sam3")            # ⇒ segment-1
min_priority = dspy.Model.authored("leaves/priority_gam.R",  # ⇒ r, fit-1, rds
                                   effects="pure")           # purity: a CLAIM, never inferred


class MinAnswerer(dspy.Module):
    def __init__(self):
        # Adapters default from the resolved face: chat ⇒ preset chat,
        # embed/predict ⇒ identity, segment ⇒ the face's preset.
        # `adapter=` appears only to override.
        self.draft = dspy.Predict(
            "question, passages, priority -> answer, reasoning @reasoning",
            lm=min_drafter)
        self.polish = dspy.Predict(
            "question, draft_answer -> answer @citations",
            lm=min_polisher, adapter=dspy.adapters.preset("json"))  # explicit: overrides chat default
        self.search = dspy.Tool(search_policies)
        self.embed = dspy.Predict("text -> embedding: list[float]",
                                  model=min_embedder)
        self.priority = dspy.Predict("question, customer_tier -> priority: float",
                                     model=min_priority)
        self.locate = dspy.Predict("screenshot @media, ui_query -> region: dspy.Mask",
                                   model=min_sam)

    forward = Answerer.forward                                    # identical logic, verbatim


# Data: omitted revision/snapshot ⇒ resolved NOW and pinned at bake —
# the manifest always carries the pinned identity; syntax may say "current".
min_tickets = (dspy.Dataset.hf("acme/support-tickets")            # ⇒ revision pinned at bake
               .filter(col.language == "en")
               .mutate(question=col.subject + "\n" + col.body,
                       answer=col.resolved_answer)
               .select("question", "answer", "product", "customer_tier")
               .slice_sample(n=800, seed=7)
               .split(train=0.7, dev=0.3, seed=7)
               .with_inputs("question", "product", "customer_tier"))

# Objectives stay DECLARED (fixed point 2 — human-owned; suggestion ok,
# silent choice never). One trainer surface, contract-dispatched:
# Train ⇒ sgd-1 (four verbs; loss= required) or fit-1 (no loss) by the
# target entry's declared training contract.
min_metric = dspy.Metric(ticket_metric, judge=dspy.LM("claude-sonnet-4-5"),
                         devset=min_tickets.dev)
prog = MinAnswerer()
prog = dspy.optim.BootstrapFewShot(metric=min_metric).compile(prog, trainset=min_tickets.train)
prog = dspy.optim.Train(target=prog.draft, trainset=min_tickets.train,
                        loss="cross_entropy", lr=1e-4, rank=32).compile(prog)   # ⇒ sgd-1
prog = dspy.optim.Train(target=prog.embed, trainset=embed_pairs,
                        loss="contrastive").compile(prog)                       # ⇒ sgd-1
prog = dspy.optim.Train(target=prog.priority,
                        trainset=priority_labels).compile(prog)                 # ⇒ fit-1

dspy.export(prog, "artifacts/answerer.ir", metric=min_metric, devset=min_tickets.dev)
# Identical artifact class as §7: the terse syntax never produces an
# implicit manifest — minimal to write, explicit to read.
