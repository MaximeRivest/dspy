"""Ticket assistant — the exemplar program for ProgramIR representability.

Rev 4: plain functions as tools — deps declared by comment inside the
function body, right where the import happens; no decorator. LM/adapter
wiring in current dspy style, presuming set_adapter exists alongside
set_lm. Adds LLM-driven tool use (dspy.ReAct over the same tool pool),
the baked in-process Python interpreter with a declared scope, a second
interpreter in another language (bash, subprocess-floored), and a custom
authored LM whose weights BAKE INTO the artifact (rung 0) — shared by two
predictors, so the artifact carries one weights blob with two bindings.
"""
# 3 news things: the deps comments, set_adapter, and dspy.export(...)
import dspy


# ---------------------------------------------------------------------------
# Tools — leaves. Plain functions, fully self-contained: identity from the
# type hints, deps from the `# deps:` comment in the body, imports local,
# source baked. No declaration ceremony beyond the comment.

def fetch_account(customer_id: str) -> dict:
    """Look up a customer account in the billing API."""
    # deps: httpx
    import httpx

    r = httpx.get(f"https://billing.internal/api/accounts/{customer_id}", timeout=5)
    r.raise_for_status()
    return r.json()


def kb_search(query: str, k: int = 3) -> list[str]:
    """Search the internal knowledge base, best-k passages."""
    # deps: httpx
    import httpx

    hits = httpx.get("https://kb.internal/search", params={"q": query, "k": k}).json()
    return [h["text"] for h in hits["results"]]


def extract_order_ids(text: str) -> list[str]:
    """Pull ORD-xxxxxx ids out of free text. Stdlib only — nothing to declare."""
    import re

    return re.findall(r"ORD-\d{6}", text)


# ---------------------------------------------------------------------------
# A custom LM that OWNS its weights — authored class, baked at export:
# the class source travels in the artifact (origin: authored, language:
# python, deps from the comment), the weights travel as safetensors
# (component 8b: model.safetensors + rebuild_config + tokenizer +
# tying.json + device.json), engine: transformers — training-capable, so
# the weights themselves are an optimizable field of the shipped program.

class TinyTriageLM(dspy.BaseLM):
    """A 321M local model for cheap classification; weights ship inside."""

    def __init__(self, model_dir: str = "PleIAs/Baguettotron", device: str = "cpu"):
        # deps: torch, transformers, safetensors
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model=model_dir)
        self.weights_identity = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        self.device = device

    def programir_weight_spec(self):
        return {
            "model_attribute": "model",
            "tokenizer_attribute": "tokenizer",
            "weights_identity": self.weights_identity,
            "engine": "transformers",
            "device": self.device,
            "frozen": False,
            "weight_ref": "base",
            "ties": [
                {"target": "lm_head.weight", "source": "model.embed_tokens.weight"}
            ],
        }

    def forward(self, request):
        # typed_lm contract: forward(LMRequest) -> LMResponse. Leaf code —
        # full Python, torch, `with` blocks: none of this is whitelist
        # territory.
        import torch

        text = self.tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **ids, max_new_tokens=request.max_tokens or 256
            )
        completion = self.tokenizer.decode(
            out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return dspy.LMResponse(text=completion)


# ---------------------------------------------------------------------------
# Signatures

class Triage(dspy.Signature):
    """Classify a support ticket and plan retrieval actions."""

    ticket: str = dspy.InputField()
    category: str = dspy.OutputField(desc="one of: billing, technical, account, other")
    urgency: int = dspy.OutputField(desc="1 (low) to 5 (page someone)")
    actions: list[dict] = dspy.OutputField(
        desc='retrieval steps, each {"name": tool, "args": {kwargs}}'
    )


class DraftReply(dspy.Signature):
    """Write the reply. Quote the KB passages you relied on."""

    ticket: str = dspy.InputField()
    findings: dict = dspy.InputField(desc="everything gathered about this ticket")
    reply: str = dspy.OutputField()
    quotes: dspy.Citations = dspy.OutputField()


# ---------------------------------------------------------------------------
# Modules

class PolicyCheck(dspy.Module):
    """Nested module: is a drafted reply allowed to go out?"""

    def __init__(self):
        self.assess = dspy.Predict(
            "reply, account_tier: str -> compliant: bool, violation: str"
        )

    def forward(self, reply, account_tier):
        return self.assess(reply=reply, account_tier=account_tier)


class TicketAssistant(dspy.Module):
    def __init__(self):
        self.triage = dspy.Predict(Triage)
        self.draft = dspy.Predict(DraftReply)
        self.policy = PolicyCheck()

        # LLM-driven tool use: the model decides which tool to call and with
        # what args, loop + dispatch inside the shipped module.
        self.investigate = dspy.ReAct(
            "ticket, context -> summary",
            tools=[kb_search, fetch_account],
            max_iters=4,
        )

        # Baked in-process interpreter; scope declares exactly the names
        # generated code may touch — nothing else exists in its namespace.
        self.py = dspy.PythonInterpreter(scope={"round": round})

        # Second interpreter, different language: bash. `allow` scopes the
        # command surface; bash never runs in-process — its isolation floor
        # is a subprocess with an empty environment.
        self.sh = dspy.BashInterpreter(allow=["grep"])

        self.fetch_account = fetch_account
        self.extract_order_ids = extract_order_ids
        self.actions = {"lookup_account": fetch_account, "search_kb": kb_search}

    def forward(self, ticket: str, customer_id: str):
        t = self.triage(ticket=ticket)

        findings = {"category": t.category}

        account = self.fetch_account(customer_id=customer_id)
        findings["tier"] = account["tier"]

        orders = self.extract_order_ids(text=ticket)
        if t.category == "billing" and len(orders) > 0:
            findings["orders"] = orders
            code = f"result = round({account['open_balance']} * 0.10, 2)"
            findings["refund_cap"] = self.py(code=code)
            findings["refund_mentions"] = self.sh(
                code=f"grep -c '{orders[0]}' /var/log/refunds.log"
            )

        for step in t.actions:
            name = step["name"]
            if name in self.actions:
                findings[name] = self.actions[name](**step["args"])

        if t.urgency >= 4:
            deep = self.investigate(ticket=ticket, context=findings)
            findings["deep_dive"] = deep.summary

        attempts = 0
        approved = None
        while attempts < 3:
            d = self.draft(ticket=ticket, findings=findings)
            check = self.policy(reply=d.reply, account_tier=findings["tier"])
            if check.compliant:
                approved = d
                break
            findings["violation"] = check.violation
            attempts = attempts + 1

        if approved is None:
            return dspy.Prediction(reply="Escalating to a human agent.", quotes=[])
        return approved


# ---------------------------------------------------------------------------
# Metric — component 12: leaf code (full Python), travels with the artifact.

def quality(example, prediction) -> float:
    score = 0.0
    if example.must_mention in prediction.reply:
        score = score + 0.5
    if len(prediction.quotes) > 0:
        score = score + 0.5
    return score


# ---------------------------------------------------------------------------
# Wiring — current dspy style throughout. Keys come from the environment
# (litellm convention); export converts them into declared credential_ref
# names, never values.

if __name__ == "__main__":
    router = dspy.LM("openai/gpt-oss-120b", api_base="https://gw.internal/v1")
    writer = dspy.LM("anthropic/claude-sonnet-5")
    tiny = TinyTriageLM()   # weights load here; export bakes them

    dspy.configure(lm=router, adapter=dspy.JSONAdapter())

    program = TicketAssistant()

    # Three LMs, three rungs: tiny baked in-process, router over the
    # gateway (ambient — ReAct uses it), writer at a hosted provider.
    # triage and policy SHARE the tiny entry: one weights blob, two
    # bindings — a joint fine-tune moves both; a per-binding LoRA delta
    # specializes one without duplicating 1.3 GB.
    program.triage.set_lm(tiny)
    program.policy.assess.set_lm(tiny)
    program.draft.set_lm(writer)
    program.draft.set_adapter(dspy.ChatAdapter())   # presumed: set_lm's twin

    program.draft.demos = [
        dspy.Example(
            ticket="I was double-charged on ORD-482113.",
            findings={"category": "billing", "tier": "pro"},
            reply="I can confirm the duplicate charge on ORD-482113 was reversed...",
            quotes=["Refunds for duplicate charges post within 3-5 business days."],
        ).with_inputs("ticket", "findings")
    ]

    devset = [
        dspy.Example(
            ticket="Cancel my subscription, nothing works.",
            customer_id="C-99120",
            must_mention="cancel",
        ).with_inputs("ticket", "customer_id"),
    ]

    trainset = [
        dspy.Example(
            ticket="Where is my refund for ORD-102938? It's been two weeks.",
            customer_id="C-55801",
            must_mention="refund",
        ).with_inputs("ticket", "customer_id"),
        # ...a few hundred of these in practice
    ]

    # Train the baked LM. tiny's engine is `transformers`, so its weights
    # are a live optimizable field (weight-ref): BootstrapFinetune runs the
    # program over the trainset, keeps the traces the metric scores well,
    # and takes gradient steps on the in-process module — JOINTLY for
    # triage and policy.assess, because they share the one pool entry.
    # Every step is a full checkpoint (checkpoint = save); the store shares
    # every blob except the weights, so N steps cost 1.3 GB + kilobytes.
    tuner = dspy.BootstrapFinetune(metric=quality, epochs=2, lr=1e-5)
    program = tuner.compile(program, trainset=trainset)

    # To specialize policy WITHOUT moving triage's shared base:
    #   dspy.BootstrapFinetune(metric=quality, delta="lora", rank=4)
    # → base stays frozen (hash unchanged), a ~2.4 MB delta rides
    # policy.assess's binding; triage is untouched.

    dspy.export(program, "ticket_assistant.ir", metric=quality, devset=devset)
