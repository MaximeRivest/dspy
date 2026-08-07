"""Ticket assistant — signed-modules exemplar, no ambient anything.

Close to dspy-native (the class form of exemplar-program.py), with two
deliberate changes:

1. CUSTOM MODULES DECLARE SIGNATURES. A module's external contract is a
   Signature class on the module, exactly as Predict declares one. The
   compiler checks forward against it: parameters = declared inputs,
   every return path = declared outputs, no more, no less (PIR-013
   becomes checkable; sub-module leaves become typed like every other
   leaf; interface-preserving lowerings get their per-node seam).

2. NO AMBIENT SETTINGS SYNTAX. There is no dspy.configure and no
   dspy.context anywhere. Every predictor carries an explicit lm and
   adapter binding; export refuses an unbound predictor loudly, naming
   it — `settings.X or Default()` does not exist in this dialect.
"""

import dspy


# ---------------------------------------------------------------------------
# Tools — unchanged: plain, self-contained functions.

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
    """Pull ORD-xxxxxx ids out of free text. Stdlib only."""
    import re

    return re.findall(r"ORD-\d{6}", text)


# ---------------------------------------------------------------------------
# Custom LM with baked weights — unchanged.

class TinyTriageLM(dspy.BaseLM):
    """A 321M local model for cheap classification; weights ship inside."""

    def __init__(self, model_dir: str = "PleIAs/Baguettotron", device: str = "cpu"):
        # deps: torch, transformers, safetensors
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model=model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
        self.device = device

    def forward(self, request):
        import torch

        text = self.tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True
        )
        ids = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=request.max_tokens or 256)
        return dspy.LMResponse(
            text=self.tokenizer.decode(
                out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
            )
        )


# ---------------------------------------------------------------------------
# Signatures — predictor-level (prompted) AND module-level (contractual).
# A module signature's docstring describes the contract; it is never
# rendered into a prompt — instructions belong to predictors.

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


class Policy(dspy.Signature):
    """Is a drafted reply allowed to go out?"""

    reply: str = dspy.InputField()
    account_tier: str = dspy.InputField()
    compliant: bool = dspy.OutputField()
    violation: str = dspy.OutputField()


class Assistant(dspy.Signature):
    """Resolve a support ticket end to end: triage, retrieve, draft, verify."""

    ticket: str = dspy.InputField()
    customer_id: str = dspy.InputField()
    reply: str = dspy.OutputField()
    quotes: dspy.Citations = dspy.OutputField()


# ---------------------------------------------------------------------------
# Modules — each declares its signature; forward is checked against it.

class PolicyCheck(dspy.Module):
    signature = Policy

    def __init__(self):
        self.assess = dspy.Predict(Policy)

    def forward(self, reply, account_tier):
        return self.assess(reply=reply, account_tier=account_tier)


class TicketAssistant(dspy.Module):
    signature = Assistant

    def __init__(self):
        self.triage = dspy.Predict(Triage)
        self.draft = dspy.Predict(DraftReply)
        self.policy = PolicyCheck()
        self.investigate = dspy.ReAct(
            "ticket, context -> summary",
            tools=[kb_search, fetch_account],
            max_iters=4,
        )
        self.py = dspy.PythonInterpreter(allow=["round"])
        self.sh = dspy.BashInterpreter(allow=["grep"])

        self.fetch_account = fetch_account
        self.extract_order_ids = extract_order_ids
        self.actions = {"lookup_account": fetch_account, "search_kb": kb_search}

    def forward(self, ticket, customer_id):
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
            # Must satisfy `signature = Assistant` exactly: reply + quotes,
            # no extra fields — the compiler enforces it on every return path.
            return dspy.Prediction(reply="Escalating to a human agent.", quotes=[])
        return dspy.Prediction(reply=approved.reply, quotes=approved.quotes)


# ---------------------------------------------------------------------------
# Metric — unchanged.

def quality(example, prediction) -> float:
    score = 0.0
    if example.must_mention in prediction.reply:
        score = score + 0.5
    if len(prediction.quotes) > 0:
        score = score + 0.5
    return score


# ---------------------------------------------------------------------------
# Wiring — fully explicit. No configure, no context, no defaults: every
# predictor states its lm and adapter; export refuses anything unbound,
# naming the predictor path.

if __name__ == "__main__":
    router = dspy.LM("openai/gpt-oss-120b", api_base="https://gw.internal/v1")
    writer = dspy.LM("anthropic/claude-sonnet-5")
    tiny = TinyTriageLM()
    json_a = dspy.JSONAdapter()
    chat_a = dspy.ChatAdapter()

    program = TicketAssistant()

    program.triage.set_lm(tiny)
    program.triage.set_adapter(json_a)

    program.policy.assess.set_lm(tiny)
    program.policy.assess.set_adapter(json_a)

    program.draft.set_lm(writer)
    program.draft.set_adapter(chat_a)

    program.investigate.set_lm(router)       # propagates to ReAct's internal
    program.investigate.set_adapter(json_a)  # react + extract predictors

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

    dspy.export(program, "ticket_assistant.ir", metric=quality, devset=devset)
