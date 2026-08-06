"""Ticket assistant — function-form exemplar (sibling of exemplar-program.py).

Same program, no classes for modules and no `self.`: a module is a plain
decorated function. The decorator deduces the init — it parses the body,
resolves every free name against the enclosing namespace at decoration
time, and every name that holds a leaf (Predict, decorated module, tool
function, interpreter, dict-of-leaves) becomes a declared child; its
variable name becomes its tree name. The function's own parameters and
return annotation are the module's external signature.
"""

import dspy


# ---------------------------------------------------------------------------
# Tools — unchanged from the class version: plain, self-contained functions.

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
# Custom LM with baked weights — unchanged: an LM is legitimately a class
# (it is an engine, not a module).

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
        completion = self.tokenizer.decode(
            out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return dspy.LMResponse(text=completion)


# ---------------------------------------------------------------------------
# Signatures — unchanged.

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
# Leaves — declared once at module level, used by bare name below.

triage = dspy.Predict(Triage)
draft = dspy.Predict(DraftReply)
assess = dspy.Predict("reply, account_tier: str -> compliant: bool, violation: str")

investigate = dspy.ReAct(
    "ticket, context -> summary",
    tools=[kb_search, fetch_account],
    max_iters=4,
)

py = dspy.PythonInterpreter(scope={"round": round})
sh = dspy.BashInterpreter(allow=["grep"])

actions = {"lookup_account": fetch_account, "search_kb": kb_search}


# ---------------------------------------------------------------------------
# Modules — plain functions. The decorator deduces the init from the body.

@dspy.module
def policy_check(reply: str, account_tier: str):
    """Nested module: is a drafted reply allowed to go out?"""
    return assess(reply=reply, account_tier=account_tier)


@dspy.module
def ticket_assistant(ticket: str, customer_id: str) -> DraftReply:
    t = triage(ticket=ticket)

    findings = {"category": t.category}

    account = fetch_account(customer_id=customer_id)
    findings["tier"] = account["tier"]

    orders = extract_order_ids(text=ticket)
    if t.category == "billing" and len(orders) > 0:
        findings["orders"] = orders
        code = f"result = round({account['open_balance']} * 0.10, 2)"
        findings["refund_cap"] = py(code=code)
        findings["refund_mentions"] = sh(
            code=f"grep -c '{orders[0]}' /var/log/refunds.log"
        )

    for step in t.actions:
        name = step["name"]
        if name in actions:
            findings[name] = actions[name](**step["args"])

    if t.urgency >= 4:
        deep = investigate(ticket=ticket, context=findings)
        findings["deep_dive"] = deep.summary

    attempts = 0
    approved = None
    while attempts < 3:
        d = draft(ticket=ticket, findings=findings)
        check = policy_check(reply=d.reply, account_tier=findings["tier"])
        if check.compliant:
            approved = d
            break
        findings["violation"] = check.violation
        attempts = attempts + 1

    if approved is None:
        return dspy.Prediction(reply="Escalating to a human agent.", quotes=[])
    return approved


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
# Wiring — set_lm/set_adapter directly on the leaf objects; no paths.

if __name__ == "__main__":
    router = dspy.LM("openai/gpt-oss-120b", api_base="https://gw.internal/v1")
    writer = dspy.LM("anthropic/claude-sonnet-5")
    tiny = TinyTriageLM()

    dspy.configure(lm=router, adapter=dspy.JSONAdapter())

    triage.set_lm(tiny)
    assess.set_lm(tiny)
    draft.set_lm(writer)
    draft.set_adapter(dspy.ChatAdapter())

    draft.demos = [
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

    dspy.export(ticket_assistant, "ticket_assistant.ir", metric=quality, devset=devset)
