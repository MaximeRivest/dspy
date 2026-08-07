"""Ticket assistant — flow-form exemplar (keras/dplyr/ggplot/pyspark style).

Same program, third Python surface: the module is BUILT BY COMPOSITION,
never lifted from source. Verbs chain like dplyr/pyspark; conditions are
column expressions (pyspark's `F.col` idiom: `&`, `~`, `>=`, `.isin`
overload to IR expression nodes); control flow is combinators (`when`,
`foreach`, `until`) — the If/For/While nodes authored declaratively.

The deep property of this surface: THERE IS NO COMPILER. The imperative
exemplars need ast.parse + the whitelist; every verb here constructs the
core tree directly, so `export` is nearly the identity function — and the
identical builder API can exist in any language with operator overloading
or method chaining. This is the Spark/DuckDB stance made literal.
"""

import dspy

c = dspy.col  # column references into the flow's state


# ---------------------------------------------------------------------------
# Tools, custom LM, signatures — identical to exemplar-program.py rev 6.

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
# Leaves — declared once, as in the function-form exemplar.

triage = dspy.Predict(Triage)
draft = dspy.Predict(DraftReply)
assess = dspy.Predict("reply, account_tier: str -> compliant: bool, violation: str")
investigate = dspy.ReAct(
    "ticket, context -> summary", tools=[kb_search, fetch_account], max_iters=4
)
py = dspy.PythonInterpreter(allow=["round"])
sh = dspy.BashInterpreter(allow=["grep"])
actions = {"lookup_account": fetch_account, "search_kb": kb_search}


# ---------------------------------------------------------------------------
# Modules — built by composition. Verbs add named values to the flow state;
# combinators are the control-flow nodes; `.returns` is the signature.

policy_check = (
    dspy.flow("policy_check", inputs=["reply", "account_tier"])
    .predict("check", assess, reply=c.reply, account_tier=c.account_tier)
    .returns(compliant=c.check.compliant, violation=c.check.violation)
)

ticket_assistant = (
    dspy.flow("ticket_assistant", inputs=["ticket", "customer_id"])
    .predict("t", triage, ticket=c.ticket)
    .call("account", fetch_account, customer_id=c.customer_id)
    .call("orders", extract_order_ids, text=c.ticket)
    .derive(findings=dspy.record(category=c.t.category, tier=c.account["tier"]))
    .when(
        (c.t.category == "billing") & (c.orders.length() > 0),
        then=dspy.steps()
        .set(c.findings["orders"], c.orders)
        .exec("cap", py, code=dspy.fmt(
            "result = round({bal} * 0.10, 2)", bal=c.account["open_balance"]
        ))
        .set(c.findings["refund_cap"], c.cap)
        .exec("mentions", sh, code=dspy.fmt(
            "grep -c '{o}' /var/log/refunds.log", o=c.orders[0]
        ))
        .set(c.findings["refund_mentions"], c.mentions),
    )
    .foreach(
        "step", in_=c.t.actions,
        do=dspy.steps().when(
            c.step["name"].isin(actions),
            then=dspy.steps()
            .dispatch("obs", tools=actions, name=c.step["name"], args=c.step["args"])
            .set(c.findings[c.step["name"]], c.obs),
        ),
    )
    .when(
        c.t.urgency >= 4,
        then=dspy.steps()
        .predict("deep", investigate, ticket=c.ticket, context=c.findings)
        .set(c.findings["deep_dive"], c.deep.summary),
    )
    .until(
        c.check.compliant, max_iters=3,
        do=dspy.steps()
        .predict("d", draft, ticket=c.ticket, findings=c.findings)
        .subflow("check", policy_check, reply=c.d.reply, account_tier=c.findings["tier"])
        .when(
            ~c.check.compliant,
            then=dspy.steps().set(c.findings["violation"], c.check.violation),
        ),
    )
    .when(
        ~c.check.compliant,
        then=dspy.returns(reply="Escalating to a human agent.", quotes=[]),
    )
    .returns(reply=c.d.reply, quotes=c.d.quotes)
)


# ---------------------------------------------------------------------------
# Metric + wiring + export — unchanged from the other Python exemplars.

def quality(example, prediction) -> float:
    score = 0.0
    if example.must_mention in prediction.reply:
        score = score + 0.5
    if len(prediction.quotes) > 0:
        score = score + 0.5
    return score


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
