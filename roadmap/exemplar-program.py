"""Ticket assistant — the exemplar program for ProgramIR representability.

A complete, naturally-written dspy program: nested modules, tools with
foreign deps, an interpreter, dynamic dispatch, mixed LM/adapter bindings,
demos, a metric, and export. This file is the iteration target: everything
here should either compile to the IR or teach us what to change.
"""

import re
from typing import Annotated

import dspy


# ---------------------------------------------------------------------------
# Tools — leaves. Full Python, any deps. Introspected at compile:
# identity from the type hints, deps from the `# deps:` line, body baked.

# deps: httpx
import httpx


def fetch_account(customer_id: str) -> dict:
    """Look up a customer account in the billing API."""
    r = httpx.get(f"https://billing.internal/api/accounts/{customer_id}", timeout=5)
    r.raise_for_status()
    return r.json()


def kb_search(query: str, k: int = 3) -> list[str]:
    """Search the internal knowledge base, best-k passages."""
    hits = httpx.get("https://kb.internal/search", params={"q": query, "k": k}).json()
    return [h["text"] for h in hits["results"]]


def extract_order_ids(text: str) -> list[str]:
    """Pull ORD-xxxxxx ids out of free text. Pure."""
    return re.findall(r"ORD-\d{6}", text)


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
    quotes: Annotated[list[str], dspy.citations] = dspy.OutputField()


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
        self.py = dspy.PythonInterpreter()

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

        for step in t.actions:
            name = step["name"]
            if name in self.actions:
                findings[name] = self.actions[name](**step["args"])

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
# Metric — component 12: tool-shaped, pure, travels with the artifact.

def quality(example, prediction) -> float:
    score = 0.0
    if example.must_mention in prediction.reply:
        score = score + 0.5
    if len(prediction.quotes) > 0:
        score = score + 0.5
    return score


# ---------------------------------------------------------------------------
# Wiring + export

if __name__ == "__main__":
    program = TicketAssistant()

    program.pools(
        lms={
            "router": dspy.LM(
                "openai/gpt-oss-120b",
                endpoint_ref="LM_ENDPOINT",
                credential_ref="LM_API_KEY",
            ),
            "writer": dspy.LM(
                "anthropic/claude-sonnet-5",
                credential_ref="ANTHROPIC_API_KEY",
            ),
        },
        adapters={"chat": dspy.presets.chat, "json": dspy.presets.json},
    )
    program.triage.bind(adapter="json", lm="router")
    program.policy.assess.bind(adapter="json", lm="router")
    program.draft.bind(adapter="chat", lm="writer")

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
