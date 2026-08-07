"""Ticket assistant — graph-visible exemplar (plain-Python dialect, restructured).

Same zero-subclassing dialect as exemplar-program-plain.py, reorganized so
the GRAPH is what you see while reading:

- RESOURCES → NODES → GRAPH, in that order. Every predictor carries its
  lm/adapter/demos AT THE DECLARATION (decorator kwargs) — the wiring
  section is deleted; a node's full config is visible where the node is.
- The top-level module is five lines, one node application per line — the
  program's adjacency list. Each stage is itself a small module, so every
  level of zoom reads the same way: names applied to names.
- Stages are single-assignment where possible; `findings` flows visibly
  from stage to stage instead of being mutated all over one big body.
"""

from dataclasses import dataclass
from typing import Annotated

import dspy


# ═══════════════════════════════════════════════════════════════════════════
# TOOLS — leaves, unchanged.

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


def tiny_triage_lm(model_dir: str = "PleIAs/Baguettotron", device: str = "cpu"):
    """A 321M local model for cheap classification; weights ship inside."""
    # deps: torch, transformers, safetensors
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)

    def forward(request):
        import torch

        text = tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=request.max_tokens or 256)
        return dspy.LMResponse(
            text=tokenizer.decode(out[0][ids["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
        )

    return dspy.lm(forward, weights=model, weights_identity=model_dir)


# ═══════════════════════════════════════════════════════════════════════════
# SHAPES — output records; Annotated metadata is the field description.

@dataclass
class TriageOut:
    category: Annotated[str, "one of: billing, technical, account, other"]
    urgency: Annotated[int, "1 (low) to 5 (page someone)"]
    actions: Annotated[list[dict], 'retrieval steps, each {"name": tool, "args": {kwargs}}']


@dataclass
class ReplyOut:
    reply: str
    quotes: dspy.Citations


@dataclass
class PolicyOut:
    compliant: bool
    violation: str


@dataclass
class ResearchOut:
    summary: str


# ═══════════════════════════════════════════════════════════════════════════
# RESOURCES — every lm, adapter, interpreter, and tool-map the nodes bind.
# (tiny loads weights at import — a lazy variant is an open question.)

tiny = tiny_triage_lm()
router = dspy.LM("openai/gpt-oss-120b", api_base="https://gw.internal/v1")
writer = dspy.LM("anthropic/claude-sonnet-5")
json_a = dspy.JSONAdapter()
chat_a = dspy.ChatAdapter()

py = dspy.PythonInterpreter(allow=["round"])
sh = dspy.BashInterpreter(allow=["grep"])
actions = {"lookup_account": fetch_account, "search_kb": kb_search}

REPLY_DEMOS = [
    {
        "ticket": "I was double-charged on ORD-482113.",
        "findings": {"category": "billing", "tier": "pro"},
        "reply": "I can confirm the duplicate charge on ORD-482113 was reversed...",
        "quotes": ["Refunds for duplicate charges post within 3-5 business days."],
    }
]


# ═══════════════════════════════════════════════════════════════════════════
# NODES — each predictor fully configured where it is declared.

@dspy.predict(lm=tiny, adapter=json_a)
def classify(ticket: str) -> TriageOut:
    """Classify a support ticket and plan retrieval actions."""


@dspy.predict(lm=writer, adapter=chat_a, demos=REPLY_DEMOS)
def write_reply(
    ticket: str,
    findings: Annotated[dict, "everything gathered about this ticket"],
) -> ReplyOut:
    """Write the reply. Quote the KB passages you relied on."""


@dspy.predict(lm=tiny, adapter=json_a)
def check_policy(reply: str, account_tier: str) -> PolicyOut:
    """Is a drafted reply allowed to go out?"""


def research(ticket: str, context: dict) -> ResearchOut:
    """Dig into a ticket with tools."""

investigate = dspy.react(
    research, tools=[kb_search, fetch_account], max_iters=4, lm=router, adapter=json_a
)


# ═══════════════════════════════════════════════════════════════════════════
# STAGES — each a small module; single-assignment; findings flows through.

@dspy.module
def gather_context(ticket: str, customer_id: str, triage: TriageOut) -> dict:
    findings = {"category": triage.category}
    account = fetch_account(customer_id=customer_id)
    findings["tier"] = account["tier"]

    orders = extract_order_ids(text=ticket)
    if triage.category == "billing" and len(orders) > 0:
        findings["orders"] = orders
        code = f"result = round({account['open_balance']} * 0.10, 2)"
        findings["refund_cap"] = py(code=code)
        findings["refund_mentions"] = sh(
            code=f"grep -c '{orders[0]}' /var/log/refunds.log"
        )
    return findings


@dspy.module
def run_planned_actions(triage: TriageOut, findings: dict) -> dict:
    for step in triage.actions:
        name = step["name"]
        if name in actions:
            findings[name] = actions[name](**step["args"])
    return findings


@dspy.module
def deep_dive_if_urgent(ticket: str, triage: TriageOut, findings: dict) -> dict:
    if triage.urgency >= 4:
        deep = investigate(ticket=ticket, context=findings)
        findings["deep_dive"] = deep.summary
    return findings


@dspy.module
def draft_until_compliant(ticket: str, findings: dict) -> ReplyOut:
    attempts = 0
    while attempts < 3:
        d = write_reply(ticket=ticket, findings=findings)
        check = check_policy(reply=d.reply, account_tier=findings["tier"])
        if check.compliant:
            return d
        findings["violation"] = check.violation
        attempts = attempts + 1
    return ReplyOut(reply="Escalating to a human agent.", quotes=[])


# ═══════════════════════════════════════════════════════════════════════════
# THE GRAPH — the whole program, one edge per line.

@dspy.module
def ticket_assistant(ticket: str, customer_id: str) -> ReplyOut:
    t        = classify(ticket=ticket)
    findings = gather_context(ticket=ticket, customer_id=customer_id, triage=t)
    findings = run_planned_actions(triage=t, findings=findings)
    findings = deep_dive_if_urgent(ticket=ticket, triage=t, findings=findings)
    return draft_until_compliant(ticket=ticket, findings=findings)


# ═══════════════════════════════════════════════════════════════════════════
# METRIC + EXPORT

def quality(example: dict, prediction: ReplyOut) -> float:
    score = 0.0
    if example["must_mention"] in prediction.reply:
        score = score + 0.5
    if len(prediction.quotes) > 0:
        score = score + 0.5
    return score


if __name__ == "__main__":
    devset = [
        {
            "ticket": "Cancel my subscription, nothing works.",
            "customer_id": "C-99120",
            "must_mention": "cancel",
        }
    ]

    dspy.export(ticket_assistant, "ticket_assistant.ir", metric=quality, devset=devset)
