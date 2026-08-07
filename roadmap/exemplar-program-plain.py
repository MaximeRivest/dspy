"""Ticket assistant — plain-Python exemplar: zero subclassing.

Everything the -sig exemplar declared through class inheritance is
declared here with standard Python constructs only:

- output records are @dataclass + Annotated (no dspy.Signature subclass;
  field descriptions ARE the Annotated metadata)
- predictors are @dspy.predict TYPED STUBS — docstring = instructions,
  params = inputs, return annotation = outputs; calling one returns the
  dataclass, typed
- modules are @dspy.module functions whose OWN annotations are the module
  signature (the -sig requirement, without classes); return paths
  construct the plain dataclass — no dspy.Prediction
- the custom LM is a closure factory, not a BaseLM subclass
- examples are plain dicts; with_inputs is GONE — inputs are inferred
  from the signatures that already declare them
- no ambient settings exist; every predictor binds explicitly

The only dspy syntax left is decorator names and constructor calls.
"""

from dataclasses import dataclass
from typing import Annotated

import dspy


# ---------------------------------------------------------------------------
# Tools — unchanged: they were already plain Python.

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
# Custom LM with baked weights — a closure factory instead of a subclass:
# engine state lives in the closure; dspy.lm wraps the forward.

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


# ---------------------------------------------------------------------------
# Output records — dataclasses; Annotated metadata is the field description.

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


# ---------------------------------------------------------------------------
# Predictors — typed stubs. The body is the docstring; that's the point:
# a predictor IS a typed function contract with instructions attached.

@dspy.predict
def classify(ticket: str) -> TriageOut:
    """Classify a support ticket and plan retrieval actions."""


@dspy.predict
def write_reply(
    ticket: str,
    findings: Annotated[dict, "everything gathered about this ticket"],
) -> ReplyOut:
    """Write the reply. Quote the KB passages you relied on."""


@dspy.predict
def check_policy(reply: str, account_tier: str) -> PolicyOut:
    """Is a drafted reply allowed to go out?"""


def research(ticket: str, context: dict) -> ResearchOut:
    """Dig into a ticket with tools."""

investigate = dspy.react(research, tools=[kb_search, fetch_account], max_iters=4)

py = dspy.PythonInterpreter(allow=["round"])
sh = dspy.BashInterpreter(allow=["grep"])
actions = {"lookup_account": fetch_account, "search_kb": kb_search}


# ---------------------------------------------------------------------------
# Modules — functions whose annotations ARE the module signature; every
# return path constructs the declared dataclass, checked at compile.

@dspy.module
def policy_check(reply: str, account_tier: str) -> PolicyOut:
    return check_policy(reply=reply, account_tier=account_tier)


@dspy.module
def ticket_assistant(ticket: str, customer_id: str) -> ReplyOut:
    t = classify(ticket=ticket)

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
        d = write_reply(ticket=ticket, findings=findings)
        check = policy_check(reply=d.reply, account_tier=findings["tier"])
        if check.compliant:
            approved = d
            break
        findings["violation"] = check.violation
        attempts = attempts + 1

    if approved is None:
        return ReplyOut(reply="Escalating to a human agent.", quotes=[])
    return approved


# ---------------------------------------------------------------------------
# Metric — plain function over plain values.

def quality(example: dict, prediction: ReplyOut) -> float:
    score = 0.0
    if example["must_mention"] in prediction.reply:
        score = score + 0.5
    if len(prediction.quotes) > 0:
        score = score + 0.5
    return score


# ---------------------------------------------------------------------------
# Wiring — explicit everywhere; no ambient anything. Examples are plain
# dicts: inputs are inferred from the signatures that declare them.

if __name__ == "__main__":
    router = dspy.LM("openai/gpt-oss-120b", api_base="https://gw.internal/v1")
    writer = dspy.LM("anthropic/claude-sonnet-5")
    tiny = tiny_triage_lm()
    json_a = dspy.JSONAdapter()
    chat_a = dspy.ChatAdapter()

    classify.set_lm(tiny)
    classify.set_adapter(json_a)
    check_policy.set_lm(tiny)
    check_policy.set_adapter(json_a)
    write_reply.set_lm(writer)
    write_reply.set_adapter(chat_a)
    investigate.set_lm(router)
    investigate.set_adapter(json_a)

    write_reply.demos = [
        {
            "ticket": "I was double-charged on ORD-482113.",
            "findings": {"category": "billing", "tier": "pro"},
            "reply": "I can confirm the duplicate charge on ORD-482113 was reversed...",
            "quotes": ["Refunds for duplicate charges post within 3-5 business days."],
        }
    ]

    devset = [
        {
            "ticket": "Cancel my subscription, nothing works.",
            "customer_id": "C-99120",
            "must_mention": "cancel",
        }
    ]

    dspy.export(ticket_assistant, "ticket_assistant.ir", metric=quality, devset=devset)
