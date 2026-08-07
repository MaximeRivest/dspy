"""Ticket assistant restricted to the ProgramIR node set v0.1.

This is the executable exporter stimulus while the fuller exemplar waits for
v0.2. Values flow through predictor outputs and simple assignments; there are
no dict literals, subscripts, f-strings, membership tests, or kwargs splats.
"""

import dspy
from dspy.programir import compile, write


class Triage(dspy.Signature):
    """Classify a support ticket."""

    ticket: str = dspy.InputField()
    category: str = dspy.OutputField(desc="billing, technical, account, or other")
    urgency: int = dspy.OutputField(desc="1 (low) to 5 (urgent)")


class DraftReply(dspy.Signature):
    """Write a concise support reply."""

    ticket: str = dspy.InputField()
    category: str = dspy.InputField()
    reply: str = dspy.OutputField()


class Policy(dspy.Signature):
    """Check whether a reply is safe to send."""

    reply: str = dspy.InputField()
    compliant: bool = dspy.OutputField()
    violation: str = dspy.OutputField()


class PolicyCheck(dspy.Module):
    def __init__(self):
        self.assess = dspy.Predict(Policy)

    def forward(self, reply):
        return self.assess(reply=reply)


class TicketAssistant(dspy.Module):
    def __init__(self):
        self.triage = dspy.Predict(Triage)
        self.draft = dspy.Predict(DraftReply)
        self.policy = PolicyCheck()
        self.escalate = dspy.Predict("ticket, violation -> reply")

    def forward(self, ticket):
        triage = self.triage(ticket=ticket)
        draft = self.draft(ticket=ticket, category=triage.category)
        check = self.policy(reply=draft.reply)
        if check.compliant == True:
            answer = draft
        else:
            answer = self.escalate(ticket=ticket, violation=check.violation)
        return answer


if __name__ == "__main__":
    router = dspy.LM("openai/gpt-oss-120b")
    writer = dspy.LM("anthropic/claude-sonnet-5")
    json_adapter = dspy.JSONAdapter()

    program = TicketAssistant()
    program.set_lm(router)
    program.set_adapter(json_adapter)
    program.draft.set_lm(writer)
    program.draft.set_adapter(dspy.ChatAdapter())

    write(compile(program), "ticket_assistant_v01.ir")
