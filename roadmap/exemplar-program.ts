/**
 * Ticket assistant — TypeScript exemplar (design fiction: presumes the
 * dspy-ts engine exists). Sibling of exemplar-program.py rev 6.
 *
 * Idiomatic choices:
 * - Modules are plain async functions; `await` marks leaf calls, and the
 *   build-time frontend compiles the body from the TS AST (TS cannot lift
 *   bodies at runtime; `pir compile` is a bundler step).
 * - Signatures are typed object literals — Predictions come back TYPED,
 *   the reason a native TS surface exists at all.
 * - Signature FIELD names stay snake_case (they are prompt-visible
 *   contract identity); local code is camelCase.
 * - Interpreter scope is declared as NAMES (`allow: ["round"]`), not live
 *   objects — an improvement the Python surface should adopt.
 */

import * as dspy from "@dspy/core";

// ---------------------------------------------------------------------------
// Tools — plain functions. Node's fetch is built-in: nothing to declare;
// a `// deps:` comment appears only when a real package is imported.

/** Look up a customer account in the billing API. */
async function fetchAccount({ customer_id }: { customer_id: string }) {
  const r = await fetch(`https://billing.internal/api/accounts/${customer_id}`);
  if (!r.ok) throw new Error(`billing api: ${r.status}`);
  return (await r.json()) as Record<string, unknown>;
}

/** Search the internal knowledge base, best-k passages. */
async function kbSearch({ query, k = 3 }: { query: string; k?: number }) {
  const u = new URL("https://kb.internal/search");
  u.searchParams.set("q", query);
  u.searchParams.set("k", String(k));
  const hits = (await (await fetch(u)).json()) as { results: { text: string }[] };
  return hits.results.map((h) => h.text);
}

/** Pull ORD-xxxxxx ids out of free text. Pure, stdlib only. */
function extractOrderIds({ text }: { text: string }): string[] {
  return text.match(/ORD-\d{6}/g) ?? [];
}

// ---------------------------------------------------------------------------
// Custom LM with baked weights — engine: transformers-js (ONNX runtime).

class TinyTriageLM extends dspy.BaseLM {
  // deps: @huggingface/transformers
  private pipe: unknown;

  constructor(private modelDir = "PleIAs/Baguettotron") {
    super({ model: modelDir });
  }

  async forward(request: dspy.LMRequest): Promise<dspy.LMResponse> {
    const { pipeline } = await import("@huggingface/transformers");
    this.pipe ??= await pipeline("text-generation", this.modelDir);
    const out = await (this.pipe as any)(request.messages, {
      max_new_tokens: request.maxTokens ?? 256,
    });
    return dspy.LMResponse.text(out[0].generated_text.at(-1).content);
  }
}

// ---------------------------------------------------------------------------
// Signatures — typed object literals; inference gives typed Predictions.

const Triage = dspy.signature({
  description: "Classify a support ticket and plan retrieval actions.",
  ticket: dspy.input<string>(),
  category: dspy.output<string>({ desc: "one of: billing, technical, account, other" }),
  urgency: dspy.output<number>({ desc: "1 (low) to 5 (page someone)" }),
  actions: dspy.output<{ name: string; args: Record<string, unknown> }[]>({
    desc: "retrieval steps",
  }),
});

const DraftReply = dspy.signature({
  description: "Write the reply. Quote the KB passages you relied on.",
  ticket: dspy.input<string>(),
  findings: dspy.input<Record<string, unknown>>({
    desc: "everything gathered about this ticket",
  }),
  reply: dspy.output<string>(),
  quotes: dspy.output(dspy.Citations),
});

const Assess = dspy.signature({
  description: "Is a drafted reply allowed to go out?",
  reply: dspy.input<string>(),
  account_tier: dspy.input<string>(),
  compliant: dspy.output<boolean>(),
  violation: dspy.output<string>(),
});

// ---------------------------------------------------------------------------
// Leaves — module scope, used by bare name; variable name = tree name.

const triage = dspy.predict(Triage);
const draft = dspy.predict(DraftReply);
const assess = dspy.predict(Assess);

const investigate = dspy.react({
  signature: "ticket, context -> summary",
  tools: [kbSearch, fetchAccount],
  maxIters: 4,
});

const py = dspy.pythonInterpreter({ allow: ["round"] });
const sh = dspy.bashInterpreter({ allow: ["grep"] });

const actions = { lookup_account: fetchAccount, search_kb: kbSearch };

// ---------------------------------------------------------------------------
// Modules — plain async functions; init deduced from the body.

const policyCheck = dspy.module(
  async ({ reply, account_tier }: { reply: string; account_tier: string }) =>
    assess({ reply, account_tier }),
);

export const ticketAssistant = dspy.module(
  async ({ ticket, customer_id }: { ticket: string; customer_id: string }) => {
    const t = await triage({ ticket });

    const findings: Record<string, unknown> = { category: t.category };

    const account = await fetchAccount({ customer_id });
    findings["tier"] = account["tier"];

    const orders = extractOrderIds({ text: ticket });
    if (t.category === "billing" && orders.length > 0) {
      findings["orders"] = orders;
      const code = `result = round(${account["open_balance"]} * 0.10, 2)`;
      findings["refund_cap"] = await py({ code });
      findings["refund_mentions"] = await sh({
        code: `grep -c '${orders[0]}' /var/log/refunds.log`,
      });
    }

    for (const step of t.actions) {
      if (step.name in actions) {
        findings[step.name] = await actions[step.name](step.args as never);
      }
    }

    if (t.urgency >= 4) {
      const deep = await investigate({ ticket, context: findings });
      findings["deep_dive"] = deep.summary;
    }

    let attempts = 0;
    let approved = null;
    while (attempts < 3) {
      const d = await draft({ ticket, findings });
      const check = await policyCheck({
        reply: d.reply,
        account_tier: findings["tier"] as string,
      });
      if (check.compliant) {
        approved = d;
        break;
      }
      findings["violation"] = check.violation;
      attempts = attempts + 1;
    }

    if (approved === null) {
      return dspy.prediction({ reply: "Escalating to a human agent.", quotes: [] });
    }
    return approved;
  },
);

// ---------------------------------------------------------------------------
// Metric — leaf code, travels with the artifact.

function quality(example: dspy.Example, prediction: dspy.Prediction): number {
  let score = 0.0;
  if (prediction.reply.includes(example.must_mention)) score += 0.5;
  if (prediction.quotes.length > 0) score += 0.5;
  return score;
}

// ---------------------------------------------------------------------------
// Wiring + export.

const router = dspy.lm("openai/gpt-oss-120b", { apiBase: "https://gw.internal/v1" });
const writer = dspy.lm("anthropic/claude-sonnet-5");
const tiny = new TinyTriageLM();

dspy.configure({ lm: router, adapter: dspy.jsonAdapter() });

triage.setLm(tiny);
assess.setLm(tiny);
draft.setLm(writer);
draft.setAdapter(dspy.chatAdapter());

draft.demos = [
  dspy.example({
    ticket: "I was double-charged on ORD-482113.",
    findings: { category: "billing", tier: "pro" },
    reply: "I can confirm the duplicate charge on ORD-482113 was reversed...",
    quotes: ["Refunds for duplicate charges post within 3-5 business days."],
  }).withInputs("ticket", "findings"),
];

const devset = [
  dspy.example({
    ticket: "Cancel my subscription, nothing works.",
    customer_id: "C-99120",
    must_mention: "cancel",
  }).withInputs("ticket", "customer_id"),
];

await dspy.export(ticketAssistant, "ticket_assistant.ir", { metric: quality, devset });
