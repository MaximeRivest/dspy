// Ticket assistant — Go exemplar (design fiction: presumes dspy-go).
// Sibling of exemplar-program.py rev 6.
//
// Idiomatic choices:
//   - Signatures are STRUCTS with pir tags; Predict[T] returns T, so
//     prediction access is typed fields (t.Category), no string getters.
//     The struct doc comment is the instructions.
//   - Modules are plain functions marked //pir:module, compiled from
//     source by `pir compile ./...` (go/ast — Go cannot lift bodies at
//     runtime; compilation is a build step, like go:generate).
//   - Tools are ordinary funcs returning (T, error); the error return IS
//     the typed-error channel — the frontend lowers the
//     `v, err := ...; if err != nil` idiom into Try/ToolError semantics.
//   - Deps need NO comment: Go imports + go.mod already declare every
//     leaf's dependencies statically. Authored-Go ships PACKAGED (D-025);
//     `pir compile` records the module path + version from go.mod.
package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"

	dspy "github.com/dspy/dspy-go"
	llama "github.com/dspy/dspy-go/engines/llamacpp"
)

// ---------------------------------------------------------------------------
// Signatures — structs with tags; doc comment = instructions.

// Triage: classify a support ticket and plan retrieval actions.
type Triage struct {
	Ticket   string      `pir:"in"`
	Category string      `pir:"out,desc=one of: billing, technical, account, other"`
	Urgency  int         `pir:"out,desc=1 (low) to 5 (page someone)"`
	Actions  []dspy.Step `pir:"out,desc=retrieval steps"` // Step{Name string; Args map[string]any}
}

// DraftReply: write the reply. Quote the KB passages you relied on.
type DraftReply struct {
	Ticket   string         `pir:"in"`
	Findings dspy.Map       `pir:"in,desc=everything gathered about this ticket"`
	Reply    string         `pir:"out"`
	Quotes   dspy.Citations `pir:"out"`
}

// Assess: is a drafted reply allowed to go out?
type Assess struct {
	Reply       string `pir:"in"`
	AccountTier string `pir:"in,name=account_tier"` // wire name pinned; Go name idiomatic
	Compliant   bool   `pir:"out"`
	Violation   string `pir:"out"`
}

// Investigate: dig into a ticket with tools.
type Investigate struct {
	Ticket  string   `pir:"in"`
	Context dspy.Map `pir:"in"`
	Summary string   `pir:"out"`
}

// ---------------------------------------------------------------------------
// Tools — ordinary functions; (T, error) is the ToolError channel.

// FetchAccount looks up a customer account in the billing API.
func FetchAccount(customerID string) (map[string]any, error) {
	resp, err := http.Get("https://billing.internal/api/accounts/" + customerID)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var out map[string]any
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out, nil
}

// KBSearch searches the internal knowledge base, best-k passages.
func KBSearch(query string) ([]string, error) {
	resp, err := http.Get("https://kb.internal/search?q=" + query)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var hits struct {
		Results []struct{ Text string }
	}
	if err := json.NewDecoder(resp.Body).Decode(&hits); err != nil {
		return nil, err
	}
	out := []string{}
	for _, h := range hits.Results {
		out = append(out, h.Text)
	}
	return out, nil
}

var ordRe = regexp.MustCompile(`ORD-\d{6}`)

// ExtractOrderIDs pulls ORD-xxxxxx ids out of free text. Pure.
func ExtractOrderIDs(text string) ([]string, error) {
	return ordRe.FindAllString(text, -1), nil
}

// ---------------------------------------------------------------------------
// Custom LM with baked weights — engine: llama.cpp binding. Inference-only:
// Go cannot train, so a Go-authored artifact exports its weight-ref FROZEN
// (train in the Python engine, serve here — the engine axis at work).

type TinyTriageLM struct {
	model *llama.Model
}

func NewTinyTriageLM() (*TinyTriageLM, error) {
	m, err := llama.Load("PleIAs/Baguettotron")
	if err != nil {
		return nil, err
	}
	return &TinyTriageLM{model: m}, nil
}

func (m *TinyTriageLM) Forward(req dspy.LMRequest) (dspy.LMResponse, error) {
	text, err := m.model.Chat(req.Messages, llama.MaxTokens(req.MaxTokensOr(256)))
	if err != nil {
		return dspy.LMResponse{}, err
	}
	return dspy.TextResponse(text), nil
}

// ---------------------------------------------------------------------------
// Leaves — package scope; variable name = tree name.

var (
	triage = dspy.NewPredict[Triage]()
	draft  = dspy.NewPredict[DraftReply]()
	assess = dspy.NewPredict[Assess]()

	investigate = dspy.NewReAct[Investigate](
		dspy.Tools(KBSearch, FetchAccount),
		dspy.MaxIters(4),
	)

	py = dspy.NewPythonInterpreter(dspy.Allow("round"))
	sh = dspy.NewBashInterpreter(dspy.Allow("grep"))

	actions = dspy.ToolMap{
		"lookup_account": dspy.Tool(FetchAccount),
		"search_kb":      dspy.Tool(KBSearch),
	}
)

// ---------------------------------------------------------------------------
// Modules — plain functions; init deduced at `pir compile` time.

//pir:module
func PolicyCheck(reply, accountTier string) (Assess, error) {
	return assess.Call(Assess{Reply: reply, AccountTier: accountTier})
}

//pir:module
func TicketAssistant(ticket, customerID string) (DraftReply, error) {
	t, err := triage.Call(Triage{Ticket: ticket})
	if err != nil {
		return DraftReply{}, err
	}

	findings := dspy.Map{"category": t.Category}

	account, err := FetchAccount(customerID)
	if err != nil {
		return DraftReply{}, err
	}
	findings["tier"] = account["tier"]

	orders, _ := ExtractOrderIDs(ticket)
	if t.Category == "billing" && len(orders) > 0 {
		findings["orders"] = orders
		code := fmt.Sprintf("result = round(%v * 0.10, 2)", account["open_balance"])
		findings["refund_cap"], _ = py.Exec(code)
		findings["refund_mentions"], _ = sh.Exec(
			fmt.Sprintf("grep -c '%s' /var/log/refunds.log", orders[0]),
		)
	}

	for _, step := range t.Actions {
		if tool, ok := actions[step.Name]; ok {
			findings[step.Name], _ = tool.Call(step.Args)
		}
	}

	if t.Urgency >= 4 {
		deep, err := investigate.Call(Investigate{Ticket: ticket, Context: findings})
		if err == nil {
			findings["deep_dive"] = deep.Summary
		}
	}

	attempts := 0
	var approved *DraftReply
	for attempts < 3 {
		d, err := draft.Call(DraftReply{Ticket: ticket, Findings: findings})
		if err != nil {
			return DraftReply{}, err
		}
		check, err := PolicyCheck(d.Reply, findings["tier"].(string))
		if err != nil {
			return DraftReply{}, err
		}
		if check.Compliant {
			approved = &d
			break
		}
		findings["violation"] = check.Violation
		attempts = attempts + 1
	}

	if approved == nil {
		return DraftReply{Reply: "Escalating to a human agent.", Quotes: dspy.Citations{}}, nil
	}
	return *approved, nil
}

// ---------------------------------------------------------------------------
// Metric — leaf code, travels with the artifact.

func Quality(example dspy.Example, prediction DraftReply) float64 {
	score := 0.0
	if example.Has("must_mention") && contains(prediction.Reply, example.Str("must_mention")) {
		score += 0.5
	}
	if len(prediction.Quotes) > 0 {
		score += 0.5
	}
	return score
}

func contains(s, sub string) bool { return regexp.MustCompile(regexp.QuoteMeta(sub)).MatchString(s) }

// ---------------------------------------------------------------------------
// Wiring + export.

func main() {
	router := dspy.NewLM("openai/gpt-oss-120b", dspy.APIBase("https://gw.internal/v1"))
	writer := dspy.NewLM("anthropic/claude-sonnet-5")
	tiny, err := NewTinyTriageLM()
	if err != nil {
		panic(err)
	}

	dspy.Configure(dspy.WithLM(router), dspy.WithAdapter(dspy.JSONAdapter()))

	triage.SetLM(tiny)
	assess.SetLM(tiny)
	draft.SetLM(writer)
	draft.SetAdapter(dspy.ChatAdapter())

	draft.Demos = []dspy.Example{
		dspy.NewExample(dspy.Map{
			"ticket":   "I was double-charged on ORD-482113.",
			"findings": dspy.Map{"category": "billing", "tier": "pro"},
			"reply":    "I can confirm the duplicate charge on ORD-482113 was reversed...",
			"quotes":   []string{"Refunds for duplicate charges post within 3-5 business days."},
		}).WithInputs("ticket", "findings"),
	}

	devset := []dspy.Example{
		dspy.NewExample(dspy.Map{
			"ticket":       "Cancel my subscription, nothing works.",
			"customer_id":  "C-99120",
			"must_mention": "cancel",
		}).WithInputs("ticket", "customer_id"),
	}

	if err := dspy.Export(TicketAssistant, "ticket_assistant.ir",
		dspy.Metric(Quality), dspy.Devset(devset)); err != nil {
		panic(err)
	}
}
