# E-11 — the refusals exemplar

**Proves:** every named refusal fires loudly, with an error that names
the thing, the place, and (where relevant) both versions — never a
silent partial anything. One trigger + one expected error per refusal.
Grouped by the stage that refuses. `[R]` = ratified ground; `[D-0xx]` =
drafted. Error texts are normative *shape*, not byte-exact prose: each
MUST name the quoted elements; wording may vary.

Corpus note: every entry below becomes one refusal vector in the
fixture corpus; the IDs (R-01…) are stable and cited by FIXTURE-PLAN.

## A. Compile-time (frontend → IR)

**R-01 · un-whitelisted node** `[R: node-set]`
```python
def forward(self, inputs):
    return [self.p(x=i) for i in inputs.items]     # comprehension over leaf calls
```
> `forward: <ListComp> at modules.py:14 is not in the representable node set (node_set 0.5); For-over-list accumulation is the lawful desugar`

**R-02 · call to an undeclared leaf** `[R: §d leaf rule]`
```python
    value = self.normalize(text=inputs.question)   # no such declared leaf
```
> `forward: call to undeclared 'self.normalize' at modules.py:9 — every call must resolve to a Predict, sub-module, tool, or interpreter leaf`

**R-03 · `dspy.context` / `With` scoping** `[R: D-035]`
```python
    with dspy.context(lm=other):                   # ambient dynamic scoping
        return self.p(q=inputs.q)
```
> `forward: <With> at modules.py:21 — dspy.context-scoped forwards are non-exportable (D-035); bind the model on the leaf instead`

**R-04 · explicit concurrency** `[R: D-037]`
```python
    a, b = await asyncio.gather(self.p.acall(...), self.q.acall(...))
```
> `aforward: asyncio.gather at modules.py:30 — execution mode belongs to the engine, never the artifact; sequential await lowers to the same graph, explicit fan-out refuses`

**R-05 · shape with no JSON-schema meaning** `[R: component 2]`
```python
    handler: Callable = dspy.InputField()          # a tool pretending to be data
```
> `signature 'draft': field 'handler' shape Callable has no JSON-schema meaning — declare it as a tool leaf, not a data field`

**R-06 · conflicting role spellings** `[R: D-011]`
```python
    answer: Annotated[str, citations] = dspy.OutputField(role="reasoning")
```
> `signature: field 'answer' carries conflicting roles 'citations' (Annotated) and 'reasoning' (role=) — one field, one role`

**R-07 · unknown role, eager** `[R: D-011]`
```python
    "question -> answer @groundedness"
```
> `unknown role 'groundedness'; vocabulary (roles 1.0): plain, reasoning, tools, tool_calls, citations, history, media, code`

**R-08 · LM class capture failure** `[R: §e0-class]`
```python
    lm = make_lm()          # closure over a C-extension handle; source not introspectable
```
> `compile: authored model entry 'house_lm' — source cannot be captured (un-introspectable closure); capture-or-refuse (DT-003): no dotted path is ever emitted`

**R-09 · deduction ambiguity** `[D-050]`
```python
    m = dspy.Model.authored(source="f <- function(x) x + 1")   # no path; R? Julia?
```
> `Model.authored: language of inline source is ambiguous (candidates: r, julia) — pass language=; deductions must be deterministic or refused`

**R-10 · trainable claim without a training contract** `[D-045/DT-003 pattern]`
```python
    dspy.optim.Train(target=prog.locate, trainset=pairs)   # sam entry declares no verbs
```
> `Train: model entry 'sam' declares no training contract (neither sgd-1 verbs nor fit-1) — capture-or-refuse; nothing is silently frozen-but-pretending`

**R-11 · loss mismatch with the resolved grain** `[D-045]`
```python
    dspy.optim.Train(target=prog.priority, trainset=labels, loss="cross_entropy")
```
> `Train: target entry 'priority_gam' trains under fit-1 — loss= applies to sgd-1 only; remove it (fit-1 objectives live in the fit itself)`

## B. Adapter construction (entry build / registration)

**R-12 · non-invertible template slot** `[R: adapter-rfc lens law]`
```
{% for f in outputs %}
{f.value}
{% endfor %}
```
> `template 'bare': output slot for field values has no anchoring literals — the lens cannot invert it; anchor the slot or declare a pipeline parser`

**R-13 · aggregate JSON in the exchange layer** `[R: adapter-rfc]`
```
Respond with one JSON object containing all output fields.
```
> `template 'json_prose': whole-object aggregate is not lensable (semantic format, many spellings) — declare the json pipeline recipe or gate on native_structured_output`

**R-14 · predicate over an undeclared fact** `[R: adapter-rfc six-fact ceiling]`
```jsonc
{"predicate": {"capability": "context_length_gt_100k"}}
```
> `strategy rule: unknown capability fact 'context_length_gt_100k'; vocabulary: instruct, completion, native_reasoning, native_function_calling, native_citations, image_input`

**R-15 · turns/routing drift** `[R: adapter-rfc probe]`
```jsonc
{"turns": {"assistant": {"kind": "template", "content": "{c.name}({c.args})"}},
 "routings": [{"kind": "text", "pattern": "<call name=…>"}]}   // reads a different spelling
```
> `strategy 'heredoc_tools': turns.assistant renders a call its own routing cannot read back — parse(render(call)) != call; refused at registration`

**R-16 · face/config mismatch** `[D-049]`
```python
    dspy.Model("hf:BAAI/bge-small-en-v1.5", temperature=0.7)
```
> `model 'embedder': 'temperature' is chat-face config; resolved face is embed-1 (resolved_from: hf:pipeline_tag) — face-scoped config only`

**R-17 · chat-sugar face assertion** `[D-049]`
```python
    dspy.LM("hf:BAAI/bge-small-en-v1.5")
```
> `dspy.LM asserts the chat face; 'BAAI/bge-small-en-v1.5' resolves to embed-1 — use dspy.Model(...)`

## C. Link step (load: pools ⇄ bindings)

**R-18 · dangling binding** `[R: §b-pools]`
```jsonc
{"kind": "predict", "name": "draft", "bindings": {"adapter": "terse", "model": "drafter"}}
// no adapter pool entry named "terse"
```
> `link: predictor 'draft' binds adapter 'terse' — no such entry in 4_adapter (have: chat, json, identity, sam_points)`

**R-19 · dangling delta** `[D-049]`
```jsonc
{"bindings": {"model": "drafter", "delta": "sha256:0000…"}}
```
> `link: binding 'draft'.delta sha256:0000… does not resolve in the store; a delta must resolve to a blob over its entry's declared base`

**R-20 · ETL result-hash mismatch** `[D-047]`
> `datasets 'tickets': pipeline re-run produced sha256:7c02…, manifest declares sha256:41d9… — source revision or verb semantics drifted; refusing the dataset`

**R-21 · unknown ETL verb** `[D-047]`
```jsonc
{"verb": "group_by", "fields": ["product"]}
```
> `datasets 'tickets': verb 'group_by' is not in the ETL vocabulary (etl 0.1: rename, select, filter, mutate, slice_sample, shuffle, split, designate_inputs)`

## D. Load-time verification (declared tier)

**R-22 · missing versions block** `[R: D-024]`
> `manifest has no 'versions' block — malformed; no unversioned grandfathering exists`

**R-23 · unknown major version** `[R: D-024]`
> `versions.adapter_ir: artifact declares 2.1.0, engine implements 0.2.0 — same-major accepts, otherwise refuse naming both (this message is the naming)`

**R-24 · unresolved credential** `[R: §e0-binding]`
> `load: credential 'PRIORITY_API_KEY' (scope: priority-gam service calls) is not resolvable in this environment`

**R-25 · endpoint serves the wrong identity** `[R: ex-09/10]`
> `load: LM_ENDPOINT backend serves ['qwen3-8b'] — none matches weights_identity 'openai/gpt-4o-mini' or served_aliases []; refusing before first use`

**R-26 · isolation floor under-run** `[R upstream: D-042]`
> `materialize: leaf 'priority' demands isolation_floor 'fork'; receiver envelope provides 'none' — floors tighten, envelopes may exceed, never under-run`

**R-27 · interpreter profile unsatisfied** `[R: D-033]`
> `interpreter 'main': profile requires cpython 3.12 isolated_no_builtins; this engine offers pyodide 3.11 — identity deviation, not a placement; record a deviation or refuse (refusing)`

**R-28 · rung-walk exhausted** `[R: D-022]`
> `component 8_model.'priority_gam' (language r): no endpoint binding, sidecar toolchain 'R>=4.3' not present, rung 0 unsatisfiable — refusing; unmet rungs listed`

**R-29 · credential value found in bytes** `[R: export scan]`
> `export: literal value of credential 'LM_API_KEY' found in tools/search_policies.py:12 (docstring) — credentials are names only, byte-absence is checked positively`

## E. Warranty & objective rules

**R-30 · warranted score over an unpinned dataset** `[D-046]`
```jsonc
{"source": {"kind": "ref", "endpoint_ref": "DW_PAIRS"}}   // no snapshot
// + provenance carrying {"score": 0.83, "devset": "embed_pairs"}
```
> `score 0.83 claims devset 'embed_pairs' which has no pinned identity — unpinned dataset ⇒ unwarranted score; store the score as informational or pin a snapshot`

**R-31 · inner loop touching the objective** `[R: §e2 fixed point 2; D-048]`
```python
    dspy.optim.Train(target=prog.draft, trainset=t, mutate=["12_metric"])
```
> `optimizer may not rewrite its own objective: 12_metric/devset are fixed points of the inner loop; objective evolution is the outer loop's recorded deviation (D-048)`

**R-32 · deferred training grains, named not silent** `[D-045 + STUDY F6/F7]`
```jsonc
{"training": {"contract": "online-1"}}
```
> `training contract 'online-1' is named but not ratified (training 0.1: sgd-1, fit-1) — serving-time state mutation breaks checkpoint=save; schedule outer-loop fit-1 snapshots (cadence law), or await ratification`

## F. Meta

Refusals themselves are versioned surface: each R-id will map to an
error code in the contract's error table; a code no fixture fires is
dead spec (FIXTURE-PLAN rule). Gaps G11/G19 close with this file;
per-refusal fixture vectors are this file sliced, one artifact or
snippet each, with the expected code asserted.

**R-33 · dynamic dispatch outside the declared pool** `[E-06]`
```python
    obs = self.helpers[pred.name](**pred.args)   # helpers is not a declared tool map
```
> `forward: dynamic call at modules.py:31 dispatches over 'self.helpers' which is not a declared tool pool — model-dispatched calls resolve only within declared leaves`

**R-34 · undeclared grant use** `[E-07; D-042/D-043]`
```jsonc
// entry grants: []  — but the envelope passes an open TCP fd anyway
```
> `materialize: envelope supplies grant 'fd:tcp:169.254.0.1:80' not present in leaf 'main' grants row — grants are declared in the artifact, never improvised by the envelope`

**R-35 · dangling checkpoint hash** `[E-09; §c1]`
> `store: checkpoints/step_07 references sha256:e5a8… which is not present in objects/ — a checkpoint that cannot reconstruct is not a checkpoint`

**R-36 · score curve crossing an objective boundary** `[E-09; D-048]`
> `trajectory: step_09 (objective 0c44…/b201…) compared against step_07's score 0.66 (objective 03fa…/41d9…) — scores attach to configurations; re-read step_07 as 0.63 (rescored) or do not compare`

**R-37 · diff naming a frozen field** `[E-09; §e2 fixed points]`
> `trajectory: diff at step_04 rewrites '2_signature/draft' (external signature) — not in the optimizable set; structure moves live within the §d grammar, the task's type is a fixed point`

**R-38 · training a frozen blob** `[E-08; §b-pools]`
> `training_run at step_12 names blob sha256:ab3f… which entry 'shared' declares frozen — flip the tag as a recorded diff first, or attach a delta; frozen means the hash may not move`

**R-39 · lowering that adds an external input** `[E-05; Law 1]`
> `lowering 'hint_injector' on 'answer': expansion adds external input 'hint_' — interface-preserving strictly; added inputs break drop-in substitution (declare an honest program maker instead)`

**R-40 · lowering that smuggles a call** `[E-05; Law 2]`
> `lowering 'clean_json' on 'answer': expansion performs an LM exchange with no emitted leaf (parser-internal call) — the single-shot law; every exchange is a leaf, authored, lowered, or refused`

**R-41 · ambiguous lowering composition** `[E-05; Law 4]`
> `lowerings 'retry' and 'two_step' both annotate 'answer' with no declared order — composition is nesting, declared; write Retry(TwoStep(p)) or TwoStep(p, extract=Retry(…))`

**R-42 · authored adapter on a foreign profile engine** `[E-03; D-026/D-040]`
> `load (declared-tier engine, ts): entry 'house_ansi' requires python authored-codec evaluation — outside the declared-tier profile; unmet requirement named, template half remains readable (explain works, format refuses)`

**R-43 · strategy resolution with no admissible rule** `[E-03]`
> `strategies: role 'reasoning' bound 'auto' — no registered rule's predicate passes for capabilities {instruct: false, completion: true, native_reasoning: false}; refusing (a base model with no reasoning conduct is a configuration, not a fallback)`

**R-44 · constructed object fails the contract probe** `[E-04; D-036]`
> `admission: 'house' constructor returned an object with no callable 'complete' — authored entries are admitted by implementing the declared contract 'complete(Request)->Response', not by name or base class`

**R-45 · packaged class missing from the env block** `[E-04; §e0-class]`
> `compile: model entry 'team_lm' declares origin packaged (corp_llm.TeamLM) but no python env-block dependency provides 'corp_llm' — packaged means uv sync reconstructs it; declare the dep or bake the source`

**R-46 · delta outside the face grammar** `[E-10]`
> `stream (face embed-1): received delta kind 'TextDelta' — embed-1 declares no stream grammar; a streaming embed backend is nonconformant, buffer it`

**R-47 · declared-field delta after end** `[E-10; lm15 grammar]`
> `stream: ThinkingDelta received after 'end' event — grammar lm15/stream-1 is start delta* end|error; late deltas are a protocol violation, not extra data`

**R-48 · MCP server without return-schema evidence** `[E-12; D-027]`
> `load: KB_MCP_SERVER lists 'kb_search' without outputSchema — identity verification requires the return leg; name+input alone under-verifies (the known bridge gap D-027 records)`

**R-49 · undeclared sampling request** `[E-12; D-027]`
> `mcp: server 'kb' issued a sampling request but tool 'kb_search' declares no sampling binding — a sidecar LM call must be a named binding (model + credential_ref) or it is refused`
