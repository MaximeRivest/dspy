# Cross-language — postures, profiles, and the question bank

**Status:** planning document (2026-08-06). Companion to `IR-program-spec.md`
(§e0-lang, ratified this date) and `adapter-ir-spec.md` (§4/§9 additions).
Sources: two survey sweeps over this repo, `~/Projects/lm15-dev/`, the golden
corpus, and the MCP projects (`mcp2py`, `mcptocli`, `rat`), plus the D-022–027
ratification session. This file is the *don't-lose-the-question* list for a
TypeScript/Go port campaign — it decides nothing; decisions get D-numbers.

---

## The three postures — "port" means one of these, and they cost differently

1. **IR executor** — an engine that loads and runs shipped `ProgramIR`
   artifacts. The declared-tier profile (D-023) is its v1; the rung-walk rule
   (D-022) is its growth path. Feasible against today's specs + corpora once
   the promotion gaps below close.
2. **Conformant adapter library** — render/parse over SignatureCore, held to
   the adapter-IR corpus (the lm15-of-adapters). Blocked on Epic D landing
   and the corpus promotion checklist below.
3. **Authoring framework** — signatures/modules/optimizers native in TS/Go.
   Has *no story at all* under current specs (forwards are captured from
   Python source; optimizers have no contract surface). Largest open ground.

Every question below is tagged where it bites: [1], [2], [3], or [all].

## What is already decided (the spine a port builds on)

- D-015/Epic E: lm15 is the LM substrate; its contract repo + 304-check
  harness is the one graduated cross-language surface today.
- D-020/D-021: the adapter layer graduates to its own contract repo post-D/E;
  until then `_engine/` is reference behavior under an import boundary.
- D-022: rung-walk — foreign-language bake degrades to declare, recorded.
- D-023: the declared-tier profile is a named conformance class.
- D-024: `ir_version` + vocabulary versions block in artifacts and presets.
- D-025: `language` tag on authored code; per-language env blocks.
- D-026: authored adapter code does not rung-walk; templates-as-data are the
  portable customization path.
- D-027: sidecar wire-contract candidates (rat kernel protocol; MCP with
  outputSchema; bearer_env≡credential_ref); text deferred to E/F.

## What already exists (assets, with their known holes)

- **lm15-go and lm15-ts exist and pass the full 304-check harness** (request
  110, response 102, stream 8, error 16, serde 68 — go/ts pass status per
  their own READMEs at the pre-review freeze; only python-shim harness
  reports are archived in the contract repo). Holes: unpublished (no
  npm/module release engineering), no CONTRACT_PIN discipline (sibling-path
  checkout), frozen at pre-review API. Corpus holes that matter downstream:
  no streamed-tool-call/thinking/mid-stream-error fixtures, no openai_chat
  error fixtures, compat presets unpinned, serde/mapping rules live outside
  the contract repo.
- **Golden corpus**: 205 pure-JSON case fixtures (request 168 / parse 31 /
  callbacks 6), plus `request/_metadata.json` recording library versions.
  Fatal for foreign use today: **case stimuli are Python code** — no
  SignatureCore serialization exists in any fixture.
- **rat**: kernel protocol (JSON-lines execute with sessions, streaming,
  events) + runtime.yaml materialization + daemon lifecycle. Missing: vars
  marshaling, typed errors, version negotiation, auth.
- **mcptocli**: Go proof of MCP-tool-as-subprocess with JSON I/O; bearer_env
  credential seam; 8-category exit-code taxonomy. Drops outputSchema.
- **mcp2py**: Python consuming side + bidirectional (sampling/elicitation/
  roots). Sampling = the D-027 hazard; no return schemas; no stdio env seam.

---

## The question bank

### 1 · Scope & product definition

- **Optimizer scope** [3]: is optimization in scope for any foreign port —
  metrics-as-data (same three-origin problem as tools, never spec'd), the
  Evaluate/parallel-executor semantics, dataset loading, and cross-language
  RNG determinism (must Go MIPROv2 seed=0 reproduce Python's stream)? Gates
  the product definition: execute-only is a serving story; optimizing needs
  contract surfaces that don't exist.
- **Which dspy** [all]: port of the fork's contract stack, of upstream
  behavior, or a conformance layer offered to existing community ports
  (Ax, dspy-go)? Naming/namespace/trademark questions ride on this.
- **The 3×3 interop matrix** [3]: which author-language × engine-language
  cells are promised? If TS authoring must emit the closed AST grammar, the
  grammar needs respecifying as language-neutral (node semantics defined
  operationally, not "what `ast.dump` emits") — or foreign frontends author
  in a builder DSL.
- **Legacy migration** [1]: existing `.json`/`.pkl` saves are not ProgramIR;
  ruling needed — "ports read ProgramIR only; migration is Python tooling"
  — and someone owns the exporter. The realistic first Go customer holds an
  already-optimized Python program.
- **Public compatibility wording** [all]: the promise is "declared-tier
  profile + rung-walk", never "run any DSPy program in Go". The preflight
  checker (D-023) and the engines must share one predicate.

### 2 · Governance, versioning, staffing

- **Contract-repo home** [all]: post-graduation, do D-numbers govern all
  implementations (contract-repo-centric governance) or does each port fork
  its own log? Shared home or the vocabularies fork — the D-010 disease.
- **Release train** [all]: feature-lag SLA per port (N or N-1), a
  required-capabilities block engines check, and named maintainers per
  language *before* any port is announced. The Julia port is the cautionary
  steady state: silently divergent, no harness to notice.
- **Conformance CI location** [2]: where does public multi-language
  conformance run so external ports can self-certify without Maxime's
  server? A green claim that can't be reproduced is not a claim.
- **lm15 residue** [1,2]: extract serde-rules.md/mapping-rules.md into
  lm15-contract (one pinnable repo); add CONTRACT_PIN to lm15-go/ts; publish
  both; decide whether ports track the post-review API renames.

### 3 · Artifact format

- **manifest.json JSON Schema** [1]: publish the full schema (12 components,
  pools/bindings, placement blocks) as JSON Schema — nothing normative
  exists; the only full artifacts live on a private server. D-γ's
  regenerated examples 01–04 are the natural first public artifact corpus.
- **Forward-AST JSON encoding** [1,3]: the node-set grammar is proven but
  its serialized encoding exists only in off-repo example manifests; needs a
  schema + operational semantics (truthiness of empty containers, iteration
  order, exception hierarchy for Try match) that CPython currently supplies
  implicitly.
- **Packed single-file form** [1]: the directory layout assumes a POSIX
  disk; TS edge runtimes need a packed encoding (tar? bundle?) with
  equivalence rules. Small spec now vs three de-facto formats later.
- **Trust & signing** [1]: artifacts are mobile code; exec-at-load needs an
  integrity story (canonical hash, signature block, trust tiers for
  authored-origin code) before Go/TS serving stacks can adopt.
- **Refusal taxonomy as data** [1,2]: error codes for link/verify/refuse
  paths — Python type names and English format strings are pinned in
  fixtures today; Go has no exceptions. Needed for L5 to be testable at all.

### 4 · Execution semantics (beyond render/parse)

- **settings/context model** [1]: is ambient override (dspy.context) part of
  portable execution semantics or does the closed grammar guarantee forwards
  can't observe it? If the latter, state it as a law; if the former, every
  engine imports dynamic scoping.
- **Prediction/completions contract** [1]: n>1 ordering, first-completion
  law, the temperature-bump heuristic (0.15/0.7 magic numbers) — runtime
  policy or portable semantics?
- **Sync/async fixture ruling** [2]: single-color languages cannot exhibit
  both modes; which mode is normative per fixture, and do legitimate
  sync/async divergences get erased before corpus promotion?
- **Cache-key portability** [1]: is "same program + same inputs ⇒ same key
  across languages" promised (then the key algorithm is spec'd like serde)
  or is caching per-language idiom? Mixed fleets (Python optimizer, Go
  server) want shared hits.
- **String-semantics profile** [1,2]: offsets in code points, casing tables
  (infer_prefix uses Python's capitalize/isupper), character classes — 
  UTF-16 TS vs code-point Python makes fixtures near non-BMP content
  undefined without a two-page profile.

### 5 · Adapter byte-parity (the hard surface for posture 2)

Each of these is "Python semantics leaked into prompt bytes" and needs a
neutral definition + fixtures, or an explicit bug-compat/cleanup ruling
before corpus promotion:

- **text-pythonish rendering**: str() semantics (True/False/None, shortest
  float repr), json.dumps separators (`', '`/`': '` vs JS's none).
- **Guillemet blob grammar** for list[str] inputs (escaping, indent,
  numbering).
- **serialize_for_json coercion table** (pydantic dump_python + str()
  fallback) enumerated per supported type.
- **parse_value chain**: json_repair's algorithm (third-party!), which
  Python-literal forms are contract (`"{'a': 1}"`, tuples?), pydantic lax
  coercion rules. The most divergence-prone surface — a port doing
  JSON.parse-only hard-fails where Python succeeds, changing eval scores.
- **JSON-schema dialect**: pydantic's exact output (anyOf ordering, $defs
  naming, titles, prefixItems) is embedded in prompt strings; bless it
  per-version, or regenerate fixtures from a spec-owned schema printer.
- **Structured-output schema** for response_format incl. enforce_required
  rewrite — provider-facing, so divergence changes completions.
- **Regex portability**: Go RE2 lacks backreferences/recursion; parsers need
  an acceptance-language spec + corpus oracle, not regex translation.
- **Template grammar EBNF** (wait for post-D revision, then formalize):
  token syntax, escaping, separator semantics, directive expansion order —
  two independent parsers must agree.
- **Instruction normalization**: cleandoc/dedent specified self-contained;
  pick the canonical Python minor for existing fixture bytes.
- **String-signature grammar** [3]: which type expressions are contract; no
  frame-walking/import resolution in ports — custom types become a registry.

### 6 · Corpus promotion checklist (the batch gate before graduation)

Nothing graduates until each has a ruling; quirks pinned into a graduated
corpus become permanent cross-language obligations:

- [ ] **Case stimuli become data** (`core.json` per case: SignatureCore +
      values + preset) — the single highest-leverage work item; today a
      foreign implementation cannot run one case.
- [ ] Enum repr fixture (`<GoldenPriority.HIGH: 'high'>`) → neutral encoding;
      `__type__`/`__repr__` escape hatches formally banned + linted.
- [ ] Error expectations: Python type names/format strings → neutral codes;
      fix or drop the dead ImportError probe (5 fixtures pin an accident).
- [ ] python_sensitive dedent cases regenerated with explicit instructions.
- [ ] Sync/async divergence ruling (see §4).
- [ ] Byte-vs-structural equality ruling (float repr, key sorting) written
      down before someone adds 1e17 or -0.0.
- [ ] Callback family: double-fire/async-no-parse quirks — bug-compat or
      replaced by a spec'd observability contract?
- [ ] Missing families: codec round-trip battery, template-vocabulary
      fixtures, streaming event sequences, per-preset (not per-class)
      fixtures speaking the spec's vocabulary.
- [ ] Corpus versioning: frozen spec-owned artifact with a version number,
      not regenerable-at-will Python output.
- [ ] Generative layer: shared case-generator + differential comparison
      across implementations (fixed corpora can't catch lenient-parse drift
      on novel inputs).

### 7 · Tools, interpreter, sidecar (posture 1 growth path)

- **Interpreter wire contract** (D-027 base: rat): add execute(code, vars) →
  typed result marshaling, typed error codes, protocol version negotiation,
  auth beyond loopback; then spec it in E/F.
- **Tool rung**: the `outputSchema` declaration must be added to whichever
  bridge ships (all three drop it); `structuredContent` already exists in
  rat (emits) and mcptocli (passthrough + render), absent only in mcp2py.
  Identity-verify = name + input schema + return schema against tools/list.
- **Metric leaf wire form** [1,3]: (example, prediction) → score over the
  same sidecar plumbing; needed for re-scorable-anywhere off-Python.
- **Sidecar handshake/lifecycle spec**: spawn → initialize → enumerate →
  verify identity → bind (mcp2py's sequence + rat's health/stale-replacement
  as reference); who owns restart policy.
- **Bidirectional channel policy**: sampling refused-or-declared (D-027);
  elicitation/roots = receiver UX, allowed.
- **Rung-0 ceiling ruling** [1]: is rung 0 (in-process weights, tokenizer
  parity) formally out of scope for foreign engines — declared-tier as
  permanent ceiling — or is there a designated inference-backend story?
  Declaring the ceiling stops ports budgeting for an engine they'll never
  ship.

### 8 · Language-specific (posture 2/3)

- **TS**: zod/typebox as SignatureCore surface? typed Prediction via
  inference (the reason a native TS framework beats calling Python)? ALS
  caveats in edge runtimes; 2^53 number limits; UTF-16 offsets.
- **Go**: struct tags vs builders for signatures; map-ordering discipline
  (slice-of-pairs for render paths); which JSON-schema library/draft;
  float-preserving JSON codec (adopt lm15-go's jsonutil).
- **Shared-kernel option** [2]: cost the three-clean-rooms maintenance bill
  against one WASM/codegen kernel for the worst byte-parity layers
  (json_repair equivalence, schema printer, template renderer) before
  defaulting to clean-rooms-because-lm15-did.

### 9 · Explicitly deferred (with the wait reason)

- Template grammar EBNF + preset serde JSON Schema — Epic D is the proving
  ground; fix-spec-first forbids freezing before D-α.
- Sidecar wire-contract text — E/F, with D-027's candidates as the base.
- WASM as universal authored-code target — revisit when authored-Go demand
  is real (D-025 notes it).
- Trust/signing spec — needs its own design pass; placeholder in §3 above.
- Starlark re-evaluation — NOT triggered; D-022 keeps the goal soft. §d's
  ratified trigger is execution hardening ("forwards must run natively in a
  Rust dspy with no re-implementation"); foreign *authoring* of forwards in
  native syntax would be a second, distinct trigger (the 3×3 matrix
  question).
