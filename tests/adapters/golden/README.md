# Adapter golden parity corpus

This directory freezes the behavior of DSPy's adapters as data. Every case in
`cases.py` runs each adapter against a stub LM that records the exact
messages and kwargs crossing the adapter-to-LM boundary (plus call counts,
outcomes, and selected intermediate render surfaces). The recordings live in
`request/*.json` and are enforced byte-for-byte by
`tests/adapters/test_golden_parity.py`.

The corpus is the primary merge gate for the adapter engine migration
(`scratch/newadapter/epic-quiet-compiler.md`): a refactor is correct exactly
when the corpus cannot tell it apart from the code it replaced.

## Layout

- `harness.py` — recording stub LM (`StubLM`), shared ordered `Recorder`,
  `canonicalize` (deterministic JSON-able rendering; pydantic model classes
  compared structurally via `model_json_schema()`), surface capture.
- `cases.py` — the case registry (~145 cases × sync+async). Case ids follow
  `<family>--<adapter>--<slug>`; families are documented in the module
  docstring.
- `generate_fixtures.py` — writes/checks `request/*.json`; every run
  generates twice in-process and fails on any nondeterminism.
- `request/` — generated fixtures plus `_metadata.json` (recorded library
  versions). Generated files: do not edit by hand.

## Rules

1. **Capability flags are explicit constructor data.** Stub LMs never consult
   litellm's model-capability map. Model *names* (e.g. `anthropic/...`,
   `openai/gpt-5`) are varied independently of flags because adapter code
   sniffs names for provider-specific behavior.
2. **Determinism.** Cases use fixed inputs only: no network, no file reads,
   no randomness, no timestamps. The generator's double-run check enforces
   this.
3. **Bug-for-bug.** The corpus pins current behavior including known quirks
   (e.g. JSONAdapter's falsy-result double call happens on the sync path but
   NOT on the async path, where `_json_adapter_call_common` returns an
   always-truthy coroutine — see the `sync-async-divergence` tag). Fixing a
   pinned quirk is a deliberate later change with its own corpus update,
   never a drive-by.

## Regeneration protocol (rebase protocol)

Fixture regeneration is allowed **only in dedicated corpus-update commits
containing zero `dspy/` source changes.** This makes any fixture diff inside
a feature PR a behavior change by construction.

When rebasing the engine stack over upstream adapter changes:

1. Rebase onto the new base; do NOT touch fixtures in the same commit as code.
2. In a dedicated commit, regenerate from the legacy path at the new base:
   `python tests/adapters/golden/generate_fixtures.py`.
3. Review that commit's fixture diff as the upstream behavior delta — it is
   the human-readable answer to "what did upstream change?"

Verification commands:

```bash
python tests/adapters/golden/generate_fixtures.py --check   # zero-drift gate
pytest tests/adapters/test_golden_parity.py                  # full parity gate
```

## Version pinning

Byte-level parity is authoritative in the pinned CI job (constraints in
`parity-constraints.txt`, recorded versions in `request/_metadata.json`).
The floating test matrix runs the same tests and emits a warning (not a
failure) when library versions drift, because schema-derived strings (JSON
schemas from pydantic) can legitimately change across versions.

## Adding cases

Add a `Case` to `build_registry()` in `cases.py` (unique id, deterministic
payload builder), regenerate in a dedicated commit, and state in the commit
message which behavior family the case pins. Widening the corpus is always
welcome; narrowing or editing existing fixtures requires the protocol above.
