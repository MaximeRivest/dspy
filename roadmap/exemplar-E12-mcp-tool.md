# E-12 — the MCP tool rung: bytes now, execution with epics E/F

**Proves (gap G12, DEFER-marked):** the outermost tool rung as bytes —
identity verification incl. `return_schema` via MCP `outputSchema`, and
the sampling-as-declared-binding rule (D-027). The wire contract text
itself remains deferred to E/F (D-027's ruling); this exemplar pins the
ARTIFACT side only, so the byte shape cannot drift while execution
waits.

## 1. The entry (component 6 at the outer rung)

```jsonc
"6_tools": {"kb_search": {
  "name": "kb_search",
  "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
  "return_schema": {"type": "object", "properties": {"passages": {"type": "array"}}},
  "source": null, "deps": [], "language": null,          // nothing local travels — the body
                                                          //   is the server's
  "kind": "call", "dispatch": ["model", "program"],
  "placement": {"rung": "mcp",
                "contract": "mcp/tools-1",
                "endpoint_ref": "KB_MCP_SERVER",
                "credential_ref": "KB_MCP_TOKEN",
                "isolation": "remote"},
  "identity_verification": {                              // D-027: tools/list name + input schema
    "requires": ["outputSchema"],                          //   under-verifies; the return leg is
    "match": "schema_equal"}                               //   REQUIRED evidence — a server not
}}                                                         //   exposing it fails load (R-48)
```

## 2. The sampling rule (the smuggled-binding disease, closed)

An MCP server may issue sampling requests (asking OUR side to run an
LM). Lawful ONLY as a declared binding — the tool entry carries it or
the engine refuses the request:

```jsonc
"6_tools": {"summarize_kb": {"…": "…",
  "sampling": {"model": "sampler_lm",                     // an 8_model pool entry (declared face,
               "credential_ref": "SAMPLER_KEY"}}}         //   endpoint binds) — visible to the
                                                          //   placement census, priced in View 2
```

Leaves declare; representation layers lower (TwoStep had to expand —
a leaf's LM call only has to be NAMED). Elicitation/roots stay
receiver-side UX, no declaration needed (D-027).

## 3. Refusal hooks (cross-ref E-11)

**R-48 · MCP server without return-schema evidence** `[E-12; D-027]`
> `load: KB_MCP_SERVER lists 'kb_search' without outputSchema — identity verification requires the return leg; name+input alone under-verifies (the known bridge gap D-027 records)`

**R-49 · undeclared sampling request** `[E-12; D-027]`
> `mcp: server 'kb' issued a sampling request but tool 'kb_search' declares no sampling binding — a sidecar LM call must be a named binding (model + credential_ref) or it is refused`

**DEFER:** execution semantics (kernel-protocol marshaling, session
negotiation, auth handshake) ride epics E/F per D-027; this file's
shapes are the fixture seed so the manifest side is already fixed.
