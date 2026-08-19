# E-04 — the authored LM class: structural admission, constructor, capture-or-refuse

**Proves (gap G5):** an authored model-pool entry whose ENGINE is user
code — baked source, structural admission (D-036: implements the
contract, never inherits a base), `constructor: {ref, args}`, the
language tag, and the two capture-or-refuse edges. Successor to the
authored halves of server examples 09/15.

## 1. The frontend

```python
# house_lm.py
# deps: httpx
def make_house_lm(base_url: str, timeout: float = 30.0):
    """A closure-factory LM: complete(Request) -> Response over our gateway."""
    client = httpx.Client(base_url=base_url, timeout=timeout)
    def complete(request):                    # the STRUCTURAL contract — no BaseLM,
        ...                                   #   no inheritance, just the verb
        return response
    return SimpleNamespace(complete=complete,
                           capabilities={"instruct": True})

lm = dspy.Model(make_house_lm("https://llm.corp.internal/v1"))   # admitted structurally
```

## 2. The entry bytes (component 8)

```jsonc
"8_model": {"house": {
  "face": "lm15/chat-1",
  "forward_contract": "typed_lm",
  "class": {
    "origin": "authored",
    "language": "python",
    "source": "models/house_lm.py",              // captured, introspectable — NEVER a dotted path
    "deps": ["httpx"],                           // derived from inline "# deps:"
    "authored_by": "human",
    "constructor": {"ref": "make_house_lm",      // D-036: closure factories admitted
                    "args": {"base_url": {"endpoint_ref": "HOUSE_LM_ENDPOINT"},   // an ARG that is
                             "timeout": 30.0}}},                                  //   a binding: value
                                                                                  //   resolved at load,
                                                                                  //   never baked as URL
  "weights_identity": "corp/house-mix-7b@2026-07",   // the gateway's declared identity —
  "served_aliases": ["house-mix"],                   //   verified against the backend (R-25)
  "placement": {"rung": "http_remote", "contract": "complete(Request)->Response",
                "endpoint_ref": "HOUSE_LM_ENDPOINT", "credential_ref": "HOUSE_LM_KEY"}
}}
```

Load admission (structural, D-036): exec the source in an isolated
namespace → call `constructor.ref` with resolved args → verify the
RESULT implements `complete(Request) -> Response` (a probe against the
declared contract) → bind. Never `isinstance(x, BaseLM)` — admission
verifies the contract, not the spelling. Trust flows from the baked
source + lockfile (the artifact provides the class), not a runtime
flag: there is no `allow_custom_lm_class=True` anywhere.

## 3. The two axes, kept orthogonal (§e0-class)

`class.origin` says how the DEFINITION ships (authored source here;
`packaged` = import path + env-block distribution). `placement.rung`
says where the INSTANCE runs (http_remote here; an authored class can
equally own in-process weights — ex-15's inproc LM was exactly that).
Any origin × any rung.

## 4. Refusal hooks (cross-ref E-11; R-08 is this exemplar's compile edge)

**R-44 · constructed object fails the contract probe** `[E-04; D-036]`
> `admission: 'house' constructor returned an object with no callable 'complete' — authored entries are admitted by implementing the declared contract 'complete(Request)->Response', not by name or base class`

**R-45 · packaged class missing from the env block** `[E-04; §e0-class]`
> `compile: model entry 'team_lm' declares origin packaged (corp_llm.TeamLM) but no python env-block dependency provides 'corp_llm' — packaged means uv sync reconstructs it; declare the dep or bake the source`
