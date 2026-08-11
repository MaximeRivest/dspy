# Adapter-as-data audit (2026-08-10) — verdict, evidence, direction

Verification pass over the Epic D claim "the adapter is fully data",
prompted by the capability/trust work (the claim is load-bearing for
`spec/trust.md`'s guarantees). Per-field verdict with evidence:

| entry field | verdict | evidence |
|---|---|---|
| template | **(a) pure interpreted data** — closed vocabulary, no eval anywhere in `_engine/template/*`; `{field('name')}` is an escape spelling, not a call | serde.py:66; template/vocabulary.py |
| parser | **(a) closed enum** {chat, json, xml, full_text}; authored parsers have no slot to travel in — refuse by construction | serde.py:159–163; D-026 |
| strategies | **(b) name-reference only** — strings validated against builtin vocabulary or locally *registered* names; code never travels; dangling name = loud load error | serde.py:147–153; strategies/vocabulary.py:69–118 |
| config | **(a) pure data** — shape-checked dict | serde.py:156–157 |
| codecs | **(b) OR (c)** — string refs are (b); origin-tagged dicts: `builtin` (b), `packaged` (b, version-checked import), **`authored` = (c): Python source travels in the artifact and `load_preset` runs it via `exec`** | serde.py:177–205; admission.py:220–319, exec at admission.py:312 |

Sharpenings that matter:

- **The sha256 identity (ADP-011) is integrity, not authorship** — any
  author computes a matching hash for malicious code. Signatures +
  transparency logs are the authorship story (future, spec/trust.md's
  origin fact).
- **The admission battery executes the code it certifies.** The
  round-trip probe battery is not a sandbox; by the time it certifies,
  the authored code has run.
- **Eager materialization breaks the grade-1 claim**: `_check_codec_ref`
  execs authored codecs at *link* time (serde.py:196–200), so
  "read+link executes nothing" is false for artifacts carrying an
  authored codec. Fix direction (spec/trust.md pairing rule,
  ratification ask 2): link validates shape + hash only; execution is
  an explicit grade-2 materialize act at the required isolation rung.

Accurate claim wording from the audit: *"adapters are data, except
codecs may opt into shipping executable Python under the authored
origin; everything else is data or a pre-existing-code reference."*

## Direction (Maxime, 2026-08-10)

Custom parsers, custom strategies, and custom LMs are **wanted, not
refused**. The path is the pairing rule (`spec/trust.md`): trust
deficit paid with isolation —

- an authored **parser** is an authored leaf on the injection path by
  definition (its input is raw LM output); forced isolation rung +
  integrity-mode awareness;
- an authored **strategy** is the same shape (it forms prompts and
  parses fragments);
- a custom **LM** is just another leaf whose placement crosses the
  trust boundary — endpoint-bound = secrecy sink + integrity source;
  weight-baked rung-0 = neither. Nothing new needed beyond stating it.

Self-declared `effects` on authored code are worth nothing (hostile
code lies); the rung enforces what the declaration claims. Composition
disclosure is aggregated (one bill-of-materials report per artifact),
never per-component prompts — consent fatigue makes many small
warnings equal zero. Receiver postures (`workbench`/`reviewed`/
`hardened`, non-transitive) decide; `workbench` is the full-trust
local data-science mode with zero ceremony.
