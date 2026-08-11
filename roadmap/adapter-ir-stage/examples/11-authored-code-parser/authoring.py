# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import dspy


class Extract(dspy.Signature):
    """Extract the ledger entries."""

    document: str = dspy.InputField()
    entries: list[dict] = dspy.OutputField()


# The honest code tail: a stateful, grammar-coupled recovery parser
# that no combinator pipeline expresses (balance-tracking across
# partially truncated records). It is CODE, so it walks the trust
# ladder like every other component — language env block, forced
# isolation (a parser's exposure is total by definition), identity.
def parse_ledger(text: str, fields: dict) -> dict:
    # ... 60 lines of bespoke recovery heuristics, match statements,
    # running-balance validation (python>=3.12 syntax) ...
    ...


ledger = dspy.make_adapter(
    name="ledger_recovery",
    template=[
        {"role": "system", "content": "{instruction}"},
        {
            "role": "user",
            "content": "{% for f in inputs separator='\\n\\n' %}{f.name}:\n{f.value}{% endfor %}\n\nEmit the ledger records, one per line.",
        },
    ],
    parser=dspy.adapters.authored_parser(  # NEW: the level-3 door
        parse_ledger,
        language={"name": "python", "requires": ">=3.12"},
        # isolation is NOT optional here — the pairing rule forces the
        # floor for parsers: raw LM output in, always on the injection
        # path.
        isolation="sandboxed",
    ),
)

# Export does NOT refuse (the D-026 amendment): the entry ships with
# its source baked in and its requirement set declared. Receivers that
# meet the requirements run it isolated; receivers that don't refuse
# NAMING the requirement — see notes.md for the exact message.
entry = ledger.dump_entry()
assert entry["requires"]["languages"][0]["name"] == "python"
