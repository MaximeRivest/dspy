# IDEALIZED SURFACE — parts marked NEW do not exist in dspy today.

import duckdb

import dspy


class Analyze(dspy.Signature):
    """Answer the analytics question from the database."""

    db: duckdb.DuckDBPyConnection = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()


# `duckdb.DuckDBPyConnection` has NO neutral shape: it is an arbitrary
# host class with compute at the boundary (rendering it means running
# queries: schema listing, sample rows). The two-layer rule's criterion
# says: no neutral shape -> code on the trust ladder. So the codec
# stays DATA — a reference — and the compute is a declared
# transformation LEAF with effects and a placement, exactly like any
# other leaf in the ProgramIR.
def render_duckdb(db: duckdb.DuckDBPyConnection) -> str:
    """Schema + 5 sample rows per table, as the model-facing text."""
    ...


analytics = dspy.ChatAdapter.with_codecs(  # NEW: with_codecs
    per_field={
        "db": dspy.adapters.leaf_codec(  # NEW: the leaf-rule door
            render_duckdb,
            language={"name": "python", "requires": ">=3.11", "packages": ["duckdb>=1.0"]},
            effects=["read:db"],  # it queries; it never writes
            # No isolation floor forced here (unlike a parser): this
            # leaf sees YOUR object, not raw LM output. Placement is
            # unbound in the artifact; the receiver binds it.
        )
    }
)

# The composition report states the residue as fact, receiver decides:
#   "portable except field `db` requires python>=3.11 (+duckdb) at
#    leaf `render_duckdb`"
