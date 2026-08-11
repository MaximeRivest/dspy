"""Private adapter-engine internals. No stability promise whatsoever.

What survives of the legacy engine is the constrained template language
(`_engine/template/`): eager parsing with teaching errors, pure rendering,
the walker, and the vocabulary-as-data. Everything else — formats, plan
builder, patch machinery — died with the greenfield carve; the entry IS
the adapter now (`dspy.adapters.adapter`).
"""
