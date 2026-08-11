"""Adapter-layer refusals: host/authoring errors, not program errors.

The runtime parse failure (`AdapterParseError`) lives in the canonical
typed-error table (`dspy.core.errors`) because a program's `Try` may catch
it. The errors here are the other channel: a malformed entry, a dangling
reference, an unmet requirement, or an unserializable annotation are
authoring/host mistakes refused eagerly with teaching messages — never
caught by program handlers, never silently degraded.
"""

from dspy.core.errors import AdapterParseError

__all__ = ["AdapterError", "AdapterParseError", "EntryError", "UnserializableTypeError"]


class AdapterError(ValueError):
    """An adapter construction or binding the engine refuses, loudly."""


class EntryError(AdapterError):
    """An entry the serde refuses: malformed shape, incompatible version,
    dangling reference, or an unmet declared requirement — always naming
    the offender."""


class UnserializableTypeError(TypeError):
    """A field annotation with no wire meaning reached the render/parse
    boundary."""
