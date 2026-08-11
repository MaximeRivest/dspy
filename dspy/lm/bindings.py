"""Explicit bindings: one module-level table, overrides passed by hand.

The whole configuration story is a plain dict. `dspy.configure(lm=...)`
writes the default binding table; a predictor holding its own bindings
passes them as `overrides` at resolve time and wins. There are no
thread-locals, no `dspy.context`, and no ambient settings object — what
a program runs with is always visible in one of two dicts.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = ["BINDINGS", "BindingError", "configure", "resolve"]

#: The default binding table. A plain dict: read it, print it, test it.
BINDINGS: dict[str, Any] = {}


class BindingError(Exception):
    """A required binding is missing. Host configuration, not program error.

    Deliberately outside the catchable typed-error table: a program's
    `Try` handles runtime failures, not the host forgetting to bind an LM.
    """


def configure(**bindings: Any) -> None:
    """Write the default binding table.

    Args:
        **bindings: Named bindings, e.g. `lm=dspy.LM(...)`. Passing
            `None` for a name removes that binding.

    Examples:
        ```python
        dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))
        dspy.configure(lm=None)  # unbind
        ```
    """
    for name, value in bindings.items():
        if value is None:
            BINDINGS.pop(name, None)
        else:
            BINDINGS[name] = value


def resolve(name: str = "lm", overrides: Mapping[str, Any] | None = None) -> Any:
    """Resolve one binding: per-predictor overrides win, else the default table.

    Args:
        name: The binding name, `"lm"` by default.
        overrides: A predictor's own bindings, if it holds any.

    Returns:
        The bound object.

    Raises:
        BindingError: When neither the overrides nor the default table
            bind `name`.
    """
    if overrides is not None and overrides.get(name) is not None:
        return overrides[name]
    if name in BINDINGS:
        return BINDINGS[name]
    raise BindingError(
        f"No {name!r} binding is configured. Call dspy.configure({name}=...) once at "
        f"startup, or pass a per-predictor binding. There is no ambient default."
    )
