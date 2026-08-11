"""Typed ProgramIR refusal errors."""

from __future__ import annotations

from typing import Any


class ProgramIRRefusal(ValueError):  # noqa: N818 — "refusal" is the contract's own word for this act
    """Refuse a malformed or unlinked ProgramIR value with a stable code."""

    def __init__(self, code: str, component: str, detail: dict[str, Any], message: str):
        super().__init__(message)
        self.code = code
        self.component = component
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        """Return the language-neutral refusal record."""
        return {"code": self.code, "component": self.component, "detail": self.detail}
