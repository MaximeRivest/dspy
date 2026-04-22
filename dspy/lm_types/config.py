"""LMConfig — typed generation parameters that replace untyped kwargs."""

from __future__ import annotations

from typing import Any

import pydantic


class LMToolDef(pydantic.BaseModel):
    """A tool the model can invoke.

    This is the *definition* sent to the LM (name + JSON schema).
    Separate from ``dspy.Tool`` which wraps a callable.
    """

    name: str
    description: str | None = None
    parameters: dict[str, Any] = pydantic.Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )

    model_config = pydantic.ConfigDict(frozen=True)


class LMConfig(pydantic.BaseModel):
    """Generation parameters passed from Adapter to LM.

    This replaces ``lm_kwargs: dict[str, Any]``.  Every parameter the
    adapter pipeline uses is typed.  Provider-specific settings that
    don't fit the universal schema go in ``extensions``.
    """

    # Generation control
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    n: int = 1
    """Number of completions to request."""

    # Tool calling
    tools: list[LMToolDef] | None = None

    # Structured output
    response_format: dict[str, Any] | None = None
    """e.g., ``{"type": "json_object"}`` or a JSON schema."""

    # Reasoning
    reasoning_effort: str | None = None
    """e.g., ``"low"``, ``"medium"``, ``"high"`` (provider-dependent)."""

    # Provider-specific extras (never inspected by adapter/BaseLM contract)
    extensions: dict[str, Any] = pydantic.Field(default_factory=dict)

    model_config = pydantic.ConfigDict(extra="forbid")

    def merge(self, overrides: "LMConfig") -> "LMConfig":
        """Merge another config on top of this one.

        Non-None fields in ``overrides`` win.  ``extensions`` dicts are merged.
        """
        base = self.model_dump()
        over = overrides.model_dump()

        merged_ext = {**(base.pop("extensions", None) or {}), **(over.pop("extensions", None) or {})}

        merged: dict[str, Any] = dict(base)
        for k, v in over.items():
            if v is not None:
                merged[k] = v
        # ``n`` has a default of 1 rather than None; respect overrides' n
        # only if it differs from the default (tests merge with empty configs).
        if overrides.n != 1:
            merged["n"] = overrides.n
        merged["extensions"] = merged_ext
        return LMConfig(**merged)


__all__ = ["LMConfig", "LMToolDef"]
