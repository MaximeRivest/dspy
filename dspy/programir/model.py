"""Plain-data values shared by ProgramIR frontends and the compiler."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class FrontendProgram:
    """Hold one framework-neutral program snapshot before compilation.

    Frontends resolve live framework state into these plain component values.
    The compiler never receives a live module, LM, adapter, or settings object.
    """

    versions: Mapping[str, str]
    module_tree: Mapping[str, Any]
    signatures: Mapping[str, Any]
    instructions: Mapping[str, str]
    demos: Mapping[str, Any]
    predictor_config: Mapping[str, Any]
    adapters: Mapping[str, Any]
    forwards: Mapping[str, Any]
    tools: Mapping[str, Any]
    interpreters: Mapping[str, Any]
    lms: Mapping[str, Any]
    environment: Mapping[str, Any]
    credentials: tuple[Mapping[str, Any], ...]
    ambient_policy: Mapping[str, Any]
    evaluation: Mapping[str, Any] | None = None
    provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ProgramIR:
    """Hold one compiled ProgramIR manifest and its pending sidecars."""

    manifest: Mapping[str, Any]
    sidecars: Mapping[str, bytes] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        """Return a detached manifest dictionary safe for serialization."""
        return deepcopy(dict(self.manifest))
