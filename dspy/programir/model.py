"""Plain-data values shared by ProgramIR frontends and the compiler."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class ProgramIR:
    """Hold one compiled ProgramIR manifest and its pending sidecars."""

    manifest: Mapping[str, Any]
    sidecars: Mapping[str, bytes] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        """Return a detached manifest dictionary safe for serialization."""
        return deepcopy(dict(self.manifest))
