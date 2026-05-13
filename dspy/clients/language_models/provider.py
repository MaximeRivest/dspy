"""Provider request boundary for SDK-backed language models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ProviderRequest:
    """An SDK call prepared from an `LMRequest`.

    `ProviderRequest` is deliberately small. JSON APIs usually use `kwargs`,
    positional SDKs can use `args`, and unusual integrations can keep arbitrary
    provider-native data in `data`. `preview` is optional display data for
    diagnostics when `args`/`kwargs` contain objects that are hard to read.
    """

    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    data: Any | None = None
    preview: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
