"""Composable contributions to an ``AdapterPlan``.

``AdapterPatch`` COMPOSES the public :class:`dspy.core.types.LMRequestPatch`
(request-side channels: messages, parts, tools, config, field deletions)
rather than mirroring it, and adds the response-side channels a request-only
patch cannot express: parser hooks, field transforms, warnings, and debug
links. A request-only patch was found fatally incomplete in design review —
every representation decision must carry its parser.

Merging is deterministic and associative: list channels append in
contribution order; config merges via ``LMRequestPatch``'s own rules;
metadata conflicts on differing values are explicit errors.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from dspy.adapters._engine.ir import AdapterPlan
from dspy.adapters._engine.transforms import HideInputField, HideOutputField
from dspy.core.types import LMRequestPatch


@dataclass(frozen=True)
class DebugLink:
    """Provenance: how one element moved from source to destination.

    Destinations use the rich dotted slot vocabulary from the design docs
    (e.g. ``user.input_parts.LMImagePart``, ``LMRequest.config.reasoning``)
    even though the plan stores collapsed slots — links are where the richer
    vocabulary lives.
    """

    source: str
    strategy: str
    destination: str


@dataclass(frozen=True)
class StrategyTrace:
    """One strategy-selection decision, recorded for explainability.

    ``resolved_by`` records which registry key answered the double-key
    lookup (``"role"`` or ``"annotation"``, epic-C §6 stage 2); None for
    entries that predate resolution or self-reported traces.
    """

    strategy: str
    field: str | None
    decision: Literal["candidate", "selected", "skipped", "fallback", "conflict"]
    reason: str
    resolved_by: str | None = None


class PatchMergeError(ValueError):
    """Two contributions wrote conflicting values to an exclusive channel."""


@dataclass
class AdapterPatch:
    """One composable contribution to a plan.

    ``replace_render_signature`` is a legacy-compat channel used only by the
    auto-wrapped third-party type hook (``LegacyTypeHookStrategy``): the
    documented hook contract returns a whole rewritten signature, which a
    delete-only reconstruction could silently truncate. The builder consumes
    it in place of transform-derived deletions; it never lands on the plan.
    """

    request: LMRequestPatch = field(default_factory=LMRequestPatch)
    field_transforms: list[Any] = field(default_factory=list)
    parsers: list[Any] = field(default_factory=list)
    debug_links: list[DebugLink] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategy_trace: list[StrategyTrace] = field(default_factory=list)
    replace_render_signature: Any = None

    def merge(self, other: "AdapterPatch") -> "AdapterPatch":
        """Return a new patch equal to this patch followed by ``other``."""
        if self.replace_render_signature is not None and other.replace_render_signature is not None:
            raise PatchMergeError("Two contributions both replace the render signature")
        return AdapterPatch(
            request=self.request.merge(other.request),
            field_transforms=[*self.field_transforms, *other.field_transforms],
            parsers=[*self.parsers, *other.parsers],
            debug_links=[*self.debug_links, *other.debug_links],
            warnings=[*self.warnings, *other.warnings],
            strategy_trace=[*self.strategy_trace, *other.strategy_trace],
            replace_render_signature=self.replace_render_signature or other.replace_render_signature,
        )

    def merge_into(self, plan: AdapterPlan) -> None:
        """Apply this patch's channels onto a plan's collapsed slots.

        ``LMRequestPatch`` interop happens here and only here:
        ``delete_input_fields`` / ``delete_output_fields`` become Hide
        transforms (the engine's canonical representation of native-feature
        field removal, with None-backfill metadata); ``messages`` insert
        between system and the current user turn (the history slot);
        ``assistant_parts`` map to the assistant prefill slot.
        """
        plan.system_parts.extend(self.request.system_parts)
        plan.history_messages.extend(self.request.messages)
        plan.user_parts.extend(self.request.user_parts)
        plan.assistant_prefill_parts.extend(self.request.assistant_parts)
        plan.tools.extend(self.request.tools)

        if self.request.config is not None:
            if plan.config is None:
                plan.config = self.request.config
            else:
                merged = LMRequestPatch(config=plan.config).merge(LMRequestPatch(config=self.request.config))
                plan.config = merged.config

        for key, value in self.request.metadata.items():
            if key in plan.metadata and plan.metadata[key] != value:
                raise PatchMergeError(f"Conflicting metadata for key {key!r}: {plan.metadata[key]!r} vs {value!r}")
            plan.metadata[key] = value

        plan.field_transforms.extend(
            HideInputField(name, reason="lm_request_patch.delete_input_fields")
            for name in self.request.delete_input_fields
        )
        plan.field_transforms.extend(
            HideOutputField(name, reason="lm_request_patch.delete_output_fields")
            for name in self.request.delete_output_fields
        )

        plan.field_transforms.extend(self.field_transforms)
        plan.parsers.extend(self.parsers)
        plan.debug_links.extend(self.debug_links)
        plan.warnings.extend(self.warnings)
        plan.strategy_trace.extend(self.strategy_trace)
