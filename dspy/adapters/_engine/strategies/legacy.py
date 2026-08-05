"""The legacy type hook adapted to the strategy contract.

A third-party Type in ``native_response_types`` with no registered strategy
is auto-wrapped in :class:`LegacyTypeHookStrategy`, so the builder runs one
uniform loop over strategies with no special-cased third-party branch. The
documented ``Type.adapt_to_native_lm_feature`` contract is honored silently
and byte-identically to the inline block this replaces: hook always invoked,
whole-signature rewrite respected (via the patch's
``replace_render_signature`` channel), Hide transforms and the
``parse_lm_response`` parser recorded on the plan, and the strategy trace
self-reported with the observed-effects decision rule.
"""

from dspy.adapters._engine.parser_hook import ThirdPartyNativeParserHook
from dspy.adapters._engine.patch import AdapterPatch, StrategyTrace
from dspy.adapters._engine.transforms import HideOutputField


class LegacyTypeHookStrategy:
    """Adapts ``Type.adapt_to_native_lm_feature`` to ``TypeStrategy``."""

    priority = 0
    exclusive_group = None

    def __init__(self, annotation):
        self.annotation = annotation
        self.name = f"type_hook:{annotation.__name__}"

    def applies(self, ctx) -> bool:
        """The legacy hook was always invoked; the hook itself decides
        internally whether to act, reported post-hoc in the trace."""
        return True

    def contribute(self, ctx) -> AdapterPatch:
        lm_kwargs = ctx.lm_kwargs
        kwargs_before = dict(lm_kwargs)
        fields_before = set(ctx.signature.output_fields)

        adapted = self.annotation.adapt_to_native_lm_feature(ctx.signature, ctx.field.name, ctx.lm, lm_kwargs)

        deleted = sorted(fields_before - set(adapted.output_fields))
        return AdapterPatch(
            replace_render_signature=adapted,
            field_transforms=[
                HideOutputField(deleted_name, reason=f"native:{self.annotation.__name__}") for deleted_name in deleted
            ],
            parsers=[ThirdPartyNativeParserHook(self.annotation, ctx.field.name)],
            strategy_trace=[
                StrategyTrace(
                    strategy=self.name,
                    field=ctx.field.name,
                    decision="selected" if deleted or kwargs_before != lm_kwargs else "skipped",
                    reason="third-party adapt_to_native_lm_feature",
                )
            ],
        )
