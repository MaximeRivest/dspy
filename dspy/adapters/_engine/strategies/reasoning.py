"""Native reasoning as a per-field strategy.

The decision logic lives in ``dspy.adapters.types.reasoning``'s shared
predicate (imported by both this strategy and the legacy hook, so the two
paths cannot drift); this class is the engine-side representation of its
effects: the ``reasoning_effort`` request kwarg, the hidden output field,
and the plan-carried parser for the ``reasoning_content`` channel.
"""

from dspy.adapters._engine.patch import AdapterPatch, DebugLink
from dspy.adapters._engine.transforms import HideOutputField


class NativeReasoningStrategy:
    name = "reasoning.native"
    priority = 500
    exclusive_group = "reasoning.representation"

    def applies(self, ctx) -> bool:
        from dspy.adapters.types.reasoning import resolve_native_reasoning_effort

        return resolve_native_reasoning_effort(ctx.lm, ctx.lm_kwargs) is not None

    def contribute(self, ctx) -> AdapterPatch:
        from dspy.adapters.types.reasoning import resolve_native_reasoning_effort

        effort = resolve_native_reasoning_effort(ctx.lm, ctx.lm_kwargs)
        ctx.lm_kwargs["reasoning_effort"] = effort

        destination = ctx.field.destination_name
        return AdapterPatch(
            field_transforms=[HideOutputField(ctx.field.name, reason="native:Reasoning")],
            parsers=[NativeReasoningParserHook(destination)],
            debug_links=[
                DebugLink(
                    source=f"Signature.outputs.{destination}",
                    strategy=self.name,
                    destination="lm.config.reasoning (legacy kwarg: reasoning_effort)",
                )
            ],
        )


class NativeReasoningParserHook:
    """Plan-carried record: Prediction.<field> fills from the
    ``reasoning_content`` response channel."""

    name = "reasoning.native.parser"

    def __init__(self, destination: str):
        self.destination = destination

    def parse(self, response_view, ctx):
        from dspy.adapters.types.reasoning import Reasoning

        content = response_view.channel("reasoning_content")
        if content is None:
            return {}
        return {self.destination: Reasoning(content=content)}
