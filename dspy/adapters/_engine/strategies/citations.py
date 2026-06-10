"""Native citations as a per-field strategy.

The provider gate lives in ``dspy.adapters.types.citation``'s shared
predicate (imported by both this strategy and the legacy hook); this class
is the engine-side representation: hide the output field, parse the
``citations`` response channel.
"""

from dspy.adapters._engine.patch import AdapterPatch, DebugLink
from dspy.adapters._engine.transforms import HideOutputField


class NativeCitationsStrategy:
    name = "citations.native"
    priority = 500
    exclusive_group = "citations.representation"

    def applies(self, ctx) -> bool:
        from dspy.adapters.types.citation import native_citations_supported

        return native_citations_supported(ctx.lm)

    def contribute(self, ctx) -> AdapterPatch:
        destination = ctx.field.destination_name
        return AdapterPatch(
            field_transforms=[HideOutputField(ctx.field.name, reason="native:Citations")],
            parsers=[NativeCitationsParserHook(destination)],
            debug_links=[
                DebugLink(
                    source=f"Signature.outputs.{destination}",
                    strategy=self.name,
                    destination="provider citations channel",
                )
            ],
        )


class NativeCitationsParserHook:
    """Plan-carried record: Prediction.<field> fills from the provider
    ``citations`` channel."""

    name = "citations.native.parser"

    def __init__(self, destination: str):
        self.destination = destination

    def parse(self, response_view, ctx):
        from dspy.experimental import Citations

        data = response_view.channel("citations")
        if isinstance(data, list):
            return {self.destination: Citations.from_dict_list(data)}
        return {}
