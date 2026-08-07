"""The adapter engine's intermediate representation.

``AdapterPlan`` is the explicit, inspectable artifact between a signature
call and the final ``LMRequest``: request slots (collapsed schema), the
model-facing field layer (``RenderField``), response-side parser hooks, and
provenance (debug links, warnings, strategy trace). Planning is pure — a
plan can be built, inspected, and tested without any LM call.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from dspy.core.types import LMConfig, LMMessage, LMPart, LMToolSpec


@dataclass
class RenderField:
    """A model-facing field, decoupled from the semantic signature field.

    ``name`` is what the model sees; ``original_name`` is the semantic
    DSPy/Prediction key it maps back to (``None`` for auxiliary fields that
    never become prediction values). ``hidden`` fields are skipped by
    ordinary text rendering but may still be filled by parser hooks — hiding
    is a rendering decision, never a semantic deletion.
    """

    name: str
    original_name: str | None
    role: Literal["input", "output"]
    annotation: Any
    value: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)
    hidden: bool = False

    @property
    def destination_name(self) -> str:
        """The key parsed values are stored under in the prediction."""
        return self.original_name if self.original_name is not None else self.name


@dataclass
class AdapterPlan:
    """Inspectable plan for one adapter call.

    Request slots use the COLLAPSED schema only. The richer dotted slot
    vocabulary (``system.strategy_instructions``, ``user.input_parts``, ...)
    is used in ``DebugLink`` destinations and documentation, not as
    structure: collapsing keeps the IR small while provenance keeps it
    explainable.
    """

    # Request slots.
    system_parts: list[LMPart] = field(default_factory=list)
    developer_parts: list[LMPart] = field(default_factory=list)
    history_messages: list[LMMessage] = field(default_factory=list)
    user_parts: list[LMPart] = field(default_factory=list)
    assistant_prefill_parts: list[LMPart] = field(default_factory=list)
    tools: list[LMToolSpec] = field(default_factory=list)
    config: LMConfig | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Strategy-contributed template fragments, keyed by fragment target
    # ("system" | "user"); the preset's positional {fragments(...)} slots
    # interpolate them, and empty targets cost zero bytes.
    fragments: dict[str, list[str]] = field(default_factory=dict)

    # Model-facing field layer.
    input_fields: list[RenderField] = field(default_factory=list)
    output_fields: list[RenderField] = field(default_factory=list)
    field_transforms: list[Any] = field(default_factory=list)

    # Response side.
    parsers: list[Any] = field(default_factory=list)

    # Provenance.
    debug_links: list[Any] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    strategy_trace: list[Any] = field(default_factory=list)

    @classmethod
    def from_signature(cls, signature, inputs: dict[str, Any] | None = None) -> "AdapterPlan":
        """Initialize render fields one-to-one from a signature.

        The signature itself is never mutated by planning; all edits to the
        model-facing view happen through field transforms.
        """
        from dspy.adapters._engine.roles import resolve_semantic_role

        inputs = inputs or {}
        plan = cls()
        for name, field_info in signature.input_fields.items():
            plan.input_fields.append(
                RenderField(
                    name=name,
                    original_name=name,
                    role="input",
                    annotation=field_info.annotation,
                    value=inputs.get(name),
                    metadata={"semantic_role": resolve_semantic_role(field_info, field_name=name)},
                )
            )
        for name, field_info in signature.output_fields.items():
            plan.output_fields.append(
                RenderField(
                    name=name,
                    original_name=name,
                    role="output",
                    annotation=field_info.annotation,
                    metadata={"semantic_role": resolve_semantic_role(field_info, field_name=name)},
                )
            )
        return plan

    def visible_input_fields(self) -> list[RenderField]:
        return [f for f in self.input_fields if not f.hidden]

    def visible_output_fields(self) -> list[RenderField]:
        return [f for f in self.output_fields if not f.hidden]

    def find_field(self, role: str, name: str) -> RenderField | None:
        fields = self.input_fields if role == "input" else self.output_fields
        for render_field in fields:
            if render_field.name == name:
                return render_field
        return None
