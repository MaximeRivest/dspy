"""DSPy type strategies that render to normalized LM request pieces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from dspy.clients.language_models.types import LMOutput, LMRequestPatch

T = TypeVar("T")


@dataclass(frozen=True)
class TypeStrategy(Generic[T]):
    """Strategy for rendering and parsing one DSPy marker/value type.

    A DSPy type such as `Image`, `Reasoning`, `Code`, or `History` expresses
    user intent in a signature. A `TypeStrategy` expresses how that type should
    be represented for one LM call: as normal adapter text, as native LM parts,
    as config, as extra messages, or as parsed response parts.
    """

    marker_type: type[T]

    def matches(self, annotation: Any) -> bool:
        """Return whether this strategy applies to a signature annotation."""
        try:
            return annotation == self.marker_type or (
                isinstance(annotation, type) and issubclass(annotation, self.marker_type)
            )
        except TypeError:
            return False

    def render_input(self, *, field_name: str, field: Any, value: Any, adapter: Any) -> LMRequestPatch:
        """Render one input field into a partial normalized LM request."""
        return LMRequestPatch()

    def render_output(self, *, field_name: str, field: Any, adapter: Any) -> LMRequestPatch:
        """Render one output field into a partial normalized LM request."""
        return LMRequestPatch()

    def parse_output(
        self,
        *,
        field_name: str,
        output: LMOutput | dict[str, Any] | str,
        field: Any | None = None,
        adapter: Any | None = None,
    ) -> Any | None:
        """Parse one output field from a normalized or legacy LM output."""
        return None

    def prepare(
        self,
        *,
        signature: Any,
        lm: Any,
        lm_kwargs: dict[str, Any],
        inputs: dict[str, Any],
        adapter: Any | None = None,
    ) -> LMRequestPatch:
        """Return the normalized request patch implied by this strategy."""
        patch = LMRequestPatch()

        for name, field in signature.input_fields.items():
            if self.matches(field.annotation) and name in inputs:
                patch = patch.merge(self.render_input(field_name=name, field=field, value=inputs[name], adapter=adapter))

        for name, field in signature.output_fields.items():
            if self.matches(field.annotation):
                patch = patch.merge(self.render_output(field_name=name, field=field, adapter=adapter))

        return patch
