"""Field transforms: ordered edits to the model-facing field view.

Transforms are the only sanctioned way to change what the model sees — the
original signature is never mutated during planning. v1 ships the minimum
set (hide, rename, add); ``ReplaceField`` and ``MoveField`` are deliberately
absent (deferred per the epic's "rejected or deferred" list).

Application order is phased and deterministic: hide → rename → add →
validate. Conflicts (contradictory renames, duplicate visible names) are
explicit errors raised before any LM call.
"""

from dataclasses import dataclass, field
from typing import Any

from dspy.adapters._engine.ir import RenderField


class FieldTransformError(ValueError):
    """A field-transform conflict detected during plan application."""


@dataclass(frozen=True)
class HideInputField:
    """Hide an input from ordinary rendering (it is represented elsewhere,
    e.g. as a native media part or provider tool spec)."""

    name: str
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    role = "input"
    phase = "hide"


@dataclass(frozen=True)
class HideOutputField:
    """Hide an output from the ordinary response schema.

    ``metadata={"backfill": None}`` (the default) records that legacy
    postprocessing backfills the semantic key with ``None`` when no parser
    fills it — the engine representation of base.py's ``setdefault(None)``
    for native-feature fields.
    """

    name: str
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=lambda: {"backfill": None})

    role = "output"
    phase = "hide"


@dataclass(frozen=True)
class RenameInputField:
    original_name: str
    rendered_name: str

    role = "input"
    phase = "rename"


@dataclass(frozen=True)
class RenameOutputField:
    """Render an output under a different name; parse back to the original."""

    original_name: str
    rendered_name: str

    role = "output"
    phase = "rename"


@dataclass(frozen=True)
class AddInputField:
    """Add an auxiliary model-facing input absent from the signature."""

    name: str
    annotation: Any
    value: Any = None
    original_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    role = "input"
    phase = "add"


@dataclass(frozen=True)
class AddOutputField:
    """Add an auxiliary model-facing output absent from the signature.

    ``metadata["store"]`` decides where a parsed value goes
    (``"prediction"`` | ``"metadata"`` | ``"internal"`` | ``"discard"``).
    """

    name: str
    annotation: Any
    original_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    role = "output"
    phase = "add"


_PHASES = ("hide", "rename", "add")


def apply_field_transforms(
    input_fields: list[RenderField],
    output_fields: list[RenderField],
    transforms: list[Any],
) -> tuple[list[RenderField], list[RenderField], list[str]]:
    """Apply transforms in phase order and validate the result.

    Returns new field lists plus human-readable warnings (e.g. renaming a
    hidden field, which has no visible effect). Raises
    :class:`FieldTransformError` on conflicts.
    """
    inputs = [_copy_field(f) for f in input_fields]
    outputs = [_copy_field(f) for f in output_fields]
    warnings: list[str] = []

    for phase in _PHASES:
        for transform in transforms:
            if getattr(transform, "phase", None) != phase:
                continue
            target = inputs if transform.role == "input" else outputs
            if phase == "hide":
                _apply_hide(target, transform)
            elif phase == "rename":
                _apply_rename(target, transform, warnings)
            elif phase == "add":
                _apply_add(target, transform)

    _validate(inputs, "input")
    _validate(outputs, "output")
    return inputs, outputs, warnings


def _copy_field(render_field: RenderField) -> RenderField:
    return RenderField(
        name=render_field.name,
        original_name=render_field.original_name,
        role=render_field.role,
        annotation=render_field.annotation,
        value=render_field.value,
        metadata=dict(render_field.metadata),
        hidden=render_field.hidden,
    )


def _apply_hide(fields: list[RenderField], transform) -> None:
    for render_field in fields:
        if render_field.name == transform.name:
            render_field.hidden = True
            if transform.reason:
                render_field.metadata.setdefault("hide_reason", transform.reason)
            for key, value in transform.metadata.items():
                render_field.metadata.setdefault(key, value)
            return
    raise FieldTransformError(f"Cannot hide unknown {transform.role} field {transform.name!r}")


def _apply_rename(fields: list[RenderField], transform, warnings: list[str]) -> None:
    for render_field in fields:
        if render_field.name == transform.original_name:
            if render_field.metadata.get("renamed_from") is not None:
                raise FieldTransformError(
                    f"Conflicting renames for {transform.role} field "
                    f"{render_field.metadata['renamed_from']!r}: already rendered as "
                    f"{render_field.name!r}, cannot also render as {transform.rendered_name!r}"
                )
            if render_field.hidden:
                warnings.append(
                    f"Renaming hidden {transform.role} field {transform.original_name!r} has no visible effect"
                )
            render_field.metadata["renamed_from"] = render_field.name
            render_field.name = transform.rendered_name
            return
    raise FieldTransformError(f"Cannot rename unknown {transform.role} field {transform.original_name!r}")


def _apply_add(fields: list[RenderField], transform) -> None:
    field_kwargs = {
        "name": transform.name,
        "original_name": transform.original_name,
        "role": transform.role,
        "annotation": transform.annotation,
        "metadata": dict(transform.metadata),
    }
    if transform.role == "input":
        field_kwargs["value"] = transform.value
    fields.append(RenderField(**field_kwargs))


def _validate(fields: list[RenderField], role: str) -> None:
    seen: dict[str, RenderField] = {}
    for render_field in fields:
        if render_field.hidden:
            continue
        if render_field.name in seen:
            raise FieldTransformError(f"Duplicate visible {role} field name {render_field.name!r} after transforms")
        seen[render_field.name] = render_field
