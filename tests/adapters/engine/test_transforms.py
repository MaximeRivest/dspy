"""Field-transform semantics: phased application, original_name mapping,
and explicit conflict errors."""

import pytest

from dspy.adapters._engine.ir import RenderField
from dspy.adapters._engine.transforms import (
    AddInputField,
    AddOutputField,
    FieldTransformError,
    HideInputField,
    HideOutputField,
    RenameInputField,
    RenameOutputField,
    apply_field_transforms,
)


def _fields():
    inputs = [
        RenderField(name="question", original_name="question", role="input", annotation=str, value="Q?"),
        RenderField(name="image", original_name="image", role="input", annotation=str, value="IMG"),
    ]
    outputs = [
        RenderField(name="reasoning", original_name="reasoning", role="output", annotation=str),
        RenderField(name="answer", original_name="answer", role="output", annotation=str),
    ]
    return inputs, outputs


def test_hide_marks_without_deleting():
    inputs, outputs = _fields()
    new_inputs, new_outputs, warnings = apply_field_transforms(
        inputs, outputs, [HideInputField("image", reason="native part"), HideOutputField("reasoning")]
    )
    assert [f.name for f in new_inputs] == ["question", "image"]
    assert new_inputs[1].hidden and new_inputs[1].metadata["hide_reason"] == "native part"
    assert new_outputs[0].hidden and new_outputs[0].metadata == {"backfill": None}
    assert warnings == []
    # Inputs to the function are never mutated.
    assert not inputs[1].hidden and not outputs[0].hidden


def test_rename_preserves_semantic_destination():
    inputs, outputs = _fields()
    _, new_outputs, _ = apply_field_transforms(
        inputs, outputs, [RenameOutputField(original_name="answer", rendered_name="final")]
    )
    renamed = next(f for f in new_outputs if f.name == "final")
    assert renamed.original_name == "answer"
    assert renamed.destination_name == "answer"


def test_rename_input_field():
    inputs, outputs = _fields()
    new_inputs, _, _ = apply_field_transforms(
        inputs, outputs, [RenameInputField(original_name="question", rendered_name="q")]
    )
    assert [f.name for f in new_inputs] == ["q", "image"]
    assert new_inputs[0].original_name == "question"


def test_add_output_field_with_store_policy():
    inputs, outputs = _fields()
    _, new_outputs, _ = apply_field_transforms(
        inputs,
        outputs,
        [AddOutputField(name="confidence", annotation=float, metadata={"store": "prediction"})],
    )
    added = next(f for f in new_outputs if f.name == "confidence")
    assert added.original_name is None
    assert added.metadata["store"] == "prediction"


def test_add_input_field_carries_value():
    inputs, outputs = _fields()
    new_inputs, _, _ = apply_field_transforms(
        inputs, outputs, [AddInputField(name="context_note", annotation=str, value="aux")]
    )
    assert next(f for f in new_inputs if f.name == "context_note").value == "aux"


def test_phase_order_hide_then_rename_then_add():
    inputs, outputs = _fields()
    # Listed in "wrong" order on purpose: phases, not list order, decide.
    transforms = [
        AddOutputField(name="confidence", annotation=float),
        RenameOutputField(original_name="answer", rendered_name="final"),
        HideOutputField("reasoning"),
    ]
    _, new_outputs, _ = apply_field_transforms(inputs, outputs, transforms)
    assert [f.name for f in new_outputs] == ["reasoning", "final", "confidence"]
    assert new_outputs[0].hidden


def test_conflicting_renames_error():
    inputs, outputs = _fields()
    with pytest.raises(FieldTransformError, match="Conflicting renames"):
        apply_field_transforms(
            inputs,
            outputs,
            [
                RenameOutputField(original_name="answer", rendered_name="final"),
                RenameOutputField(original_name="final", rendered_name="result"),
            ],
        )


def test_duplicate_visible_names_error():
    inputs, outputs = _fields()
    with pytest.raises(FieldTransformError, match="Duplicate visible"):
        apply_field_transforms(inputs, outputs, [AddOutputField(name="answer", annotation=str)])


def test_duplicate_name_allowed_when_original_hidden():
    inputs, outputs = _fields()
    new_inputs, new_outputs, _ = apply_field_transforms(
        inputs, outputs, [HideOutputField("answer"), AddOutputField(name="answer", annotation=str)]
    )
    visible = [f for f in new_outputs if not f.hidden and f.name == "answer"]
    assert len(visible) == 1


def test_rename_hidden_field_warns():
    inputs, outputs = _fields()
    _, _, warnings = apply_field_transforms(
        inputs,
        outputs,
        [HideOutputField("answer"), RenameOutputField(original_name="answer", rendered_name="final")],
    )
    assert warnings and "no visible effect" in warnings[0]


def test_unknown_field_errors():
    inputs, outputs = _fields()
    with pytest.raises(FieldTransformError, match="unknown"):
        apply_field_transforms(inputs, outputs, [HideOutputField("missing")])
    with pytest.raises(FieldTransformError, match="unknown"):
        apply_field_transforms(inputs, outputs, [RenameInputField(original_name="missing", rendered_name="x")])
