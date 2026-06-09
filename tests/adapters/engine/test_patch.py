"""AdapterPatch composition: deterministic, associative merging, and the
LMRequestPatch interop boundary (delete_* fields become Hide transforms)."""

import pytest

from dspy.adapters._engine.ir import AdapterPlan
from dspy.adapters._engine.patch import AdapterPatch, DebugLink, PatchMergeError
from dspy.adapters._engine.transforms import HideInputField, HideOutputField
from dspy.core.types import LMConfig, LMRequestPatch, LMTextPart


def _patch(tag, **request_kwargs):
    return AdapterPatch(
        request=LMRequestPatch(**request_kwargs),
        warnings=[f"warning-{tag}"],
        debug_links=[DebugLink(source=f"src-{tag}", strategy=f"strat-{tag}", destination=f"dst-{tag}")],
    )


def test_merge_preserves_contribution_order():
    a = _patch("a", system_parts=[LMTextPart(text="A")])
    b = _patch("b", system_parts=[LMTextPart(text="B")])
    merged = a.merge(b)
    assert [part.text for part in merged.request.system_parts] == ["A", "B"]
    assert merged.warnings == ["warning-a", "warning-b"]
    assert [link.strategy for link in merged.debug_links] == ["strat-a", "strat-b"]


def test_merge_is_associative_on_plan_application():
    a = _patch("a", user_parts=[LMTextPart(text="A")], delete_input_fields=("image",))
    b = _patch("b", user_parts=[LMTextPart(text="B")], config=LMConfig(temperature=0.1))
    c = _patch("c", user_parts=[LMTextPart(text="C")], metadata={"k": "v"})

    def apply(patch):
        plan = AdapterPlan()
        patch.merge_into(plan)
        return (
            [part.text for part in plan.user_parts],
            plan.config.temperature if plan.config else None,
            plan.metadata,
            [type(t).__name__ for t in plan.field_transforms],
            plan.warnings,
        )

    assert apply(a.merge(b).merge(c)) == apply(a.merge(b.merge(c)))


def test_merge_into_maps_request_channels_to_collapsed_slots():
    plan = AdapterPlan()
    patch = AdapterPatch(
        request=LMRequestPatch(
            system_parts=[LMTextPart(text="sys")],
            user_parts=[LMTextPart(text="usr")],
            assistant_parts=[LMTextPart(text="prefill")],
            delete_input_fields=("image",),
            delete_output_fields=("reasoning",),
            metadata={"trace": "on"},
        ),
        parsers=["sentinel-parser"],
    )
    patch.merge_into(plan)

    assert [part.text for part in plan.system_parts] == ["sys"]
    assert [part.text for part in plan.user_parts] == ["usr"]
    assert [part.text for part in plan.assistant_prefill_parts] == ["prefill"]
    assert plan.metadata == {"trace": "on"}
    assert plan.parsers == ["sentinel-parser"]

    # The interop boundary: delete_* appears in the plan ONLY as Hide
    # transforms — the engine's canonical representation, with backfill
    # metadata on the output side.
    hides = {(type(t).__name__, t.name) for t in plan.field_transforms}
    assert hides == {("HideInputField", "image"), ("HideOutputField", "reasoning")}
    output_hide = next(t for t in plan.field_transforms if isinstance(t, HideOutputField))
    assert output_hide.metadata == {"backfill": None}
    assert not any(isinstance(t, HideInputField) and t.metadata.get("backfill") for t in plan.field_transforms)


def test_config_merges_via_lm_request_patch_rules():
    plan = AdapterPlan()
    AdapterPatch(request=LMRequestPatch(config=LMConfig(temperature=0.2))).merge_into(plan)
    AdapterPatch(request=LMRequestPatch(config=LMConfig(max_tokens=64))).merge_into(plan)
    assert plan.config.temperature == 0.2
    assert plan.config.max_tokens == 64


def test_metadata_conflicts_are_explicit_errors():
    plan = AdapterPlan()
    AdapterPatch(request=LMRequestPatch(metadata={"k": "v1"})).merge_into(plan)
    AdapterPatch(request=LMRequestPatch(metadata={"k": "v1"})).merge_into(plan)  # same value: fine
    with pytest.raises(PatchMergeError):
        AdapterPatch(request=LMRequestPatch(metadata={"k": "v2"})).merge_into(plan)
