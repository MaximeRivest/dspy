"""Method-level parity: ChatFormat vs the legacy ChatAdapter bodies.

The dual-run corpus gates end-to-end output; these tests compare each
Format surface directly against the legacy method it replaces, across the
golden signatures, so a divergence points at the exact method."""

import pytest
from golden.cases import CASES

from dspy.adapters._engine.formats.chat import ChatFormat
from dspy.adapters.chat_adapter import ChatAdapter

FMT = ChatFormat()
LEGACY = ChatAdapter()

CHAT_PAYLOAD_CASES = sorted(case_id for case_id, case in CASES.items() if case.adapter == "chat" and not case.surfaces)


def _signatures():
    seen = set()
    for case_id in CHAT_PAYLOAD_CASES:
        payload = CASES[case_id].payload()
        signature = payload["signature"]
        key = (
            tuple(signature.input_fields),
            tuple(signature.output_fields),
            signature.instructions,
        )
        if key not in seen:
            seen.add(key)
            yield case_id, payload


SIGNATURE_PAYLOADS = dict(_signatures())


@pytest.mark.parametrize("case_id", sorted(SIGNATURE_PAYLOADS))
def test_system_sections_match_legacy(case_id):
    signature = SIGNATURE_PAYLOADS[case_id]["signature"]
    assert FMT.render_field_description(signature) == LEGACY.format_field_description(signature)
    assert FMT.render_field_structure(signature) == LEGACY.format_field_structure(signature)
    assert FMT.render_task_description(signature) == LEGACY.format_task_description(signature)
    assert FMT.render_system(signature) == LEGACY.format_system_message(signature)


@pytest.mark.parametrize("case_id", sorted(SIGNATURE_PAYLOADS))
def test_user_content_matches_legacy(case_id):
    payload = SIGNATURE_PAYLOADS[case_id]
    signature, inputs = payload["signature"], payload["inputs"]
    assert FMT.render_user_content(signature, inputs, main_request=True) == LEGACY.format_user_message_content(
        signature, inputs, main_request=True
    )
    assert FMT.render_user_content(signature, inputs, prefix="P ", suffix=" S") == LEGACY.format_user_message_content(
        signature, inputs, prefix="P ", suffix=" S"
    )
    assert FMT.output_requirements(signature) == LEGACY.user_message_output_requirements(signature)


@pytest.mark.parametrize("case_id", sorted(SIGNATURE_PAYLOADS))
def test_assistant_content_matches_legacy(case_id):
    payload = SIGNATURE_PAYLOADS[case_id]
    signature = payload["signature"]
    outputs = payload.get("surface_outputs") or {}
    assert FMT.render_assistant_content(signature, outputs) == LEGACY.format_assistant_message_content(
        signature, outputs
    )
    assert FMT.render_assistant_content(
        signature, {}, missing_field_message="Not supplied for this particular example. "
    ) == LEGACY.format_assistant_message_content(
        signature, {}, missing_field_message="Not supplied for this particular example. "
    )
