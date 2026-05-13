import pytest

import dspy


class TextOnlyLM(dspy.LanguageModel):
    def __init__(self):
        super().__init__(model="test/text-only", cache=False)

    def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
        return dspy.LMResponse.from_text("ok", model=request.model)


class ImageLM(TextOnlyLM):
    def __init__(self):
        super().__init__()
        self.support = self.support.with_updates(
            images=dspy.ImageSupport(urls=True, base64=True, placement="any_message")
        )


def test_request_and_response_support_are_programmatically_accessible():
    lm = TextOnlyLM()

    assert lm.features.request.text
    assert lm.features.request.input_image.status == "unsupported"
    assert lm.features.response.text
    assert lm.features.response.tool_calls.status == "unsupported"
    assert lm.features.supports("request.text")
    assert not lm.features.supports("request.input_image")


def test_report_includes_request_and_response_sections():
    report = TextOnlyLM().features.report()

    assert "Request support:" in report
    assert "Response support:" in report
    assert "input_image" in report
    assert "tool_calls" in report


def test_json_report_includes_request_and_response_support():
    data = TextOnlyLM().features.report(format="json")

    assert data["request"]["text"]["status"] == "inferred"
    assert data["request"]["input_image"]["status"] == "unsupported"
    assert data["response"]["text"]["status"] == "inferred"


def test_validate_request_reports_unsupported_request_shapes():
    lm = TextOnlyLM()
    request = lm.normalize_request("describe", dspy.Image("data:image/png;base64,abc"))

    support = lm.validate_request(request)

    assert support.status == "unsupported"
    assert not support
    assert [issue.feature for issue in support.issues] == ["request.input_image"]


def test_require_request_support_raises_structured_error():
    lm = TextOnlyLM()
    request = lm.normalize_request("describe", dspy.Image("data:image/png;base64,abc"))

    with pytest.raises(dspy.LMUnsupportedFeatureError) as exc_info:
        lm.require_request_support(request)

    assert exc_info.value.features == ["request.input_image"]
    assert "does not declare request.input_image" in exc_info.value.issues[0]


def test_validate_request_passes_when_shape_is_supported():
    lm = ImageLM()
    request = lm.normalize_request("describe", dspy.Image("data:image/png;base64,abc"))

    support = lm.validate_request(request)

    assert support.status == "supported"
    assert support.issues == []
    lm.require_request_support(request)
