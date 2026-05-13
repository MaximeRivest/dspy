import base64
from dataclasses import dataclass

import pytest

import dspy


@dataclass
class FakeImage:
    url: str | None = None
    content: bytes | None = None
    format: str | None = None


class FakeResponse:
    def __init__(self, text):
        self.text = text


def test_from_sdk_text_wrapper_uses_provider_request_without_subclassing():
    calls = []

    def call(provider_request):
        calls.append(provider_request)
        return FakeResponse(f"echo: {provider_request.args[0]}")

    lm = dspy.LM.from_sdk(
        model="toy/text",
        call=call,
        extract_text=lambda response: response.text,
        cache=False,
    )

    response = lm("hello")

    assert response.text == "echo: hello"
    assert calls[0].args == ("hello",)
    assert isinstance(lm.explain_provider_request("hello"), dspy.ProviderRequest)


def test_explain_provider_request_does_not_call_sdk():
    calls = 0

    def call(provider_request):
        nonlocal calls
        calls += 1
        return FakeResponse("ok")

    lm = dspy.LM.from_sdk(model="toy/text", call=call, extract_text=lambda response: response.text, cache=False)

    provider_request = lm.explain_provider_request("hello")

    assert provider_request.args == ("hello",)
    assert calls == 0


def test_agno_style_top_level_images_are_supported_and_previewable():
    def map_image(image):
        if image.url:
            return FakeImage(url=image.url)
        return FakeImage(content=base64.b64decode(image.data), format=image.media_type.removeprefix("image/"))

    def place_images(request, images, provider_request):
        return dspy.ProviderRequest(args=(request.messages[-1].text,), kwargs={"images": images})

    lm = dspy.LM.from_sdk(
        model="agno/openai:gpt-5.4",
        call=lambda provider_request: FakeResponse("bridge"),
        extract_text=lambda response: response.text,
        cache=False,
    ).with_images(
        map_image=map_image,
        place_images=place_images,
        support=dspy.ImageSupport(urls=True, base64=True, placement="latest_user_message_only"),
    )

    provider_request = lm.explain_provider_request(
        dspy.User("describe", dspy.Image("https://example.com/dog.png"))
    )

    assert provider_request.args == ("describe",)
    assert provider_request.kwargs == {"images": [FakeImage(url="https://example.com/dog.png")]}
    assert "latest_user_message_only" in lm.support.report()


def test_latest_user_only_image_support_rejects_older_turn_images():
    lm = dspy.LM.from_sdk(
        model="agno/openai:gpt-5.4",
        call=lambda provider_request: FakeResponse("ok"),
        extract_text=lambda response: response.text,
        cache=False,
    ).with_images(
        map_image=lambda image: image,
        support=dspy.ImageSupport(urls=True, base64=True, placement="latest_user_message_only"),
    )

    with pytest.raises(dspy.LMUnsupportedFeatureError) as exc_info:
        lm(
            dspy.User("first", dspy.Image("https://example.com/a.png")),
            dspy.Assistant("ok"),
            dspy.User("now compare"),
        )

    assert exc_info.value.features == ["request.input_image"]
    assert "latest user message" in exc_info.value.issues[0]


def test_with_request_mapper_escape_hatch_declares_support():
    lm = dspy.LM.from_sdk(
        model="weird/model",
        call=lambda provider_request: FakeResponse("ok"),
        extract_text=lambda response: response.text,
        cache=False,
    ).with_request_mapper(
        lambda request: dspy.ProviderRequest(kwargs={"payload": [message.model_dump() for message in request.messages]}),
        support=dspy.LMSupport(images=dspy.ImageSupport(urls=True, placement="any_message")),
    )

    response = lm(dspy.User("describe", dspy.Image("https://example.com/a.png")))

    assert response.text == "ok"
    assert lm.support.images.placement == "any_message"
