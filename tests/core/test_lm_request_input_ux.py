import pytest

import dspy
from dspy.core import types as lm


def make_lm():
    return dspy.BaseLM(model="test/model", cache=True, temperature=0.1)


def test_normalize_image_audio_file_and_reasoning_parts_in_one_user_message():
    request = make_lm().normalize_request(
        "summarize these inputs",
        dspy.Image("data:image/png;base64,image-bytes"),
        dspy.Audio(data="audio-bytes", audio_format="wav"),
        dspy.File(file_data="data:application/pdf;base64,file-bytes", filename="paper.pdf"),
        dspy.Reasoning(content="Prior reasoning supplied by the caller."),
    )

    assert request.messages == [
        dspy.User(
            "summarize these inputs",
            lm.LMImagePart(data="image-bytes", media_type="image/png"),
            lm.LMAudioPart(data="audio-bytes", media_type="audio/wav"),
            lm.LMBinaryPart(data="file-bytes", media_type="application/pdf", filename="paper.pdf"),
            lm.LMThinkingPart(text="Prior reasoning supplied by the caller."),
        )
    ]


def test_normalize_remote_image_url_part():
    request = make_lm().normalize_request("describe this", dspy.Image("https://example.com/dog.png"))

    assert request.messages == [
        dspy.User(
            "describe this",
            lm.LMImagePart(url="https://example.com/dog.png", media_type="image/png"),
        )
    ]


def test_normalize_previous_lm_response_as_assistant_message():
    previous = dspy.LMResponse(
        model="test/model",
        outputs=[
            lm.LMOutput(
                parts=[
                    lm.LMThinkingPart(text="I should answer concisely."),
                    lm.LMTextPart(text="DSPy is a programming framework."),
                    lm.LMCitationPart(text="DSPy documentation", title="DSPy Docs", url="https://dspy.ai"),
                    lm.LMImagePart(data="image-bytes", media_type="image/png"),
                ]
            )
        ],
    )

    request = make_lm().normalize_request(
        dspy.User("What is DSPy?"),
        previous,
        dspy.User("Say that in five words."),
    )

    assert request.messages == [
        dspy.User("What is DSPy?"),
        dspy.Assistant(
            lm.LMThinkingPart(text="I should answer concisely."),
            "DSPy is a programming framework.",
            lm.LMCitationPart(text="DSPy documentation", title="DSPy Docs", url="https://dspy.ai"),
            lm.LMImagePart(data="image-bytes", media_type="image/png"),
        ),
        dspy.User("Say that in five words."),
    ]


def test_normalize_tool_call_and_tool_result_messages():
    request = make_lm().normalize_request(
        dspy.User("What is the weather in Paris?"),
        dspy.Assistant(dspy.ToolCall(id="call_1", name="get_weather", args={"location": "Paris"})),
        dspy.ToolResult(
            call_id="call_1",
            name="get_weather",
            content=[
                "Here is the tool result: ",
                lm.LMTextPart(text='{"temperature": "22", "unit": "celsius"}'),
                dspy.Image("data:image/png;base64,weather-chart"),
            ],
        ),
        dspy.User("Summarize."),
    )

    assert request.messages == [
        dspy.User("What is the weather in Paris?"),
        dspy.Assistant(lm.LMToolCallPart(id="call_1", name="get_weather", args={"location": "Paris"})),
        dspy.ToolResult(
            lm.LMToolResultPart(
                call_id="call_1",
                name="get_weather",
                content=[
                    lm.LMTextPart(text="Here is the tool result: "),
                    lm.LMTextPart(text='{"temperature": "22", "unit": "celsius"}'),
                    lm.LMImagePart(data="weather-chart", media_type="image/png"),
                ],
            )
        ),
        dspy.User("Summarize."),
    ]


def test_normalize_bare_lm_parts_into_user_message():
    request = make_lm().normalize_request(
        lm.LMTextPart(text="describe this generated image"),
        lm.LMImagePart(data="generated-image", media_type="image/png"),
        lm.LMCitationPart(text="source text", title="Source", url="https://example.com/source"),
    )

    assert request.messages == [
        dspy.User(
            "describe this generated image",
            lm.LMImagePart(data="generated-image", media_type="image/png"),
            lm.LMCitationPart(text="source text", title="Source", url="https://example.com/source"),
        )
    ]


def test_normalize_positional_tool_sugar():
    def crop_image(x1: int, y1: int, x2: int, y2: int) -> str:
        """Crop the image to the given bounding box."""
        return "cropped-image-id"

    request = make_lm().normalize_request(
        "crop the dog",
        dspy.Image("data:image/png;base64,image-bytes"),
        dspy.Tool(crop_image),
    )

    assert request.messages == [dspy.User("crop the dog", lm.LMImagePart(data="image-bytes", media_type="image/png"))]
    assert request.tools == [
        lm.LMToolSpec(
            name="crop_image",
            description="Crop the image to the given bounding box.",
            parameters={
                "type": "object",
                "properties": {
                    "x1": {"type": "integer"},
                    "y1": {"type": "integer"},
                    "x2": {"type": "integer"},
                    "y2": {"type": "integer"},
                },
                "required": ["x1", "y1", "x2", "y2"],
            },
        )
    ]


def test_explicit_lm_request_group_overrides_preserve_unspecified_subfields():
    explicit = dspy.LMRequest(
        model="test/explicit",
        messages=[dspy.User("hello")],
        config=dspy.LMConfig(
            cache=lm.LMCacheConfig(enabled=True, rollout_id="old"),
            prompt_cache=lm.LMPromptCacheConfig(enabled=True, key="prefix-old"),
            tool_choice=lm.LMToolChoice(mode="auto", parallel=True),
            reasoning=lm.LMReasoningConfig(effort="low", max_tokens=100, summary="auto"),
        ),
    )

    request = make_lm().normalize_request(
        explicit,
        rollout_id="new",
        prompt_cache_key="prefix-new",
        parallel_tool_calls=False,
        reasoning_effort="high",
    )

    assert request.config.cache == lm.LMCacheConfig(enabled=True, rollout_id="new")
    assert request.config.prompt_cache == lm.LMPromptCacheConfig(enabled=True, key="prefix-new")
    assert request.config.tool_choice == lm.LMToolChoice(mode="auto", parallel=False)
    assert request.config.reasoning == lm.LMReasoningConfig(effort="high", max_tokens=100, summary="auto")


def test_normalize_rejects_request_mixed_with_direct_inputs():
    explicit = dspy.LMRequest(model="test/explicit", messages=[dspy.User("hello")])

    with pytest.raises(ValueError, match="either an LMRequest or direct-call inputs"):
        make_lm().normalize_request(explicit, "extra text")
