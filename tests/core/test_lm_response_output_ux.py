from dspy.core import types as lm


def test_text_only_response_is_list_like_and_text_friendly():
    response = lm.LMResponse.from_text(
        "Hello!",
        model="test/model",
        usage=lm.LMUsage(input_tokens=1, output_tokens=2, total_tokens=3),
        cost=0.00001,
    )

    assert response[0] == "Hello!"
    assert list(response) == ["Hello!"]
    assert response.output == response.outputs[0]
    assert response.parts == [lm.LMTextPart(text="Hello!")]
    assert response.text == "Hello!"
    assert response.reasoning_content is None
    assert response.tool_calls == []
    assert response.citations == []
    assert response.images == []
    assert response.audio == []
    assert response.binaries == []
    assert response.usage.total_tokens == 3
    assert response.cost == 0.00001
    assert response.cache_hit is False
    assert response.to_outputs() == ["Hello!"]


def test_text_plus_logprobs_keeps_output_level_metadata():
    logprobs = {"tokens": ["A"], "token_logprobs": [-0.1]}
    response = lm.LMResponse(
        model="test/model",
        outputs=[lm.LMOutput(parts=[lm.LMTextPart(text="A")], logprobs=logprobs)],
    )

    assert response.text == "A"
    assert response.outputs[0].logprobs == logprobs
    assert response.to_outputs() == [{"text": "A", "logprobs": logprobs}]


def test_reasoning_text_and_tool_call_output_views():
    thinking = lm.LMThinkingPart(text="I should call the weather tool.")
    tool_call = lm.LMToolCallPart(id="call_1", name="get_weather", args={"location": "Paris"})
    response = lm.LMResponse(
        model="test/model",
        outputs=[
            lm.LMOutput(
                parts=[
                    thinking,
                    lm.LMTextPart(text="I will check Paris now."),
                    tool_call,
                ],
                finish_reason="tool_calls",
            )
        ],
    )

    assert response[0] == [thinking, "I will check Paris now.", tool_call]
    assert response.reasoning_content == "I should call the weather tool."
    assert response.text == "I will check Paris now."
    assert response.tool_calls == [tool_call]
    assert response.outputs[0].finish_reason == "tool_calls"
    assert response.to_outputs() == [
        {
            "text": "I will check Paris now.",
            "reasoning_content": "I should call the weather tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
                }
            ],
        }
    ]


def test_citations_are_projected_from_output_parts():
    citation = lm.LMCitationPart(
        text="Water boils at 100°C.",
        title="Physics Handbook",
        url="https://example.com/physics",
    )
    response = lm.LMResponse(
        model="test/model",
        outputs=[lm.LMOutput(parts=[lm.LMTextPart(text="Water boils at 100°C."), citation])],
    )

    assert response.text == "Water boils at 100°C."
    assert response.citations == [citation]
    assert response[0] == ["Water boils at 100°C.", citation]


def test_generated_images_audio_and_binaries_are_return_parts():
    image = lm.LMImagePart(data="image-bytes", media_type="image/png")
    audio = lm.LMAudioPart(data="audio-bytes", media_type="audio/wav")
    binary = lm.LMBinaryPart(data="file-bytes", media_type="application/pdf", filename="paper.pdf")
    response = lm.LMResponse(
        model="test/model",
        outputs=[
            lm.LMOutput(
                parts=[
                    lm.LMTextPart(text="Here are the generated artifacts."),
                    image,
                    audio,
                    binary,
                ]
            )
        ],
    )

    assert response.text == "Here are the generated artifacts."
    assert response.images == [image]
    assert response.audio == [audio]
    assert response.binaries == [binary]
    assert response[0] == ["Here are the generated artifacts.", image, audio, binary]


def test_multiple_outputs_keep_candidate_level_metadata_separate():
    tool_call = lm.LMToolCallPart(id="call_1", name="search", args={"query": "DSPy"})
    response = lm.LMResponse(
        model="test/model",
        outputs=[
            lm.LMOutput(parts=[lm.LMTextPart(text="first")], finish_reason="stop"),
            lm.LMOutput(parts=[lm.LMTextPart(text="second")], finish_reason="length", truncated=True),
            lm.LMOutput(parts=[tool_call], finish_reason="tool_calls"),
        ],
    )

    assert list(response) == ["first", "second", [tool_call]]
    assert response.outputs[0].finish_reason == "stop"
    assert response.outputs[1].truncated is True
    assert response.outputs[2].tool_calls == [tool_call]
    assert response.to_outputs() == [
        "first",
        "second",
        {
            "text": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query": "DSPy"}'},
                }
            ],
        },
    ]


def test_streaming_events_assemble_into_lm_output_and_lm_response():
    builder = lm.LMOutputBuilder()

    builder.apply(lm.LMStreamStartEvent(model="test/model"))
    builder.apply(lm.LMStreamDeltaEvent(output_index=0, part_index=0, delta=lm.LMThinkingDelta(text="Think. ")))
    builder.apply(lm.LMStreamDeltaEvent(output_index=0, part_index=1, delta=lm.LMTextDelta(text="Hello")))
    builder.apply(lm.LMStreamDeltaEvent(output_index=0, part_index=1, delta=lm.LMTextDelta(text="!")))
    builder.apply(
        lm.LMStreamDeltaEvent(
            output_index=0,
            part_index=2,
            delta=lm.LMToolCallDelta(id="call_1", name="search", args_delta='{"query": "DSPy"}'),
        )
    )
    builder.apply(lm.LMStreamOutputEndEvent(output_index=0, finish_reason="tool_calls"))
    final = builder.apply(
        lm.LMStreamEndEvent(
            usage=lm.LMUsage(input_tokens=5, output_tokens=7, total_tokens=12),
            cost=0.0002,
        )
    )

    assert final == lm.LMResponse(
        model="test/model",
        outputs=[
            lm.LMOutput(
                parts=[
                    lm.LMThinkingPart(text="Think. "),
                    lm.LMTextPart(text="Hello!"),
                    lm.LMToolCallPart(
                        id="call_1",
                        name="search",
                        args={"query": "DSPy"},
                        provider_data={"args_buffer": '{"query": "DSPy"}'},
                    ),
                ],
                finish_reason="tool_calls",
            )
        ],
        usage=lm.LMUsage(input_tokens=5, output_tokens=7, total_tokens=12),
        cost=0.0002,
    )
