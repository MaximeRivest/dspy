import dspy
from dspy.clients.language_models import (
    LMCapabilities,
    LMImagePart,
    LMOutput,
    LMResponse,
    LMTextPart,
    LMThinkingPart,
    LanguageModel,
)


class CapturingLM(LanguageModel):
    def __init__(self, response, *, capabilities=None):
        super().__init__(model="test/type-strategy-integration", cache=False)
        self.response = response
        self._capabilities = capabilities or LMCapabilities()
        self.request = None

    def get_capabilities(self):
        return self._capabilities

    def forward(self, request):
        self.request = request
        if callable(self.response):
            return self.response(request)
        return self.response


def _request_text(request):
    return "\n".join(
        part.text
        for message in request.messages
        for part in message.parts
        if getattr(part, "type", None) == "text"
    )


def test_native_reasoning_uses_lm_reasoning_config_and_parses_thinking_part():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    lm = CapturingLM(
        LMResponse(
            model="test/type-strategy-integration",
            outputs=[
                LMOutput(
                    parts=[
                        LMThinkingPart(text="Native hidden reasoning."),
                        LMTextPart(text="[[ ## answer ## ]]\nParis\n\n[[ ## completed ## ]]"),
                    ]
                )
            ],
        ),
        capabilities=LMCapabilities(reasoning=True),
    )
    adapter = dspy.ChatAdapter(
        adapter_types=[dspy.types.NativeReasoning(reasoning_effort="high")],
        use_json_adapter_fallback=False,
    )

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.Predict(QA)(question="What is the capital of France?")

    assert lm.request.config.reasoning.effort == "high"
    prompt_text = _request_text(lm.request)
    assert "[[ ## reasoning ## ]]" not in prompt_text
    assert "[[ ## answer ## ]]" in prompt_text
    assert pred.reasoning == "Native hidden reasoning."
    assert pred.answer == "Paris"


def test_text_reasoning_stays_in_adapter_text_format():
    class QA(dspy.Signature):
        question: str = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        answer: str = dspy.OutputField()

    lm = CapturingLM(
        LMResponse.from_text(
            "[[ ## reasoning ## ]]\nThe question asks for France's capital.\n\n"
            "[[ ## answer ## ]]\nParis\n\n"
            "[[ ## completed ## ]]",
            model="test/type-strategy-integration",
        ),
        capabilities=LMCapabilities(reasoning=True),
    )
    adapter = dspy.ChatAdapter(
        adapter_types=[dspy.types.TextReasoning()],
        use_json_adapter_fallback=False,
    )

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.Predict(QA)(question="What is the capital of France?")

    assert lm.request.config.reasoning is None
    prompt_text = _request_text(lm.request)
    assert "[[ ## reasoning ## ]]" in prompt_text
    assert "[[ ## answer ## ]]" in prompt_text
    assert pred.reasoning == "The question asks for France's capital."
    assert pred.answer == "Paris"


def test_native_code_parses_code_from_normalized_lm_text_part_without_adapter_field_wrapping():
    class GenerateCode(dspy.Signature):
        task: str = dspy.InputField()
        code: dspy.Code["python"] = dspy.OutputField()

    lm = CapturingLM(
        LMResponse(
            model="test/type-strategy-integration",
            outputs=[
                LMOutput(
                    parts=[
                        LMTextPart(
                            text="def add_one(x):\n    return x + 1\n",
                            metadata={"dspy_field": "code"},
                        )
                    ]
                )
            ],
        )
    )
    adapter = dspy.ChatAdapter(
        adapter_types=[dspy.types.NativeCode()],
        use_json_adapter_fallback=False,
    )

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.Predict(GenerateCode)(task="Write a function that adds one.")

    prompt_text = _request_text(lm.request)
    assert "[[ ## code ## ]]" not in prompt_text
    assert isinstance(pred.code, dspy.Code)
    assert pred.code.code == "def add_one(x):\n    return x + 1\n"


def test_text_code_uses_adapter_field_and_markdown_code_fence():
    class GenerateCode(dspy.Signature):
        task: str = dspy.InputField()
        code: dspy.Code["python"] = dspy.OutputField()

    lm = CapturingLM(
        LMResponse.from_text(
            "[[ ## code ## ]]\n"
            "```python\n"
            "def add_one(x):\n"
            "    return x + 1\n"
            "```\n\n"
            "[[ ## completed ## ]]",
            model="test/type-strategy-integration",
        )
    )
    adapter = dspy.ChatAdapter(
        adapter_types=[dspy.types.TextCode()],
        use_json_adapter_fallback=False,
    )

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.Predict(GenerateCode)(task="Write a function that adds one.")

    prompt_text = _request_text(lm.request)
    assert "[[ ## code ## ]]" in prompt_text
    assert "```python" in prompt_text
    assert isinstance(pred.code, dspy.Code)
    assert pred.code.code == "def add_one(x):\n    return x + 1"


def test_native_image_round_trips_via_lm_image_parts():
    class EditImage(dspy.Signature):
        edit: str = dspy.InputField()
        image: dspy.Image = dspy.InputField()
        reasoning: dspy.Reasoning = dspy.OutputField()
        edited_image: dspy.Image = dspy.OutputField()
        edit_summaries: list[str] = dspy.OutputField()

    lm = CapturingLM(
        LMResponse(
            model="test/type-strategy-integration",
            outputs=[
                LMOutput(
                    parts=[
                        LMImagePart(url="https://example.com/edited.png", media_type="image/png"),
                        LMTextPart(
                            text="[[ ## reasoning ## ]]\nThe edit changes the background.\n\n"
                            "[[ ## edit_summaries ## ]]\n"
                            '["Changed background to blue."]\n\n'
                            "[[ ## completed ## ]]"
                        ),
                    ]
                )
            ],
        ),
        capabilities=LMCapabilities(input_image=True, output_image=True),
    )
    adapter = dspy.ChatAdapter(
        adapter_types=[dspy.types.NativeImage(detail="high"), dspy.types.TextReasoning()],
        use_json_adapter_fallback=False,
    )

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.Predict(EditImage)(
            edit="Make the background blue.",
            image=dspy.Image("https://example.com/source.png"),
        )

    request_parts = [part for msg in lm.request.messages for part in msg.parts]
    assert any(part.type == "image" and part.url == "https://example.com/source.png" for part in request_parts)
    prompt_text = "\n".join(part.text for part in request_parts if part.type == "text")
    assert "[[ ## edited_image ## ]]" not in prompt_text
    assert isinstance(pred.edited_image, dspy.Image)
    assert pred.edited_image.url == "https://example.com/edited.png"
    assert pred.edit_summaries == ["Changed background to blue."]


def test_native_history_expands_history_into_prior_messages():
    class ConversationalQA(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    lm = CapturingLM(
        LMResponse.from_text(
            "[[ ## answer ## ]]\nParis is still the capital.\n\n[[ ## completed ## ]]",
            model="test/type-strategy-integration",
        )
    )
    adapter = dspy.ChatAdapter(
        adapter_types=[dspy.types.NativeHistory()],
        use_json_adapter_fallback=False,
    )
    history = dspy.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.Predict(ConversationalQA)(question="Are you sure about France?", history=history)

    assert [message.role for message in lm.request.messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    all_text = [part.text for msg in lm.request.messages for part in msg.parts if part.type == "text"]
    assert any("What is the capital of France?" in text for text in all_text)
    assert any("Paris" in text for text in all_text)
    assert any("Are you sure about France?" in text for text in all_text)
    assert "[[ ## history ## ]]" not in lm.request.messages[-1].text
    assert pred.answer == "Paris is still the capital."


def test_text_history_embeds_history_inside_current_user_message():
    class ConversationalQA(dspy.Signature):
        question: str = dspy.InputField()
        history: dspy.History = dspy.InputField()
        answer: str = dspy.OutputField()

    lm = CapturingLM(
        LMResponse.from_text(
            "[[ ## answer ## ]]\nParis is still the capital.\n\n[[ ## completed ## ]]",
            model="test/type-strategy-integration",
        )
    )
    adapter = dspy.ChatAdapter(
        adapter_types=[dspy.types.TextHistory()],
        use_json_adapter_fallback=False,
    )
    history = dspy.History(messages=[{"question": "What is the capital of France?", "answer": "Paris"}])

    with dspy.context(lm=lm, adapter=adapter):
        pred = dspy.Predict(ConversationalQA)(question="Are you sure about France?", history=history)

    assert [message.role for message in lm.request.messages] == ["system", "user"]
    user_text = lm.request.messages[-1].text
    assert "Conversation history" in user_text
    assert "What is the capital of France?" in user_text
    assert "Paris" in user_text
    assert "Are you sure about France?" in user_text
    assert pred.answer == "Paris is still the capital."
