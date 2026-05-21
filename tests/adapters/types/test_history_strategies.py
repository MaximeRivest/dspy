import dspy


def test_native_history_expands_history_into_messages_and_deletes_input_field():
    strategy = dspy.types.NativeHistory()
    history = dspy.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
            {"question": "What is the capital of Germany?", "answer": "Berlin"},
        ]
    )

    patch = strategy.render_input(field_name="history", field=None, value=history, adapter=None)

    assert patch.delete_input_fields == ("history",)
    assert [message.role for message in patch.messages] == ["user", "assistant", "user", "assistant"]
    text = "\n".join(message.text for message in patch.messages)
    assert "What is the capital of France?" in text
    assert "Paris" in text
    assert "What is the capital of Germany?" in text
    assert "Berlin" in text


def test_text_history_embeds_history_as_user_text_and_deletes_input_field():
    strategy = dspy.types.TextHistory()
    history = dspy.History(
        messages=[
            {"question": "What is the capital of France?", "answer": "Paris"},
        ]
    )

    patch = strategy.render_input(field_name="history", field=None, value=history, adapter=None)

    assert patch.delete_input_fields == ("history",)
    assert patch.messages == []
    text = patch.user_parts[0].text
    assert "Conversation history:" in text
    assert "Turn 1:" in text
    assert "question: What is the capital of France?" in text
    assert "answer: Paris" in text
