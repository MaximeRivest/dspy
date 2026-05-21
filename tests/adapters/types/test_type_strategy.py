import dspy
from dspy.clients.language_models import LMConfig, LMReasoningConfig, LMRequestPatch, LMTextPart, LMToolSpec


def test_type_strategy_matches_subclasses():
    strategy = dspy.types.TextCode()
    PythonCode = dspy.Code["python"]

    assert strategy.matches(dspy.Code)
    assert strategy.matches(PythonCode)
    assert not strategy.matches(dspy.Image)


def test_lm_request_patch_merges_request_pieces_and_config_extensions():
    left = LMRequestPatch(
        system_parts=[LMTextPart(text="system")],
        config=LMConfig(temperature=0.2, extensions={"left": 1}),
        delete_input_fields=("history",),
    )
    right = LMRequestPatch(
        user_parts=[LMTextPart(text="user")],
        tools=[LMToolSpec(name="search", parameters={})],
        config=LMConfig(reasoning=LMReasoningConfig(effort="high"), extensions={"right": 2}),
        delete_output_fields=("reasoning",),
    )

    patch = left.merge(right)

    assert [part.text for part in patch.system_parts] == ["system"]
    assert [part.text for part in patch.user_parts] == ["user"]
    assert [tool.name for tool in patch.tools] == ["search"]
    assert patch.config.temperature == 0.2
    assert patch.config.reasoning.effort == "high"
    assert patch.config.extensions == {"left": 1, "right": 2}
    assert patch.delete_input_fields == ("history",)
    assert patch.delete_output_fields == ("reasoning",)


def test_lm_request_patch_as_lm_kwargs_bridges_config_and_tools():
    patch = LMRequestPatch(
        tools=[LMToolSpec(name="search", parameters={})],
        config=LMConfig(reasoning=LMReasoningConfig(effort="high")),
    )

    kwargs = patch.as_lm_kwargs()

    assert kwargs["reasoning"] == {"effort": "high"}
    assert kwargs["tools"][0].name == "search"
