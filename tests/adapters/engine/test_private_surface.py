"""The engine is private machinery: zero public-surface change, ever lazy.

These assertions are merge gates for every PR in the engine stack — the
public adapter surface must remain byte-equivalent to pre-engine main, and
importing dspy must not even load the engine package.
"""

import subprocess
import sys
import types

import dspy.adapters

# The exact public (non-module) surface of dspy.adapters before the engine
# existed. Submodule attributes are import-order noise, not API, so they are
# excluded from the comparison.
EXPECTED_PUBLIC_SURFACE = {
    "Adapter",
    "Audio",
    "ChatAdapter",
    "Code",
    "File",
    "History",
    "Image",
    "JSONAdapter",
    "Reasoning",
    "Tool",
    "ToolCallResults",
    "ToolCalls",
    "TwoStepAdapter",
    "Type",
    "XMLAdapter",
}


def test_public_surface_unchanged():
    actual = {
        name
        for name in dir(dspy.adapters)
        if not name.startswith("_") and not isinstance(getattr(dspy.adapters, name), types.ModuleType)
    }
    assert actual == EXPECTED_PUBLIC_SURFACE, (
        f"public dspy.adapters surface changed (added: {sorted(actual - EXPECTED_PUBLIC_SURFACE)}, "
        f"removed: {sorted(EXPECTED_PUBLIC_SURFACE - actual)})"
    )
    # BAMLAdapter has never been exported; it must stay that way.
    assert "BAMLAdapter" not in actual


def test_importing_dspy_does_not_load_the_engine():
    """Run in a fresh interpreter: other tests in this process legitimately
    import the engine."""
    code = "import sys, dspy; assert 'dspy.adapters._engine' not in sys.modules, 'engine eagerly imported'"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_no_remaining_adapters_plan_todos():
    """All four TODO(adapters-plan) seams are closed; the marker must be gone
    from dspy/ source."""
    import pathlib
    import subprocess

    result = subprocess.run(
        ["grep", "-rn", "TODO(adapters-plan)", str(pathlib.Path("dspy"))],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.stdout == "", f"lingering TODO(adapters-plan) markers:\n{result.stdout}"


def test_external_compat_keep_list_importable():
    """Symbols external code (incl. dspy-community/dspy-template-adapter)
    imports must stay importable from their historical paths."""
    from dspy.adapters.types.base_type import split_message_content_for_custom_types  # noqa: F401
    from dspy.adapters.utils import (  # noqa: F401
        get_field_description_string,
        parse_value,
        serialize_for_json,
        translate_field_type,
    )
    from dspy.utils.mcp import convert_input_schema_to_tool_args  # noqa: F401


def test_kill_list_inventory():
    """Legacy bodies retained ONLY for override orchestration (they dispatch
    through overridable hooks, so delegating would bypass user overrides).
    This is the retirement epic's work list — additions/removals must be
    deliberate."""
    import inspect

    from dspy.adapters.baml_adapter import BAMLAdapter
    from dspy.adapters.base import Adapter
    from dspy.adapters.chat_adapter import ChatAdapter
    from dspy.adapters.two_step_adapter import TwoStepAdapter
    from dspy.adapters.xml_adapter import XMLAdapter

    def delegates(cls, name, needle):
        return needle in inspect.getsource(inspect.unwrap(vars(cls)[name]))

    # True leaves delegate to their Format (strings live once)...
    assert delegates(ChatAdapter, "format_field_description", "_chat_format()")
    assert delegates(ChatAdapter, "format_task_description", "_chat_format()")
    assert delegates(ChatAdapter, "user_message_output_requirements", "_chat_format()")
    assert delegates(ChatAdapter, "parse", "_chat_format()")
    assert delegates(XMLAdapter, "user_message_output_requirements", "_xml_format()")
    assert delegates(XMLAdapter, "_parse_field_value", "_xml_format()")
    assert delegates(TwoStepAdapter, "format_task_description", "_two_step_format()")
    assert delegates(TwoStepAdapter, "format_user_message_content", "_two_step_format()")
    assert delegates(TwoStepAdapter, "format_assistant_message_content", "_two_step_format()")

    # ...while hook-dispatching bodies are deliberately retained (the kill
    # list): delegation would bypass subclass overrides of the inner hooks.
    assert not delegates(ChatAdapter, "format_field_structure", "_chat_format()")
    assert not delegates(ChatAdapter, "format_user_message_content", "_chat_format()")
    assert not delegates(ChatAdapter, "format_assistant_message_content", "_chat_format()")
    assert not delegates(ChatAdapter, "format_field_with_value", "_chat_format()")
    assert "self._parse_field_value" in inspect.getsource(inspect.unwrap(vars(XMLAdapter)["parse"]))
    assert "_legacy_async_quirks_postprocess" in vars(TwoStepAdapter)
    # BAML's user content dispatches the overridable output-requirements hook.
    assert "user_message_output_requirements" in inspect.getsource(
        inspect.unwrap(vars(BAMLAdapter)["format_user_message_content"])
    )
    assert "format_demos" in inspect.getsource(inspect.unwrap(vars(Adapter)["format"]))
