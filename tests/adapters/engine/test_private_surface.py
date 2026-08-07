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
# existed, plus Epic D's sanctioned additions. Submodule attributes are
# import-order noise, not API, so they are excluded from the comparison.
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
    # Epic D-5 (epic-D-adapter-serializer.md, acceptance test #5): the
    # component-4 entry loader. Its dump twin lives on Adapter instances.
    "load_entry",
    # Epic D-7 (sanctioned public surface): the template authoring adapter,
    # the section-9 registration APIs with admission gates, and the
    # language-discoverability call the contract requires.
    "TemplateAdapter",
    # D-δ: load_entry's return type and register_preset's consumer —
    # users already hold instances of it (design-eval C1 verdict).
    "PresetAdapter",
    "describe_template_language",
    "register_codec",
    "register_preset",
    "register_strategy",
    "unregister_codec",
    "unregister_preset",
    "unregister_strategy",
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
