"""The vendored lm15 copy must import as ``dspy._vendor.lm15`` from any install."""

from pathlib import Path


def test_vendored_lm15_imports_under_dspy_namespace():
    from dspy._vendor import lm15
    from dspy._vendor.lm15 import Message, Request
    from dspy._vendor.lm15.providers import openai

    assert "_vendor" in Path(lm15.__file__).parts
    assert openai.__name__ == "dspy._vendor.lm15.providers.openai"
    assert Request(model="x", messages=[Message.user("hi")]).model == "x"


def test_vendored_lm15_carries_provenance_and_license():
    import dspy._vendor.lm15 as lm15

    pkg = Path(lm15.__file__).parent
    provenance = pkg.parent / "lm15-provenance.txt"
    if provenance.exists():
        marker = provenance.read_text()
        assert all(f"{key}=" in marker for key in ("source", "commit", "contract", "split"))
        license_file = pkg.parent / "lm15-LICENSE"
    else:
        # The initial copied snapshot remains valid until the first subtree update.
        marker = (pkg / "VENDORED").read_text()
        assert "commit=" in marker and "digest=" in marker
        license_file = pkg / "LICENSE"
    assert "MIT" in license_file.read_text()
