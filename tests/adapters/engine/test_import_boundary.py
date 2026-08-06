"""Mechanical import boundary for the adapter engine (D-021, spec section 8).

``dspy/adapters/_engine/`` must stay extraction-ready: a later move to a
standalone rendering/parsing library has to be a move, not a surgery. The
engine may import only the signature core, the types layer (adapter types,
the core LM contract types, the adapter value utils, contract exceptions),
and — post Epic E — lm15. It may never import settings, modules, clients,
or teleprompt.

Enforcement is AST-based over every import statement in the package,
including function-local ones. Back-edges that predate the boundary are
pinned in ``KNOWN_BACK_EDGES`` as a SHRINKING allowlist: a new violation
fails immediately, and removing a back-edge without deleting its pin also
fails, so the list can only shrink.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ENGINE_ROOT = REPO_ROOT / "dspy" / "adapters" / "_engine"

#: Import targets the boundary allows. Everything is matched by module
#: prefix after normalizing ``from package import submodule`` forms.
ALLOWED_PREFIXES = (
    "dspy.adapters._engine",  # the engine itself
    "dspy.adapters.types",  # adapter types layer (Image, Tool, History, ...)
    "dspy.adapters.utils",  # value render/parse helpers (codec substrate)
    "dspy.signatures",  # the signature core
    "dspy.core",  # the typed LM contract (LMMessage, LMRequestPatch, ...)
    "dspy.utils.exceptions",  # contract exceptions (AdapterParseError, ...)
)

#: Pre-existing boundary violations, pinned exactly. Each is a back-edge
#: into adapter classes, clients, or re-export namespaces documented in
#: ``roadmap/epic-D-adapter-serializer.md``. Delete the pin when the
#: back-edge dies; never add to this set.
KNOWN_BACK_EDGES = frozenset(
    {
        # Adapter-class back-edges: the migration staging table and the
        # legacy helpers the engine still shares with base.py.
        ("builder.py", "dspy.adapters.base"),
        ("render.py", "dspy.adapters.base"),
        ("postprocess.py", "dspy.adapters.base"),
        ("migrated.py", "dspy.adapters.baml_adapter"),
        ("migrated.py", "dspy.adapters.chat_adapter"),
        ("migrated.py", "dspy.adapters.json_adapter"),
        ("migrated.py", "dspy.adapters.two_step_adapter"),
        ("migrated.py", "dspy.adapters.xml_adapter"),
        ("formats/baml.py", "dspy.adapters.baml_adapter"),
        # TwoStep's extraction stage builds a ChatAdapter and calls an LM in
        # parse; dies when TwoStep expands as a lowering (Epic F).
        ("formats/twostep.py", "dspy.adapters.chat_adapter"),
        # Clients back-edge: legacy-output derivation from LMResponse;
        # retired by the lm15 veneer (Epic E).
        ("parser_hook.py", "dspy.clients.openai_format"),
        # Citations is re-exported through dspy.experimental; the class
        # itself lives in the allowed types layer.
        ("roles.py", "dspy.experimental"),
        ("strategies/__init__.py", "dspy.experimental"),
        ("strategies/citations.py", "dspy.experimental"),
    }
)


def _is_submodule(module: str, name: str) -> bool:
    """True when ``from module import name`` imports a submodule."""
    base = REPO_ROOT / Path(*module.split("."))
    return (base / f"{name}.py").exists() or (base / name / "__init__.py").exists()


def _import_targets(tree: ast.AST) -> set[str]:
    """Every dspy module targeted by an import statement in ``tree``."""
    targets = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative: stays inside the engine package
                continue
            module = node.module or ""
            for alias in node.names:
                if _is_submodule(module, alias.name):
                    targets.add(f"{module}.{alias.name}")
                else:
                    targets.add(module)
    return {t for t in targets if t == "dspy" or t.startswith("dspy.")}


def _allowed(target: str) -> bool:
    return any(target == prefix or target.startswith(prefix + ".") for prefix in ALLOWED_PREFIXES)


def _scan_engine() -> set[tuple[str, str]]:
    violations = set()
    for path in sorted(ENGINE_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(ENGINE_ROOT).as_posix()
        for target in _import_targets(tree):
            if not _allowed(target):
                violations.add((relative, target))
    return violations


def test_engine_import_boundary():
    observed = _scan_engine()

    new_violations = observed - KNOWN_BACK_EDGES
    assert not new_violations, (
        "new engine import-boundary violations (D-021): "
        f"{sorted(new_violations)}. The engine imports only the signature "
        f"core, the types layer, and (post-E) lm15 — allowed prefixes: "
        f"{ALLOWED_PREFIXES}. Do not extend KNOWN_BACK_EDGES."
    )

    stale_pins = KNOWN_BACK_EDGES - observed
    assert not stale_pins, (
        f"stale KNOWN_BACK_EDGES entries (the back-edge is gone — delete "
        f"the pin so the allowlist shrinks): {sorted(stale_pins)}"
    )


def test_engine_never_imports_banned_layers_even_via_pins():
    """The pin list itself may never contain settings/modules/teleprompt.

    Clients back-edges are pinned (they die in Epic E), but ambient
    settings, dspy modules, and optimizers were never imported by the
    engine and must never appear even as pins.
    """
    banned_roots = ("dspy.dsp", "dspy.predict", "dspy.primitives", "dspy.teleprompt", "dspy.evaluate", "dspy.streaming")
    for _, target in KNOWN_BACK_EDGES:
        assert not any(target == root or target.startswith(root + ".") for root in banned_roots), target
