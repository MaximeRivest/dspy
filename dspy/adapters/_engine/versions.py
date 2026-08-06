"""Version constants for the adapter IR serde (D-024), single-sourced.

Every closed vocabulary carries its version constant next to its data;
this module is the ONE aggregation point the serde (and later the ProgramIR
exporter, D-γ) reads. A serialized preset carries ``adapter_ir_version``
plus the ``versions`` block from its first dump/load shape — no
versionless shape may ever exist — and a loader meeting an unknown major
(or, while a version is 0.x, an unknown minor) refuses loudly naming both
versions.
"""

from dspy.adapters._engine.codecs import CODECS_VERSION
from dspy.adapters._engine.strategies.vocabulary import STRATEGIES_VERSION
from dspy.adapters._engine.template.vocabulary import TEMPLATE_LANGUAGE_VERSION
from dspy.signatures.field import SEMANTIC_ROLES_VERSION

#: Version of the adapter IR contract itself (roadmap/adapter-ir-spec.md;
#: provisional, hence 0.x — minor bumps are breaking until 1.0.0).
ADAPTER_IR_VERSION = "0.1.0"


def versions_block() -> dict:
    """The vocabulary-versions block a serialized preset carries (D-024)."""
    return {
        "roles": SEMANTIC_ROLES_VERSION,
        "strategies": STRATEGIES_VERSION,
        "codecs": CODECS_VERSION,
        "template_language": TEMPLATE_LANGUAGE_VERSION,
    }


def parse_version(version, *, what: str) -> tuple[int, int, int]:
    """Parse ``"major.minor.patch"``; refuse malformed naming ``what``."""
    if isinstance(version, str):
        parts = version.split(".")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            return tuple(int(part) for part in parts)  # type: ignore[return-value]
    raise ValueError(f"malformed {what} version {version!r} — expected 'major.minor.patch'")


def check_version_compatible(kind: str, theirs, ours: str) -> None:
    """Refuse loudly when ``theirs`` is not readable by ``ours``.

    Unknown majors refuse; while a vocabulary is 0.x, unknown minors refuse
    too (the semver-0 convention). The refusal names both versions and the
    block (L5 applied to time).
    """
    their_major, their_minor, _ = parse_version(theirs, what=kind)
    our_major, our_minor, _ = parse_version(ours, what=kind)
    incompatible = their_major != our_major or (our_major == 0 and their_minor != our_minor)
    if incompatible:
        raise ValueError(
            f"incompatible {kind} version: entry carries {theirs!r}, this engine speaks {ours!r} — "
            "refusing to load rather than misread (D-024)"
        )
