"""Versions implemented by the ProgramIR compiler."""

IMPLEMENTED_VERSIONS = {
    "ir_version": "0.1",
    "node_set": "0.3",
    "roles": "0.1",
    "strategies": "0.1",
    "codecs": "0.1",
    "adapter_ir": "0.1",
    "lm15": "0.1",
    "interpreter_profile": "1.0",
}

#: Pre-1.0 version-set rule (spec/versions.md, D-034): an implementation
#: declares a SET of supported exact versions; membership accepts. The
#: compiler stamps the newest member (IMPLEMENTED_VERSIONS); readers
#: accept any member.
SUPPORTED_VERSION_SETS = {
    "node_set": ("0.1", "0.2", "0.3"),
}

REQUIRED_VERSION_KEYS = (
    "ir_version",
    "node_set",
    "roles",
    "strategies",
    "codecs",
    "adapter_ir",
    "lm15",
)


def supported_versions(name: str) -> tuple[str, ...] | None:
    """Return the exact versions this implementation accepts for `name`."""
    implemented = IMPLEMENTED_VERSIONS.get(name)
    if implemented is None:
        return None
    return SUPPORTED_VERSION_SETS.get(name, (implemented,))
