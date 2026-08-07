"""Pure structural, version, and link checks for ProgramIR manifests."""

from __future__ import annotations

import re
from typing import Any

from dspy.programir.errors import ProgramIRRefusal
from dspy.programir.interpreters import validate_interpreter_profile
from dspy.programir.versions import IMPLEMENTED_VERSIONS, REQUIRED_VERSION_KEYS

_REQUIRED_COMPONENTS = (
    "1_module_tree",
    "2_signature",
    "3a_instructions",
    "3b_demos",
    "3c_predictor_config",
    "4_adapter",
    "5_forward",
    "6_tools",
    "7_interpreter",
    "8_lm",
    "9_environment",
    "10_credentials",
    "11_ambient_policy",
)
_OPTIONAL_COMPONENTS = ("12_metric",)
_TOP_LEVEL_KEYS = {"versions", "components", "provenance"}


def validate_manifest(manifest: Any) -> None:
    """Validate the closed manifest envelope and load-bearing component shape."""
    if not isinstance(manifest, dict):
        _manifest_refusal("manifest must be a JSON object", {"type": _json_type(manifest)})
    unknown = sorted(set(manifest) - _TOP_LEVEL_KEYS)
    if unknown:
        _manifest_refusal("manifest contains unknown top-level keys", {"unknown": unknown})
    missing = sorted({"versions", "components"} - set(manifest))
    if missing:
        _manifest_refusal("manifest is missing required top-level keys", {"missing": missing})

    check_versions(manifest)
    components = manifest["components"]
    if not isinstance(components, dict):
        _manifest_refusal("components must be an object", {"field": "components"})
    missing_components = [name for name in _REQUIRED_COMPONENTS if name not in components]
    unknown_components = sorted(set(components) - set(_REQUIRED_COMPONENTS) - set(_OPTIONAL_COMPONENTS))
    if missing_components or unknown_components:
        _manifest_refusal(
            "components do not match the closed component set",
            {"missing": missing_components, "unknown": unknown_components},
        )

    object_components = _REQUIRED_COMPONENTS[1:9] + ("9_environment", "11_ambient_policy")
    for name in object_components:
        if not isinstance(components[name], dict):
            _manifest_refusal(f"{name} must be an object", {"component": name})
    if not isinstance(components["10_credentials"], list):
        _manifest_refusal("10_credentials must be an array", {"component": "10_credentials"})
    for name, profile in components["7_interpreter"].items():
        if isinstance(profile, dict) and isinstance(profile.get("kind"), str):
            continue  # legacy fused profile
        if manifest["versions"].get("interpreter_profile") != "1.0":
            _manifest_refusal(
                "structural interpreters require versions.interpreter_profile=1.0",
                {"component": "7_interpreter", "entry": name},
            )
        try:
            validate_interpreter_profile(profile, name=name)
        except ValueError as error:
            _manifest_refusal(str(error), {"component": "7_interpreter", "entry": name})
    if "12_metric" in components:
        evaluation = components["12_metric"]
        if not isinstance(evaluation, dict) or set(evaluation) != {"metrics", "devset"}:
            _manifest_refusal(
                "12_metric must contain exactly metrics and devset",
                {"component": "12_metric"},
            )
        if not isinstance(evaluation["metrics"], dict) or not isinstance(evaluation["devset"], list):
            _manifest_refusal("12_metric metrics/devset have invalid types", {"component": "12_metric"})

    predictors = _collect_predictors(components["1_module_tree"])
    expected = set(predictors)
    for name in ("2_signature", "3a_instructions", "3b_demos", "3c_predictor_config"):
        actual = set(components[name])
        if actual != expected:
            _manifest_refusal(
                f"{name} keys do not match the module tree's predictors",
                {"component": name, "missing": sorted(expected - actual), "extra": sorted(actual - expected)},
            )


def check_versions(manifest: dict[str, Any]) -> dict[str, str]:
    """Check required version entries using the contract's pre-1.0 rule."""
    versions = manifest.get("versions")
    if not isinstance(versions, dict):
        _manifest_refusal("versions must be an object", {"field": "versions"})
    missing = [name for name in REQUIRED_VERSION_KEYS if name not in versions]
    if missing:
        _manifest_refusal("versions is missing required entries", {"missing": missing})
    for name, declared in versions.items():
        if not isinstance(declared, str) or not declared:
            _manifest_refusal("version entries must be non-empty strings", {"entry": name})
        implemented = IMPLEMENTED_VERSIONS.get(name)
        if implemented is None:
            continue
        declared_major = _major(declared)
        implemented_major = _major(implemented)
        compatible = declared_major == implemented_major and declared_major is not None
        if compatible and declared_major == 0:
            compatible = declared == implemented
        if not compatible:
            raise ProgramIRRefusal(
                "PIR-E-VERSION-001",
                "versions",
                {"entry": name, "declared": declared, "implemented": implemented},
                f"versions.{name} declares {declared!r}, but this reader implements {implemented!r}",
            )
    return versions


def resolve_bindings(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Resolve every predictor's adapter and LM names into a binding table."""
    validate_manifest(manifest)
    components = manifest["components"]
    adapters = components["4_adapter"]
    lms = components["8_lm"]
    table: dict[str, dict[str, Any]] = {}
    for path, node in _collect_predictors(components["1_module_tree"]).items():
        bindings = node.get("bindings")
        if not isinstance(bindings, dict) or set(bindings) != {"adapter", "lm", "delta"}:
            _link_refusal(path, "bindings", "tree", bindings)
        adapter = bindings["adapter"]
        lm = bindings["lm"]
        if adapter not in adapters:
            _link_refusal(path, "adapter", "4_adapter", adapter)
        if lm not in lms:
            _link_refusal(path, "lm", "8_lm", lm)
        if bindings["delta"] is not None:
            _link_refusal(path, "delta", "unratified", bindings["delta"])
        table[path] = dict(bindings)
    return table


def _collect_predictors(tree: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(tree, dict):
        _manifest_refusal("1_module_tree must be an object", {"component": "1_module_tree"})
    found: dict[str, dict[str, Any]] = {}

    def walk(node: Any, path: str) -> None:
        if not isinstance(node, dict):
            _manifest_refusal("module tree children must be objects", {"component": "1_module_tree"})
        name = node.get("name")
        kind = node.get("kind")
        children = node.get("children")
        if not isinstance(name, str) or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            _manifest_refusal("module tree node has an invalid name", {"name": name})
        if not isinstance(kind, str) or not isinstance(children, list):
            _manifest_refusal("module tree node lacks kind or children", {"name": name})
        if kind == "Predict":
            if children:
                _manifest_refusal("Predict leaves cannot have children", {"predictor": path})
            found[path] = node
            return
        for child in children:
            child_name = child.get("name") if isinstance(child, dict) else None
            child_path = child_name if path == "self" else f"{path}.{child_name}"
            walk(child, child_path)

    walk(tree, "self")
    return found


def _major(version: str) -> int | None:
    match = re.match(r"^(\d+)(?:\.|$)", version)
    return int(match.group(1)) if match else None


def _manifest_refusal(message: str, detail: dict[str, Any]) -> None:
    raise ProgramIRRefusal("PIR-E-MANIFEST-001", "manifest", detail, message)


def _link_refusal(predictor: str, binding: str, pool: str, entry: Any) -> None:
    raise ProgramIRRefusal(
        "PIR-E-LINK-002",
        "1_module_tree",
        {"predictor": predictor, "binding": binding, "pool": pool, "entry": entry},
        f"predictor {predictor!r} has unresolved {binding} binding {entry!r} in {pool}",
    )


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__
