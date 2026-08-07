"""Bake structurally declared in-process LM weights."""

from __future__ import annotations

import ast
import inspect
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dspy.programir.leaves import parse_deps
from dspy.programir.write import canonical_json_bytes


@dataclass(frozen=True)
class BakedLM:
    """Hold one baked LM entry and its binary and text sidecars."""

    entry: dict[str, Any]
    sidecars: dict[str, bytes]
    dependencies: list[str]


def has_weight_spec(lm: Any) -> bool:
    """Return whether an LM declares the structural weight-baking protocol."""
    return callable(getattr(lm, "programir_weight_spec", None))


def bake_lm(lm: Any, *, name: str, weights_root: str = "weights") -> BakedLM:
    """Bake one LM that declares `programir_weight_spec()`.

    The hook returns plain metadata and names the attributes that own the model
    and tokenizer. This keeps detection structural: the exporter never depends
    on a particular Transformers model or an application-defined LM class.
    """
    hook = getattr(lm, "programir_weight_spec", None)
    if not callable(hook):
        raise ValueError(f"ProgramIR LM {type(lm).__name__} does not declare programir_weight_spec()")
    spec = hook()
    if not isinstance(spec, Mapping):
        raise ValueError("ProgramIR programir_weight_spec() must return a mapping")

    required = {
        "model_attribute",
        "tokenizer_attribute",
        "weights_identity",
        "engine",
        "frozen",
        "ties",
    }
    missing = sorted(required - spec.keys())
    if missing:
        raise ValueError(f"ProgramIR weight spec for LM {name!r} is missing {missing}")
    unknown = sorted(
        set(spec) - required - {"device", "rebuild_config", "weight_ref"}
    )
    if unknown:
        raise ValueError(f"ProgramIR weight spec for LM {name!r} has unknown fields {unknown}")

    model = _owned_attribute(lm, spec["model_attribute"], role="model", name=name)
    tokenizer = _owned_attribute(lm, spec["tokenizer_attribute"], role="tokenizer", name=name)
    identity = _nonempty_string(spec["weights_identity"], field="weights_identity", name=name)
    engine = _nonempty_string(spec["engine"], field="engine", name=name)
    if not isinstance(spec["frozen"], bool):
        raise ValueError(f"ProgramIR weight spec for LM {name!r} field 'frozen' must be boolean")
    ties = _validate_ties(spec["ties"], name=name)

    state = model.state_dict() if callable(getattr(model, "state_dict", None)) else None
    if not isinstance(state, Mapping):
        raise ValueError(f"ProgramIR weight-owning LM {name!r} model must provide state_dict()")
    tensors = dict(state)
    absent_ties = [tie for tie in ties if tie["source"] not in tensors or tie["target"] not in tensors]
    if absent_ties:
        tie = absent_ties[0]
        raise ValueError(
            f"ProgramIR weight tie for LM {name!r} names absent tensor "
            f"{tie['target']!r} or {tie['source']!r}"
        )
    for tie in ties:
        del tensors[tie["target"]]

    config = spec.get("rebuild_config")
    if config is None:
        model_config = getattr(model, "config", None)
        config = model_config.to_dict() if callable(getattr(model_config, "to_dict", None)) else None
    if not isinstance(config, Mapping):
        raise ValueError(
            f"ProgramIR weight-owning LM {name!r} must declare rebuild_config or own a model.config.to_dict()"
        )

    root = weights_root.rstrip("/")
    files = {
        "tensors": f"{root}/model.safetensors",
        "rebuild_config": f"{root}/rebuild_config.json",
        "tying": f"{root}/tying.json",
        "tokenizer": f"{root}/tokenizer/",
        "device": f"{root}/device.json",
    }
    sidecars = {
        files["tensors"]: _save_safetensors(tensors),
        files["rebuild_config"]: canonical_json_bytes(dict(config)),
        files["tying"]: canonical_json_bytes(ties),
        files["device"]: canonical_json_bytes({"device": str(spec.get("device", "cpu"))}),
    }
    sidecars.update(_save_tokenizer(tokenizer, root=f"{root}/tokenizer"))

    source, dependencies = _authored_lm_source(type(lm), name=name)
    source_path = f"lm/{name}.py"
    sidecars[source_path] = source.encode("utf-8")
    placement = _in_process_placement()
    weights = {
        "format": "safetensors",
        "files": files,
        "frozen": spec["frozen"],
        "placement": placement,
    }
    if "weight_ref" in spec:
        weights["weight_ref"] = _nonempty_string(spec["weight_ref"], field="weight_ref", name=name)
    entry = {
        "forward_contract": getattr(type(lm), "forward_contract", "legacy"),
        "class": {
            "identity": f"{type(lm).__module__}.{type(lm).__qualname__}",
            "origin": "authored",
            "language": "python",
            "source": source_path,
            "deps": dependencies,
        },
        "weights_identity": identity,
        "engine": engine,
        "weights": weights,
        "placement": placement,
    }
    if entry["forward_contract"] != "typed_lm":
        raise ValueError(f"ProgramIR weight-owning LM {name!r} must declare forward_contract = 'typed_lm'")
    return BakedLM(entry=entry, sidecars=sidecars, dependencies=dependencies)


def _owned_attribute(lm: Any, attribute: Any, *, role: str, name: str) -> Any:
    if not isinstance(attribute, str) or not attribute:
        raise ValueError(f"ProgramIR weight spec for LM {name!r} field '{role}_attribute' must name an attribute")
    if not hasattr(lm, attribute):
        raise ValueError(f"ProgramIR weight-owning LM {name!r} has no {role} attribute {attribute!r}")
    return getattr(lm, attribute)


def _nonempty_string(value: Any, *, field: str, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"ProgramIR weight spec for LM {name!r} field {field!r} must be a non-empty string")
    return value


def _validate_ties(value: Any, *, name: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError(f"ProgramIR weight spec for LM {name!r} field 'ties' must be a list")
    ties: list[dict[str, str]] = []
    targets: set[str] = set()
    for tie in value:
        if not isinstance(tie, Mapping) or set(tie) != {"target", "source"}:
            raise ValueError(f"ProgramIR weight ties for LM {name!r} must contain only target and source")
        target = _nonempty_string(tie["target"], field="ties.target", name=name)
        source = _nonempty_string(tie["source"], field="ties.source", name=name)
        if target == source or target in targets:
            raise ValueError(f"ProgramIR weight ties for LM {name!r} contain an invalid duplicate target {target!r}")
        targets.add(target)
        ties.append({"target": target, "source": source})
    return ties


def _save_safetensors(tensors: Mapping[str, Any]) -> bytes:
    try:
        from safetensors.torch import save
    except ImportError as error:
        raise ValueError("ProgramIR weight baking requires the 'safetensors' package") from error
    try:
        return save(dict(tensors))
    except Exception as error:
        raise ValueError(f"ProgramIR could not serialize model state as safetensors: {error}") from error


def _save_tokenizer(tokenizer: Any, *, root: str) -> dict[str, bytes]:
    save_pretrained = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_pretrained):
        raise ValueError("ProgramIR weight-owning LM tokenizer must provide save_pretrained()")
    with tempfile.TemporaryDirectory(prefix="programir-tokenizer-") as temporary:
        directory = Path(temporary) / "tokenizer"
        directory.mkdir()
        save_pretrained(directory)
        files = [path for path in sorted(directory.rglob("*")) if path.is_file()]
        if not files:
            raise ValueError("ProgramIR tokenizer save_pretrained() produced no files")
        return {f"{root}/{path.relative_to(directory).as_posix()}": path.read_bytes() for path in files}


def _authored_lm_source(cls: type, *, name: str) -> tuple[str, list[str]]:
    try:
        source = textwrap.dedent(inspect.getsource(cls)).strip() + "\n"
    except (OSError, TypeError) as error:
        raise ValueError(f"ProgramIR authored LM {name!r} source is not introspectable") from error
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise ValueError(f"ProgramIR authored LM {name!r} source is not one class definition")
    if tree.body[0].decorator_list:
        raise ValueError(f"ProgramIR authored LM {name!r} uses decorators; bake the undecorated class instead")
    dependencies = parse_deps(source)
    source = "import dspy\nfrom dspy import BaseLM, LMRequest, LMResponse\n\n" + source
    return source, dependencies


def _in_process_placement() -> dict[str, Any]:
    return {
        "rung": "in_process",
        "contract": "forward(LMRequest)->LMResponse",
        "endpoint_ref": None,
        "isolation": "none",
        "credential_ref": None,
    }
