"""Regenerate contract fixtures from artifacts emitted by `corpus_programs.py`."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path

DATE = "2026-08-07"
ARTIFACT_NAMES = {
    "01": "ap-01-predict-chat",
    "02": "ap-02-predict-xml",
    "03": "ap-03-predict-json",
    "09": "ap-09-xml-gemma-http",
    "10": "ap-10-xml-cerebras-remote",
    "12": "ap-12-step0-shared-lm",
    "13": "ap-13-nested-modules",
    "14": "ap-14-mini-react",
    "15": "ap-15-mini-rlm",
}


def _read(path: Path):
    return json.loads(path.read_text())


def _write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def _provenance(sha: str, source: str, detail: str):
    return {
        "source": source,
        "date": DATE,
        "evidence": f"exported by dspy at {sha}; {detail}",
    }


def _case(case_id, op, case_input, expect, sha, source, detail, notes, cites):
    return {
        "id": case_id,
        "op": op,
        "input": case_input,
        "expect": expect,
        "provenance": _provenance(sha, source, detail),
        "notes": notes,
        "cites": cites,
    }


def _predictor_bindings(tree):
    bindings = {}

    def walk(node, path):
        if node.get("kind") == "Predict":
            bindings[path or node["name"]] = copy.deepcopy(node["bindings"])
            return
        for child in node.get("children", []):
            child_path = f"{path}.{child['name']}" if path else child["name"]
            walk(child, child_path)

    walk(tree, "")
    return bindings


def _find_predictor(tree, path):
    found = None

    def walk(node, current):
        nonlocal found
        if node.get("kind") == "Predict" and (current or node["name"]) == path:
            found = node
        for child in node.get("children", []):
            child_path = f"{current}.{child['name']}" if current else child["name"]
            walk(child, child_path)

    walk(tree, "")
    if found is None:
        raise KeyError(path)
    return found


def emit(artifacts: Path, contract: Path, sha: str):
    manifests = {name: _read(artifacts / name / "manifest.json") for name in ARTIFACT_NAMES}
    cases = contract / "cases"
    provisional = cases / "artifacts-provisional"
    if provisional.exists():
        shutil.rmtree(provisional)
    artifact_dir = cases / "artifacts"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)

    for name, case_id in ARTIFACT_NAMES.items():
        manifest = manifests[name]
        source = f"dspy.export corpus program {name}"
        case = _case(
            case_id,
            "load_manifest",
            {"manifest": manifest},
            {"ok": True, "result": {"manifest": manifest}},
            sha,
            source,
            "public exporter output, no migration",
            "Exporter-emitted canonical manifest. The expected normalized form is the manifest itself.",
            ["PIR-001", "PIR-004", "PIR-009", "spec/manifest.md (D-029)"],
        )
        _write(artifact_dir / f"{case_id}.json", case)

    # Link success and refusal vectors re-derived from exporter bytes.
    for case_id, name in (("ls-01-single", "01"), ("ls-13-nested", "13")):
        manifest = manifests[name]
        result = {"bindings": _predictor_bindings(manifest["components"]["1_module_tree"])}
        _write(
            cases / "link-success" / f"{case_id}.json",
            _case(
                case_id,
                "link",
                {"manifest": manifest},
                {"ok": True, "result": result},
                sha,
                "mutated-artifact",
                f"link result re-derived from exported example {name}",
                "Successful predictor bindings from an exporter-emitted artifact.",
                ["PIR-009", "spec/linking.md"],
            ),
        )

    dangling_adapter = copy.deepcopy(manifests["01"])
    root = dangling_adapter["components"]["1_module_tree"]
    root["bindings"]["adapter"] = "missing-adapter"
    _write(
        cases / "link-errors" / "le-dangling-adapter.json",
        _case(
            "le-dangling-adapter",
            "link",
            {"manifest": dangling_adapter},
            {
                "ok": False,
                "error": {
                    "code": "PIR-E-LINK-001",
                    "predictor": "self",
                    "binding": "adapter",
                    "pool": "4_adapter",
                    "entry": "missing-adapter",
                },
            },
            sha,
            "mutated-artifact",
            "example 01 root adapter binding changed to an absent pool entry",
            "A predictor binding must resolve in its named pool.",
            ["PIR-009", "spec/errors.md LINK-001"],
        ),
    )
    dangling_lm = copy.deepcopy(manifests["13"])
    _find_predictor(dangling_lm["components"]["1_module_tree"], "polish")["bindings"]["lm"] = "missing-lm"
    _write(
        cases / "link-errors" / "le-dangling-lm.json",
        _case(
            "le-dangling-lm",
            "link",
            {"manifest": dangling_lm},
            {
                "ok": False,
                "error": {
                    "code": "PIR-E-LINK-001",
                    "predictor": "polish",
                    "binding": "lm",
                    "pool": "8_lm",
                    "entry": "missing-lm",
                },
            },
            sha,
            "mutated-artifact",
            "example 13 polish LM binding changed to an absent pool entry",
            "A nested predictor's LM binding must resolve in component 8.",
            ["PIR-009", "spec/errors.md LINK-001"],
        ),
    )

    # Manifest refusal vectors from the current exported envelope.
    old_unversioned = [
        cases / "manifest-errors" / "me-unversioned-01-flat.json",
        cases / "manifest-errors" / "me-unversioned-11-map.json",
        cases / "manifest-errors" / "me-unversioned-13-pool.json",
    ]
    for path in old_unversioned:
        path.unlink(missing_ok=True)
    for name in ("01", "12", "13"):
        manifest = copy.deepcopy(manifests[name])
        del manifest["versions"]
        case_id = f"me-unversioned-{name}-exported"
        _write(
            cases / "manifest-errors" / f"{case_id}.json",
            _case(
                case_id,
                "load_manifest",
                {"manifest": manifest},
                {"ok": False, "error": {"code": "PIR-E-MANIFEST-003", "component": "versions"}},
                sha,
                "mutated-artifact",
                f"versions removed from exported example {name}",
                "Every artifact carries the version block before any component is interpreted.",
                ["PIR-001", "spec/errors.md MANIFEST-003"],
            ),
        )
    malformed_interpreter = copy.deepcopy(manifests["15"])
    del malformed_interpreter["components"]["7_interpreter"]["interpreter"]["runtime"]
    _write(
        cases / "manifest-errors" / "me-structural-interpreter-missing-runtime.json",
        _case(
            "me-structural-interpreter-missing-runtime",
            "load_manifest",
            {"manifest": malformed_interpreter},
            {"ok": False, "error": {"code": "PIR-E-MANIFEST-002"}},
            sha,
            "mutated-artifact",
            "runtime removed from exported example 15 interpreter profile",
            "A structural interpreter profile must identify its runtime.",
            ["PIR-004", "spec/manifest.md"],
        ),
    )

    # Version vectors use exported example 09 as their complete envelope.
    for case_id, mutation, expect, detail in (
        (
            "ve-missing-entry-node-set",
            ("remove", "node_set", None),
            {
                "ok": False,
                "error": {"code": "PIR-E-MANIFEST-002", "component": "versions", "entry": "node_set"},
            },
            "node_set removed",
        ),
        (
            "ve-wrong-major-adapter-ir",
            ("set", "adapter_ir", "99.0"),
            {"ok": False, "error": {"code": "PIR-E-VERSION-001", "entry": "adapter_ir", "declared": "99.0"}},
            "adapter_ir changed to unknown major 99",
        ),
        (
            "ve-wrong-major-ir",
            ("set", "ir_version", "99.0"),
            {"ok": False, "error": {"code": "PIR-E-VERSION-001", "entry": "ir_version", "declared": "99.0"}},
            "ir_version changed to unknown major 99",
        ),
        (
            "ve-zero-minor-refuse",
            ("set", "node_set", "0.2"),
            {"ok": False, "error": {"code": "PIR-E-VERSION-001", "entry": "node_set", "declared": "0.2"}},
            "node_set changed from 0.1 to 0.2 under exact pre-1.0 matching",
        ),
    ):
        manifest = copy.deepcopy(manifests["09"])
        action, key, value = mutation
        if action == "remove":
            del manifest["versions"][key]
        else:
            manifest["versions"][key] = value
        _write(
            cases / "versions" / f"{case_id}.json",
            _case(
                case_id,
                "check_versions",
                {"manifest": manifest},
                expect,
                sha,
                "mutated-artifact",
                f"exported example 09; {detail}",
                "Version compatibility is checked before linking or interpretation.",
                ["PIR-001", "spec/versions.md (D-024)", "spec/errors.md VERSION-001"],
            ),
        )

    # Profile vectors now point directly at exported manifests.
    for case_id, name, in_profile, violations in (
        ("pf-in-09-gemma-http", "09", True, []),
        ("pf-in-10-cerebras-remote", "10", True, []),
        (
            "pf-out-01-inproc-baked",
            "01",
            False,
            [
                {"clause": "DT-001", "subject": "8_lm/PleIAs-Baguettotron"},
                {"clause": "DT-003", "subject": "8_lm/PleIAs-Baguettotron"},
                {"clause": "DT-004", "subject": "8_lm/PleIAs-Baguettotron"},
            ],
        ),
    ):
        _write(
            cases / "profile" / f"{case_id}.json",
            _case(
                case_id,
                "profile_check",
                {"manifest": manifests[name], "profile": "declared-tier"},
                {
                    "ok": True,
                    "result": {
                        "profile": "declared-tier",
                        "in_profile": in_profile,
                        "violations": violations,
                    },
                },
                sha,
                "dspy.export",
                f"profile result for exported example {name}",
                "Declared examples remain in profile; baked in-process weights remain out of profile.",
                ["PIR-011", "spec/placement.md (D-023)"],
            ),
        )

    # Compiler-oracle forwards: each result echoes the exporter-produced node body.
    node_dir = cases / "node-set" / "compile"
    for path in node_dir.glob("nc-accept-13-*.json"):
        path.unlink()
    for name in ("13", "14", "15"):
        for forward_name, forward in manifests[name]["components"]["5_forward"].items():
            suffix = forward_name.replace(".", "-")
            case_id = f"nc-accept-{name}-{suffix}"
            _write(
                node_dir / f"{case_id}.json",
                _case(
                    case_id,
                    "node_compile",
                    {"forward": forward},
                    {"ok": True, "result": {"forward": forward}},
                    sha,
                    "dspy.export",
                    f"5_forward/{forward_name} from exported example {name}",
                    "The live Python forward compiled to this ratified v0.1 node body.",
                    ["spec/node-set.md SEM-8"],
                ),
            )

    change = contract / "changes" / "2026-08-07-dspy-exported-corpus.md"
    change.write_text(
        "# DSPy-exported ProgramIR corpus\n\n"
        f"The nine provisional migrated manifests were replaced by public `dspy.export` output at DSPy `{sha}`. "
        "Link, manifest-error, version, profile, and node-compile derivatives were regenerated from those bytes "
        "in this dedicated corpus-only commit (L8). No contract source or specification changed.\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--dspy-sha", required=True)
    args = parser.parse_args()
    emit(args.artifacts, args.contract, args.dspy_sha)
