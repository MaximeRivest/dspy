"""Child-process scoring harness for FlexIR `eval_mode="artifact"`.

The parent exports a candidate as a normal ProgramIR artifact, then runs
this module in a subprocess under the artifact's own environment:

    <artifact-env-python> -m dspy.optim._score_harness <job.json>

The job file carries the artifact directory, the devset examples, the
metric's self-contained source, and serialized DummyLM state. The
harness loads the artifact (`dspy.load` + bindings), runs the shared
`evaluate` over the examples, and prints ONE JSON object on stdout:
`{"score", "lm_calls", "results", "consumed"}`.

The sacred error distinction crosses the process boundary unchanged:
- catchable program errors are handled INSIDE `evaluate` (that example
  scores 0.0) and the harness exits 0 with a payload;
- anything that escapes — an unloadable artifact, an uncatchable engine
  guard, a malformed job — is INFRASTRUCTURE: the harness prints a
  `FLEXIR-INFRA` line on stderr and exits non-zero, and the parent
  raises instead of scoring the candidate.

Trust note: the harness GRANTS the artifact's isolation-required tool
leaves from their sidecars. This is the optimizer scoring its OWN
candidate — exactly the code that would run in-process under
`eval_mode="in_process"` — not a receiver-side bypass of the grant gate.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any


def _load_function(source: str, *, tag: str) -> Any:
    from dspy.modules._generate import load_generated

    name = ast.parse(source).body[0].name
    return load_generated(source, tag=tag, name=name)


def _bindings(job: dict[str, Any]) -> dict[str, dict[str, Any]]:
    import dspy

    lms: dict[str, Any] = {}
    for name, spec in job["lm"].items():
        if "script" in spec:
            lms[name] = dspy.DummyLM(spec["script"])
        else:
            lms[name] = dspy.DummyLM(_load_function(spec["function_source"], tag="flex-harness-lm"))
    artifact = Path(job["artifact"])
    manifest = json.loads((artifact / "manifest.json").read_text())
    tools: dict[str, Any] = {}
    for name, entry in manifest["components"]["6_tools"].items():
        if entry.get("placement", {}).get("rung") == "isolation_required":
            source = (artifact / entry["source"]).read_text()
            tools[name] = _load_function(source, tag="flex-harness-tool")
    return {"lm": lms, "tool": tools}


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())

    import dspy
    from dspy.core.example import Example
    from dspy.optim.base import evaluate

    metric = _load_function(job["metric_source"], tag="flex-harness-metric")
    bindings = _bindings(job)
    program = dspy.load(job["artifact"], bindings=bindings)
    examples = [Example(**record["values"]).with_inputs(*record["input_keys"]) for record in job["examples"]]
    result = evaluate(program, examples, metric)
    payload = {
        "score": result.score,
        "lm_calls": result.lm_calls,
        "results": [
            {"prediction": None if prediction is None else prediction.toDict(), "value": value}
            for _example, prediction, value in result.results
        ],
        "consumed": {name: len(lm.calls) for name, lm in bindings["lm"].items()},
    }
    print(json.dumps(payload, default=str))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as error:  # anything escaping evaluate is infrastructure
        print(f"FLEXIR-INFRA {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(3)
