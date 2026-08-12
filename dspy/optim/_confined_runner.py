"""Child runner for `dspy.confined`: run one callable under the envelope.

The parent (already having placed the child under the isolation envelope
via `preexec_fn`) launches this with a job file carrying the callable's
source and JSON args. The child defines the one function, calls it, and
prints one JSON line: `{"value": ...}` on success or `{"error": "..."}`
on a caught exception. Anything that escapes exits non-zero — the parent
raises rather than returning a value.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    namespace: dict[str, Any] = {}
    exec(compile(job["source"], "<dspy-confined>", "exec"), namespace)  # the author's own function, in the child
    fn = namespace[job["name"]]
    payload = json.loads(job["payload"])
    try:
        value = fn(*payload["args"], **payload["kwargs"])
    except Exception as error:  # a caught callable error is DATA, not an abort
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        return
    print(json.dumps({"value": value}, default=str))


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as error:  # infrastructure — the parent raises
        print(f"CONFINED-INFRA {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(3)
