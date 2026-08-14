"""Turn spelling: how a past tool exchange is written into the next prompt.

The `turns` strategy face is the render dual of `routings`: routings say
where field values arrive from in a reply; `turns` says how a prior
exchange's tool calls and results are spelled into the next prompt's
messages. A template slot here speaks a closed vocabulary — loops over
`calls` / `results` with `{c.name}`, `{c.args}`, `{c.id}`, `{r.name}`,
`{r.value}`, `{r.id}` — small enough to render with no engine context.
"""

import json
import re
from typing import Any

_TURNS_LOOP = re.compile(r"\{% for (?:c|r) in (calls|results) %\}\n?(.*?)\{% endfor %\}", re.DOTALL)


def render_turn_content(content: str, *, calls: Any = (), results: Any = ()) -> str:
    """Render one turns template: loops stamp per item, literals stay.

    `{c.args}` spells the arguments as canonical JSON, so the spelled call
    is exactly what a tool-calls routing reads back.
    """

    def stamp(match: re.Match) -> str:
        collection, body = match.groups()
        stamped = []
        if collection == "calls":
            for item in calls:
                text = body.replace("{c.name}", str(item.get("name", "")))
                text = text.replace("{c.args}", json.dumps(item.get("args", {}), ensure_ascii=False))
                text = text.replace("{c.id}", str(item.get("id") or ""))
                stamped.append(text.strip("\n"))
        else:
            for item in results:
                text = body.replace("{r.name}", str(item.get("name", "")))
                text = text.replace("{r.value}", result_text(item.get("value")))
                text = text.replace("{r.id}", str(item.get("id") or ""))
                stamped.append(text.strip("\n"))
        return "\n".join(stamped)

    return _TURNS_LOOP.sub(stamp, content).strip("\n")


def result_text(value: Any) -> str:
    """One tool result as text: strings verbatim, everything else JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def normalized_results(calls: list[dict], raw: Any) -> list[dict]:
    """Tool results as `{id, name, value}` items aligned with the calls.

    Accepts a list (zipped with the calls in order), a dict keyed by tool
    name, or a single bare value (attributed to the first call).
    """
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [
            {"id": call["id"], "name": call["name"], "value": raw[call["name"]]}
            for call in calls
            if call["name"] in raw
        ]
    if isinstance(raw, (list, tuple)):
        return [{"id": call["id"], "name": call["name"], "value": value} for call, value in zip(calls, raw)]
    first = calls[0] if calls else {"id": None, "name": ""}
    return [{"id": first["id"], "name": first["name"], "value": raw}]
