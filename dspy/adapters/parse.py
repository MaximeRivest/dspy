"""Parse combinators: declared parse-data, one vocabulary (level 1).

Authoring helpers build the exact data spellings of the adapter-ir-stage
examples; `validate_pipeline` refuses unknown ops and options eagerly with
teaching errors; `run_pipeline` executes a pipeline purely over a
completion string. The same vocabulary serves entry-level parsers and
strategy routings — parse combinators are ONE language wherever they
appear.

Regex combinators are restricted to the RE2 subset (no backreferences, no
lookaround, no atomic/conditional groups): cross-language identical, and a
no-backtracking DoS guard for free.

The origin collapse governs this whole module: pipelines are data with
total inspectability and zero authority, so loading one asks only "do you
speak `parse_combinators` 0.1", never "who wrote this".
"""

import json
import re
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import json_repair

from dspy.adapters.codecs import COERCE_SHAPES, coerce_shape
from dspy.adapters.errors import AdapterParseError, EntryError

#: Version of the parse-combinator vocabulary; present in an entry's
#: versions block only when the entry uses a pipeline.
PARSE_COMBINATORS_VERSION = "0.1.0"


# ---------------------------------------------------------------------------
# Authoring helpers: data in, data out
# ---------------------------------------------------------------------------


def pipeline(*steps: dict) -> dict:
    """A level-1 parser: an ordered list of combinator steps."""
    return {"kind": "pipeline", "steps": list(steps)}


def fenced_block(language: str | None = None, policy: str = "prefer") -> dict:
    """Extract a fenced code block. `policy`: prefer (use it when present)
    or require (refuse when absent)."""
    step: dict = {"op": "fenced_block"}
    if language is not None:
        step["language"] = language
    step["policy"] = policy
    return step


def alternatives(*branches: dict) -> dict:
    """Try each branch step in order; the first success wins."""
    return {"op": "alternatives", "try": list(branches)}


def json_object(repair: str = "none") -> dict:
    """Parse the text as one JSON object. `repair`: none (strict) or
    json_repair (the pinned repair policy)."""
    return {"op": "json_object", "repair": repair}


def fields_from_object(unknown_keys: str = "exhaust") -> dict:
    """Map object keys to output fields. `unknown_keys`: exhaust, ignore,
    or refuse."""
    return {"op": "fields_from_object", "unknown_keys": unknown_keys}


def regex(
    pattern: str,
    *,
    dialect: str = "re2",
    mode: str = "search",
    group: str | None = None,
    join: str | None = None,
) -> dict:
    """RE2-subset regex captures. `mode`: search (named groups) or findall
    (all matches); `group`+`join` collect one group across matches into a
    single string."""
    step: dict = {"op": "regex", "dialect": dialect, "pattern": pattern, "mode": mode}
    if group is not None:
        step["group"] = group
    if join is not None:
        step["join"] = join
    return step


def fields_from_groups() -> dict:
    """Named regex groups become same-named fields, coerced by the field
    schema."""
    return {"op": "fields_from_groups"}


def coerce(shape: str) -> dict:
    """Coerce the current value to a named shape from the closed vocabulary."""
    return {"op": "coerce", "shape": shape}


def tool_calls(*, name_group: str, args_group: str, args: str = "json") -> dict:
    """Pair regex findall groups into ToolCalls (name + parsed args)."""
    return {"op": "tool_calls", "name_group": name_group, "args_group": args_group, "args": args}


def citations(*, span_group: str, doc_group: str) -> dict:
    """Pair regex findall groups into Citations (claim span + doc index)."""
    return {"op": "citations", "span_group": span_group, "doc_group": doc_group}


# ---------------------------------------------------------------------------
# Validation: the closed vocabulary, refused eagerly
# ---------------------------------------------------------------------------

#: Per-op option table: required keys, optional keys, and enum values.
_OP_SPECS: dict[str, dict] = {
    "fenced_block": {
        "required": (),
        "optional": ("language", "policy"),
        "enums": {"policy": ("prefer", "require")},
    },
    "alternatives": {"required": ("try",), "optional": (), "enums": {}},
    "json_object": {
        "required": (),
        "optional": ("repair",),
        "enums": {"repair": ("none", "json_repair")},
    },
    "fields_from_object": {
        "required": (),
        "optional": ("unknown_keys",),
        "enums": {"unknown_keys": ("exhaust", "ignore", "refuse")},
    },
    "regex": {
        "required": ("pattern",),
        "optional": ("dialect", "mode", "group", "join"),
        "enums": {"dialect": ("re2",), "mode": ("search", "findall")},
    },
    "fields_from_groups": {"required": (), "optional": (), "enums": {}},
    "coerce": {"required": ("shape",), "optional": (), "enums": {"shape": COERCE_SHAPES}},
    "tool_calls": {
        "required": ("name_group", "args_group"),
        "optional": ("args",),
        "enums": {"args": ("json",)},
    },
    "citations": {"required": ("span_group", "doc_group"), "optional": (), "enums": {}},
}

#: RE2-subset refusals: constructs whose presence makes a pattern
#: non-portable (backtracking or engine-specific).
_NON_RE2 = (
    ("(?=", "lookahead"),
    ("(?!", "negative lookahead"),
    ("(?<=", "lookbehind"),
    ("(?<!", "negative lookbehind"),
    ("(?>", "atomic group"),
    ("(?(", "conditional group"),
    ("(?P=", "named backreference"),
    ("\\g<", "backreference"),
)


def validate_re2_pattern(pattern: str, *, where: str) -> None:
    """Refuse a regex outside the RE2 subset, naming the construct."""
    if not isinstance(pattern, str) or not pattern:
        raise EntryError(f"{where}: regex 'pattern' must be a non-empty string")
    for needle, name in _NON_RE2:
        if needle in pattern:
            raise EntryError(
                f"{where}: pattern uses {name} ({needle!r}) — outside the RE2 subset; "
                "RE2-subset patterns run identically in every receiver and cannot backtrack"
            )
    if re.search(r"\\[1-9]", pattern):
        raise EntryError(
            f"{where}: pattern uses a numeric backreference — outside the RE2 subset; "
            "RE2-subset patterns run identically in every receiver and cannot backtrack"
        )
    try:
        re.compile(pattern)
    except re.error as error:
        raise EntryError(f"{where}: pattern does not compile: {error}") from None


def validate_pipeline(spec: Any, *, where: str = "parser") -> None:
    """Refuse a malformed pipeline eagerly, naming the offending step."""
    if not isinstance(spec, dict) or spec.get("kind") != "pipeline":
        raise EntryError(f"{where}: a pipeline is a dict with kind='pipeline', got {spec!r}")
    unknown = set(spec) - {"kind", "steps"}
    if unknown:
        raise EntryError(f"{where}: unknown pipeline keys {sorted(unknown)} — valid keys: kind, steps")
    steps = spec.get("steps")
    if not isinstance(steps, list) or not steps:
        raise EntryError(f"{where}: a pipeline carries a non-empty 'steps' list")
    for index, step in enumerate(steps):
        _validate_step(step, where=f"{where}.steps[{index}]")


def _validate_step(step: Any, *, where: str) -> None:
    if not isinstance(step, dict) or "op" not in step:
        raise EntryError(f"{where}: a step is a dict with an 'op' key, got {step!r}")
    op = step["op"]
    if op not in _OP_SPECS:
        raise EntryError(f"{where}: unknown combinator {op!r} — parse_combinators vocabulary: {', '.join(_OP_SPECS)}")
    spec = _OP_SPECS[op]
    valid_keys = {"op", *spec["required"], *spec["optional"]}
    unknown = set(step) - valid_keys
    if unknown:
        raise EntryError(
            f"{where}: unknown {op} options {sorted(unknown)} — valid options: {', '.join(sorted(valid_keys - {'op'})) or '(none)'}"
        )
    missing = [key for key in spec["required"] if key not in step]
    if missing:
        raise EntryError(f"{where}: {op} requires options {missing}")
    for key, allowed in spec["enums"].items():
        if key in step and step[key] not in allowed:
            raise EntryError(f"{where}: unknown {op} {key} {step[key]!r} — valid values: {', '.join(allowed)}")
    if op == "regex":
        validate_re2_pattern(step["pattern"], where=where)
    if op == "alternatives":
        branches = step["try"]
        if not isinstance(branches, list) or not branches:
            raise EntryError(f"{where}: alternatives carries a non-empty 'try' list")
        for index, branch in enumerate(branches):
            _validate_step(branch, where=f"{where}.try[{index}]")


# ---------------------------------------------------------------------------
# Execution: pure over a completion string
# ---------------------------------------------------------------------------


@dataclass
class PipelineState:
    """The running state one pipeline threads through its steps."""

    text: str
    base_offset: int = 0
    value: Any = None
    obj: Any = None
    groups: dict[str, str] | None = None
    matches: list[dict[str, str]] | None = None
    fields: dict[str, Any] | None = None
    exhaust: dict[str, Any] = field(default_factory=dict)
    spans: list[tuple[int, int]] = field(default_factory=list)


def run_pipeline(spec: dict, completion: str, *, signature=None, output_codec=None) -> PipelineState:
    """Execute a validated pipeline over one completion, purely.

    Args:
        spec: The pipeline data (`{"kind": "pipeline", "steps": [...]}`).
        completion: The raw completion text.
        signature: Required by `fields_from_object`/`fields_from_groups`
            (they coerce through field schemas).
        output_codec: The value codec field coercion routes through;
            defaults to the shared text codec.

    Returns:
        The final `PipelineState`: `fields` when a fields terminal ran,
        `value` for value pipelines, `spans` naming the consumed regions of
        the original completion.

    Raises:
        AdapterParseError: When a step cannot parse its input.
    """
    if output_codec is None:
        from dspy.adapters.codecs import TEXT_PYTHONISH

        output_codec = TEXT_PYTHONISH
    state = PipelineState(text=completion, value=completion)
    for step in spec["steps"]:
        state = _run_step(step, state, signature, output_codec)
    return state


def _run_step(step: dict, state: PipelineState, signature, output_codec) -> PipelineState:
    op = step["op"]
    if op == "fenced_block":
        return _run_fenced_block(step, state)
    if op == "alternatives":
        failures = []
        for branch in step["try"]:
            try:
                return _run_step(branch, deepcopy(state), signature, output_codec)
            except AdapterParseError as error:
                failures.append(f"{branch['op']}: {error}")
        raise AdapterParseError("every alternative failed — " + "; ".join(failures))
    if op == "json_object":
        return _run_json_object(step, state)
    if op == "fields_from_object":
        return _run_fields_from_object(step, state, signature, output_codec)
    if op == "regex":
        return _run_regex(step, state)
    if op == "fields_from_groups":
        return _run_fields_from_groups(state, signature, output_codec)
    if op == "coerce":
        state.value = coerce_shape(state.value, step["shape"])
        return state
    if op == "tool_calls":
        return _run_tool_calls(step, state)
    if op == "citations":
        return _run_citations(step, state)
    raise AdapterParseError(f"unknown combinator {op!r}")  # pragma: no cover - validation prevents


def _run_fenced_block(step: dict, state: PipelineState) -> PipelineState:
    language = step.get("language")
    lang_pattern = re.escape(language) if language else r"\w*"
    pattern = re.compile(rf"```(?:{lang_pattern})[ \t]*\n(.*?)```", re.DOTALL)
    match = pattern.search(state.text)
    if match is None:
        if step.get("policy", "prefer") == "require":
            raise AdapterParseError(
                f"no fenced {language or 'code'} block in the completion (fenced_block policy 'require')"
            )
        return state
    state.spans.append((state.base_offset + match.start(), state.base_offset + match.end()))
    state.base_offset += match.start(1)
    state.text = match.group(1)
    state.value = state.text
    return state


def _run_json_object(step: dict, state: PipelineState) -> PipelineState:
    text = state.text.strip()
    repair = step.get("repair", "none")
    if repair == "none":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as error:
            raise AdapterParseError(f"completion is not strict JSON: {error}") from None
    else:  # json_repair
        try:
            obj = json_repair.loads(text)
        except Exception as error:
            raise AdapterParseError(f"json_repair could not recover an object: {error}") from None
    if not isinstance(obj, dict):
        raise AdapterParseError(f"expected one JSON object, got {type(obj).__name__}")
    state.obj = obj
    state.value = obj
    return state


def _run_fields_from_object(step: dict, state: PipelineState, signature, output_codec) -> PipelineState:
    if state.obj is None:
        raise AdapterParseError("fields_from_object needs a parsed object — run json_object first")
    if signature is None:
        raise AdapterParseError("fields_from_object needs a signature to map keys against")
    unknown_keys = step.get("unknown_keys", "exhaust")
    fields: dict[str, Any] = {}
    for key, value in state.obj.items():
        if key in signature.output_fields:
            annotation = signature.output_fields[key].annotation
            try:
                fields[key] = output_codec.parse_value(value, annotation)
            except Exception as error:
                raise AdapterParseError(f"failed to parse field {key!r} with value {value!r}: {error}") from error
        elif unknown_keys == "refuse":
            raise AdapterParseError(
                f"unknown key {key!r} in the completion object — declared output fields: "
                f"{', '.join(signature.output_fields)}"
            )
        elif unknown_keys == "exhaust":
            state.exhaust[key] = value
    state.fields = fields
    return state


def _run_regex(step: dict, state: PipelineState) -> PipelineState:
    pattern = re.compile(step["pattern"])
    mode = step.get("mode", "search")
    if mode == "search":
        match = pattern.search(state.text)
        if match is None:
            raise AdapterParseError(f"pattern {step['pattern']!r} matched nothing in the completion")
        state.groups = match.groupdict()
        state.spans.append((state.base_offset + match.start(), state.base_offset + match.end()))
        state.value = match.group(0)
        return state
    # findall
    matches = list(pattern.finditer(state.text))
    state.matches = [m.groupdict() for m in matches]
    state.spans.extend((state.base_offset + m.start(), state.base_offset + m.end()) for m in matches)
    group = step.get("group")
    if group is not None:
        values = [m.groupdict().get(group, "") for m in matches]
        join = step.get("join")
        state.value = (join if join is not None else "").join(values) if values else ""
    else:
        state.value = state.matches
    return state


def _run_fields_from_groups(state: PipelineState, signature, output_codec) -> PipelineState:
    if state.groups is None:
        raise AdapterParseError("fields_from_groups needs named groups — run a regex with mode 'search' first")
    if signature is None:
        raise AdapterParseError("fields_from_groups needs a signature to coerce against")
    fields: dict[str, Any] = {}
    for name, value in state.groups.items():
        if name in signature.output_fields:
            annotation = signature.output_fields[name].annotation
            try:
                fields[name] = output_codec.parse_value(value, annotation)
            except Exception as error:
                raise AdapterParseError(f"failed to parse field {name!r} with value {value!r}: {error}") from error
        else:
            state.exhaust[name] = value
    state.fields = fields
    return state


def _run_tool_calls(step: dict, state: PipelineState) -> PipelineState:
    if state.matches is None:
        raise AdapterParseError("tool_calls needs findall matches — run a regex with mode 'findall' first")
    calls = []
    for match in state.matches:
        name = match.get(step["name_group"])
        raw_args = match.get(step["args_group"], "")
        if name is None:
            raise AdapterParseError(f"tool_calls: match has no group {step['name_group']!r}")
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = json_repair.loads(raw_args)
        if not isinstance(args, dict):
            raise AdapterParseError(f"tool call {name!r}: arguments {raw_args!r} are not a JSON object")
        calls.append({"name": name, "args": args})
    state.value = coerce_shape({"tool_calls": calls}, "ToolCalls")
    return state


def _run_citations(step: dict, state: PipelineState) -> PipelineState:
    if state.matches is None:
        raise AdapterParseError("citations needs findall matches — run a regex with mode 'findall' first")
    items = []
    for match in state.matches:
        span = match.get(step["span_group"])
        doc = match.get(step["doc_group"])
        if span is None or doc is None:
            raise AdapterParseError(f"citations: match lacks group {step['span_group']!r} or {step['doc_group']!r}")
        items.append({"span": span.strip(), "doc": int(doc)})
    state.value = coerce_shape(items, "Citations")
    return state


def remove_spans(text: str, spans: list[tuple[int, int]]) -> str:
    """Remove the given spans from `text` (the `consume` mechanic)."""
    if not spans:
        return text
    keep: list[str] = []
    cursor = 0
    for start, end in sorted(spans):
        start = max(start, cursor)
        keep.append(text[cursor:start])
        cursor = max(cursor, end)
    keep.append(text[cursor:])
    return "".join(keep)
