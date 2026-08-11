"""The lens: a parser derived mechanically from the bound template.

`{"kind": "lens", "of": "template"}` is the level-0 parser: no vocabulary
of its own, because the template already pins the layout. The derivation
reads the template's own output-emission pattern — labels become section
boundaries, slots become captures, values decode through the bound output
codec — so `parse(render(x)) == x` holds by construction and a template
edit can never desynchronize rendering from parsing.

Where the derivation looks, in order:

1. the demos directive's authored assistant pattern (what an example
   completion looks like IS what the parser reads back);
2. an authored assistant message;
3. an outputs loop or outputs aggregate anywhere in the template (the
   base-model case: the trailing label the template renders is the
   boundary the completion continues from);
4. an inputs loop in the last user message (the exchange's labeling
   convention, generalized to outputs — the instructed-marker case);
5. nothing found: the degenerate full-text lens (one plain output field).

With one output field and no labels in the completion, every labeled lens
degenerates to full_text: everything after the last label.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import json_repair

from dspy.adapters._engine.template.parser import (
    AggregateSlot,
    ContentMessage,
    Loop,
    LoopVarSlot,
    ParsedTemplate,
    Text,
)
from dspy.adapters.errors import AdapterError, AdapterParseError

#: Loop attributes the derivation can invert (static per field).
_INVERTIBLE_ATTRS = {"name", "marker"}


@dataclass(frozen=True)
class Lens:
    """A derived parser: the template read backwards.

    Attributes:
        mode: `labeled`, `json_object`, or `full_text`.
        boundary: The boundary regex (labeled mode), with a `name` group
            when the label carries the field name.
        named: Whether boundaries carry the field name; unnamed boundaries
            assign sections to output fields positionally.
        suffix: A per-field literal cut from the end of each section
            (`<NAME>` marks the field-name hole), or None.
        source: Where the derivation found the pattern (for explain).
    """

    mode: str
    boundary: str | None = None
    named: bool = True
    suffix: str | None = None
    source: str = "template"

    def describe(self) -> dict[str, Any]:
        """The lens as data, for explain and cross-language readers."""
        return {
            "mode": self.mode,
            "boundary": self.boundary,
            "named": self.named,
            "suffix": self.suffix,
            "source": self.source,
        }

    def parse(self, signature, completion: str, *, codec) -> dict[str, Any]:
        """Read one completion back through the lens.

        Returns the fields it found; completeness is the caller's check
        (the adapter knows which fields strategies routed elsewhere).
        """
        if self.mode == "json_object":
            return _parse_json_object(signature, completion, codec)
        if self.mode == "labeled":
            return _parse_labeled(self, signature, completion, codec)
        return _parse_full_text(signature, completion, codec)


class LensError(AdapterError):
    """A template whose lens derivation is ambiguous or impossible."""


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def derive_lens(template: ParsedTemplate) -> Lens:
    """Derive the parser from a parsed template, mechanically.

    Raises:
        LensError: When the output pattern uses a loop attribute the lens
            cannot invert (a weak lens is refused, never silent).
    """
    for nodes, source in _pattern_sources(template):
        aggregate = _find_outputs_json_aggregate(nodes)
        if aggregate is not None:
            return Lens(mode="json_object", source=source)
        loop = _find_loop(nodes, "outputs")
        if loop is not None:
            return _lens_from_loop(loop, source)
    # The labeling-convention fallback: the last user message's inputs loop.
    convention = _inputs_convention(template)
    if convention is not None:
        return _lens_from_loop(convention, "user.inputs_loop")
    return Lens(mode="full_text", source="degenerate")


def _pattern_sources(template: ParsedTemplate):
    demos = template.demos_directive
    if demos is not None and demos.assistant is not None:
        yield demos.assistant, "demos.assistant"
    for message in template.messages:
        if isinstance(message, ContentMessage) and message.role == "assistant":
            yield message.nodes, "assistant"
    for message in reversed(template.messages):
        if isinstance(message, ContentMessage):
            yield message.nodes, f"{message.role}"


def _inputs_convention(template: ParsedTemplate):
    for message in reversed(template.messages):
        if isinstance(message, ContentMessage) and message.role == "user":
            return _find_loop(message.nodes, "inputs")
    return None


def _find_loop(nodes, collection: str):
    for node in nodes or ():
        if isinstance(node, Loop) and node.collection == collection:
            return node
    return None


def _find_outputs_json_aggregate(nodes):
    for node in nodes or ():
        if isinstance(node, AggregateSlot) and node.kind == "outputs" and node.style == "json_object":
            return node
    return None


def _lens_from_loop(loop: Loop, source: str) -> Lens:
    prefix_nodes: list = []
    suffix_nodes: list = []
    seen_value = False
    for node in loop.body:
        if isinstance(node, LoopVarSlot) and node.attr == "value":
            if seen_value:
                raise LensError(
                    "the lens cannot invert an output pattern with two {f.value} slots — author a parser instead"
                )
            seen_value = True
            continue
        (suffix_nodes if seen_value else prefix_nodes).append(node)

    prefix_parts = _pattern_parts(prefix_nodes, where="output-pattern prefix")
    suffix_parts = _pattern_parts(suffix_nodes, where="output-pattern suffix")

    boundary, named = _boundary_regex(prefix_parts)
    if boundary is None:
        # A bare {f.value} body (the historical full_text assistant shape).
        return Lens(mode="full_text", source=source)
    suffix = _literal_template(suffix_parts) or None
    return Lens(mode="labeled", boundary=boundary, named=named, suffix=suffix, source=source)


def _pattern_parts(nodes, *, where: str) -> list[tuple[str, str]]:
    """Nodes -> [("lit", text) | ("name", "")] parts, refusing the
    non-invertible."""
    parts: list[tuple[str, str]] = []
    for node in nodes:
        if isinstance(node, Text):
            parts.append(("lit", node.text))
        elif isinstance(node, LoopVarSlot) and node.attr in _INVERTIBLE_ATTRS:
            if node.attr == "marker":
                parts.append(("lit", "[[ ## "))
                parts.append(("name", ""))
                parts.append(("lit", " ## ]]"))
            else:
                parts.append(("name", ""))
        else:
            label = getattr(node, "attr", None) or type(node).__name__
            raise LensError(
                f"the lens cannot invert {{f.{label}}} in the {where} — a parse boundary derives from "
                "literal text and {f.name}/{f.marker} only; spell the pattern with those, or author a "
                "pipeline parser"
            )
    return parts


def _boundary_regex(parts: list[tuple[str, str]]) -> tuple[str | None, bool]:
    """Escape the prefix parts into a line-anchored boundary regex."""
    # Trim trailing whitespace: the gap between a label and its value
    # belongs to neither.
    while parts and parts[-1][0] == "lit" and not parts[-1][1].strip():
        parts = parts[:-1]
    if parts and parts[-1][0] == "lit":
        parts = parts[:-1] + [("lit", parts[-1][1].rstrip())]
    if not parts:
        return None, False
    named = any(kind == "name" for kind, _ in parts)
    pieces: list[str] = ["(?m)^[ \\t]*"]
    seen_name = False
    for kind, text in parts:
        if kind == "lit":
            pieces.append(re.escape(text))
        elif not seen_name:
            pieces.append(r"(?P<name>\w+)")
            seen_name = True
        else:
            pieces.append(r"(?P=name)")
    return "".join(pieces), named


def _literal_template(parts: list[tuple[str, str]]) -> str:
    """Suffix parts as a literal template with `<NAME>` holes."""
    out: list[str] = []
    for kind, text in parts:
        out.append(text if kind == "lit" else "<NAME>")
    return "".join(out).strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_labeled(lens: Lens, signature, completion: str, codec) -> dict[str, Any]:
    boundary = re.compile(lens.boundary)
    matches = list(boundary.finditer(completion))
    output_names = list(signature.output_fields)

    sections: dict[str, str] = {}
    positional = 0
    for index, match in enumerate(matches):
        if lens.named:
            name = match.group("name")
        else:
            if positional >= len(output_names):
                break
            name = output_names[positional]
            positional += 1
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(completion)
        content = completion[match.end() : content_end]
        content = _cut_suffix(content, lens.suffix, name)
        if name in signature.output_fields and name not in sections:
            sections[name] = content.strip()

    if not sections and len(output_names) == 1:
        # The degenerate rule: one output field, no labels — everything
        # after the last label is the value (here: the whole completion).
        return _parse_full_text(signature, completion, codec)

    fields: dict[str, Any] = {}
    for name, raw in sections.items():
        annotation = signature.output_fields[name].annotation
        try:
            fields[name] = codec.parse_value(raw, annotation)
        except Exception as error:
            raise AdapterParseError(
                f"Failed to parse field {name} with value {raw} from the LM response. Error message: {error}"
            ) from error
    return fields


def _cut_suffix(content: str, suffix: str | None, name: str) -> str:
    if not suffix:
        return content
    literal = suffix.replace("<NAME>", name)
    cut = content.rstrip().rfind(literal)
    if cut != -1:
        return content[:cut]
    return content


def _parse_json_object(signature, completion: str, codec) -> dict[str, Any]:
    """The json lens: the pinned object-reading of a json_object template.

    Prefer a fenced json block; strict parse, then the repair policy; keys
    map to output fields; unknown keys are exhaust (ADP-004).
    """
    text = completion
    fence = re.search(r"```(?:json)?[ \t]*\n(.*?)```", text, re.DOTALL)
    if fence is not None:
        text = fence.group(1)
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        obj = json_repair.loads(text)
    if not isinstance(obj, dict):
        raise AdapterParseError(
            f"expected the completion to carry one JSON object with the output fields, got: {completion!r}"
        )
    fields: dict[str, Any] = {}
    for key, value in obj.items():
        if key not in signature.output_fields:
            continue  # exhaust
        annotation = signature.output_fields[key].annotation
        try:
            fields[key] = codec.parse_value(value, annotation)
        except Exception as error:
            raise AdapterParseError(
                f"Failed to parse field {key} with value {value!r} from the LM response. Error message: {error}"
            ) from error
    return fields


def _parse_full_text(signature, completion: str, codec) -> dict[str, Any]:
    names = list(signature.output_fields)
    if len(names) != 1:
        raise AdapterParseError(
            f"the full-text lens maps the whole completion into exactly one output field, but this "
            f"signature declares {len(names)} ({', '.join(names) or 'none'}) — give the template a "
            "labeled output pattern, or author a pipeline parser"
        )
    name = names[0]
    annotation = signature.output_fields[name].annotation
    try:
        return {name: codec.parse_value(completion.strip(), annotation)}
    except Exception as error:
        raise AdapterParseError(f"Failed to parse field {name} from the LM response. Error message: {error}") from error
