"""Strategy authoring helpers: rules as data, one face per helper.

A strategy rule conducts one semantic role through the exchange. Its faces
are all data (adapter-north-star): a `predicate` over declared
LM-capability facts, `hides` (rendering decisions, never semantic
deletions), `transforms` (field renames), render `fragments`
(template-language strings targeting the template's fragment slots),
`engine_controls` (request-side data), and parse `routings` (channel ->
field, or a combinator pipeline over the completion text).

These helpers only build the canonical dict spellings; validation and
resolution live in `dspy.adapters.strategies`.
"""

from typing import Any

__all__ = [
    "all_of",
    "any_of",
    "capability",
    "channel",
    "fragment",
    "negate",
    "rename",
    "rule",
    "text",
]


def rule(
    *,
    predicate: dict,
    hides: list[str] | tuple = (),
    transforms: list[dict] | None = None,
    fragments: list[dict] | tuple = (),
    engine_controls: dict | None = None,
    routings: list[dict] | tuple = (),
) -> dict:
    """One strategy rule, in the canonical face order."""
    built: dict[str, Any] = {"kind": "rule", "predicate": predicate, "hides": list(hides)}
    if transforms:
        built["transforms"] = list(transforms)
    built["fragments"] = list(fragments)
    built["engine_controls"] = dict(engine_controls or {})
    built["routings"] = list(routings)
    return built


def capability(name: str) -> dict:
    """A predicate atom: the LM declares this capability fact."""
    return {"capability": name}


def all_of(*predicates: dict) -> dict:
    """Every predicate holds."""
    return {"all": list(predicates)}


def any_of(*predicates: dict) -> dict:
    """At least one predicate holds."""
    return {"any": list(predicates)}


def negate(predicate: dict) -> dict:
    """The predicate does not hold."""
    return {"not": predicate}


def fragment(target: str, content: str) -> dict:
    """A render fragment for the template's `{fragments(target)}` slot.

    `content` is template-language text: it may use slots such as
    `{field('tools')}` and renders against the call's signature and values.
    """
    return {"target": target, "content": content}


def channel(channel_name: str, *, field: str, coerce: str = "str") -> dict:
    """A channel routing: a provider response channel fills a field."""
    return {"channel": channel_name, "field": field, "coerce": coerce}


def text(pipeline: dict, *, field: str, consume: bool = False, materialize: dict | None = None) -> dict:
    """A text routing: a combinator pipeline over the completion fills a
    field. `consume` removes the matched spans from the text the main
    parser sees."""
    built: dict[str, Any] = {"text": pipeline, "field": field, "consume": consume}
    if materialize is not None:
        built["materialize"] = materialize
    return built


def rename(from_field: str, to_field: str) -> dict:
    """A transform face entry: rename a field for this exchange."""
    return {"rename": {"from": from_field, "to": to_field}}
