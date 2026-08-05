"""Codec seam tests: round-trip properties and binding wiring.

The corpus is the byte-level integration gate; these tests pin the codec
contract itself — that `parse_value(render_value(x)) == x` holds for the
value shapes the seam claims to carry, including adversarial ones, and
that the format bindings resolve to the documented defaults.
"""

import enum

import pydantic
import pytest
from pydantic.fields import FieldInfo

from dspy.adapters._engine.codecs import PYDANTIC_JSON, TEXT_PYTHONISH
from dspy.adapters._engine.formats.baml import BAMLFormat
from dspy.adapters._engine.formats.chat import ChatFormat
from dspy.adapters._engine.formats.json import JSONFormat
from dspy.adapters._engine.formats.xml import XMLFormat


class Inner(pydantic.BaseModel):
    label: str
    weight: float


class Outer(pydantic.BaseModel):
    name: str
    inner: Inner
    tags: list[str]


class Priority(enum.Enum):
    LOW = "low"
    HIGH = "high"


def _field(annotation):
    fi = FieldInfo(annotation=annotation)
    fi.json_schema_extra = {"desc": "", "__dspy_field_type": "output", "prefix": ""}
    return fi


ROUND_TRIP_VALUES = [
    ("int", int, 42),
    ("float", float, 0.25),
    ("bool", bool, True),
    ("dict-int", dict[str, int], {"retries": 3, "depth": 2}),
    ("dict-empty", dict[str, int], {}),
    ("list-str-empty", list[str], []),
    ("list-nested", list[list[int]], [[1, 2], [], [3]]),
    ("model", Outer, Outer(name="x", inner=Inner(label="a", weight=0.5), tags=["t"])),
    ("model-list", list[Inner], [Inner(label="a", weight=0.1), Inner(label="b", weight=0.9)]),
    ("optional-none", int | None, None),
    ("optional-some", int | None, 7),
    ("enum", Priority, Priority.HIGH),
    ("unicode", str, "cœur brisé 💔 «naïve» ✓"),
    ("str-multiline", str, "line one\nline two\n  indented"),
    ("str-empty", str, ""),
    ("str-jsonish", str, '{"looks": "like json"}'),
]


@pytest.mark.parametrize("codec", [TEXT_PYTHONISH, PYDANTIC_JSON], ids=lambda c: c.name)
@pytest.mark.parametrize("annotation,value", [(a, v) for _, a, v in ROUND_TRIP_VALUES], ids=[i for i, _, _ in ROUND_TRIP_VALUES])
def test_round_trip(codec, annotation, value):
    rendered = codec.render_value(value, _field(annotation))
    assert isinstance(rendered, str)
    assert codec.parse_value(rendered, annotation) == value


def test_directional_independence():
    """BAML overrides only the input side; parsing is the shared chain."""
    model = Outer(name="x", inner=Inner(label="a", weight=0.5), tags=["t"])
    assert PYDANTIC_JSON.render_value(model, _field(Outer)) == model.model_dump_json(indent=2, by_alias=True)
    assert TEXT_PYTHONISH.render_value(model, _field(Outer)) != PYDANTIC_JSON.render_value(model, _field(Outer))
    rendered = PYDANTIC_JSON.render_value(model, _field(Outer))
    assert PYDANTIC_JSON.parse_value(rendered, Outer) == model
    # Non-model values fall back to the shared syntax.
    assert PYDANTIC_JSON.render_value({"k": 1}, _field(dict[str, int])) == TEXT_PYTHONISH.render_value(
        {"k": 1}, _field(dict[str, int])
    )


def test_format_bindings():
    assert ChatFormat().input_codec is TEXT_PYTHONISH
    assert ChatFormat().output_codec is TEXT_PYTHONISH
    assert XMLFormat().input_codec is TEXT_PYTHONISH
    assert JSONFormat().input_codec is TEXT_PYTHONISH
    assert BAMLFormat().input_codec is PYDANTIC_JSON
    assert BAMLFormat().output_codec is TEXT_PYTHONISH


def test_known_non_round_trips_documented():
    """Pinned quirks, not bugs: shapes the legacy syntax cannot round-trip.

    A list VALUE arriving in a str-annotated field renders as the guillemet
    blob (reader-friendly, not machine-reversible) — the input-side nicety
    for passing document lists to plain-text fields. A fact of the
    text-pythonish syntax the corpus already pins; a future codec may do
    better, which is the point of the seam.
    """
    rendered = TEXT_PYTHONISH.render_value(["a", "b"], _field(str))
    assert rendered == "[1] «a»\n[2] «b»"
    assert TEXT_PYTHONISH.parse_value(rendered, str) == rendered  # str parse is identity: blob stays a blob
