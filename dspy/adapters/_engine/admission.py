"""Admission gates for registered and shipped adapter-layer code (spec §9).

Three gates, all eager and loud:

- ``run_codec_battery`` — the schema-generated round-trip probe battery
  (ADP-003): a codec failing ``parse(render(x)) == x`` on adversarial
  probes (nesting, unicode, empty collections, None-bearing shapes) is
  refused before it touches any program.
- ``check_strategy_registrable`` — a registrable strategy self-declares its
  capability requirements and carries its parser (L9): declaration is
  verified mechanically at registration, never discovered later.
- ``materialize_codec_entry`` — the three-origin loader for codec entries
  shipped inside artifacts: ``builtin`` resolves internally, ``packaged``
  imports at the declared distribution version, ``authored`` execs baked
  source after identity verification (ADP-011). Code origins run the
  battery at link, so a shipped codec meets the same gate a registered one
  does.
"""

import hashlib
import json
from typing import Any, Optional

import pydantic


class _ProbeInner(pydantic.BaseModel):
    label: str
    values: list[int] = []


class _ProbeModel(pydantic.BaseModel):
    text: str
    note: str | None = None
    inner: _ProbeInner
    tags: list[str] = []


def _battery_probes() -> list[tuple[Any, Any]]:
    """(annotation, value) probes, generated fresh per run.

    Adversarial on purpose — unicode, empties, nesting, None inside an
    optional shape — while staying inside the round-trip territory the
    builtin codecs themselves honor (each builtin passes this battery;
    a conformance test pins that).
    """
    return [
        (str, "Héllo — Ünïcode ✓ 中文"),
        (str, ""),
        (int, 0),
        (int, -7),
        (float, 3.5),
        (bool, True),
        (list[int], [1, 2, 3]),
        (list[int], []),
        (dict[str, int], {"a": 1, "Α-α": 2}),
        (dict[str, int], {}),
        (Optional[int], 5),
        (
            _ProbeModel,
            _ProbeModel(text="MiXeD Case — ✓", note=None, inner=_ProbeInner(label="Alpha-α", values=[]), tags=[]),
        ),
        (
            list[_ProbeInner],
            [_ProbeInner(label="Nested — ✓", values=[0, -1])],
        ),
        (list[_ProbeInner], []),
    ]


def _probe_field_info(annotation):
    """A FieldInfo shaped like the ones real signatures produce."""
    from dspy.signatures.field import OutputField

    info = OutputField()
    info.annotation = annotation
    return info


def run_codec_battery(codec, *, name: str) -> None:
    """Refuse a codec that is not a working render/parse/schema triple.

    Every probe must round-trip exactly and every schema spelling must be a
    string; the refusal names the codec, the probe's annotation, and what
    came back (teaching errors, like the template language's).

    ``parse_value`` is probed with the rendered TEXT and with the DECODED
    value (D-δ): structured parse lanes (the json parser) hand codecs the
    already-decoded JSON value rather than a string, so admission certifies
    the contract the runtime actually exercises — a string-only codec is
    refused here instead of crashing on its first json-mode parse.
    """
    for surface in ("render_value", "parse_value", "render_typed_placeholder"):
        if not callable(getattr(codec, surface, None)):
            raise ValueError(
                f"codec {name!r} is not registrable: it lacks a callable {surface!r} — a codec is a "
                "render/parse/schema triple (how a value is shown, how an emission is recovered, and "
                "how the expected shape is described)"
            )
    for annotation, value in _battery_probes():
        field_info = _probe_field_info(annotation)
        try:
            rendered = codec.render_value(value, field_info)
        except Exception as error:
            raise ValueError(
                f"codec {name!r} refused at registration: render_value failed on probe "
                f"{annotation!r} = {value!r} ({type(error).__name__}: {error})"
            ) from error
        if not isinstance(rendered, str):
            raise ValueError(
                f"codec {name!r} refused at registration: render_value returned {type(rendered).__name__} "
                f"for probe {annotation!r}; a codec renders values to wire text"
            )
        try:
            recovered = codec.parse_value(rendered, annotation)
        except Exception as error:
            raise ValueError(
                f"codec {name!r} refused at registration: parse_value failed to invert its own render "
                f"on probe {annotation!r} = {value!r} ({type(error).__name__}: {error}) — every "
                "representation decision carries its parser (ADP-003)"
            ) from error
        if recovered != value:
            raise ValueError(
                f"codec {name!r} refused at registration: round trip broke on probe {annotation!r} — "
                f"rendered {rendered!r}, parsed back {recovered!r} != {value!r} (ADP-003: "
                "parse(render(x)) == x on adversarial probes gates admissibility)"
            )
        decoded = _decoded_probe_value(value)
        try:
            recovered_decoded = codec.parse_value(decoded, annotation)
        except Exception as error:
            raise ValueError(
                f"codec {name!r} refused at registration: parse_value failed on the DECODED form of "
                f"probe {annotation!r} = {decoded!r} ({type(error).__name__}: {error}) — structured "
                "parse lanes (json) hand parse_value the decoded value, not a string; a codec must "
                "accept both"
            ) from error
        if recovered_decoded != value:
            raise ValueError(
                f"codec {name!r} refused at registration: the DECODED form of probe {annotation!r} "
                f"parsed to {recovered_decoded!r} != {value!r} — structured parse lanes (json) hand "
                "parse_value the decoded value, not a string; a codec must accept both"
            )
        schema = codec.render_typed_placeholder("probe", field_info)
        if not isinstance(schema, str):
            raise ValueError(
                f"codec {name!r} refused at registration: render_typed_placeholder returned "
                f"{type(schema).__name__} for probe {annotation!r}; the schema spelling is wire text"
            )


def _decoded_probe_value(value: Any) -> Any:
    """A probe value as a structured lane would hand it to ``parse_value``:
    the JSON-decoded form (what ``json.loads`` yields for the value's
    serialization)."""
    from dspy.adapters.utils import serialize_for_json

    return json.loads(json.dumps(serialize_for_json(value)))


def check_strategy_registrable(strategy) -> None:
    """Refuse a strategy that does not declare what registration requires.

    The TypeStrategy contract (``name``, ``applies``, ``contribute``), a
    ``capability_requirements`` self-declaration (which LM capability flags
    the strategy consults; an empty tuple declares none), and the carried
    parser (``parser`` — a hook exposing ``parse(response_view, ctx)``, or
    ``None`` declared explicitly for request-only strategies).
    """
    name = getattr(strategy, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("a registrable strategy declares a non-empty string `name`")
    for method in ("applies", "contribute"):
        if not callable(getattr(strategy, method, None)):
            raise ValueError(
                f"strategy {name!r} is not registrable: it lacks a callable {method!r} "
                "(the TypeStrategy contract: applies(ctx) -> bool, contribute(ctx) -> AdapterPatch)"
            )
    capabilities = getattr(strategy, "capability_requirements", None)
    if (
        capabilities is None
        or isinstance(capabilities, str)
        or not all(isinstance(item, str) for item in capabilities)
    ):
        raise ValueError(
            f"strategy {name!r} is not registrable: declare `capability_requirements` — a tuple of the "
            "LM capability flags it consults (an empty tuple declares none). Capabilities are declared, "
            "never discovered, and the declaration is checked at plan time"
        )
    if not hasattr(strategy, "parser"):
        raise ValueError(
            f"strategy {name!r} is not registrable: declare `parser` — the hook that recovers what its "
            "representation decision moves (an object with parse(response_view, ctx)), or None for a "
            "request-only strategy. Every representation decision carries its parser (ADP-003)"
        )
    parser = strategy.parser
    if parser is not None and not callable(getattr(parser, "parse", None)):
        raise ValueError(
            f"strategy {name!r} is not registrable: its declared parser has no callable "
            "`parse(response_view, ctx)`"
        )


# ---------------------------------------------------------------------------
# The three-origin codec loader (spec section 9; D-025/D-026, ADP-011)
# ---------------------------------------------------------------------------

_MATERIALIZED: dict[str, Any] = {}


def _require_keys(entry: dict, keys: tuple, *, origin: str, name: str) -> None:
    missing = [key for key in keys if not entry.get(key)]
    if missing:
        raise ValueError(
            f"{origin} codec entry {name!r} is missing {missing} — required keys for origin "
            f"{origin!r}: {', '.join(keys)}"
        )


def materialize_codec_entry(entry: dict):
    """Load one origin-tagged codec entry into a live codec, gated.

    ``builtin`` resolves internally; ``packaged`` imports ``module:attr``
    and refuses a distribution version mismatch naming both versions;
    ``authored`` verifies the source's sha256 identity, then execs it in an
    isolated namespace. Code origins run the round-trip battery before the
    codec touches anything. Results are cached per canonical entry.
    """
    from dspy.adapters._engine.codecs import resolve_codec

    if not isinstance(entry, dict):
        raise ValueError(f"a codec entry is a dict, got {type(entry).__name__}")
    origin = entry.get("origin")
    name = entry.get("name") or "(unnamed)"
    if origin == "builtin":
        _require_keys(entry, ("name",), origin="builtin", name=name)
        return resolve_codec(entry["name"])
    if origin not in ("packaged", "authored"):
        raise ValueError(
            f"codec entry {name!r} carries unknown origin {origin!r} — valid origins: builtin, "
            "packaged, authored"
        )
    if "language" not in entry:
        raise ValueError(
            f"{origin} codec entry {name!r} carries no 'language' — malformed (D-025: packaged and "
            "authored code entries declare their language)"
        )
    if entry["language"] != "python":
        raise ValueError(
            f"{origin} codec entry {name!r} declares language {entry['language']!r}; this engine "
            "executes python only — foreign authored adapter code is refusable at the profile level "
            "and never rung-walks (D-026)"
        )

    cache_key = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
    cached = _MATERIALIZED.get(cache_key)
    if cached is not None:
        return cached

    if origin == "packaged":
        codec = _load_packaged(entry, name)
    else:
        codec = _load_authored(entry, name)
    run_codec_battery(codec, name=name)
    _MATERIALIZED[cache_key] = codec
    return codec


def _load_packaged(entry: dict, name: str):
    import importlib
    import importlib.metadata

    _require_keys(entry, ("import", "dist", "version"), origin="packaged", name=name)
    module_name, _, attr = entry["import"].partition(":")
    if not attr:
        raise ValueError(
            f"packaged codec entry {name!r}: 'import' must be 'module.path:attr', got {entry['import']!r}"
        )
    try:
        installed = importlib.metadata.version(entry["dist"])
    except importlib.metadata.PackageNotFoundError:
        raise ValueError(
            f"packaged codec entry {name!r}: distribution {entry['dist']!r} is not installed — the "
            "program's env manifest provides it"
        ) from None
    if installed != entry["version"]:
        raise ValueError(
            f"packaged codec entry {name!r}: distribution {entry['dist']!r} is installed at "
            f"{installed!r} but the entry declares {entry['version']!r} — a version mismatch refuses "
            "loudly naming both versions"
        )
    try:
        target = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as error:
        raise ValueError(
            f"packaged codec entry {name!r}: cannot import {entry['import']!r} "
            f"({type(error).__name__}: {error})"
        ) from error
    return target() if isinstance(target, type) else target


def _load_authored(entry: dict, name: str):
    _require_keys(entry, ("source", "attr", "authored_by", "identity"), origin="authored", name=name)
    identity = entry["identity"]
    digest = "sha256:" + hashlib.sha256(entry["source"].encode("utf-8")).hexdigest()
    if identity != digest:
        raise ValueError(
            f"authored codec entry {name!r}: identity mismatch — entry declares {identity!r}, baked "
            f"source hashes to {digest!r}; refusing to exec unverified source (ADP-011)"
        )
    namespace: dict[str, Any] = {}
    exec(compile(entry["source"], f"<authored codec {name}>", "exec"), namespace)
    if entry["attr"] not in namespace:
        raise ValueError(
            f"authored codec entry {name!r}: baked source defines no {entry['attr']!r} — the entry's "
            "'attr' names the codec object its source builds"
        )
    target = namespace[entry["attr"]]
    return target() if isinstance(target, type) else target
