"""Public registration for adapter-layer extension points (spec section 9).

Third parties extend the system at the pools — codecs, strategies, presets
— and every registration carries an admission gate, run eagerly at the
door:

- ``register_codec``: the round-trip probe battery (ADP-003) — a codec
  failing ``parse(render(x)) == x`` on adversarial probes (nesting,
  unicode, empty collections, None-bearing shapes) is refused before it
  touches any program.
- ``register_strategy``: capability self-declaration plus the carried
  parser (L9) are verified mechanically; the strategy binds by semantic
  role or by annotation type — the double-key registry's two doors.
- ``register_preset``: the template parses eagerly with the language's
  teaching errors, codec refs link, and strategy bindings validate exactly
  as adapter constructors validate them.

Builtin names (codecs ``text_pythonish``/``pydantic_json``/``baml``,
presets ``chat``/``json``/``xml``) can be neither shadowed nor removed —
overwriting one would silently change every adapter that resolves it.
"""

from typing import Any


def describe_template_language() -> dict:
    """The template language's entire vocabulary, as data.

    The language is closed, so everything it accepts is enumerable — slots,
    aggregate styles, loop variables and options, directive roles, parse
    modes, fragment targets. This is the same structure the validator and
    the teaching errors read, so what it describes is exactly what parses
    (adapter IR contract, section 3: discoverability).
    """
    from dspy.adapters._engine.template import describe_template_language as _describe

    return _describe()


def register_codec(name: str, codec) -> None:
    """Register a value codec under ``name``, gated by the probe battery.

    The codec must be a working render/parse/schema triple: every battery
    probe must round-trip exactly (``parse_value(render_value(x)) == x``)
    or registration refuses naming the failing probe. Re-registration
    refuses too — ``unregister_codec`` first, deliberately.

    Args:
        name: The registry name preset bindings will reference.
        codec: The codec object (`render_value`, `parse_value`,
            `render_typed_placeholder`).
    """
    from dspy.adapters._engine.admission import run_codec_battery
    from dspy.adapters._engine.codecs import BUILTIN_CODEC_NAMES, CODECS

    if not isinstance(name, str) or not name:
        raise ValueError(f"a codec registers under a non-empty string name, got {name!r}")
    if name in BUILTIN_CODEC_NAMES:
        raise ValueError(
            f"codec name {name!r} is builtin and cannot be shadowed — every adapter resolving it "
            "would silently change meaning"
        )
    if name in CODECS:
        raise ValueError(f"codec {name!r} is already registered — unregister_codec({name!r}) first")
    run_codec_battery(codec, name=name)
    CODECS[name] = codec


def unregister_codec(name: str) -> None:
    """Remove a registered codec; builtin names refuse."""
    from dspy.adapters._engine.codecs import BUILTIN_CODEC_NAMES, CODECS

    if name in BUILTIN_CODEC_NAMES:
        raise ValueError(f"codec name {name!r} is builtin and cannot be removed")
    if name not in CODECS:
        raise KeyError(f"no registered codec {name!r} — registered codecs: {', '.join(CODECS)}")
    del CODECS[name]


def register_strategy(strategy, *, role: str | None = None, annotation: type | None = None) -> None:
    """Register a field strategy under a semantic role or an annotation.

    Exactly one of ``role``/``annotation`` keys the registration (the
    double-key registry resolves role first, annotation second). The
    admission gate verifies the TypeStrategy contract, the
    ``capability_requirements`` self-declaration, and the carried
    ``parser`` (ADP-003) before anything binds.

    Args:
        strategy: The strategy object.
        role: A semantic-role name from the closed vocabulary.
        annotation: A type the strategy serves.
    """
    from dspy.adapters._engine.admission import check_strategy_registrable
    from dspy.adapters._engine.strategies import register_field_strategy, register_role_strategy

    if (role is None) == (annotation is None):
        raise ValueError(
            "register_strategy takes exactly one of role= (a semantic-role name) or annotation= "
            "(a type) — the key the double-key registry resolves the strategy under"
        )
    check_strategy_registrable(strategy)
    if role is not None:
        register_role_strategy(role, strategy)
    else:
        register_field_strategy(annotation, strategy)


def unregister_strategy(*, role: str | None = None, annotation: type | None = None) -> None:
    """Remove a registered strategy; the key falls back to built-ins."""
    from dspy.adapters._engine.strategies import unregister_field_strategy, unregister_role_strategy

    if (role is None) == (annotation is None):
        raise ValueError("unregister_strategy takes exactly one of role= or annotation=")
    if role is not None:
        unregister_role_strategy(role)
    else:
        unregister_field_strategy(annotation)


def register_preset(
    name: str,
    messages: list[dict[str, Any]],
    *,
    parser: str,
    codecs: dict[str, str] | None = None,
    strategies: dict[str, str] | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    """Register a named preset, validated exactly like a built one.

    The template parses eagerly (unknown constructs refuse with the
    language's teaching errors), codec refs link, strategy bindings
    validate against the closed vocabulary, and capacity derives — the
    same mint every preset comes from.

    Args:
        name: The preset name adapters and entries will reference.
        messages: The template as a message list.
        parser: The parser binding (``chat``/``json``/``xml``/``full_text``).
        codecs: Directional codec bindings; unbound directions use the
            shared text codec.
        strategies: Per-role strategy bindings; unbound roles are ``auto``.
        config: Extra preset config.
    """
    from dspy.adapters._engine.presets import _AUTO_STRATEGIES, BUILTIN_PRESET_NAMES, PRESETS, _make_preset
    from dspy.adapters._engine.strategies.vocabulary import check_binding_name

    if not isinstance(name, str) or not name:
        raise ValueError(f"a preset registers under a non-empty string name, got {name!r}")
    if name in BUILTIN_PRESET_NAMES:
        raise ValueError(
            f"preset name {name!r} is builtin and cannot be shadowed — every adapter resolving it "
            "would silently change meaning"
        )
    if name in PRESETS:
        raise ValueError(f"preset {name!r} is already registered — unregister_preset({name!r}) first")
    bindings = {**_AUTO_STRATEGIES, **(strategies or {})}
    for role, strategy_name in bindings.items():
        check_binding_name(role, strategy_name)
    PRESETS[name] = _make_preset(
        name=name,
        template_messages=messages,
        parser=parser,
        codecs={"input": "text_pythonish", "output": "text_pythonish", **(codecs or {})},
        strategies=bindings,
        config=dict(config or {}),
    )


def unregister_preset(name: str) -> None:
    """Remove a registered preset; builtin names refuse."""
    from dspy.adapters._engine.presets import BUILTIN_PRESET_NAMES, PRESETS

    if name in BUILTIN_PRESET_NAMES:
        raise ValueError(f"preset name {name!r} is builtin and cannot be removed")
    if name not in PRESETS:
        raise KeyError(f"no registered preset {name!r} — registered presets: {', '.join(PRESETS)}")
    del PRESETS[name]
