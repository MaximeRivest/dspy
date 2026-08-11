"""The Adapter: one entry, executable. The entry IS the adapter.

An adapter is a signature-independent inference-time language for
rendering and parsing: a template (the whole-exchange form), a parser
(derived lens or declared pipeline), codec bindings (the type boundary),
and strategy bindings (per-role conduct) — all data. `format()` renders
the messages and the request-side data for one call; `parse()` reads a
completion back; `preview()` and `parse_preview()` are the same two
functions with nothing bound — pure, byte-testable, no LM anywhere.

Modules act on signatures; adapters act on LM families. The same program
must run unchanged across adapter choices — the adapter choice follows
the model, the token budget, and the inference strategy, never the
program.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from dspy.adapters._engine.template.parser import parse_message_template
from dspy.adapters._engine.template.preview import render_template_messages
from dspy.adapters.codecs import (
    CODECS,
    coerce_shape,
    encode_shape_value,
    resolve_codec,
    validate_shape_codec,
)
from dspy.adapters.errors import AdapterError, AdapterParseError, EntryError
from dspy.adapters.lens import Lens, derive_lens
from dspy.adapters.parse import remove_spans, run_pipeline, validate_pipeline
from dspy.adapters.strategies import (
    StrategyEffects,
    active_effects,
    capability_fact,
    check_binding,
    render_fragment,
)

#: The default strategy block, in the canonical role order.
DEFAULT_STRATEGIES = {
    "reasoning": "auto",
    "citations": "auto",
    "tools": "auto",
    "media": "auto",
    "history": "directive_turns",
}

#: Sentinel: derive `requires` from the entry body at dump.
DERIVED = object()


@dataclass
class AdapterCall:
    """One formatted call: the messages and the request-side data.

    The executor sends `lm(messages=call.messages, **call.request)`.
    """

    messages: list[dict[str, Any]]
    request: dict[str, Any] = field(default_factory=dict)


class _HybridMethod:
    """Callable on the class (builds a fresh instance first) or on an
    instance — `ChatAdapter.with_strategies(...)` and
    `adapter.with_strategies(...)` both work."""

    def __init__(self, fn):
        self.fn = fn
        self.__doc__ = fn.__doc__

    def __get__(self, obj, cls):
        if obj is not None:
            return lambda *args, **kwargs: self.fn(obj, *args, **kwargs)
        return lambda *args, **kwargs: self.fn(cls(), *args, **kwargs)


class Adapter:
    """An adapter entry, executable.

    Args:
        name: The entry name.
        template: The message-list template (raw data; parsed eagerly).
        parser: `{"kind": "lens", "of": "template"}` (the default when
            omitted), or `{"kind": "pipeline", "steps": [...]}`. Authored
            code parsers are refused naming their requirement.
        codecs: `{"input": name, "output": name, "per_field": {...}}`;
            both directions default to `text_pythonish`.
        strategies: Role -> name-or-rule bindings; defaults to the auto
            block.
        config: Open configuration (`engine_controls` lives here).
        requires: The declared requirement set; omitted = derived from
            the entry body at dump.
    """

    def __init__(
        self,
        *,
        name: str,
        template: list[dict[str, Any]],
        parser: dict | None = None,
        codecs: dict | None = None,
        strategies: dict | None = None,
        config: dict | None = None,
        requires: Any = DERIVED,
    ):
        if not isinstance(name, str) or not name:
            raise AdapterError(f"an adapter needs a non-empty string name, got {name!r}")
        self.name = name
        self.template_raw = deepcopy(list(template))
        self.template = parse_message_template(self.template_raw)

        self.parser_spec = deepcopy(parser) if parser is not None else {"kind": "lens", "of": "template"}
        self._lens: Lens | None = None
        self._validate_parser()

        codecs = dict(codecs or {})
        unknown = set(codecs) - {"input", "output", "per_field"}
        if unknown:
            raise EntryError(f"unknown codecs keys {sorted(unknown)} — valid keys: input, output, per_field")
        self.codec_bindings = {
            "input": codecs.get("input", "text_pythonish"),
            "output": codecs.get("output", "text_pythonish"),
        }
        self.input_codec = resolve_codec(self.codec_bindings["input"])
        self.output_codec = resolve_codec(self.codec_bindings["output"])
        self.per_field_codecs: dict[str, dict] = deepcopy(codecs.get("per_field") or {})
        for field_name, spec in self.per_field_codecs.items():
            self._validate_per_field_codec(field_name, spec)

        self.strategies = dict(strategies) if strategies is not None else dict(DEFAULT_STRATEGIES)
        for role, binding in self.strategies.items():
            check_binding(role, binding)

        self.config = deepcopy(config or {})
        self.requires = deepcopy(requires) if requires is not DERIVED and requires is not None else requires

    # ------------------------------------------------------------------
    # Construction-time validation
    # ------------------------------------------------------------------

    def _validate_parser(self) -> None:
        spec = self.parser_spec
        if isinstance(spec, str):
            raise EntryError(
                f"parser {spec!r} is the retired 0.2.0 enum spelling — the builtin parsers are lens "
                "derivations of their own templates now; write {'kind': 'lens', 'of': 'template'} "
                "or a {'kind': 'pipeline'} object"
            )
        if not isinstance(spec, dict) or "kind" not in spec:
            raise EntryError(f"a parser is a dict with a 'kind' key, got {spec!r}")
        kind = spec["kind"]
        if kind == "lens":
            unknown = set(spec) - {"kind", "of"}
            if unknown:
                raise EntryError(f"unknown lens-parser keys {sorted(unknown)} — valid keys: kind, of")
            if spec.get("of", "template") != "template":
                raise EntryError(f"a lens derives 'of' the template — got of={spec.get('of')!r}")
            self._lens = derive_lens(self.template)
            return
        if kind == "pipeline":
            validate_pipeline(spec, where=f"adapter {self.name!r} parser")
            has_terminal = any(
                step.get("op") in ("fields_from_object", "fields_from_groups")
                or (
                    step.get("op") == "alternatives"
                    and any(b.get("op") in ("fields_from_object", "fields_from_groups") for b in step["try"])
                )
                for step in spec["steps"]
            )
            if not has_terminal:
                raise EntryError(
                    f"adapter {self.name!r} parser: an entry-level pipeline ends in a fields terminal "
                    "(fields_from_object or fields_from_groups) — value pipelines belong in routings"
                )
            return
        if kind == "authored":
            language = spec.get("language", {})
            requirement = f"{language.get('name', 'code')}{language.get('requires', '')}"
            raise EntryError(
                f"requires {requirement} sidecar for `4_adapter/{self.name}/parser` — unbound; "
                "refuse or bind one (authored code parsers walk the trust ladder; this engine "
                "ships the data levels only)"
            )
        raise EntryError(f"unknown parser kind {kind!r} — valid kinds: lens, pipeline, authored")

    def _validate_per_field_codec(self, field_name: str, spec: Any) -> None:
        where = f"codecs.per_field.{field_name}"
        if not isinstance(spec, dict) or "kind" not in spec:
            raise EntryError(f"{where}: a per-field codec is a dict with a 'kind' key, got {spec!r}")
        kind = spec["kind"]
        if kind == "family":
            unknown = set(spec) - {"kind", "family", "options", "parse_chain"}
            if unknown:
                raise EntryError(f"{where}: unknown family-codec keys {sorted(unknown)}")
            family = spec.get("family")
            if family not in CODECS:
                raise EntryError(f"{where}: unknown codec family {family!r} — registered families: {', '.join(CODECS)}")
            return
        if kind == "shape":
            validate_shape_codec(spec, where=where)
            return
        if kind == "leaf":
            language = spec.get("language", {})
            requirement = f"{language.get('name', 'code')}{language.get('requires', '')}"
            raise EntryError(
                f"requires {requirement} sidecar for `4_adapter/{self.name}/codecs/{field_name}` — "
                "unbound; refuse or bind one (leaf codecs reference declared compute; this engine "
                "ships the data levels only)"
            )
        raise EntryError(f"{where}: unknown per-field codec kind {kind!r} — valid kinds: family, shape, leaf")

    # ------------------------------------------------------------------
    # The exchange, forward: format / preview
    # ------------------------------------------------------------------

    def format(
        self,
        signature,
        inputs: dict[str, Any] | None = None,
        demos: list[dict[str, Any]] | tuple = (),
        *,
        lm=None,
        capabilities=None,
    ) -> AdapterCall:
        """Render one call: messages plus request-side data. Pure.

        Args:
            signature: The call's signature.
            inputs: The call's input values.
            demos: Few-shot demos (dicts of field values).
            lm: An LM whose declared capabilities the strategies read.
            capabilities: Capability facts directly (overrides `lm`).

        Returns:
            An `AdapterCall`; the executor sends
            `lm(messages=call.messages, **call.request)`.
        """
        caps = self._capabilities(lm, capabilities)
        self._check_requires(caps)
        inputs = dict(inputs or {})
        effects = active_effects(self.strategies, signature, caps)

        rendered_fragments = {
            target: [render_fragment(content, signature, inputs, self.input_codec) for content in contents]
            for target, contents in effects.fragments.items()
        }

        exchange_signature = self._exchange_signature(signature, effects)
        exchange_inputs = {k: v for k, v in inputs.items() if k in exchange_signature.input_fields}
        exchange_inputs = self._encode_media(exchange_signature, exchange_inputs)

        messages = render_template_messages(
            self.template,
            exchange_signature,
            demos=demos,
            inputs=exchange_inputs,
            input_codec=self.input_codec,
            output_codec=self.output_codec,
            fragments=rendered_fragments,
        )
        messages = self._split_media_parts(messages)

        request = self._build_request(effects, signature, inputs)
        return AdapterCall(messages=messages, request=request)

    def preview(
        self,
        signature,
        inputs: dict[str, Any] | None = None,
        demos: list[dict[str, Any]] | tuple = (),
        *,
        capabilities=None,
    ) -> list[dict[str, Any]]:
        """The rendered messages, no LM call — `format()`'s message list."""
        return self.format(signature, inputs, demos, capabilities=capabilities).messages

    # ------------------------------------------------------------------
    # The exchange, backward: parse / parse_preview
    # ------------------------------------------------------------------

    def parse(
        self,
        signature,
        completion: str,
        *,
        channels: dict[str, Any] | None = None,
        lm=None,
        capabilities=None,
    ) -> dict[str, Any]:
        """Read one completion back into the signature's output fields.

        Args:
            signature: The call's signature.
            completion: The completion text.
            channels: Provider response channels (`reasoning`,
                `tool_calls`, `citations`, ...) for channel routings; a
                routed channel that did not fire yields None.
            lm: An LM whose declared capabilities the strategies read.
            capabilities: Capability facts directly (overrides `lm`).

        Raises:
            AdapterParseError: When the completion does not carry every
                output field.
        """
        caps = self._capabilities(lm, capabilities)
        effects = active_effects(self.strategies, signature, caps)
        channels = channels or {}

        routed: dict[str, Any] = {}
        text = completion
        for routing in effects.text_routings:
            state = run_pipeline(routing["text"], text, signature=signature, output_codec=self.output_codec)
            value = state.value
            if "coerce" in routing:
                value = coerce_shape(value, routing["coerce"])
            routed[routing["field"]] = value
            if routing.get("consume"):
                text = remove_spans(text, state.spans)

        for routing in effects.channel_routings:
            value = channels.get(routing["channel"])
            if value is None:
                routed[routing["field"]] = None
            else:
                routed[routing["field"]] = coerce_shape(value, routing.get("coerce", "str"))

        parse_signature = self._exchange_signature(signature, effects)
        fields = self._run_main_parser(parse_signature, text)
        for source, target in effects.renames:
            if target in fields:
                fields[source] = fields.pop(target)
        fields.update(routed)

        missing = [name for name in signature.output_fields if name not in fields]
        if missing:
            raise AdapterParseError(
                f"the completion did not carry output field(s) {', '.join(missing)} — parsed: "
                f"{sorted(fields)}; completion: {completion!r}"
            )
        return {name: fields[name] for name in signature.output_fields}

    def parse_preview(
        self,
        signature,
        completion: str,
        *,
        channels: dict[str, Any] | None = None,
        capabilities=None,
    ) -> dict[str, Any]:
        """The parse dual of `preview()`: pure, byte-testable, no LM."""
        return self.parse(signature, completion, channels=channels, capabilities=capabilities)

    def _run_main_parser(self, signature, text: str) -> dict[str, Any]:
        if self._lens is not None:
            return self._lens.parse(signature, text, codec=self.output_codec)
        state = run_pipeline(self.parser_spec, text, signature=signature, output_codec=self.output_codec)
        return dict(state.fields or {})

    # ------------------------------------------------------------------
    # Derived views
    # ------------------------------------------------------------------

    def lens(self) -> dict[str, Any] | None:
        """The derived lens as data (None for pipeline parsers)."""
        return self._lens.describe() if self._lens is not None else None

    def explain_strategies(self, signature, *, lm=None, capabilities=None) -> dict[str, str]:
        """How each bound role resolved against these capability facts."""
        caps = self._capabilities(lm, capabilities)
        return dict(active_effects(self.strategies, signature, caps).resolutions)

    def dump_entry(self, *, for_signature=None) -> dict[str, Any]:
        """The adapter as its canonical entry dict (see `serde`)."""
        from dspy.adapters.serde import build_entry

        return build_entry(self, for_signature=for_signature)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _capabilities(self, lm, capabilities):
        if capabilities is not None:
            return capabilities
        if lm is not None:
            return lm.capabilities
        return None

    def _check_requires(self, caps) -> None:
        stated = self.requires if isinstance(self.requires, dict) else None
        for name in (stated or {}).get("lm_capabilities", []):
            if not capability_fact(caps, name):
                raise AdapterError(
                    f"adapter {self.name!r} requires LM capability {name!r} — the bound LM does not "
                    "declare it; refuse or bind an LM that does"
                )

    def _exchange_signature(self, signature, effects: StrategyEffects):
        adapted = signature
        for source, target in effects.renames:
            adapted = _rename_field(adapted, source, target)
        for name in effects.hides:
            if name in adapted.fields:
                adapted = adapted.delete(name)
        return adapted

    def _media_fields(self, signature) -> list[str]:
        from dspy.adapters.strategies import fields_with_role

        shape_fields = [
            name
            for name, spec in self.per_field_codecs.items()
            if spec.get("kind") == "shape" and name in signature.input_fields
        ]
        role_fields = [name for name in fields_with_role(signature, "media") if name in signature.input_fields]
        return list(dict.fromkeys([*shape_fields, *role_fields]))

    def _encode_media(self, signature, inputs: dict[str, Any]) -> dict[str, Any]:
        media = self._media_fields(signature)
        if not media:
            return inputs
        encoded = dict(inputs)
        for name in media:
            if name not in encoded:
                continue
            spec = self.per_field_codecs.get(name) or {"kind": "shape", "shape": "image"}
            encoded[name] = encode_shape_value(spec, encoded[name], field_name=name)
        return encoded

    def _split_media_parts(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        from dspy.adapters.types.base_type import (
            CUSTOM_TYPE_START_IDENTIFIER,
            split_message_content_for_custom_types,
        )

        if any(
            isinstance(message.get("content"), str) and CUSTOM_TYPE_START_IDENTIFIER in message["content"]
            for message in messages
        ):
            return split_message_content_for_custom_types(messages)
        return messages

    def _build_request(self, effects: StrategyEffects, signature, inputs: dict[str, Any]) -> dict[str, Any]:
        request: dict[str, Any] = {}
        controls = dict(self.config.get("engine_controls") or {})
        controls.update(effects.engine_controls)
        for key, value in controls.items():
            if key == "stop_sequences":
                request["stop"] = list(value)
            else:
                request[key] = deepcopy(value)
        patch = _splice_from(deepcopy(effects.request_patch), signature, inputs, self.name)
        request.update(patch)
        return request

    # ------------------------------------------------------------------
    # Derivation copies: the with_* doors
    # ------------------------------------------------------------------

    def _copy(self, **overrides) -> "Adapter":
        kwargs = {
            "name": self.name,
            "template": self.template_raw,
            "parser": self.parser_spec,
            "codecs": {
                **self.codec_bindings,
                **({"per_field": self.per_field_codecs} if self.per_field_codecs else {}),
            },
            "strategies": self.strategies,
            "config": self.config,
            "requires": self.requires,
        }
        kwargs.update(overrides)
        return Adapter(**kwargs)

    @_HybridMethod
    def with_parser(self, parser: dict) -> "Adapter":
        """This adapter with a different parser (a pipeline, usually)."""
        return self._copy(parser=parser)

    @_HybridMethod
    def with_strategies(self, **bindings) -> "Adapter":
        """This adapter with role bindings replaced (name or rule object)."""
        merged = dict(self.strategies)
        merged.update(bindings)
        return self._copy(strategies=merged)

    @_HybridMethod
    def with_codecs(self, *, input=None, output=None, per_field=None) -> "Adapter":
        """This adapter with codec bindings replaced or extended."""
        codecs: dict[str, Any] = dict(self.codec_bindings)
        if input is not None:
            codecs["input"] = input
        if output is not None:
            codecs["output"] = output
        merged_per_field = dict(self.per_field_codecs)
        if per_field:
            merged_per_field.update(per_field)
        if merged_per_field:
            codecs["per_field"] = merged_per_field
        return self._copy(codecs=codecs)

    def __repr__(self) -> str:
        parser = self.parser_spec.get("kind")
        return f"{type(self).__name__}(name={self.name!r}, parser={parser!r})"


def _rename_field(signature, source: str, target: str):
    """A signature with one field renamed, in place (order preserved)."""
    if source not in signature.fields:
        raise AdapterError(
            f"cannot rename field {source!r}: this signature declares no such field "
            f"(fields: {', '.join(signature.fields) or 'none'})"
        )
    info = signature.fields[source]
    if source in signature.input_fields:
        index = list(signature.input_fields).index(source)
    else:
        index = list(signature.output_fields).index(source)
    return signature.delete(source).insert(index, target, info)


def _splice_from(patch: dict, signature, inputs: dict[str, Any], adapter_name: str) -> dict:
    """Resolve `{"$from": "field:name"}` splices against the call's values."""

    def resolve(value):
        if isinstance(value, dict):
            if set(value) == {"$from"}:
                ref = value["$from"]
                if not (isinstance(ref, str) and ref.startswith("field:")):
                    raise EntryError(f"unknown $from reference {ref!r} — the one form is 'field:<name>'")
                name = ref[len("field:") :]
                if name not in signature.fields:
                    raise AdapterError(
                        f"adapter {adapter_name!r}: request patch splices field {name!r}, but this "
                        "signature declares no such field"
                    )
                if name not in inputs:
                    raise AdapterError(
                        f"adapter {adapter_name!r}: request patch splices field {name!r}, but the call "
                        "provides no value for it"
                    )
                return _lower_field_value(inputs[name])
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    return resolve(patch)


def _lower_field_value(value: Any) -> Any:
    """Lower a spliced field value to request-side data (tools become
    provider function schemas)."""
    from dspy.adapters.types.tool import Tool

    if isinstance(value, Tool):
        return [value.format_as_litellm_function_call()]
    if isinstance(value, list) and value and all(isinstance(item, Tool) for item in value):
        return [item.format_as_litellm_function_call() for item in value]
    from dspy.adapters.utils import serialize_for_json

    return serialize_for_json(value)
