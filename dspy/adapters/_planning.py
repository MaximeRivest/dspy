"""Private adapter-call planning and strategy composition.

This module is the bridge between legacy adapters and the target strategy-based
architecture. It is intentionally private while the public strategy API settles,
but the internal model is already the long-term one:

```text
original signature + inputs + adapter + LM + strategies
    -> _AdapterPlan
    -> LMRequest
    -> LMResponse
    -> parsed output dictionaries
```

The important mental model is **partitioning**. Planning assigns responsibility
for each field/call concern to exactly one owner whenever possible:

* fields left in ``plan.render_signature`` are rendered and parsed by the
  adapter's ordinary text/JSON/XML grammar;
* replacement strategies remove fields from ``plan.render_signature`` and
  record how to render/parse them elsewhere;
* augment/configuration strategies add request overlays without taking field
  ownership.

The responsibilities are divided as follows:

* ``Adapter`` owns prompt grammar and ordinary text parsing;
* ``_Strategy`` owns a special representation policy for a type or protocol;
* ``_AdapterPlan`` records who owns every field and how to reconstruct outputs;
* ``LMRequestPatch`` contains actual LM request pieces, while
  ``_StrategyPatch`` also carries DSPy adapter bookkeeping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any, Generic, TypeVar, get_args, get_origin

from dspy.adapters.types import Audio, File, History, Image
from dspy.adapters.types.document import Document
from dspy.adapters.types.reasoning import Reasoning
from dspy.adapters.types.tool import Tool, ToolCalls
from dspy.clients.base_lm import BaseLM
from dspy.core.types import (
    LMAudioPart,
    LMBinaryPart,
    LMCitationPart,
    LMConfig,
    LMDocumentPart,
    LMImagePart,
    LMMessage,
    LMOutput,
    LMPart,
    LMReasoningConfig,
    LMRequestPatch,
    LMTextPart,
    LMToolCallPart,
    LMToolSpec,
)
from dspy.experimental import Citations
from dspy.signatures.signature import Signature

_STRATEGY_TEXT_OWNER = "text"


@dataclass(frozen=True)
class _UserPartSegment:
    """Native user-message parts associated with one input field.

    Strategies should not hard-code an adapter's delimiters. Instead, they emit
    semantic parts for a field (for example an ``LMImagePart`` for ``image``),
    and the adapter later decides how to introduce/place those parts with its
    own grammar via ``Adapter.place_user_part_segments()``.
    """

    field_name: str
    parts: list[LMPart]


@dataclass(frozen=True)
class _PlannedOutputParser:
    """Parser selected during planning for a strategy-owned output field.

    Output parsing is plan-driven on purpose: we use the strategy selected at
    request time, not whichever strategy happens to match the annotation after
    the response arrives. This avoids non-determinism and makes debugging
    strategy selection possible.
    """

    field_name: str
    field: Any
    strategy: _Strategy


@dataclass(frozen=True)
class _StrategyTrace:
    """Human-readable explanation of a planning decision.

    This is private today, but it is the seed of future debugging output such as
    "reasoning was handled by NativeReasoning because the LM supports native
    reasoning" or "image stayed in the text path because no native strategy was
    selected".
    """

    field_name: str | None
    strategy: str
    action: str
    reason: str = ""


@dataclass(frozen=True)
class _StrategyPatch:
    """Composable result returned by private strategies.

    ``LMRequestPatch`` only describes actual LM request pieces: messages, parts,
    tools, and config. Strategies also need adapter bookkeeping: which signature
    fields to remove from ordinary rendering, which native segments to place,
    and which parser should reconstruct deleted output fields. This wrapper
    keeps those concerns explicit instead of smuggling everything through
    metadata.

    ``keep_*_fields`` is the private form of an explicit text/no-op strategy:
    it records that a strategy deliberately left a field in the adapter text
    path and blocks later fallback replacement strategies from taking ownership.
    """
    request: LMRequestPatch = dataclass_field(default_factory=LMRequestPatch)
    delete_input_fields: tuple[str, ...] = ()
    delete_output_fields: tuple[str, ...] = ()
    keep_input_fields: tuple[str, ...] = ()
    keep_output_fields: tuple[str, ...] = ()
    user_part_segments: tuple[_UserPartSegment, ...] = ()
    output_parsers: tuple[_PlannedOutputParser, ...] = ()
    trace: tuple[_StrategyTrace, ...] = ()

    def merge(self, other: _StrategyPatch) -> _StrategyPatch:
        return _StrategyPatch(
            request=self.request.merge(other.request),
            delete_input_fields=(*self.delete_input_fields, *other.delete_input_fields),
            delete_output_fields=(*self.delete_output_fields, *other.delete_output_fields),
            keep_input_fields=(*self.keep_input_fields, *other.keep_input_fields),
            keep_output_fields=(*self.keep_output_fields, *other.keep_output_fields),
            user_part_segments=(*self.user_part_segments, *other.user_part_segments),
            output_parsers=(*self.output_parsers, *other.output_parsers),
            trace=(*self.trace, *other.trace),
        )


@dataclass
class _AdapterPlan:
    """Complete private plan for one adapter call.

    ``render_signature`` and ``inputs`` are the text-path partition consumed by
    ``adapter.format()`` and ``adapter.parse()``. The other attributes record
    strategy-owned request pieces and response reconstruction. If a field is
    deleted from ``render_signature``, maintainers should expect either a user
    part segment, a request/config overlay, an output parser, or a documented
    missing-value policy to explain where that field went.
    """

    adapter: Any
    original_signature: type[Signature]
    render_signature: type[Signature]
    inputs: dict[str, Any]
    lm_kwargs: dict[str, Any]
    request_patch: LMRequestPatch = dataclass_field(default_factory=LMRequestPatch)
    user_part_segments: list[_UserPartSegment] = dataclass_field(default_factory=list)
    output_parsers: list[_PlannedOutputParser] = dataclass_field(default_factory=list)
    strategy_trace: list[_StrategyTrace] = dataclass_field(default_factory=list)
    input_field_owners: dict[str, str] = dataclass_field(default_factory=dict)
    output_field_owners: dict[str, str] = dataclass_field(default_factory=dict)
    output_parser_owners: dict[str, str] = dataclass_field(default_factory=dict)

    @property
    def messages(self) -> list[LMMessage]:
        return self.request_patch.messages

    @property
    def tools(self) -> list[LMToolSpec]:
        return self.request_patch.tools

    def apply(self, patch: _StrategyPatch) -> None:
        """Merge one strategy decision into this plan.

        Applying a patch is where partitioning becomes concrete: deleted input
        fields are removed from ``render_signature`` and ``inputs`` so the
        adapter cannot render them a second time; deleted output fields are
        removed so adapter parsing does not expect them in text; parsers and
        native segments are saved for later request rendering/response parsing.
        """
        strategy_name = _patch_strategy_name(patch)
        self._validate_patch(patch, strategy_name)

        for field_name in patch.keep_input_fields:
            self.input_field_owners.setdefault(field_name, _STRATEGY_TEXT_OWNER)
        for field_name in patch.keep_output_fields:
            self.output_field_owners.setdefault(field_name, _STRATEGY_TEXT_OWNER)

        self.request_patch = self.request_patch.merge(patch.request)
        for field_name in (*patch.request.delete_input_fields, *patch.delete_input_fields):
            self.input_field_owners[field_name] = strategy_name
            if field_name in self.render_signature.input_fields:
                self.render_signature = self.render_signature.delete(field_name)
            self.inputs.pop(field_name, None)
        for field_name in (*patch.request.delete_output_fields, *patch.delete_output_fields):
            self.output_field_owners[field_name] = strategy_name
            if field_name in self.render_signature.output_fields:
                self.render_signature = self.render_signature.delete(field_name)
        if patch.request.config is not None:
            patch_kwargs = patch.request.as_lm_kwargs()
            patch_kwargs.pop("tools", None)
            self.lm_kwargs.update(patch_kwargs)
        self.user_part_segments.extend(patch.user_part_segments)
        for parser in patch.output_parsers:
            self.output_parser_owners[parser.field_name] = strategy_name
        self.output_parsers.extend(patch.output_parsers)
        self.strategy_trace.extend(patch.trace)

    def _validate_patch(self, patch: _StrategyPatch, strategy_name: str) -> None:
        _raise_on_config_conflict(self.request_patch.config, patch.request.config, strategy_name)
        for field_name in (*patch.request.delete_input_fields, *patch.delete_input_fields):
            owner = self.input_field_owners.get(field_name)
            if owner is not None:
                raise ValueError(f"Input field {field_name!r} is already owned by strategy {owner}; {strategy_name} cannot also replace it.")
        for field_name in (*patch.request.delete_output_fields, *patch.delete_output_fields):
            owner = self.output_field_owners.get(field_name)
            if owner is not None:
                raise ValueError(f"Output field {field_name!r} is already owned by strategy {owner}; {strategy_name} cannot also replace it.")
        for parser in patch.output_parsers:
            owner = self.output_parser_owners.get(parser.field_name)
            if owner is not None:
                raise ValueError(f"Output field {parser.field_name!r} already has parser strategy {owner}; {strategy_name} cannot add another parser.")


@dataclass(frozen=True)
class _StrategyContext:
    """Shared immutable context passed to planning strategies."""

    adapter: Any
    lm: BaseLM
    lm_kwargs: dict[str, Any]
    signature: type[Signature]
    inputs: dict[str, Any]


@dataclass(frozen=True)
class _InputFieldStrategyContext(_StrategyContext):
    field_name: str
    field: Any
    value: Any


@dataclass(frozen=True)
class _OutputFieldStrategyContext(_StrategyContext):
    field_name: str
    field: Any


@dataclass(frozen=True)
class _ParseStrategyContext:
    """Context passed back to the selected strategy during response parsing."""

    field_name: str
    field: Any
    output: LMOutput
    plan: _AdapterPlan
    lm: BaseLM


class _Strategy:
    """Private strategy protocol for call planning.

    A strategy can operate at three scopes:

    * signature-level, for multi-field protocols such as native tools;
    * input-field-level, for replacing one input with native message parts;
    * output-field-level, for deleting an output from text and parsing it from
      native response parts.

    Most methods return an empty patch. A strategy only overrides the scopes it
    owns. Public strategy classes should eventually mirror this shape, but this
    private version lets us prove the architecture without freezing API names.
    """

    def matches_input_annotation(self, annotation: Any) -> bool:
        return False

    def matches_output_annotation(self, annotation: Any) -> bool:
        return False

    def plan_signature(self, ctx: _StrategyContext) -> _StrategyPatch:
        return _StrategyPatch()

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        return _StrategyPatch()

    def plan_output_field(self, ctx: _OutputFieldStrategyContext) -> _StrategyPatch:
        return _StrategyPatch()

    def parse_output_field(self, ctx: _ParseStrategyContext) -> Any | None:
        return None


T = TypeVar("T")


@dataclass(frozen=True)
class _TypeStrategy(_Strategy, Generic[T]):
    """Strategy specialized to annotations containing one marker type.

    ``marker_type`` is the semantic DSPy type, such as ``Image`` or
    ``Reasoning``. Matching recurses into containers, so a strategy for
    ``Image`` can handle both ``Image`` and ``list[Image]`` when its planning
    method chooses to support collection values.
    """

    marker_type: type[T]

    def matches(self, annotation: Any) -> bool:
        return _annotation_contains(annotation, self.marker_type)

    def matches_input_annotation(self, annotation: Any) -> bool:
        return self.matches(annotation)

    def matches_output_annotation(self, annotation: Any) -> bool:
        return self.matches(annotation)


class _StrategyResolver:
    """Select private strategies for the current adapter.

    This is deliberately minimal. Public strategy selection will eventually need
    explicit precedence, conflict reporting, and user-visible traces. For now,
    the resolver has two jobs: prepend adapter-supplied private experimental
    strategies and append behavior-preserving built-ins for native tools,
    reasoning, and citations.
    """

    def __init__(self, adapter: Any):
        configured = list(getattr(adapter, "_type_strategies", []) or [])
        # Keep the first private strategy pass behavior-preserving. Strategies
        # that model new representations (history-as-plan messages and native
        # media input segments) are implemented below but only run when supplied
        # explicitly through the private adapter `_type_strategies` list. Existing
        # built-in defaults cover behavior that was already native in legacy
        # preprocessing: tools, reasoning, and citations.
        self.strategies: list[_Strategy] = [
            *configured,
            _NativeToolCallsStrategy(),
            _NativeReasoningStrategy(),
            _NativeCitationsStrategy(),
        ]

    def signature_strategies(self) -> list[_Strategy]:
        return self.strategies

    def input_strategies(self, annotation: Any) -> list[_Strategy]:
        return [strategy for strategy in self.strategies if strategy.matches_input_annotation(annotation)]

    def output_strategies(self, annotation: Any) -> list[_Strategy]:
        return [strategy for strategy in self.strategies if strategy.matches_output_annotation(annotation)]


@dataclass(frozen=True)
class _HistoryStrategy(_TypeStrategy[History]):
    """Replace a ``History`` input with prior user/assistant messages.

    This strategy is currently opt-in through the private ``_type_strategies``
    adapter argument because legacy adapters already format history inside
    ``Adapter.format()``. It exists to prove the future ownership model: history
    is an input marker type whose natural representation is extra messages, not
    ordinary text in the current user turn.
    """

    marker_type: type[History] = History

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        if not self.matches(ctx.field.annotation):
            return _StrategyPatch()
        history = ctx.value if isinstance(ctx.value, History) else History.model_validate(ctx.value)
        signature_without_history = ctx.signature.delete(ctx.field_name)
        messages = _history_to_lm_messages(ctx.adapter, signature_without_history, history)
        return _StrategyPatch(
            request=LMRequestPatch(messages=messages),
            delete_input_fields=(ctx.field_name,),
            trace=(_StrategyTrace(ctx.field_name, type(self).__name__, "replace", "history converted to messages"),),
        )


@dataclass(frozen=True)
class _NativeToolCallsStrategy(_TypeStrategy[ToolCalls]):
    """Signature-level strategy for provider-native tool calling.

    Tool calling is not a single-field decision: it consumes a ``Tool`` or
    ``list[Tool]`` input field and a ``ToolCalls`` output field together. The
    input becomes normalized ``LMToolSpec`` objects, the output is removed from
    adapter text parsing, and this same strategy parses returned
    ``LMToolCallPart`` objects back into ``dspy.ToolCalls``.
    """

    marker_type: type[ToolCalls] = ToolCalls

    def plan_signature(self, ctx: _StrategyContext) -> _StrategyPatch:
        adapter = ctx.adapter
        if not getattr(adapter, "use_native_function_calling", False):
            return _StrategyPatch()
        tool_call_output_field_name = _get_tool_call_output_field_name(ctx.signature)
        if tool_call_output_field_name is None:
            return _StrategyPatch()
        tool_call_input_field_name = _get_tool_call_input_field_name(ctx.signature)
        if tool_call_input_field_name is None:
            raise ValueError(
                f"You provided an output field {tool_call_output_field_name} to receive the tool calls information, "
                "but did not provide any tools as the input. Please provide a list of tools as the input by adding an "
                "input field with type `list[dspy.Tool]`."
            )
        if not ctx.lm.supports_function_calling:
            return _StrategyPatch()
        if tool_call_input_field_name not in ctx.inputs:
            return _StrategyPatch(delete_output_fields=(tool_call_output_field_name,))
        tools = ctx.inputs[tool_call_input_field_name]
        tools = tools if isinstance(tools, list) else [tools]
        parser = _PlannedOutputParser(tool_call_output_field_name, ctx.signature.output_fields[tool_call_output_field_name], self)
        return _StrategyPatch(
            request=LMRequestPatch(tools=[_tool_to_lm_tool_spec(tool) for tool in tools]),
            delete_input_fields=(tool_call_input_field_name,),
            delete_output_fields=(tool_call_output_field_name,),
            output_parsers=(parser,),
            trace=(_StrategyTrace(tool_call_output_field_name, type(self).__name__, "replace", "native tool calling"),),
        )

    def parse_output_field(self, ctx: _ParseStrategyContext) -> ToolCalls | None:
        if not ctx.output.tool_calls:
            return None
        return ToolCalls.from_dict_list([{"name": call.name, "args": call.args} for call in ctx.output.tool_calls])


@dataclass(frozen=True)
class _NativeReasoningStrategy(_TypeStrategy[Reasoning]):
    """Use an LM's native reasoning channel for ``Reasoning`` outputs.

    If the adapter allows ``Reasoning`` as a native response type and the LM
    advertises reasoning support, this removes the reasoning field from the text
    output structure, adds normalized reasoning config, and reconstructs the
    field from ``LMOutput.reasoning_content``. If any precondition fails, the
    field remains in the ordinary adapter text path.
    """

    marker_type: type[Reasoning] = Reasoning

    def plan_output_field(self, ctx: _OutputFieldStrategyContext) -> _StrategyPatch:
        if not self.matches(ctx.field.annotation) or Reasoning not in ctx.adapter.native_response_types:
            return _StrategyPatch()
        reasoning_effort = ctx.lm_kwargs.get("reasoning_effort", ctx.lm.kwargs.get("reasoning_effort", "low"))
        if reasoning_effort is None or not ctx.lm.supports_reasoning:
            return _StrategyPatch()
        if "gpt-5" in ctx.lm.model and getattr(ctx.lm, "model_type", None) == "chat":
            return _StrategyPatch()
        parser = _PlannedOutputParser(ctx.field_name, ctx.field, self)
        return _StrategyPatch(
            request=LMRequestPatch(config=LMConfig(reasoning=LMReasoningConfig(effort=reasoning_effort))),
            delete_output_fields=(ctx.field_name,),
            output_parsers=(parser,),
            trace=(_StrategyTrace(ctx.field_name, type(self).__name__, "replace", "native reasoning channel"),),
        )

    def parse_output_field(self, ctx: _ParseStrategyContext) -> Reasoning | None:
        return Reasoning(content=ctx.output.reasoning_content) if ctx.output.reasoning_content is not None else None


@dataclass(frozen=True)
class _NativeCitationsStrategy(_TypeStrategy[Citations]):
    """Parse native citation response parts into ``dspy.Citations``.

    The current native citation path is provider-specific, so this strategy only
    takes ownership for Anthropic models and when the adapter includes
    ``Citations`` in ``native_response_types``. Otherwise citations stay as an
    ordinary text/output concern.
    """

    marker_type: type[Citations] = Citations

    def plan_output_field(self, ctx: _OutputFieldStrategyContext) -> _StrategyPatch:
        if not self.matches(ctx.field.annotation) or Citations not in ctx.adapter.native_response_types:
            return _StrategyPatch()
        if not getattr(ctx.lm, "model", "").startswith("anthropic/"):
            return _StrategyPatch()
        parser = _PlannedOutputParser(ctx.field_name, ctx.field, self)
        return _StrategyPatch(
            delete_output_fields=(ctx.field_name,),
            output_parsers=(parser,),
            trace=(_StrategyTrace(ctx.field_name, type(self).__name__, "replace", "native citations"),),
        )

    def parse_output_field(self, ctx: _ParseStrategyContext) -> Citations | None:
        if not ctx.output.citations:
            return None
        return Citations.from_dict_list([_citation_part_to_dict(citation) for citation in ctx.output.citations])


@dataclass(frozen=True)
class _TextReasoningStrategy(_TypeStrategy[Reasoning]):
    """Explicitly keep ``Reasoning`` in the adapter text path.

    This private strategy is useful in internal tests and experiments where a
    caller wants to block the automatic ``_NativeReasoningStrategy`` fallback
    while keeping the field in the ordinary adapter-rendered output grammar.
    """

    marker_type: type[Reasoning] = Reasoning

    def plan_output_field(self, ctx: _OutputFieldStrategyContext) -> _StrategyPatch:
        if not self.matches(ctx.field.annotation):
            return _StrategyPatch()
        return _StrategyPatch(
            keep_output_fields=(ctx.field_name,),
            trace=(_StrategyTrace(ctx.field_name, type(self).__name__, "keep_text", "explicit text reasoning"),),
        )


@dataclass(frozen=True)
class _TextToolCallsStrategy(_TypeStrategy[ToolCalls]):
    """Explicitly keep ``ToolCalls`` in the adapter text path."""

    marker_type: type[ToolCalls] = ToolCalls

    def plan_signature(self, ctx: _StrategyContext) -> _StrategyPatch:
        field_name = _get_tool_call_output_field_name(ctx.signature)
        if field_name is None:
            return _StrategyPatch()
        return _StrategyPatch(
            keep_output_fields=(field_name,),
            trace=(_StrategyTrace(field_name, type(self).__name__, "keep_text", "explicit text tool calls"),),
        )

    def plan_output_field(self, ctx: _OutputFieldStrategyContext) -> _StrategyPatch:
        if not self.matches(ctx.field.annotation):
            return _StrategyPatch()
        return _StrategyPatch(
            keep_output_fields=(ctx.field_name,),
            trace=(_StrategyTrace(ctx.field_name, type(self).__name__, "keep_text", "explicit text tool calls"),),
        )


@dataclass(frozen=True)
class _NativeImageStrategy(_TypeStrategy[Image]):
    """Replace ``Image`` inputs with normalized ``LMImagePart`` segments.

    This is private/opt-in for now to preserve legacy exact prompt rendering.
    Once public strategies become the default, this is the shape users will
    customize for tiling, downsampling, caption-and-send, and similar policies.
    """

    marker_type: type[Image] = Image

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        parts = _parts_for_collection(ctx.field.annotation, ctx.value, Image, lambda value: _image_to_lm_part(value if isinstance(value, Image) else Image(value)))
        return _input_segment_patch(ctx, parts, self, "native image parts")


@dataclass(frozen=True)
class _NativeAudioStrategy(_TypeStrategy[Audio]):
    """Replace ``Audio`` inputs with normalized ``LMAudioPart`` segments."""

    marker_type: type[Audio] = Audio

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        def convert(value: Any) -> LMAudioPart:
            audio = value if isinstance(value, Audio) else Audio.model_validate(value)
            return LMAudioPart(data=audio.data, media_type=f"audio/{audio.audio_format}")

        parts = _parts_for_collection(ctx.field.annotation, ctx.value, Audio, convert)
        return _input_segment_patch(ctx, parts, self, "native audio parts")


@dataclass(frozen=True)
class _NativeFileStrategy(_TypeStrategy[File]):
    """Replace ``File`` inputs with normalized binary/file segments."""

    marker_type: type[File] = File

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        def convert(value: Any) -> LMBinaryPart:
            file = value if isinstance(value, File) else File.model_validate(value)
            return _file_to_lm_part(file)

        parts = _parts_for_collection(ctx.field.annotation, ctx.value, File, convert)
        return _input_segment_patch(ctx, parts, self, "native file parts")


@dataclass(frozen=True)
class _NativeDocumentStrategy(_TypeStrategy[Document]):
    """Replace citation-capable ``Document`` inputs with document parts."""

    marker_type: type[Document] = Document

    def plan_input_field(self, ctx: _InputFieldStrategyContext) -> _StrategyPatch:
        def convert(value: Any) -> LMDocumentPart:
            document = value if isinstance(value, Document) else Document.model_validate(value)
            return LMDocumentPart(
                media_type=document.media_type,
                source={"type": "text", "media_type": document.media_type, "data": document.data},
                citations={"enabled": True},
                title=document.title,
                context=document.context,
            )

        parts = _parts_for_collection(ctx.field.annotation, ctx.value, Document, convert)
        return _input_segment_patch(ctx, parts, self, "native document parts")


def _plan_adapter_call(
    adapter: Any,
    lm: BaseLM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    inputs: dict[str, Any],
) -> _AdapterPlan:
    """Build the private plan for one adapter call.

    Planning runs in three phases:

    1. signature strategies see the whole call first, so multi-field protocols
       such as native tools can consume related input/output fields together;
    2. input field strategies may replace remaining individual inputs;
    3. output field strategies may replace remaining individual outputs.

    Field-level composition allows many augment/config patches plus at most one
    replacement owner per field. Explicit text/keep strategies block later
    replacement fallbacks while still allowing non-owning augment/config patches.
    """
    plan = _AdapterPlan(
        adapter=adapter,
        original_signature=signature,
        render_signature=signature,
        inputs=dict(inputs),
        lm_kwargs=dict(lm_kwargs),
    )
    resolver = _StrategyResolver(adapter)

    base_ctx = _StrategyContext(adapter=adapter, lm=lm, lm_kwargs=plan.lm_kwargs, signature=plan.render_signature, inputs=plan.inputs)
    for strategy in resolver.signature_strategies():
        patch = strategy.plan_signature(base_ctx)
        if _patch_is_meaningful(patch):
            if _patch_replaces_text_owned_field(plan, patch):
                plan.strategy_trace.append(_StrategyTrace(None, type(strategy).__name__, "skip", "field explicitly kept in text path"))
            else:
                plan.apply(patch)
        base_ctx = _StrategyContext(adapter=adapter, lm=lm, lm_kwargs=plan.lm_kwargs, signature=plan.render_signature, inputs=plan.inputs)

    for name, field in list(plan.render_signature.input_fields.items()):
        if name not in plan.inputs:
            continue
        ctx = _InputFieldStrategyContext(
            adapter=adapter,
            lm=lm,
            lm_kwargs=plan.lm_kwargs,
            signature=plan.render_signature,
            inputs=plan.inputs,
            field_name=name,
            field=field,
            value=plan.inputs[name],
        )
        for strategy in resolver.input_strategies(field.annotation):
            patch = strategy.plan_input_field(ctx)
            if not _patch_is_meaningful(patch):
                continue
            if _patch_owns_input_field(patch, name) and plan.input_field_owners.get(name) == _STRATEGY_TEXT_OWNER:
                plan.strategy_trace.append(_StrategyTrace(name, type(strategy).__name__, "skip", "field explicitly kept in text path"))
                continue
            plan.apply(patch)

    for name, field in list(plan.render_signature.output_fields.items()):
        ctx = _OutputFieldStrategyContext(
            adapter=adapter,
            lm=lm,
            lm_kwargs=plan.lm_kwargs,
            signature=plan.render_signature,
            inputs=plan.inputs,
            field_name=name,
            field=field,
        )
        for strategy in resolver.output_strategies(field.annotation):
            patch = strategy.plan_output_field(ctx)
            if not _patch_is_meaningful(patch):
                continue
            if _patch_owns_output_field(patch, name) and plan.output_field_owners.get(name) == _STRATEGY_TEXT_OWNER:
                plan.strategy_trace.append(_StrategyTrace(name, type(strategy).__name__, "skip", "field explicitly kept in text path"))
                continue
            plan.apply(patch)

    return plan


def _apply_planned_messages(adapter: Any, messages: list[LMMessage], plan: _AdapterPlan) -> list[LMMessage]:
    """Apply planned message/part overlays after ordinary adapter rendering.

    The adapter first renders ``plan.render_signature`` using its existing
    grammar. This function then inserts strategy-owned messages and parts around
    that rendered text. Field-associated user segments are delegated back to the
    adapter so custom adapters own their own anchors and wrappers.
    """
    messages = adapter._coerce_lm_messages(messages)
    patch = plan.request_patch

    if patch.system_parts:
        _append_parts_to_role(messages, "system", patch.system_parts, at_start=True)
    if patch.messages:
        insert_at = _last_user_message_index(messages)
        if insert_at is None:
            insert_at = len(messages)
        messages[insert_at:insert_at] = adapter._coerce_lm_messages(list(patch.messages))
    if plan.user_part_segments:
        messages = adapter.place_user_part_segments(messages, plan.user_part_segments, plan)
    if patch.user_parts:
        _append_parts_to_role(messages, "user", patch.user_parts)
    if patch.assistant_parts:
        _append_parts_to_role(messages, "assistant", patch.assistant_parts)
    return messages


def _append_parts_to_role(messages: list[LMMessage], role: str, parts: list[LMPart], *, at_start: bool = False) -> None:
    index = next((i for i, message in enumerate(messages) if message.role == role), None) if at_start else _last_role_message_index(messages, role)
    if index is None:
        message = LMMessage(role=role, parts=list(parts))
        if at_start:
            messages.insert(0, message)
        else:
            messages.append(message)
    else:
        messages[index].parts.extend(parts)


def _insert_user_part_segments_default(adapter: Any, messages: list[LMMessage], segments: list[_UserPartSegment], plan: _AdapterPlan) -> list[LMMessage]:
    """Default adapter placement for field-associated native user parts.

    This is intentionally grammar-light: it asks the adapter for anchors and
    header/footer parts instead of checking adapter class names. Custom adapters
    can override ``place_user_part_segments()`` when their prompt grammar needs
    more precise placement than "before the next rendered input field, otherwise
    before output requirements, otherwise append".
    """
    user_index = _last_user_message_index(messages)
    if user_index is None:
        parts: list[LMPart] = []
        for segment in segments:
            parts.extend(adapter.native_input_header_parts(segment.field_name))
            parts.extend(segment.parts)
            parts.extend(adapter.native_input_footer_parts(segment.field_name))
        messages.append(LMMessage(role="user", parts=parts))
        return messages

    grouped: dict[str, list[_UserPartSegment]] = {}
    order: list[str] = []
    for segment in segments:
        anchor = _next_rendered_input_field(segment.field_name, plan) or "__output_requirements__"
        if anchor not in grouped:
            grouped[anchor] = []
            order.append(anchor)
        grouped[anchor].append(segment)

    message = messages[user_index]
    for anchor in order:
        parts = []
        for segment in grouped[anchor]:
            parts.extend(adapter.native_input_header_parts(segment.field_name))
            parts.extend(segment.parts)
            parts.extend(adapter.native_input_footer_parts(segment.field_name))
        _insert_parts_before_anchor(adapter, message, anchor, parts, plan)
    return messages


def _insert_parts_before_anchor(adapter: Any, message: LMMessage, anchor: str, parts: list[LMPart], plan: _AdapterPlan) -> None:
    """Splice LM parts into one rendered user message near an adapter anchor."""
    header = adapter.output_requirements_anchor() if anchor == "__output_requirements__" else adapter.input_field_anchor(anchor)
    if header is None:
        message.parts.extend(parts)
        return
    for index, part in enumerate(message.parts):
        if not isinstance(part, LMTextPart):
            continue
        position = part.text.find(header)
        if position == -1:
            continue
        before = part.text[:position]
        after = part.text[position:]
        inserted = _trim_leading_newlines(parts) if index == 0 and not before else parts
        replacement: list[LMPart] = []
        if before:
            replacement.append(LMTextPart(text=before, metadata=part.metadata))
        replacement.extend(inserted)
        if after:
            replacement.append(LMTextPart(text="\n\n" + after, metadata=part.metadata))
        message.parts[index : index + 1] = replacement
        message.parts = _merge_adjacent_text_parts(message.parts)
        return
    message.parts.extend(parts)
    message.parts = _merge_adjacent_text_parts(message.parts)


def _merge_adjacent_text_parts(parts: list[LMPart]) -> list[LMPart]:
    merged: list[LMPart] = []
    for part in parts:
        if merged and isinstance(merged[-1], LMTextPart) and isinstance(part, LMTextPart) and not merged[-1].metadata and not part.metadata:
            merged[-1] = LMTextPart(text=merged[-1].text + part.text)
        else:
            merged.append(part)
    return merged


def _trim_leading_newlines(parts: list[LMPart]) -> list[LMPart]:
    if not parts or not isinstance(parts[0], LMTextPart):
        return parts
    return [parts[0].model_copy(update={"text": parts[0].text.lstrip()}), *parts[1:]]


def _next_rendered_input_field(field_name: str, plan: _AdapterPlan) -> str | None:
    seen = False
    for name in plan.original_signature.input_fields:
        if name == field_name:
            seen = True
            continue
        if seen and name in plan.inputs:
            return name
    return None


def _patch_is_meaningful(patch: _StrategyPatch) -> bool:
    return bool(
        patch.request.messages
        or patch.request.system_parts
        or patch.request.user_parts
        or patch.request.assistant_parts
        or patch.request.tools
        or patch.request.config is not None
        or patch.request.delete_input_fields
        or patch.request.delete_output_fields
        or patch.delete_input_fields
        or patch.delete_output_fields
        or patch.keep_input_fields
        or patch.keep_output_fields
        or patch.user_part_segments
        or patch.output_parsers
        or patch.trace
    )


def _patch_owns_input_field(patch: _StrategyPatch, field_name: str) -> bool:
    return field_name in (*patch.request.delete_input_fields, *patch.delete_input_fields)


def _patch_owns_output_field(patch: _StrategyPatch, field_name: str) -> bool:
    return field_name in (*patch.request.delete_output_fields, *patch.delete_output_fields)


def _patch_strategy_name(patch: _StrategyPatch) -> str:
    return next((trace.strategy for trace in patch.trace if trace.strategy), "<anonymous strategy>")


def _patch_replaces_text_owned_field(plan: _AdapterPlan, patch: _StrategyPatch) -> bool:
    return any(
        plan.input_field_owners.get(field_name) == _STRATEGY_TEXT_OWNER
        for field_name in (*patch.request.delete_input_fields, *patch.delete_input_fields)
    ) or any(
        plan.output_field_owners.get(field_name) == _STRATEGY_TEXT_OWNER
        for field_name in (*patch.request.delete_output_fields, *patch.delete_output_fields)
    )


def _raise_on_config_conflict(current: LMConfig | None, incoming: LMConfig | None, strategy_name: str) -> None:
    if current is None or incoming is None:
        return
    current_data = current.model_dump(exclude_none=True)
    incoming_data = incoming.model_dump(exclude_none=True)
    for key in ("temperature", "max_tokens", "top_p", "stop", "n", "logprobs", "response_format"):
        if key in current_data and key in incoming_data and current_data[key] != incoming_data[key]:
            raise ValueError(f"LM config field {key!r} is already set to {current_data[key]!r}; {strategy_name} cannot set it to {incoming_data[key]!r}.")
    for key in ("reasoning", "tool_choice", "cache", "prompt_cache"):
        if not (isinstance(current_data.get(key), dict) and isinstance(incoming_data.get(key), dict)):
            if key in current_data and key in incoming_data and current_data[key] != incoming_data[key]:
                raise ValueError(f"LM config field {key!r} is already set; {strategy_name} cannot replace it.")
            continue
        overlap = current_data[key].keys() & incoming_data[key].keys()
        for nested_key in overlap:
            if current_data[key][nested_key] != incoming_data[key][nested_key]:
                raise ValueError(
                    f"LM config field {key}.{nested_key} is already set to {current_data[key][nested_key]!r}; "
                    f"{strategy_name} cannot set it to {incoming_data[key][nested_key]!r}."
                )
    current_extensions = current_data.get("extensions", {}) or {}
    incoming_extensions = incoming_data.get("extensions", {}) or {}
    for key in current_extensions.keys() & incoming_extensions.keys():
        if current_extensions[key] != incoming_extensions[key]:
            raise ValueError(f"LM config extension {key!r} is already set to {current_extensions[key]!r}; {strategy_name} cannot set it to {incoming_extensions[key]!r}.")


def _history_to_lm_messages(adapter: Any, signature: type[Signature], history: History) -> list[LMMessage]:
    messages: list[LMMessage] = []
    for turn in history.messages:
        messages.append(LMMessage(role="user", parts=[LMTextPart(text=adapter.format_user_message_content(signature, turn))]))
        messages.append(LMMessage(role="assistant", parts=[LMTextPart(text=adapter.format_assistant_message_content(signature, turn))]))
    return messages


def _input_segment_patch(ctx: _InputFieldStrategyContext, parts: list[LMPart] | None, strategy: _Strategy, reason: str) -> _StrategyPatch:
    """Return a replacement patch for an input field rendered as native parts."""
    if not parts:
        return _StrategyPatch()
    return _StrategyPatch(
        delete_input_fields=(ctx.field_name,),
        user_part_segments=(_UserPartSegment(ctx.field_name, parts),),
        trace=(_StrategyTrace(ctx.field_name, type(strategy).__name__, "replace", reason),),
    )


def _parts_for_collection(annotation: Any, value: Any, marker_type: type, convert: Any) -> list[LMPart] | None:
    """Convert scalar or homogeneous collection marker values into LM parts."""
    if value is None:
        return None
    origin = get_origin(annotation)
    if origin in (list, tuple, set, frozenset):
        args = get_args(annotation)
        item_annotation = args[0] if args else Any
        if not _annotation_contains(item_annotation, marker_type) or not isinstance(value, (list, tuple, set, frozenset)):
            return None
        return [convert(item) for item in value]
    if _annotation_contains(annotation, marker_type):
        return [convert(value)]
    return None


def _image_to_lm_part(image: Image) -> LMImagePart:
    source = image.url
    if source.startswith("data:") and "," in source:
        header, data = source.split(",", 1)
        media_type = header.removeprefix("data:").split(";", 1)[0]
        return LMImagePart(data=data, media_type=media_type)
    return LMImagePart(url=source)


def _file_to_lm_part(file: File) -> LMBinaryPart:
    if file.file_data is not None:
        media_type = "application/octet-stream"
        data = file.file_data
        if file.file_data.startswith("data:") and "," in file.file_data:
            header, data = file.file_data.split(",", 1)
            media_type = header.removeprefix("data:").split(";", 1)[0]
        return LMBinaryPart(data=data, media_type=media_type, filename=file.filename)
    if file.file_id is not None:
        return LMBinaryPart(file_id=file.file_id, filename=file.filename)
    raise ValueError("File must have file_data or file_id.")


def _tool_to_lm_tool_spec(tool: Tool) -> LMToolSpec:
    if hasattr(tool, "to_lm_tool_spec"):
        return tool.to_lm_tool_spec()
    args = tool.args or {}
    return LMToolSpec(
        name=tool.name or "",
        description=tool.desc,
        parameters={"type": "object", "properties": args, "required": list(args.keys())},
    )


def _citation_part_to_dict(citation: LMCitationPart) -> dict[str, Any]:
    data = citation.model_dump(exclude_none=True)
    provider_data = citation.metadata.get("provider_data") if isinstance(citation.metadata, dict) else None
    if isinstance(provider_data, dict):
        data = {**provider_data, **data}
    cited_text = data.get("cited_text") or data.get("text") or data.get("supported_text") or ""
    return {
        "cited_text": cited_text,
        "document_index": data.get("document_index", 0),
        "document_title": data.get("document_title") or data.get("title"),
        "start_char_index": data.get("start_char_index", 0),
        "end_char_index": data.get("end_char_index", len(cited_text)),
        "supported_text": data.get("supported_text"),
    }


def _get_tool_call_input_field_name(signature: type[Signature]) -> str | None:
    for name, field in signature.input_fields.items():
        origin = get_origin(field.annotation)
        if origin is list and field.annotation.__args__[0] == Tool:
            return name
        if field.annotation == Tool:
            return name
    return None


def _get_tool_call_output_field_name(signature: type[Signature]) -> str | None:
    for name, field in signature.output_fields.items():
        if field.annotation == ToolCalls:
            return name
    return None


def _annotation_contains(annotation: Any, expected: type) -> bool:
    try:
        if isinstance(annotation, type) and issubclass(annotation, expected):
            return True
    except TypeError:
        pass
    origin = get_origin(annotation)
    if origin is None:
        return False
    return any(_annotation_contains(arg, expected) for arg in get_args(annotation))


def _last_user_message_index(messages: list[LMMessage]) -> int | None:
    return _last_role_message_index(messages, "user")


def _last_role_message_index(messages: list[LMMessage], role: str) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].role == role:
            return index
    return None


def _legacy_tool_call_to_part(tool_call: Any) -> LMToolCallPart:
    if isinstance(tool_call, LMToolCallPart):
        return tool_call
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    name = function.get("name") or (tool_call.get("name") if isinstance(tool_call, dict) else "") or ""
    arguments = function.get("arguments", tool_call.get("arguments", "{}") if isinstance(tool_call, dict) else "{}")
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
    except Exception:
        args = {}
    return LMToolCallPart(id=tool_call.get("id") if isinstance(tool_call, dict) else None, name=name, args=args)
