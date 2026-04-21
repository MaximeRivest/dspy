"""AdapterV2 — typed adapter built on preset templates + Style.

The adapter's two jobs are:

1. Format: signature + demos + inputs → list[LMMessage]
2. Parse: LMCompletion → dict[str, Any]

Both are driven by the Style (parse strategy + stream parser) and the
preset template (prompt structure).  Custom adapters are just custom
templates + an existing or user-registered Style.
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, get_origin

from .base_lm_v2 import BaseLMv2
from .completion import LMCompletion, LMResponse
from .config import LMConfig, LMToolDef
from .errors import AdapterParseError, InvalidRequestError
from .messages import LMMessage
from .parts import (
    AudioPart,
    CitationPart,
    DocumentPart,
    ImagePart,
    Part,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from .preset import (
    CallableFragment,
    CHAT_PRESET,
    CSV_PRESET,
    Content,
    Demos,
    Element,
    ForEach,
    Fragment,
    History,
    InputFields,
    Inputs,
    Instruction,
    JSON_PRESET,
    Message,
    OutputFields,
    OutputRequest,
    Outputs,
    PRESETS,
    Preset,
    Structure,
    Text,
    XML_PRESET,
)
from .streaming import LMStreamError, LMStreamEvent, PartAccumulator, PartDelta
from .styles import (
    ChatStyle,
    FieldChunk,
    JSONStyle,
    Style,
    StyleLike,
    XMLStyle,
    get_style,
)


# Field helpers that handle dspy.Signature types
def _get_input_fields(signature) -> dict[str, Any]:
    return getattr(signature, "input_fields", {})


def _get_output_fields(signature) -> dict[str, Any]:
    return getattr(signature, "output_fields", {})


def _get_instructions(signature) -> str:
    return getattr(signature, "instructions", None) or ""


# Field descriptions — mirrors dspy.adapters.utils.get_field_description_string
def _annotation_name(ann) -> str:
    try:
        from dspy.adapters.utils import get_annotation_name
        return get_annotation_name(ann)
    except Exception:
        if hasattr(ann, "__name__"):
            return ann.__name__
        return str(ann)


def _field_description_string(fields: dict[str, Any]) -> str:
    try:
        from dspy.adapters.utils import get_field_description_string
        return get_field_description_string(fields)
    except Exception:
        lines = []
        for idx, (k, v) in enumerate(fields.items()):
            ann = _annotation_name(getattr(v, "annotation", v))
            lines.append(f"{idx + 1}. `{k}` ({ann})")
        return "\n".join(lines)


def _format_field_value(field_info, value) -> str:
    try:
        from dspy.adapters.utils import format_field_value
        return format_field_value(field_info=field_info, value=value)
    except Exception:
        return str(value)


# ═══════════════════════════════════════════════════════════════
# Custom type detection (dspy.Image / Audio / File / Document / Tool)
# ═══════════════════════════════════════════════════════════════

def _dspy_type_to_part(value) -> Part | None:
    """Convert a DSPy custom type (Image, Audio, File) to a typed Part.

    Returns ``None`` if the value isn't a known DSPy custom type.
    """
    try:
        from dspy.adapters.types.image import Image as _Image
        if isinstance(value, _Image):
            url = value.url
            if isinstance(url, str) and url.startswith("data:"):
                # data URI: split off mime and base64
                header, _, data = url.partition(",")
                mime = header.split(";")[0].split(":", 1)[1] if ":" in header else "image/png"
                return ImagePart(data=data, media_type=mime)
            return ImagePart(url=url)
    except Exception:
        pass

    try:
        from dspy.adapters.types.audio import Audio as _Audio
        if isinstance(value, _Audio):
            return AudioPart(
                data=value.data,
                media_type=f"audio/{value.audio_format}",
            )
    except Exception:
        pass

    try:
        from dspy.adapters.types.file import File as _File
        if isinstance(value, _File):
            # Map file_data (data URI) or file_id.
            if value.file_data and value.file_data.startswith("data:"):
                header, _, data = value.file_data.partition(",")
                mime = header.split(";")[0].split(":", 1)[1] if ":" in header else "application/octet-stream"
                return DocumentPart(
                    data=data,
                    media_type=mime,
                    title=value.filename,
                )
            return DocumentPart(
                data=value.file_data,
                file_id=value.file_id,
                title=value.filename,
            )
    except Exception:
        pass

    try:
        from dspy.adapters.types.document import Document as _Document
        if isinstance(value, _Document):
            return DocumentPart(
                data=value.data,
                title=value.title,
                media_type=value.media_type or "text/plain",
            )
    except Exception:
        pass

    return None


# ═══════════════════════════════════════════════════════════════
# AdapterV2
# ═══════════════════════════════════════════════════════════════

class AdapterV2:
    """Typed DSPy adapter built on preset templates + Style."""

    def __init__(
        self,
        messages: tuple[Element, ...] | list[Element] | None = None,
        style: StyleLike = "chat",
    ):
        if messages is None:
            # Default to the style's canonical preset
            preset_name = style if isinstance(style, str) else getattr(style, "name", "chat")
            preset = PRESETS.get(preset_name, CHAT_PRESET)
            messages = preset.messages
        self.messages: tuple[Element, ...] = tuple(messages)
        self.style: Style = get_style(style)

    @classmethod
    def from_preset_obj(cls, preset: Preset) -> "AdapterV2":
        """Create an adapter from a typed ``Preset`` object."""
        return cls(messages=preset.messages, style=preset.style)

    @classmethod
    def from_preset(cls, name: str) -> "AdapterV2":
        """Create an adapter from a built-in preset name."""
        if name not in PRESETS:
            raise KeyError(f"Unknown preset: {name}. Available: {sorted(PRESETS)}")
        preset = PRESETS[name]
        return cls(messages=preset.messages, style=preset.style)

    # ─────────────────────────────────────────────────────────
    # Rendering: preset → list[LMMessage]
    # ─────────────────────────────────────────────────────────

    def format(
        self,
        signature,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[LMMessage]:
        """Render the preset into typed LM messages."""
        # Separate dspy.History input (if present) from regular inputs
        history_field_name = self._find_history_field(signature)
        history_obj = None
        inputs_copy = dict(inputs)
        if history_field_name and history_field_name in inputs_copy:
            raw_history = inputs_copy.pop(history_field_name)
            try:
                from dspy.adapters.types.history import History as _DspyHistory
                if isinstance(raw_history, _DspyHistory):
                    history_obj = raw_history
                elif isinstance(raw_history, list):
                    history_obj = _DspyHistory(messages=raw_history)
            except Exception:
                history_obj = None

        # Use a signature-without-history for field rendering
        if history_field_name:
            try:
                render_signature = signature.delete(history_field_name)
            except Exception:
                render_signature = signature
        else:
            render_signature = signature

        ctx = {
            "_signature": render_signature,
            "_original_signature": signature,
            "_inputs": inputs_copy,
            "_history": history_obj,
            "_style": self.style,
        }

        rendered: list[LMMessage] = []

        for element in self.messages:
            if isinstance(element, Message):
                msg = self._render_message(element, ctx)
                if msg is not None:
                    rendered.append(msg)
            elif isinstance(element, Demos):
                rendered.extend(self._render_demos(element, render_signature, demos, ctx))
            elif isinstance(element, History):
                rendered.extend(self._render_history(element, render_signature, history_obj, ctx))

        return rendered

    def _render_message(self, msg: Message, ctx: dict) -> LMMessage | None:
        """Render one preset Message into an LMMessage."""
        parts = self._render_content(msg.content, ctx)

        # Collapse adjacent text parts into one
        merged: list[Part] = []
        buf: list[str] = []
        for part in parts:
            if isinstance(part, TextPart):
                buf.append(part.text)
            else:
                if buf:
                    text = "".join(buf)
                    if text.strip():
                        merged.append(TextPart(text=text))
                    buf = []
                merged.append(part)
        if buf:
            text = "".join(buf)
            if text.strip():
                merged.append(TextPart(text=text))

        if not merged:
            return None
        return LMMessage(role=msg.role, parts=merged)

    def _render_content(self, content: Content, ctx: dict) -> list[Part]:
        """Render a Content value into a flat list of Parts."""
        if isinstance(content, (list, tuple)):
            result: list[Part] = []
            for frag in content:
                result.extend(self._render_fragment(frag, ctx))
            return result
        return self._render_fragment(content, ctx)

    def _render_fragment(self, frag: Fragment, ctx: dict) -> list[Part]:
        """Render a single fragment into one or more Parts."""
        if isinstance(frag, str):
            return self._render_string_template(frag, ctx)
        if isinstance(frag, Text):
            return [TextPart(text=frag.text)]
        if isinstance(frag, Instruction):
            return [TextPart(text=_get_instructions(ctx["_signature"]))]
        if isinstance(frag, InputFields):
            text = self._render_input_fields_section(ctx["_signature"])
            return [TextPart(text=text)]
        if isinstance(frag, OutputFields):
            text = self._render_output_fields_section(ctx["_signature"])
            return [TextPart(text=text)]
        if isinstance(frag, Structure):
            return [TextPart(text=self._render_structure(ctx["_signature"], frag.style))]
        if isinstance(frag, Inputs):
            return self._render_inputs_fragment(ctx, frag.style)
        if isinstance(frag, Outputs):
            return [TextPart(text=self._render_outputs_schema(ctx["_signature"], frag.style))]
        if isinstance(frag, OutputRequest):
            return [TextPart(text=self._render_output_request(ctx["_signature"], frag.style))]
        if isinstance(frag, ForEach):
            return [TextPart(text=self._render_foreach(ctx, frag))]
        if isinstance(frag, CallableFragment):
            return [TextPart(text=str(frag.fn(ctx, **frag.args)))]
        return [TextPart(text=str(frag))]

    def _render_string_template(self, template: str, ctx: dict) -> list[Part]:
        """Render a plain-string fragment with ``{field_name}`` interpolation.

        Interpolates input field names (plus ``{instruction}`` for convenience).
        """
        class _SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        mapping = _SafeDict()
        signature = ctx["_signature"]
        inputs = ctx["_inputs"]
        for name, info in _get_input_fields(signature).items():
            if name in inputs:
                mapping[name] = _format_field_value(info, inputs[name])
            else:
                mapping[name] = ""
        mapping["instruction"] = _get_instructions(signature)

        return [TextPart(text=template.format_map(mapping))]

    # ─────────────────────────────────────────────────────────
    # Rendering helpers — mirror the old adapter's formatting
    # ─────────────────────────────────────────────────────────

    def _render_input_fields_section(self, signature) -> str:
        return f"Your input fields are:\n{_field_description_string(_get_input_fields(signature))}"

    def _render_output_fields_section(self, signature) -> str:
        return f"Your output fields are:\n{_field_description_string(_get_output_fields(signature))}"

    def _render_structure(self, signature, style_name: str) -> str:
        """Render a schema-like field placeholder block for the LM."""
        style = get_style(style_name)
        intro = "All interactions will be structured in the following way, with the appropriate values filled in."

        # Build input field display
        input_items = [
            (name, f"{{{name}}}")
            for name in _get_input_fields(signature)
        ]
        output_items = []
        for name, field in _get_output_fields(signature).items():
            output_items.append((name, self._translate_field_type(name, field)))

        in_text = style.field_layout().render(input_items) if input_items else ""
        out_text = style.field_layout().render(output_items) if output_items else ""
        # For chat style, the envelope_close already has `completed` marker;
        # we don't want it twice if the envelope is part of `in_text` + `out_text`.
        # Render both sections together so the `completed` marker appears once.
        if isinstance(style, ChatStyle):
            combined = style.field_layout().render(input_items + output_items)
            return f"{intro}\n\n{combined}"
        # For other styles, concatenate.
        return f"{intro}\n\n{in_text}\n\n{out_text}".strip()

    def _render_inputs_fragment(self, ctx: dict, style_name: str) -> list[Part]:
        """Render current input values, inlining any dspy custom types as Parts."""
        style = get_style(style_name)
        signature = ctx["_signature"]
        inputs = ctx["_inputs"]

        input_fields = _get_input_fields(signature)

        # Walk inputs; if a value is a dspy custom type, emit Part for it.
        # Otherwise serialize via style field layout.
        text_items: list[tuple[str, str]] = []
        extra_parts: list[Part] = []
        for name, info in input_fields.items():
            if name not in inputs:
                continue
            value = inputs[name]
            part = _dspy_type_to_part(value)
            if part is not None:
                # Render a placeholder line in text + an attached media Part
                text_items.append((name, "<see attached>"))
                extra_parts.append(part)
            else:
                text_items.append((name, _format_field_value(info, value)))

        text = style.field_layout().render(text_items) if text_items else ""
        parts: list[Part] = []
        if text.strip():
            parts.append(TextPart(text=text))
        parts.extend(extra_parts)
        return parts

    def _render_outputs_schema(self, signature, style_name: str) -> str:
        """Render the output schema description for the system prompt."""
        if style_name == "schema":
            import json as _json
            import pydantic
            schema: dict[str, Any] = {}
            for name, field in _get_output_fields(signature).items():
                ann = getattr(field, "annotation", field)
                try:
                    schema[name] = pydantic.TypeAdapter(ann).json_schema()
                except Exception:
                    schema[name] = {"type": _annotation_name(ann)}
            return _json.dumps(schema, indent=2, ensure_ascii=False)
        # Otherwise defer to style field layout (for chat/xml/csv it's the placeholder form)
        style = get_style(style_name)
        items = []
        for name, field in _get_output_fields(signature).items():
            items.append((name, self._translate_field_type(name, field)))
        return style.field_layout().render(items)

    def _render_output_request(self, signature, style_name: str) -> str:
        style = get_style(style_name)
        output_names = list(_get_output_fields(signature))
        return style.output_request(output_names)

    def _translate_field_type(self, name, field) -> str:
        try:
            from dspy.adapters.utils import translate_field_type
            return translate_field_type(name, field)
        except Exception:
            return f"{{{name}}}"

    def _render_foreach(self, ctx: dict, frag: ForEach) -> str:
        signature = ctx["_signature"]
        if frag.fields == "input":
            field_dict = _get_input_fields(signature)
        else:
            field_dict = _get_output_fields(signature)

        inputs = ctx["_inputs"]
        chunks: list[str] = []
        for idx, (name, field) in enumerate(field_dict.items()):
            desc_raw = (getattr(field, "json_schema_extra", None) or {}).get("desc", "")
            desc = ""
            if isinstance(desc_raw, str) and not desc_raw.startswith("${"):
                desc = desc_raw
            mapping = {
                "i": str(idx + 1),
                "index": str(idx + 1),
                "name": name,
                "type": _annotation_name(getattr(field, "annotation", field)),
                "desc": desc,
                "value": (
                    _format_field_value(field, inputs.get(name, ""))
                    if frag.fields == "input" else ""
                ),
            }

            class _SafeDict(dict):
                def __missing__(self, key):
                    return "{" + key + "}"

            chunks.append(frag.template.format_map(_SafeDict(mapping)))
        return frag.separator.join(chunks)

    def _find_history_field(self, signature) -> str | None:
        try:
            from dspy.adapters.types.history import History as _H
            for name, field in _get_input_fields(signature).items():
                if getattr(field, "annotation", None) == _H:
                    return name
        except Exception:
            pass
        return None

    # ─────────────────────────────────────────────────────────
    # Demos and History expansion
    # ─────────────────────────────────────────────────────────

    def _render_demos(
        self,
        directive: Demos,
        signature,
        demos: list[dict[str, Any]],
        ctx: dict,
    ) -> list[LMMessage]:
        if not demos:
            return []
        messages: list[LMMessage] = []
        for demo in demos:
            demo_inputs, demo_outputs = self._split_demo(signature, demo)
            # Build a per-demo context for interpolation
            demo_ctx = dict(ctx, _inputs=demo_inputs)

            if directive.user is not None and directive.assistant is not None:
                user_parts = self._render_content(directive.user, demo_ctx)
                asst_parts = self._render_content_with_outputs(
                    directive.assistant, demo_ctx, demo_outputs,
                )
                if user_parts:
                    messages.append(LMMessage(role="user", parts=user_parts))
                if asst_parts:
                    messages.append(LMMessage(role="assistant", parts=asst_parts))
            else:
                # Default demo rendering: user = inputs in chat-like style,
                # assistant = outputs in chosen style.
                user_text_parts = self._render_inputs_fragment(demo_ctx, "chat")
                if user_text_parts:
                    messages.append(LMMessage(role="user", parts=user_text_parts))
                asst_text = self._render_demo_outputs(demo_outputs, signature)
                if asst_text:
                    messages.append(LMMessage(role="assistant", parts=[TextPart(text=asst_text)]))
        return messages

    def _render_content_with_outputs(
        self,
        content: Content,
        ctx: dict,
        outputs: dict[str, Any],
    ) -> list[Part]:
        """Render content where output-field placeholders resolve to demo outputs."""
        # Include outputs in the interpolation namespace
        ctx2 = dict(ctx)
        ctx2["_outputs"] = outputs
        return self._render_content(content, ctx2)

    def _render_demo_outputs(self, outputs: dict[str, Any], signature) -> str:
        columns = list(_get_output_fields(signature))
        return self.style.format_output_block(outputs, columns)

    def _render_history(
        self,
        directive: History,
        signature,
        history_obj,
        ctx: dict,
    ) -> list[LMMessage]:
        if history_obj is None or not getattr(history_obj, "messages", []):
            return []
        messages: list[LMMessage] = []
        for turn in history_obj.messages:
            turn_inputs, turn_outputs = self._split_demo(signature, turn)
            turn_ctx = dict(ctx, _inputs=turn_inputs)
            if directive.user is not None and directive.assistant is not None:
                user_parts = self._render_content(directive.user, turn_ctx)
                asst_parts = self._render_content_with_outputs(
                    directive.assistant, turn_ctx, turn_outputs,
                )
                if user_parts:
                    messages.append(LMMessage(role="user", parts=user_parts))
                if asst_parts:
                    messages.append(LMMessage(role="assistant", parts=asst_parts))
            else:
                user_parts = self._render_inputs_fragment(turn_ctx, "chat")
                if user_parts:
                    messages.append(LMMessage(role="user", parts=user_parts))
                asst_text = self._render_demo_outputs(turn_outputs, signature)
                if asst_text:
                    messages.append(LMMessage(
                        role="assistant", parts=[TextPart(text=asst_text)],
                    ))
        return messages

    @staticmethod
    def _split_demo(
        signature,
        demo: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        input_fields = _get_input_fields(signature)
        output_fields = _get_output_fields(signature)
        inputs = {k: v for k, v in demo.items() if k in input_fields}
        outputs = {k: v for k, v in demo.items() if k in output_fields}
        return inputs, outputs

    # ─────────────────────────────────────────────────────────
    # Parse: text → dict
    # ─────────────────────────────────────────────────────────

    def parse(self, signature, completion: str) -> dict[str, Any]:
        output_fields = _get_output_fields(signature)
        parsed = self.style.parse(completion, output_fields)

        # Coerce values to annotated types
        coerced: dict[str, Any] = {}
        for name, raw in parsed.items():
            if name in output_fields:
                info = output_fields[name]
                ann = getattr(info, "annotation", None)
                try:
                    coerced[name] = self._parse_value(raw, ann)
                except Exception as e:
                    raise AdapterParseError(
                        f"Failed to parse field {name!r}: {e}",
                        adapter_name=type(self).__name__,
                        lm_response=completion,
                        expected_fields=list(output_fields.keys()),
                        parsed_fields=list(parsed.keys()),
                    )

        # All required output fields must be present
        missing = set(output_fields) - set(coerced)
        if missing:
            raise AdapterParseError(
                f"Missing fields: {sorted(missing)}",
                adapter_name=type(self).__name__,
                lm_response=completion,
                expected_fields=list(output_fields.keys()),
                parsed_fields=list(coerced.keys()),
            )

        return coerced

    @staticmethod
    def _parse_value(value, annotation) -> Any:
        if annotation is None or annotation is Any:
            return value
        try:
            from dspy.adapters.utils import parse_value
            return parse_value(value, annotation)
        except Exception:
            # Best-effort
            return value

    # ─────────────────────────────────────────────────────────
    # Call pipeline
    # ─────────────────────────────────────────────────────────

    def __call__(
        self,
        lm: BaseLMv2,
        config: LMConfig,
        signature,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        effective_config, processed_sig = self._preprocess(lm, config, signature, inputs)
        messages = self.format(processed_sig, demos, inputs)
        response: LMResponse = lm(messages, effective_config)
        return self._postprocess(processed_sig, signature, response.completions)

    async def acall(
        self,
        lm: BaseLMv2,
        config: LMConfig,
        signature,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        effective_config, processed_sig = self._preprocess(lm, config, signature, inputs)
        messages = self.format(processed_sig, demos, inputs)
        response: LMResponse = await lm.acall(messages, effective_config)
        return self._postprocess(processed_sig, signature, response.completions)

    def _preprocess(
        self,
        lm: BaseLMv2,
        config: LMConfig,
        signature,
        inputs: dict[str, Any],
    ) -> tuple[LMConfig, Any]:
        """Configure native features on LMConfig; may modify the signature."""
        # Check for tool input field + tool_calls output field
        tool_input_name = self._find_tool_input_field(signature)
        tool_output_name = self._find_tool_output_field(signature)

        new_config = config
        new_signature = signature

        if tool_output_name and tool_input_name is None:
            raise InvalidRequestError(
                f"Output field {tool_output_name!r} expects tool calls, but no "
                f"input field of type list[dspy.Tool] was provided."
            )

        if tool_output_name and lm.supports_function_calling:
            tools_value = inputs.get(tool_input_name)
            tools_list = tools_value if isinstance(tools_value, list) else [tools_value]
            lm_tools: list[LMToolDef] = []
            for tool in tools_list:
                if tool is None:
                    continue
                lm_tools.append(self._tool_to_def(tool))
            new_config = new_config.model_copy(update={"tools": lm_tools})
            # Remove the tool input and tool_calls output from the signature for
            # the downstream format (they're handled natively)
            try:
                new_signature = signature.delete(tool_output_name).delete(tool_input_name)
            except Exception:
                pass

        # Reasoning native feature
        try:
            from dspy.adapters.types.reasoning import Reasoning as _Reasoning

            reasoning_name = None
            for name, field in _get_output_fields(new_signature).items():
                if getattr(field, "annotation", None) is _Reasoning:
                    reasoning_name = name
                    break
            if reasoning_name and lm.supports_reasoning:
                # Set default reasoning_effort if not already set
                updates = {}
                if new_config.reasoning_effort is None:
                    updates["reasoning_effort"] = "low"
                if updates:
                    new_config = new_config.model_copy(update=updates)
                try:
                    new_signature = new_signature.delete(reasoning_name)
                except Exception:
                    pass
        except Exception:
            pass

        return new_config, new_signature

    @staticmethod
    def _tool_to_def(tool) -> LMToolDef:
        if isinstance(tool, LMToolDef):
            return tool
        name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
        desc = getattr(tool, "desc", None) or getattr(tool, "description", None) or ""
        params = {
            "type": "object",
            "properties": getattr(tool, "args", {}) or {},
            "required": list(getattr(tool, "args", {}) or {}),
        }
        return LMToolDef(name=name, description=desc or None, parameters=params)

    @staticmethod
    def _find_tool_input_field(signature) -> str | None:
        try:
            from dspy.adapters.types.tool import Tool as _Tool
            for name, field in _get_input_fields(signature).items():
                ann = getattr(field, "annotation", None)
                origin = get_origin(ann)
                if origin is list:
                    try:
                        args = ann.__args__
                        if args and args[0] is _Tool:
                            return name
                    except Exception:
                        pass
                if ann is _Tool:
                    return name
        except Exception:
            pass
        return None

    @staticmethod
    def _find_tool_output_field(signature) -> str | None:
        try:
            from dspy.adapters.types.tool import ToolCalls as _TC
            for name, field in _get_output_fields(signature).items():
                if getattr(field, "annotation", None) is _TC:
                    return name
        except Exception:
            pass
        return None

    @staticmethod
    def _find_reasoning_field(signature) -> str | None:
        try:
            from dspy.adapters.types.reasoning import Reasoning as _R
            for name, field in _get_output_fields(signature).items():
                if getattr(field, "annotation", None) is _R:
                    return name
        except Exception:
            pass
        return None

    @staticmethod
    def _find_citations_field(signature) -> str | None:
        try:
            from dspy.experimental import Citations as _C
            for name, field in _get_output_fields(signature).items():
                if getattr(field, "annotation", None) is _C:
                    return name
        except Exception:
            pass
        return None

    def _postprocess(
        self,
        processed_signature,
        original_signature,
        completions: list[LMCompletion],
    ) -> list[dict[str, Any]]:
        """Convert typed completions into output-field dicts."""
        results: list[dict[str, Any]] = []

        tool_output_name = self._find_tool_output_field(original_signature)
        reasoning_output_name = self._find_reasoning_field(original_signature)
        citations_output_name = self._find_citations_field(original_signature)

        for completion in completions:
            # If the LM produced tool_calls and there is a matching output field,
            # drop that field from the signature we parse against so that the
            # text-parser doesn't insist on a `[[ ## tool_calls ## ]]` marker.
            text_parse_signature = processed_signature
            if completion.tool_calls and tool_output_name:
                try:
                    text_parse_signature = text_parse_signature.delete(tool_output_name)
                except Exception:
                    pass

            # Parse text into output fields (using processed_signature, so
            # native-feature fields are already excluded)
            if completion.text:
                value = self.parse(text_parse_signature, completion.text)
            elif completion.tool_calls:
                value = {
                    k: None
                    for k in _get_output_fields(text_parse_signature)
                }
            else:
                raise AdapterParseError(
                    "LM returned an empty response.",
                    adapter_name=type(self).__name__,
                    lm_response="",
                    expected_fields=list(_get_output_fields(original_signature)),
                )

            # Fill in fields absent from processed_signature with None
            for name in _get_output_fields(original_signature):
                if name not in value:
                    value[name] = None

            # Native features
            if completion.tool_calls and tool_output_name:
                from dspy.adapters.types.tool import ToolCalls
                value[tool_output_name] = ToolCalls.from_dict_list([
                    {"name": tc.name, "args": tc.input}
                    for tc in completion.tool_calls
                ])

            if completion.thinking and reasoning_output_name:
                from dspy.adapters.types.reasoning import Reasoning
                value[reasoning_output_name] = Reasoning(content=completion.thinking)

            if completion.citations and citations_output_name:
                try:
                    from dspy.experimental import Citations
                    value[citations_output_name] = Citations.from_dict_list([
                        {
                            "cited_text": c.cited_text or "",
                            "document_index": c.source_index or 0,
                            "document_title": c.title,
                            "start_char_index": 0,
                            "end_char_index": len(c.cited_text or ""),
                        }
                        for c in completion.citations
                    ])
                except Exception:
                    pass

            if completion.logprobs:
                value["logprobs"] = completion.logprobs

            results.append(value)

        return results

    # ─────────────────────────────────────────────────────────
    # Streaming
    # ─────────────────────────────────────────────────────────

    async def stream_call(
        self,
        lm: BaseLMv2,
        config: LMConfig,
        signature,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> AsyncIterator[FieldChunk]:
        """Stream adapter output as field-level ``FieldChunk`` events."""
        effective_config, processed_sig = self._preprocess(lm, config, signature, inputs)
        messages = self.format(processed_sig, demos, inputs)

        output_field_names = list(_get_output_fields(processed_sig))
        stream_parser = self.style.stream_parser(output_field_names)

        reasoning_field = self._find_reasoning_field(signature)
        citations_field = self._find_citations_field(signature)
        tool_output_field = self._find_tool_output_field(signature)

        accumulator = PartAccumulator()

        async for event in lm.stream(messages, effective_config):
            if event.type == "start":
                continue

            if event.type == "delta":
                delta = event.delta
                if delta is None:
                    continue
                accumulator.feed(delta)

                if delta.type == "text" and delta.text:
                    for chunk in stream_parser.feed(delta.text):
                        yield chunk

                elif delta.type == "thinking" and delta.text:
                    if reasoning_field:
                        yield FieldChunk(
                            field_name=reasoning_field,
                            text=delta.text,
                            is_last=False,
                        )

                elif delta.type == "citation":
                    if citations_field:
                        yield FieldChunk(
                            field_name=citations_field,
                            text=delta,
                            is_last=False,
                        )

                elif delta.type == "tool_call":
                    # Accumulated; emitted on end
                    pass

            elif event.type == "error":
                # Raise the typed exception
                if event.error is not None:
                    raise event.error.to_exception(model=getattr(lm, "model", None))

            elif event.type == "end":
                # Flush the text stream parser
                for chunk in stream_parser.finalize():
                    yield chunk

                # Emit tool calls if any
                if tool_output_field:
                    built_parts = accumulator.build()
                    tool_call_parts = [p for p in built_parts if isinstance(p, ToolCallPart)]
                    if tool_call_parts:
                        from dspy.adapters.types.tool import ToolCalls
                        tc = ToolCalls.from_dict_list([
                            {"name": p.name, "args": p.input} for p in tool_call_parts
                        ])
                        yield FieldChunk(
                            field_name=tool_output_field,
                            text=tc,
                            is_last=True,
                        )

    # ─────────────────────────────────────────────────────────
    # Backward-compat shim — expose this AdapterV2 as a dspy.Adapter
    # ─────────────────────────────────────────────────────────

    def as_legacy(self):
        """Wrap this AdapterV2 as a legacy ``dspy.adapters.Adapter`` subclass.

        Allows using the typed adapter with existing ``dspy.Predict`` and
        legacy ``BaseLM`` subclasses that still return OpenAI-shaped responses.
        """
        return _LegacyAdapterShim(self)


def from_preset(name: str) -> AdapterV2:
    """Create an adapter from a built-in preset name.

    Equivalent to ``AdapterV2.from_preset(name)``.
    """
    return AdapterV2.from_preset(name)


# ═══════════════════════════════════════════════════════════════
# Legacy compatibility shim
# ═══════════════════════════════════════════════════════════════

from dspy.adapters.base import Adapter as _LegacyAdapter  # noqa: E402


class _LegacyAdapterShim(_LegacyAdapter):
    """Wrap an ``AdapterV2`` as a legacy ``dspy.Adapter`` subclass.

    This allows using the typed adapter with the existing ``dspy.Predict``
    pipeline and legacy ``dspy.BaseLM`` subclasses (which still return
    OpenAI-shaped dotdict responses).
    """

    def __init__(self, adapter_v2: AdapterV2):
        super().__init__()
        self._v2 = adapter_v2

    # Required abstract methods: delegate to v2 rendering where sensible.
    def format_field_description(self, signature):
        return self._v2._render_input_fields_section(signature)

    def format_field_structure(self, signature):
        return self._v2._render_structure(signature, self._v2.style.name)

    def format_task_description(self, signature):
        return _get_instructions(signature)

    def format_user_message_content(
        self,
        signature,
        inputs,
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        style = self._v2.style
        input_fields = _get_input_fields(signature)
        items = [
            (name, _format_field_value(info, inputs.get(name, "")))
            for name, info in input_fields.items()
            if name in inputs
        ]
        content = style.field_layout().render(items)
        parts = [s for s in [prefix, content] if s]
        if main_request:
            parts.append(style.output_request(list(_get_output_fields(signature))))
        if suffix:
            parts.append(suffix)
        return "\n\n".join(parts).strip()

    def format_assistant_message_content(
        self,
        signature,
        outputs,
        missing_field_message=None,
    ) -> str:
        columns = list(_get_output_fields(signature))
        filled = {c: outputs.get(c, missing_field_message) for c in columns}
        return self._v2.style.format_output_block(filled, columns)

    def parse(self, signature, completion: str):
        return self._v2.parse(signature, completion)

    # Keep the legacy ``format`` behavior (uses this shim's format_* methods)
    # for demos compatibility with existing DummyLM expectations.
