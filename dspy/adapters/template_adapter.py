"""TemplateAdapter — the universal DSPy adapter primitive.

Every adapter (Chat, JSON, XML) is a thin configuration of TemplateAdapter.
Templates use ``{field_name}`` interpolation and template functions like
``{inputs()}``, ``{outputs()}``, ``{demos()}``.  Multimodal content is
collected via ``to_lm15_part()`` and sent alongside text.

Parse modes: ``"chat"``, ``"json"``, ``"xml"``, ``"full_text"``, or a callable.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Callable

import json_repair
import pydantic
import regex

from dspy.adapters.base import Adapter
from dspy.adapters.types.history import History
from dspy.adapters.utils import (
    format_field_value,
    get_annotation_name,
    get_field_description_string,
    parse_value,
    serialize_for_json,
)
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import AdapterParseError

if TYPE_CHECKING:
    from dspy.signatures.signature import Signature

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# String interpolation helpers
# ---------------------------------------------------------------------------

_ESC_OPEN = "\x00LB\x00"
_ESC_CLOSE = "\x00RB\x00"
_FUNC_RE = re.compile(r"\{(\w+)\(([^)]*)\)\}")


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _parse_func_kwargs(raw: str) -> dict[str, str]:
    kw: dict[str, str] = {}
    for m in re.finditer(r"(\w+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\")", raw):
        kw[m.group(1)] = m.group(2) if m.group(2) is not None else m.group(3)
    return kw


# ---------------------------------------------------------------------------
# TemplateAdapter
# ---------------------------------------------------------------------------


class TemplateAdapter(Adapter):
    """Universal adapter — templates *are* the prompt.

    Parameters
    ----------
    messages : list[dict]
        Message templates.  Each is ``{"role": ..., "content": ...}``
        or a directive: ``{"role": "demos", ...}`` / ``{"role": "history"}``.
    parse_mode : str | Callable
        ``"json"`` (default), ``"chat"``, ``"xml"``, ``"full_text"``,
        or ``(signature, completion) -> dict``.
    """

    def __init__(
        self,
        messages: list[dict[str, Any]],
        parse_mode: str | Callable = "json",
        callbacks: list[BaseCallback] | None = None,
        use_native_function_calling: bool = False,
        native_response_types: list[type] | None = None,
    ):
        super().__init__(
            callbacks=callbacks,
            use_native_function_calling=use_native_function_calling,
            **({"native_response_types": native_response_types} if native_response_types else {}),
        )
        if not messages:
            raise ValueError("TemplateAdapter requires at least one message.")
        self.message_templates = messages
        self.parse_mode = parse_mode
        self._helpers: dict[str, Callable] = {}

    def register_helper(self, name: str, fn: Callable):
        self._helpers[name] = fn

    # ------------------------------------------------------------------
    # format()
    # ------------------------------------------------------------------

    def format(self, signature, demos, inputs, **kw) -> list[dict[str, Any]]:
        inputs = dict(inputs)

        # Extract history
        history_messages: list[dict] = []
        history_obj: History | None = None
        hf = self._get_history_field_name(signature)
        if hf and hf in inputs:
            raw = inputs.pop(hf)
            if isinstance(raw, History):
                history_obj = raw
                history_messages = self._expand_history(signature, raw)
            elif isinstance(raw, list):
                history_obj = History(messages=raw)
                history_messages = raw

        ctx = self._build_ctx(signature, demos, inputs, history_obj)

        rendered: list[dict] = []
        demos_used = False

        for tmpl in self.message_templates:
            role = tmpl.get("role", "")

            if role == "demos":
                rendered.extend(self._expand_demos_directive(tmpl, signature, demos, ctx))
                demos_used = True
                continue

            if role == "history":
                rendered.extend(history_messages)
                continue

            content = tmpl.get("content", "")
            text, used_d, _ = self._render(content, ctx, signature, demos, history_obj)
            if used_d:
                demos_used = True
            rendered.append({"role": role, "content": text})

        # Auto-inject demos if not consumed
        if demos and not demos_used:
            demo_msgs = self._format_demos(signature, demos)
            idx = next((i for i in range(len(rendered)-1, -1, -1) if rendered[i]["role"] == "user"), 0)
            rendered = rendered[:idx] + demo_msgs + rendered[idx:]

        return rendered

    # ------------------------------------------------------------------
    # parse()
    # ------------------------------------------------------------------

    def parse(self, signature, completion, **kw):
        if callable(self.parse_mode) and not isinstance(self.parse_mode, str):
            return self.parse_mode(signature, completion)
        completion_clean = _strip_fences(completion)
        if self.parse_mode == "full_text":
            return _parse_full_text(signature, completion)
        if self.parse_mode == "chat":
            return _parse_chat(signature, completion)
        if self.parse_mode == "xml":
            return _parse_xml(signature, completion_clean)
        return _parse_json(signature, completion_clean)

    # ------------------------------------------------------------------
    # format_finetune_data()
    # ------------------------------------------------------------------

    def format_finetune_data(self, signature, demos, inputs, outputs, **kw):
        messages = self.format(signature=signature, demos=demos, inputs=inputs)
        if self.parse_mode == "full_text" and len(signature.output_fields) == 1:
            asst = str(outputs.get(next(iter(signature.output_fields)), ""))
        elif self.parse_mode == "xml":
            asst = "\n".join(f"<{k}>{v}</{k}>" for k, v in outputs.items())
        elif self.parse_mode == "chat":
            parts = []
            for k in signature.output_fields:
                parts.append(f"[[ ## {k} ## ]]\n{outputs.get(k, '')}")
            asst = "\n\n".join(parts) + "\n\n[[ ## completed ## ]]"
        else:
            asst = json.dumps({k: serialize_for_json(v) for k, v in outputs.items()}, indent=2)
        messages.append({"role": "assistant", "content": asst})
        return {"messages": messages}

    # ------------------------------------------------------------------
    # Required abstract stubs
    # ------------------------------------------------------------------

    def format_field_description(self, sig):
        return _field_descriptions(sig)

    def format_field_structure(self, sig):
        return ""

    def format_task_description(self, sig):
        return sig.instructions or ""

    def format_user_message_content(self, sig, inputs, prefix="", suffix="", main_request=False):
        parts = [prefix] if prefix else []
        for name in sig.input_fields:
            if name in inputs:
                parts.append(f"{name}: {format_field_value(sig.input_fields[name], inputs[name])}")
        if suffix:
            parts.append(suffix)
        return "\n".join(parts).strip()

    def format_assistant_message_content(self, sig, outputs, missing_field_message=None):
        d = {name: outputs.get(name, missing_field_message) for name in sig.output_fields}
        return json.dumps(serialize_for_json(d), indent=2)

    # ==================================================================
    # Private
    # ==================================================================

    def _build_ctx(self, sig, demos, inputs, history):
        ctx = {}
        for name, fi in sig.input_fields.items():
            ctx[name] = format_field_value(fi, inputs[name]) if name in inputs else ""
        ctx["instruction"] = sig.instructions or ""
        ctx["history"] = history if history else ""
        return ctx

    def _render(self, template, ctx, sig, demos, history):
        text = template.replace("{{", _ESC_OPEN).replace("}}", _ESC_CLOSE)
        text, used_d, used_h = self._eval_funcs(text, ctx, sig, demos, history)
        text = text.format_map(_SafeDict(ctx))
        text = text.replace(_ESC_OPEN, "{").replace(_ESC_CLOSE, "}")
        return text, used_d, used_h

    def _eval_funcs(self, text, ctx, sig, demos, history):
        used_d = used_h = False

        def _repl(m):
            nonlocal used_d, used_h
            fn, raw = m.group(1), m.group(2).strip()
            kw = _parse_func_kwargs(raw) if raw else {}
            r = None
            if fn == "inputs":
                r = _render_inputs(ctx, sig, **kw)
            elif fn == "outputs":
                r = _render_outputs(sig, **kw)
            elif fn == "demos":
                used_d = True; r = _render_demos_inline(demos, sig, **kw)
            elif fn == "history":
                used_h = True; r = _render_history_inline(history, sig, **kw)
            elif fn == "field_descriptions":
                r = _field_descriptions(sig, **kw)
            elif fn == "field_structure":
                r = _field_structure(sig, **kw)
            elif fn == "task_description":
                r = _task_description(sig, **kw)
            elif fn == "output_requirements":
                r = _output_requirements(sig, **kw)
            elif fn in self._helpers:
                r = str(self._helpers[fn](ctx=ctx, signature=sig, demos=demos, **kw))
            if r is not None:
                return r.replace("{", _ESC_OPEN).replace("}", _ESC_CLOSE)
            return m.group(0)

        return _FUNC_RE.sub(_repl, text), used_d, used_h

    def _expand_demos_directive(self, directive, sig, demos, ctx):
        if not demos:
            return []
        if "user" in directive and "assistant" in directive:
            msgs = []
            fields = list(sig.input_fields) + list(sig.output_fields)
            for demo in demos:
                dc = {k: str(demo.get(k, "")) for k in fields}
                dc["outputs_json"] = json.dumps(
                    serialize_for_json({k: demo.get(k, "") for k in sig.output_fields}))
                msgs.append({"role": "user", "content": directive["user"].format_map(_SafeDict(dc))})
                msgs.append({"role": "assistant", "content": directive["assistant"].format_map(_SafeDict(dc))})
            return msgs
        return self._format_demos(sig, demos)

    def _format_demos(self, sig, demos):
        msgs = []
        for demo in demos:
            has_in = any(k in demo for k in sig.input_fields)
            has_out = any(k in demo for k in sig.output_fields)
            if not (has_in and has_out):
                continue
            # User
            parts = []
            for name, fi in sig.input_fields.items():
                if name in demo and fi.annotation != History:
                    parts.append(f"{name}: {format_field_value(fi, demo[name])}")
            if parts:
                msgs.append({"role": "user", "content": "\n".join(parts)})
            # Assistant
            out = {k: demo[k] for k in sig.output_fields if k in demo}
            if out:
                if self.parse_mode == "xml":
                    msgs.append({"role": "assistant", "content": "\n".join(f"<{k}>{v}</{k}>" for k, v in out.items())})
                elif self.parse_mode == "chat":
                    parts = [f"[[ ## {k} ## ]]\n{v}" for k, v in out.items()]
                    msgs.append({"role": "assistant", "content": "\n\n".join(parts) + "\n\n[[ ## completed ## ]]"})
                elif self.parse_mode == "full_text" and len(out) == 1:
                    msgs.append({"role": "assistant", "content": str(next(iter(out.values())))})
                else:
                    msgs.append({"role": "assistant", "content": json.dumps(serialize_for_json(out), indent=2)})
        return msgs

    def _expand_history(self, sig, history):
        msgs = []
        for entry in history.messages:
            u = [f"{n}: {entry[n]}" for n, fi in sig.input_fields.items()
                 if n in entry and fi.annotation != History]
            if u:
                msgs.append({"role": "user", "content": "\n".join(u)})
            out = {k: entry[k] for k in sig.output_fields if k in entry}
            if out:
                if self.parse_mode == "xml":
                    msgs.append({"role": "assistant", "content": "\n".join(f"<{k}>{v}</{k}>" for k, v in out.items())})
                elif self.parse_mode == "chat":
                    parts = [f"[[ ## {k} ## ]]\n{v}" for k, v in out.items()]
                    msgs.append({"role": "assistant", "content": "\n\n".join(parts) + "\n\n[[ ## completed ## ]]"})
                elif self.parse_mode == "full_text" and len(out) == 1:
                    msgs.append({"role": "assistant", "content": str(next(iter(out.values())))})
                else:
                    msgs.append({"role": "assistant", "content": json.dumps(serialize_for_json(out), indent=2)})
        return msgs

    def _get_history_field_name(self, sig):
        for name, field in sig.input_fields.items():
            if field.annotation == History:
                return name
        return None


# ======================================================================
# Template functions — generate prompt fragments from signatures
# ======================================================================


def _field_descriptions(sig, **kw):
    return (
        f"Your input fields are:\n{get_field_description_string(sig.input_fields)}\n"
        f"Your output fields are:\n{get_field_description_string(sig.output_fields)}"
    )


def _task_description(sig, **kw):
    import textwrap
    instructions = textwrap.dedent(sig.instructions or "")
    objective = ("\n" + " " * 8).join([""] + instructions.splitlines())
    return f"In adhering to this structure, your objective is: {objective}"


def _field_structure(sig, style="chat", **kw):
    """Generate the expected I/O structure example."""
    if style == "chat":
        parts = ["All interactions will be structured in the following way, with the appropriate values filled in.\n"]
        for name, fi in sig.input_fields.items():
            if fi.annotation == History:
                continue
            parts.append(f"[[ ## {name} ## ]]\n{{{name}}}")
        for name, fi in sig.output_fields.items():
            desc = _type_constraint(name, fi)
            parts.append(f"[[ ## {name} ## ]]\n{{{name}}}{desc}")
        parts.append("[[ ## completed ## ]]")
        return "\n\n".join(parts)

    if style == "json":
        input_parts = ["Inputs will have the following structure:\n"]
        for name in sig.input_fields:
            input_parts.append(f"[[ ## {name} ## ]]\n{{{name}}}")
        output_obj = {name: f"{{{name}}}" for name in sig.output_fields}
        return (
            "All interactions will be structured in the following way, with the appropriate values filled in.\n\n"
            + "\n\n".join(input_parts) + "\n\n"
            + "Outputs will be a JSON object with the following fields.\n\n"
            + json.dumps(output_obj, indent=2)
        )

    if style == "xml":
        parts = ["All interactions will be structured in the following way, with the appropriate values filled in.\n"]
        for name in sig.input_fields:
            parts.append(f"<{name}>\n{{{name}}}\n</{name}>")
        for name in sig.output_fields:
            parts.append(f"<{name}>\n{{{name}}}\n</{name}>")
        return "\n\n".join(parts)

    return ""


def _type_constraint(name, field_info):
    """Type constraint note for non-str fields (used by chat adapter)."""
    ann = field_info.annotation
    if ann is str:
        return ""
    try:
        schema = pydantic.TypeAdapter(ann).json_schema()
        return f"        # note: the value you produce must adhere to the JSON schema: {json.dumps(schema)}"
    except Exception:
        return f"        # note: must be a valid Python {get_annotation_name(ann)}"


def _output_requirements(sig, style="chat", **kw):
    if style == "chat":
        def _type_info(v):
            if v.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            return ""
        msg = "Respond with the corresponding output fields, starting with the field "
        msg += ", then ".join(f"`[[ ## {f} ## ]]`{_type_info(v)}" for f, v in sig.output_fields.items())
        msg += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return msg
    if style == "json":
        fields = ", ".join(f"`{f}`" for f in sig.output_fields)
        return f"Respond with a JSON object in the following order of fields: {fields}."
    if style == "xml":
        tags = " ".join(f"`<{f}>`" for f in sig.output_fields)
        return f"Respond with the corresponding output fields wrapped in XML tags {tags}."
    return ""


def _render_inputs(ctx, sig, style="chat", **kw):
    parts = []
    for name, fi in sig.input_fields.items():
        if fi.annotation == History:
            continue
        val = ctx.get(name, "")
        if style == "chat":
            parts.append(f"[[ ## {name} ## ]]\n{val}")
        elif style == "xml":
            parts.append(f"<{name}>\n{val}\n</{name}>")
        elif style == "json":
            parts.append(f'"{name}": {json.dumps(val)}')
        else:
            parts.append(f"{name}: {val}")
    if style == "json":
        return "{\n  " + ",\n  ".join(parts) + "\n}"
    return "\n\n".join(parts)


def _render_outputs(sig, style="default", **kw):
    if style == "schema":
        schema = {}
        for name, fi in sig.output_fields.items():
            try:
                schema[name] = pydantic.TypeAdapter(fi.annotation).json_schema()
            except Exception:
                schema[name] = {"type": get_annotation_name(fi.annotation)}
        return json.dumps(schema, indent=2)
    if style == "xml":
        return "\n".join(f"<{name}>...</{name}>" for name in sig.output_fields)
    return get_field_description_string(sig.output_fields)


def _render_demos_inline(demos, sig, style="default", **kw):
    if not demos:
        return ""
    fields = list(sig.input_fields) + list(sig.output_fields)
    blocks = []
    for i, demo in enumerate(demos):
        if style == "json":
            blocks.append(json.dumps({k: demo[k] for k in fields if k in demo}, indent=2))
        elif style == "xml":
            blocks.append("\n".join(f"<{k}>{demo[k]}</{k}>" for k in fields if k in demo))
        else:
            lines = [f"Example {i+1}:"]
            for k in fields:
                if k in demo:
                    lines.append(f"  {k}: {demo[k]}")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _render_history_inline(history, sig, style="default", **kw):
    if history is None or not history.messages:
        return ""
    blocks = []
    for i, msg in enumerate(history.messages, 1):
        if style == "json":
            blocks.append(json.dumps(msg, indent=2))
        elif style == "xml":
            fields = "\n".join(f"  <{k}>{v}</{k}>" for k, v in msg.items())
            blocks.append(f"<turn>\n{fields}\n</turn>")
        else:
            lines = [f"Turn {i}:"]
            for k, v in msg.items():
                lines.append(f"  {k}: {v}")
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


# ======================================================================
# Parse modes
# ======================================================================

_CHAT_HEADER = re.compile(r"\[\[ ## (\w+) ## \]\]")


def _strip_fences(text):
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def _parse_chat(sig, completion):
    sections = [(None, [])]
    for line in completion.splitlines():
        m = _CHAT_HEADER.match(line.strip())
        if m:
            header = m.group(1)
            rest = line[m.end():].strip()
            sections.append((header, [rest] if rest else []))
        else:
            sections[-1][1].append(line)
    fields = {}
    for k, v in sections:
        if k and k not in fields and k in sig.output_fields:
            try:
                fields[k] = parse_value("\n".join(v).strip(), sig.output_fields[k].annotation)
            except Exception as e:
                raise AdapterParseError("TemplateAdapter", sig, completion, message=f"Field {k}: {e}")
    if fields.keys() != sig.output_fields.keys():
        raise AdapterParseError("TemplateAdapter", sig, completion, parsed_result=fields)
    return fields


def _parse_json(sig, completion):
    fields = json_repair.loads(completion)
    if not isinstance(fields, dict):
        m = regex.search(r"\{(?:[^{}]|(?R))*\}", completion, regex.DOTALL)
        if m:
            fields = json_repair.loads(m.group(0))
    if not isinstance(fields, dict):
        raise AdapterParseError("TemplateAdapter", sig, completion, message="No JSON object found.")
    parsed = {}
    for name, fi in sig.output_fields.items():
        if name in fields:
            try:
                parsed[name] = parse_value(fields[name], fi.annotation)
            except Exception as e:
                raise AdapterParseError("TemplateAdapter", sig, completion, message=f"Field {name}: {e}")
    if parsed.keys() != sig.output_fields.keys():
        raise AdapterParseError("TemplateAdapter", sig, completion, parsed_result=parsed)
    return parsed


def _parse_xml(sig, completion):
    parsed = {}
    for name, fi in sig.output_fields.items():
        m = re.search(rf"<{re.escape(name)}>(.*?)</{re.escape(name)}>", completion, re.DOTALL)
        if m:
            try:
                parsed[name] = parse_value(m.group(1).strip(), fi.annotation)
            except Exception as e:
                raise AdapterParseError("TemplateAdapter", sig, completion, message=f"XML field {name}: {e}")
    if parsed.keys() != sig.output_fields.keys():
        raise AdapterParseError("TemplateAdapter", sig, completion, parsed_result=parsed,
                                message=f"Missing XML tags: {set(sig.output_fields) - set(parsed)}")
    return parsed


def _parse_full_text(sig, completion):
    keys = list(sig.output_fields.keys())
    if len(keys) != 1:
        raise AdapterParseError("TemplateAdapter", sig, completion,
                                message=f"full_text requires 1 output field, got {len(keys)}")
    return {keys[0]: parse_value(completion.strip(), sig.output_fields[keys[0]].annotation)}
