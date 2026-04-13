"""Field-level streaming listener for DSPy adapters.

Detects field boundaries in streaming LM output (``[[ ## field ## ]]``,
JSON keys, XML tags) and yields per-field ``StreamResponse`` chunks.
"""

import inspect
import re
from collections import defaultdict
from queue import Queue
from typing import TYPE_CHECKING, Any

import jiter

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.types import Type
from dspy.adapters.xml_adapter import XMLAdapter
from dspy.dsp.utils.settings import settings
from dspy.streaming.messages import StreamResponse

if TYPE_CHECKING:
    from dspy.primitives.module import Module

ADAPTER_SUPPORT_STREAMING = [ChatAdapter, XMLAdapter, JSONAdapter]

# Per-adapter boundary detection config
_ADAPTER_CONFIG = {
    "ChatAdapter": lambda name: {
        "start_id": f"[[ ## {name} ## ]]",
        "end_re": re.compile(r"\[\[ ## (\w+) ## \]\]"),
        "start_char": "[",
        "end_prefixes": ["[", "[[", "[[ ", "[[ #", "[[ ##"],
        "end_contains": "[[ ##",
    },
    "JSONAdapter": lambda name: {
        "start_id": f'"{name}":',
        "end_re": re.compile(r'\w*"(,|\s*})'),
        "start_char": '"',
        "end_prefixes": ['"', '",', '" ', '"}'],
        "end_contains": "}",
    },
    "XMLAdapter": lambda name: {
        "start_id": f"<{name}>",
        "end_re": re.compile(rf"</{name}>"),
        "start_char": "<",
        "end_prefixes": ["<", "</"],
        "end_contains": "</",
    },
}


class StreamListener:
    """Captures streaming output of a specific signature field."""

    def __init__(self, signature_field_name, predict=None, predict_name=None, allow_reuse=False):
        self.signature_field_name = signature_field_name
        self.predict = predict
        self.predict_name = predict_name
        self.allow_reuse = allow_reuse
        self._reset()

    def _reset(self):
        self.field_start_queue = []
        self.field_end_queue = Queue()
        self.stream_start = self.stream_end = self.cache_hit = False
        self._json_acc = ""

    def _resp(self, token, last=False):
        return StreamResponse(self.predict_name, self.signature_field_name, token, is_last_chunk=last) if token or last else None

    def _cfg(self):
        name = (settings.adapter.__class__.__name__ if settings.adapter else "ChatAdapter")
        factory = _ADAPTER_CONFIG.get(name)
        if not factory:
            raise ValueError(f"Unsupported adapter for streaming: {name}")
        return name, factory(self.signature_field_name)

    def _could_be_end(self, buf, cfg):
        return (any(buf.endswith(p) for p in cfg["end_prefixes"]) or
                (cfg["end_contains"] and cfg["end_contains"] in buf))

    def receive(self, chunk):
        if self.stream_end:
            if self.allow_reuse:
                self._reset()
            else:
                return

        # Custom streamable types (Reasoning, Citations)
        otype = self._output_type
        if otype and inspect.isclass(otype) and issubclass(otype, Type) and otype.is_streamable():
            if parsed := otype.parse_stream_chunk(chunk):
                return StreamResponse(self.predict_name, self.signature_field_name, parsed, is_last_chunk=self.stream_end)

        try:
            text = chunk.choices[0].delta.content
            if text is None:
                return
        except Exception:
            return

        adapter_name, cfg = self._cfg()
        start_id, end_re, start_char = cfg["start_id"], cfg["end_re"], cfg["start_char"]

        # Cache hit: full response in one chunk
        if text and start_id in text and adapter_name != "JSONAdapter":
            after = text[text.find(start_id) + len(start_id):]
            if re.search(end_re, after):
                self.cache_hit = self.stream_start = self.stream_end = True
                return

        # Phase 1: detect start boundary
        if not self.stream_start:
            if not self.field_start_queue and start_char not in text:
                return
            self.field_start_queue.append(text)
            concat = "".join(self.field_start_queue)
            if start_id in concat:
                self.stream_start = True
                self.field_start_queue = []
                text = concat[concat.find(start_id) + len(start_id):].lstrip()
                if adapter_name == "JSONAdapter":
                    self._json_acc = "{" + start_id
            elif start_id.startswith(concat.strip()):
                return  # partial match, keep buffering
            else:
                self.field_start_queue = []
                return

        # Phase 2: buffer tokens, detect end boundary
        if not text:
            return
        self.field_end_queue.put(text)
        buf = "".join(self.field_end_queue.queue).strip()

        token = None
        if not self._could_be_end(buf, cfg):
            token = self._flush(adapter_name)
        elif self.field_end_queue.qsize() > 10:
            token = self.field_end_queue.get()

        if adapter_name == "JSONAdapter":
            return self._json_handle(token, text)
        return self._default_handle(token, end_re)

    def _json_handle(self, token, text):
        self._json_acc += text
        # Check if accumulated JSON is complete
        if self._json_acc.rstrip().endswith("}"):
            try:
                jiter.from_json(self._json_acc.encode())
                self.stream_end = True
                last = self._flush("JSONAdapter")
                idx = last.rfind("}")
                token = (token + last[:idx]) if token else last[:idx]
                return self._resp(token, last=True)
            except ValueError:
                pass
        # Check if next key appeared (field ended)
        try:
            parsed = jiter.from_json(self._json_acc.encode(), partial_mode="trailing-strings")
            if len(parsed) > 1:
                self.stream_end = True
                last = self._flush("JSONAdapter")
                next_key = next(k for k in parsed if k != self.signature_field_name)
                idx = last.find(next_key)
                token = (token + last[:idx]) if token else last[:idx]
        except ValueError:
            pass
        return self._resp(token, last=self.stream_end)

    def _default_handle(self, token, end_re):
        buf = "".join(self.field_end_queue.queue).strip()
        if re.search(end_re, buf):
            self.stream_end = True
            last = self._flush(settings.adapter.__class__.__name__ if settings.adapter else "ChatAdapter")
            token = ((token or "") + last).rstrip()
        return self._resp(token, last=self.stream_end)

    def _flush(self, adapter_name):
        tokens = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()
        if adapter_name == "JSONAdapter":
            return tokens
        if adapter_name == "XMLAdapter":
            idx = tokens.find(f"</{self.signature_field_name}>")
            return tokens[:idx] if idx != -1 else tokens
        # ChatAdapter
        idx = tokens.find("[[")
        return tokens[:idx] if idx != -1 else tokens

    def finalize(self):
        if self.stream_end or not self.stream_start:
            return None
        self.stream_end = True
        if self.field_end_queue.qsize() > 0:
            token = self._flush(settings.adapter.__class__.__name__ if settings.adapter else "ChatAdapter")
            if token:
                return self._resp(token, last=True)
        return None

    @property
    def _output_type(self):
        try:
            return self.predict.signature.output_fields[self.signature_field_name].annotation
        except Exception:
            return None


def find_predictor_for_stream_listeners(program, stream_listeners):
    """Auto-match listeners to predictors by field name."""
    predictors = program.named_predictors()
    field_map = {l.signature_field_name: None for l in stream_listeners if not l.predict}

    for name, pred in predictors:
        for field_name in pred.signature.output_fields:
            if field_name in field_map:
                if field_map[field_name] is not None:
                    raise ValueError(f"Field {field_name} is not unique — specify predict explicitly.")
                field_map[field_name] = (name, pred)

    result = defaultdict(list)
    for listener in stream_listeners:
        if listener.predict:
            result[id(listener.predict)].append(listener)
        else:
            if listener.signature_field_name not in field_map or field_map[listener.signature_field_name] is None:
                raise ValueError(f"Field {listener.signature_field_name} not found in any predictor.")
            listener.predict_name, listener.predict = field_map[listener.signature_field_name]
            result[id(listener.predict)].append(listener)
    return result
