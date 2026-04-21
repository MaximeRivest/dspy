"""BaseLMv2 — minimal typed contract for language models.

A ``BaseLMv2`` subclass implements one method (``forward``) and gets:
- Automatic retries on transient errors
- Default (fake) streaming via ``forward``
- Full participation in the DSPy adapter pipeline

No OpenAI response shape, no litellm dependency, no untyped kwargs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, AsyncIterator

from dspy.clients.base_lm import BaseLM
from dspy.dsp.utils.utils import dotdict
from dspy.utils.callback import with_callbacks

from .completion import LMCompletion, LMResponse, LMUsage
from .config import LMConfig
from .errors import RETRYABLE_ERRORS, LMError
from .messages import LMMessage
from .parts import AudioPart, ImagePart, Part, TextPart, ThinkingPart, ToolCallPart
from .streaming import LMStreamEvent, PartDelta


class BaseLMv2(BaseLM):
    """Minimal base class for language models in DSPy.

    Subclasses implement ``forward(messages, config) -> LMResponse``
    using the typed v2 signature.  The ``__call__`` method wraps
    ``forward`` with retries for ``RETRYABLE_ERRORS`` and respects
    ``default_config`` merging.

    BaseLMv2 inherits from legacy ``BaseLM`` so existing DSPy code
    (``dspy.Predict``, ``dspy.configure(lm=...)``) accepts it.
    When called via the legacy API (``forward(prompt, messages, **kwargs)``),
    it delegates to the v2 ``forward`` and translates between shapes.

    Streaming is opt-in via ``stream()`` override; the default
    implementation fake-streams from ``forward()``.
    """

    def __init__(
        self,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        num_retries: int = 3,
        **kwargs,
    ):
        # Skip BaseLM.__init__ to avoid its litellm-specific defaults.
        self.model = model
        self.model_type = "chat"
        self.cache = cache
        self.num_retries = num_retries
        self.history: list[dict] = []

        # Build the default config from typed kwargs (extras go to extensions)
        known = {"temperature", "max_tokens", "top_p", "n", "tools",
                 "response_format", "reasoning_effort"}
        cfg_kwargs: dict[str, Any] = {}
        extensions: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in known:
                cfg_kwargs[k] = v
            else:
                extensions[k] = v
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature
        if max_tokens is not None:
            cfg_kwargs["max_tokens"] = max_tokens
        if extensions:
            cfg_kwargs["extensions"] = extensions
        self.default_config = LMConfig(**cfg_kwargs)

        # Legacy BaseLM.kwargs is sometimes introspected; mirror defaults.
        self.kwargs = {
            "temperature": cfg_kwargs.get("temperature"),
            "max_tokens": cfg_kwargs.get("max_tokens"),
        }

    # ── The one method subclasses implement (v2 API) ──
    #
    # Subclasses override ``forward(messages, config) -> LMResponse``.
    # The ``BaseLM`` interface (``forward(prompt, messages, **kwargs)``)
    # is handled via the ``_legacy_forward`` bridge, routed through
    # ``__call__`` when called by legacy consumers (``dspy.Predict``).

    def forward(
        self,
        messages: list[LMMessage],
        config: LMConfig,
    ) -> LMResponse:
        """Send messages to the LM and return completions.

        Subclasses must implement this method.  The return type is
        always ``LMResponse``.

        Raises:
            LMError: On provider failures.  Subclasses should raise
                the appropriate ``LMError`` subclass (e.g.,
                ``RateLimitError``, ``AuthError``) for automatic retry.
        """
        raise NotImplementedError(
            "BaseLMv2 subclasses must implement forward(messages, config) -> LMResponse"
        )

    async def aforward(
        self,
        messages: list[LMMessage],
        config: LMConfig,
    ) -> LMResponse:
        """Async variant of ``forward``.

        Default implementation runs ``forward`` in a worker thread.
        Override for true async providers.
        """
        return await asyncio.to_thread(self.forward, messages, config)

    # ── Capability discovery ──

    @property
    def supports_function_calling(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def supports_response_schema(self) -> bool:
        return False

    # ── __call__ with retries (overloaded: v2 and legacy) ──

    @with_callbacks
    def __call__(self, *args, **kwargs):
        """Call the LM.

        Two calling conventions:

        - v2:      ``lm(messages: list[LMMessage], config: LMConfig) -> LMResponse``
        - legacy:  ``lm(prompt=..., messages=..., **kwargs) -> list[str | dict]``
                   (used by legacy ``dspy.BaseLM`` consumers)
        """
        # v2 path
        if (args
                and isinstance(args[0], list)
                and (not args[0] or isinstance(args[0][0], LMMessage))):
            messages = args[0]
            config = args[1] if len(args) > 1 else kwargs.get("config")
            return self._call_v2(messages, config)
        # Legacy path: delegate to BaseLM.__call__ which eventually calls forward()
        return self._legacy_call(*args, **kwargs)

    def _call_v2(
        self,
        messages: list[LMMessage],
        config: LMConfig | None = None,
    ) -> LMResponse:
        effective = self.default_config.merge(config) if config else self.default_config

        last_error: LMError | None = None
        for attempt in range(1 + self.num_retries):
            try:
                return self.forward(messages, effective)
            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < self.num_retries:
                    time.sleep(min(2 ** attempt, 60))
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("BaseLMv2.__call__ exited loop unexpectedly")

    @with_callbacks
    async def acall(self, *args, **kwargs):
        if (args
                and isinstance(args[0], list)
                and (not args[0] or isinstance(args[0][0], LMMessage))):
            messages = args[0]
            config = args[1] if len(args) > 1 else kwargs.get("config")
            return await self._acall_v2(messages, config)
        return await self._legacy_acall(*args, **kwargs)

    async def _acall_v2(
        self,
        messages: list[LMMessage],
        config: LMConfig | None = None,
    ) -> LMResponse:
        effective = self.default_config.merge(config) if config else self.default_config

        last_error: LMError | None = None
        for attempt in range(1 + self.num_retries):
            try:
                return await self.aforward(messages, effective)
            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < self.num_retries:
                    await asyncio.sleep(min(2 ** attempt, 60))
                    continue
                raise
        if last_error is not None:
            raise last_error
        raise RuntimeError("BaseLMv2.acall exited loop unexpectedly")

    # ── Legacy compatibility bridge ──
    # These methods let a BaseLMv2 be used wherever a legacy ``BaseLM`` is
    # expected (e.g., ``dspy.Predict``).  They convert legacy dict-messages
    # to typed LMMessage, call the v2 forward, and translate the response
    # back to an OpenAI-shaped dotdict.

    def _legacy_call(self, prompt=None, messages=None, **kwargs):
        legacy_response = self._legacy_forward(prompt=prompt, messages=messages, **kwargs)
        # Emulate BaseLM._process_completion: return list of strings or dicts
        return self._process_legacy_response(legacy_response, kwargs)

    async def _legacy_acall(self, prompt=None, messages=None, **kwargs):
        legacy_response = await self._legacy_aforward(prompt=prompt, messages=messages, **kwargs)
        return self._process_legacy_response(legacy_response, kwargs)

    def _legacy_forward(self, prompt=None, messages=None, **kwargs):
        typed_messages = self._legacy_messages_to_typed(prompt, messages)
        config = self._legacy_kwargs_to_config(kwargs)
        v2_response = self._call_v2(typed_messages, config)
        return self._v2_response_to_legacy(v2_response)

    async def _legacy_aforward(self, prompt=None, messages=None, **kwargs):
        typed_messages = self._legacy_messages_to_typed(prompt, messages)
        config = self._legacy_kwargs_to_config(kwargs)
        v2_response = await self._acall_v2(typed_messages, config)
        return self._v2_response_to_legacy(v2_response)

    @staticmethod
    def _legacy_messages_to_typed(prompt, messages) -> list[LMMessage]:
        if messages is None and prompt is not None:
            return [LMMessage.user(prompt)]
        messages = messages or []
        typed: list[LMMessage] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            parts: list[Part] = []
            if isinstance(content, str):
                parts.append(TextPart(text=content))
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        parts.append(TextPart(text=str(block)))
                        continue
                    btype = block.get("type")
                    if btype == "text":
                        parts.append(TextPart(text=block.get("text", "")))
                    elif btype == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        parts.append(ImagePart(url=url))
                    elif btype == "input_audio":
                        detail = block.get("input_audio", {})
                        parts.append(AudioPart(
                            data=detail.get("data"),
                            media_type=f"audio/{detail.get('format', 'wav')}",
                        ))
                    else:
                        # Fallback to text
                        parts.append(TextPart(text=str(block)))
            else:
                parts.append(TextPart(text=str(content)))
            if not parts:
                parts.append(TextPart(text=""))
            typed.append(LMMessage(role=role, parts=parts))
        return typed

    def _legacy_kwargs_to_config(self, kwargs: dict[str, Any]) -> LMConfig:
        known = {"temperature", "max_tokens", "top_p", "n",
                 "response_format", "reasoning_effort"}
        cfg: dict[str, Any] = {}
        ext: dict[str, Any] = {}
        for k, v in kwargs.items():
            if k in known:
                cfg[k] = v
            elif k == "tools":
                # Convert legacy tools dict[] list to LMToolDef
                from .config import LMToolDef
                tools_converted = []
                for t in v:
                    if isinstance(t, dict):
                        fn = t.get("function", t)
                        tools_converted.append(LMToolDef(
                            name=fn.get("name", "tool"),
                            description=fn.get("description"),
                            parameters=fn.get("parameters", {"type": "object", "properties": {}}),
                        ))
                cfg["tools"] = tools_converted
            else:
                ext[k] = v
        if ext:
            cfg["extensions"] = ext
        return LMConfig(**cfg)

    @staticmethod
    def _v2_response_to_legacy(response: LMResponse):
        """Convert a typed LMResponse to an OpenAI-shaped dotdict."""
        choices = []
        for comp in response.completions:
            content = comp.text or ""
            message_fields = {"content": content, "tool_calls": None}
            # Native features propagated in legacy shape
            if comp.thinking:
                message_fields["reasoning_content"] = comp.thinking
            if comp.tool_calls:
                message_fields["tool_calls"] = [
                    dotdict(
                        id=tc.id,
                        function=dotdict(name=tc.name, arguments=_json_dumps(tc.input)),
                        type="function",
                    )
                    for tc in comp.tool_calls
                ]
            choices.append(dotdict(
                message=dotdict(**message_fields),
                finish_reason=comp.finish_reason or "stop",
                logprobs=comp.logprobs,
            ))
        return dotdict(
            choices=choices,
            usage=dotdict(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            model=response.model or "v2",
            id=response.id,
            _hidden_params={"response_cost": None},
        )

    def _process_legacy_response(self, response, kwargs):
        """Mirror BaseLM._process_completion to produce ``list[str | dict]``."""
        outputs = []
        for choice in response.choices:
            output: dict[str, Any] = {}
            output["text"] = choice.message.content
            if getattr(choice.message, "reasoning_content", None):
                output["reasoning_content"] = choice.message.reasoning_content
            if kwargs.get("logprobs") and getattr(choice, "logprobs", None):
                output["logprobs"] = choice.logprobs
            if getattr(choice.message, "tool_calls", None):
                output["tool_calls"] = choice.message.tool_calls
            outputs.append(output)
        if all(len(o) == 1 for o in outputs):
            outputs = [o["text"] for o in outputs]
        return outputs

    # ── Streaming (default fakes from forward) ──

    async def stream(
        self,
        messages: list[LMMessage],
        config: LMConfig | None = None,
    ) -> AsyncIterator[LMStreamEvent]:
        """Stream a response as a sequence of events.

        Default implementation fake-streams from ``_aforward_v2``.
        Override for true token-by-token streaming.
        """
        effective = self.default_config.merge(config) if config else self.default_config
        response = await self.aforward(messages, effective)

        yield LMStreamEvent(type="start", model=response.model, id=response.id)
        for i, completion in enumerate(response.completions):
            for j, part in enumerate(completion.parts):
                if isinstance(part, TextPart):
                    yield LMStreamEvent(
                        type="delta",
                        completion_index=i,
                        delta=PartDelta(type="text", part_index=j, text=part.text),
                    )
                elif isinstance(part, ThinkingPart):
                    yield LMStreamEvent(
                        type="delta",
                        completion_index=i,
                        delta=PartDelta(type="thinking", part_index=j, text=part.text),
                    )
                elif isinstance(part, ToolCallPart):
                    import json as _json
                    yield LMStreamEvent(
                        type="delta",
                        completion_index=i,
                        delta=PartDelta(
                            type="tool_call",
                            part_index=j,
                            id=part.id,
                            name=part.name,
                            input=_json.dumps(part.input),
                        ),
                    )
                # Other types not commonly needed in fake streaming.
            yield LMStreamEvent(
                type="end",
                completion_index=i,
                finish_reason=completion.finish_reason or "stop",
                usage=response.usage,
                completion=completion,
            )


def _json_dumps(obj: Any) -> str:
    import json as _json
    try:
        return _json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)


__all__ = ["BaseLMv2"]
