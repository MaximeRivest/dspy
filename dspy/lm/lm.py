"""One LM class: lm15-routed chat completions with declared capabilities.

The canonical representation is lm15's `Request`/`Response`. `LM.__call__`
has two faces: the keyword `messages=`/`prompt=` path is the
dict-in/strings-out convenience for adapters, and the positional typed
path — `lm("hi")`, `lm(dspy.System(...), dspy.User(...))` — returns the
canonical `lm15.Response` (text, tool calls, usage, citations).
`LM.complete` speaks full canonical lm15 for callers that build their
own `Request`.
Routing (model string -> provider -> credential) is lm15's `LMRouter`;
`ROUTER` is the shared module-level instance, and any object with a
`complete(Request) -> Response` method can stand in for it per-LM.

Failures surface as the typed `LMError` from the contract table — carrying
lm15's canonical error code — never as provider-specific exceptions.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from lm15 import Config, Message, ModelRegistry, Request, Response, ResponseStream, TextPart
from lm15.errors import LM15Error
from lm15.router import LMRouter, RouterConfig

from dspy.core.errors import LMError
from dspy.lm.direct import System

__all__ = ["LM", "LMCapabilities", "ROUTER"]

#: The shared default router (`dspy.lm.ROUTER`): one per process, one
#: provider LM per provider inside it, built lazily. Its model catalog
#: is hydrated from installed entry-point catalogs (e.g. the `aimo`
#: registry package) — advisory metadata plus catalog-rung resolution;
#: discovery is offline-safe and never raises. Hydration costs ~1s for
#: a large catalog, so it runs on FIRST USE, not at import — `ROUTER`
#: is a module `__getattr__` attribute backed by `default_router()`.
_ROUTER: LMRouter | None = None


def default_router() -> LMRouter:
    """Build (once) and return the shared catalog-backed router."""
    global _ROUTER
    if _ROUTER is None:
        _ROUTER = LMRouter(RouterConfig(registry=ModelRegistry.discover()))
    return _ROUTER


def __getattr__(name: str) -> Any:
    if name == "ROUTER":
        return default_router()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

#: Request kwargs that are universal generation parameters (lm15
#: `Config` fields). Everything else a caller passes goes to
#: `Config.extensions` — the clearly-separated provider-specific
#: namespace — instead of pretending to be universal.
_CONFIG_FIELDS = frozenset(
    {"max_tokens", "temperature", "top_p", "top_k", "stop", "response_format", "tool_choice", "reasoning", "cache"}
)

#: Request kwargs that are endpoint configuration, not generation
#: parameters. They stay in `lm.kwargs` (the serialization layer treats
#: `api_key` as a credential there) but never travel in a request body;
#: at call time they pin the credential and, for `api_base`, the
#: OpenAI-compatible endpoint the request goes to.
_ENDPOINT_FIELDS = ("api_key", "api_base", "base_url")


@dataclass(frozen=True)
class LMCapabilities:
    """Declared capability facts of a bound model.

    These are constructor data, stated by whoever binds the model.
    Adapter strategies predicate on these facts; nothing in the system
    sniffs a live client object to guess them.

    Attributes:
        instruct: True for instruction/chat-tuned models; False for raw
            base (completion) models.
        native_reasoning: The API surfaces a native reasoning channel.
        native_fc: The API supports native function calling.
        native_citations: The API surfaces native citations.
        image_input: The API accepts image content parts.
    """

    instruct: bool = True
    native_reasoning: bool = False
    native_fc: bool = False
    native_citations: bool = False
    image_input: bool = False

    def to_dict(self) -> dict[str, bool]:
        """Return the facts as a plain dict (serialization-friendly)."""
        return asdict(self)


class LM:
    """An lm15-routed chat-completions language model.

    Args:
        model: The model string. Bare family names route by built-in
            rule (`"gpt-4o-mini"`, `"claude-sonnet-4-5"`, `"gemini-2.5-pro"`);
            a `provider:` prefix is explicit (`"openai-chat:qwen3"`,
            `"ollama:llama3.2"`). lm15's router resolves it
            (prefix > catalog > rules) and picks up the provider's API
            key from its declared env var.
        instruct: Capability fact — instruction-tuned (True) or base model.
        native_reasoning: Capability fact — native reasoning channel.
        native_fc: Capability fact — native function calling.
        native_citations: Capability fact — native citations.
        image_input: Capability fact — image content parts accepted.
        temperature: Default sampling temperature for every request.
            `None` (the default) sends nothing — lm15 and the provider
            decide.
        max_tokens: Default completion-token cap for every request.
            `None` (the default) sends nothing — lm15 and the provider
            decide.
        router: Anything with `complete(lm15.Request) -> lm15.Response`.
            Defaults to the shared module-level `ROUTER`. Pass an
            `LMRouter` with your own `RouterConfig` for explicit keys or
            a custom transport, or `lm15.testing.FakeLM` in tests.
        **kwargs: Extra default request kwargs applied to every call.
            Universal ones (`top_p`, `stop`, ...) become lm15 `Config`
            fields; anything else travels in `Config.extensions`.

    Examples:
        ```python
        lm = dspy.LM("gpt-4o-mini", native_fc=True)
        dspy.configure(lm=lm)
        outputs = lm(prompt="Say hello.")
        ```
    """

    def __init__(
        self,
        model: str,
        *,
        instruct: bool = True,
        native_reasoning: bool = False,
        native_fc: bool = False,
        native_citations: bool = False,
        image_input: bool = False,
        temperature: float | None = None,
        max_tokens: int | None = None,
        router: Any | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.router = router if router is not None else default_router()
        self.capabilities = LMCapabilities(
            instruct=instruct,
            native_reasoning=native_reasoning,
            native_fc=native_fc,
            native_citations=native_citations,
            image_input=image_input,
        )
        defaults = {"temperature": temperature, "max_tokens": max_tokens}
        self.kwargs: dict[str, Any] = {k: v for k, v in defaults.items() if v is not None} | kwargs
        self.history: list[dict[str, Any]] = []

    def __call__(
        self,
        *inputs: Any,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> list[str] | Response:
        """Run one chat completion.

        Two faces, split by how the input arrives:

        - **Typed (positional)** — strings, `dspy.System`/`dspy.User`/
          `dspy.Assistant`/`dspy.ToolResult` messages, and previous
          `lm15.Response` values (folded in as their assistant turn).
          Returns the canonical `lm15.Response`: `.text`, `.tool_calls`,
          `.usage`, `.citations`, `.provider_data`.
        - **Legacy (keyword)** — `messages=[{"role": ..., "content": ...}]`
          or `prompt="..."` (also accepted as a single positional list).
          Returns one string per choice, the adapter convenience.

        Args:
            *inputs: Typed conversation items (typed face), or one
                list of role/content dicts (legacy face).
            messages: Chat messages, `[{"role": ..., "content": ...}, ...]`.
                Roles `user`, `assistant`, and `developer` become lm15
                messages; `system` messages fold into the request's
                system prompt. Content must be text.
            prompt: Sugar for a single user message; exclusive with the rest.
            **kwargs: Per-call request kwargs; override the constructor defaults.

        Raises:
            LMError: On any routing, transport, or provider failure, and
                (legacy face) on a response with no text content. The
                message carries lm15's canonical error code.

        Examples:
            ```python
            outputs = lm(prompt="Say hello.")          # -> list[str]
            response = lm("Say hello.")                # -> lm15.Response
            follow = lm(
                dspy.System("Be concise."),
                dspy.User("What is DSPy?"),
                response,
                dspy.User("Shorter."),
            )
            print(follow.text, follow.usage)
            ```
        """
        typed = bool(inputs) and not (len(inputs) == 1 and isinstance(inputs[0], list))
        if typed:
            if messages is not None or prompt is not None:
                raise ValueError("Pass typed items positionally OR `messages=`/`prompt=`, not both.")
            return self._call_typed(inputs, kwargs)
        if inputs:
            if messages is not None:
                raise ValueError("Pass `messages` positionally or by keyword, not both.")
            messages = inputs[0]
        messages = self._coerce_messages(messages, prompt)
        request_kwargs = {**self.kwargs, **kwargs}
        outputs = self._request(messages, request_kwargs)
        self.history.append(
            {
                "model": self.model,
                "messages": messages,
                "kwargs": request_kwargs,
                "outputs": outputs,
                "timestamp": time.time(),
            }
        )
        return outputs

    def complete(self, request: Request) -> Response:
        """Run one canonical lm15 request — the full-surface escape hatch.

        Use this when the dict-in/strings-out convenience is too small:
        tool calls, image parts, citations, usage accounting, or
        provider extensions. The request routes through this LM's
        router; errors map to the same typed `LMError`.

        Args:
            request: A complete `lm15.Request`, including the model string.

        Returns:
            The canonical `lm15.Response`.

        Raises:
            LMError: On any routing, transport, or provider failure.
        """
        try:
            return self._complete(self.router, request)
        except LM15Error as e:
            raise self._wrap(e) from e

    def stream(
        self,
        *inputs: Any,
        messages: list[dict[str, Any]] | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> ResponseStream:
        """Run one completion as a stream — same inputs as `lm(...)`.

        A separate method, never a flag: `stream()` accepts every input
        face `__call__` accepts and always returns an lm15
        `ResponseStream`. Iterate it for text as it arrives, iterate
        `.events()` for the canonical typed events, then read
        `.response` (or `.text`, `.usage`, ...) for the same `Response`
        a buffered call returns. Engines that cannot stream natively
        replay the finished response as events — one vocabulary, no
        branching on the backend. History records once, when the
        stream completes, exactly as a buffered call.

        Raises:
            LMError: On routing/transport failures at call time, and
                during iteration when the provider streams an error.

        Examples:
            ```python
            stream = lm.stream("Write a haiku about rivers.")
            for text in stream:
                print(text, end="", flush=True)
            print(stream.response.usage)
            ```
        """
        typed = bool(inputs) and not (len(inputs) == 1 and isinstance(inputs[0], list))
        if typed:
            if messages is not None or prompt is not None:
                raise ValueError("Pass typed items positionally OR `messages=`/`prompt=`, not both.")
            items = inputs
        else:
            if inputs:
                if messages is not None:
                    raise ValueError("Pass `messages` positionally or by keyword, not both.")
                messages = inputs[0]
            items = tuple(self._dict_to_typed(m) for m in self._coerce_messages(messages, prompt))

        try:
            engine, request, request_kwargs = self._prepare_typed(items, kwargs)
        except LM15Error as e:
            raise self._wrap(e) from e

        def record(response: Response) -> None:
            self._record_typed(request, request_kwargs, response)

        return _RecordingStream(self._guard_events(engine, request), request, record)

    def _call_typed(self, items: tuple[Any, ...], kwargs: dict[str, Any]) -> Response:
        """The typed positional face: lm15 messages in, `Response` out."""
        try:
            engine, request, request_kwargs = self._prepare_typed(items, kwargs)
            response = self._complete(engine, request)
        except LM15Error as e:
            raise self._wrap(e) from e

        self._record_typed(request, request_kwargs, response)
        return response

    def _prepare_typed(
        self, items: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[Any, Request, dict[str, Any]]:
        """Build the canonical request (and pick the engine) for one typed call."""
        request_kwargs = {**self.kwargs, **kwargs}
        endpoint = {k: request_kwargs.pop(k) for k in _ENDPOINT_FIELDS if request_kwargs.get(k) is not None}

        system_texts: list[str] = []
        lm15_messages: list[Message] = []
        for item in items:
            if isinstance(item, str):
                lm15_messages.append(Message.user(item))
            elif isinstance(item, System):
                system_texts.append(item.text)
            elif isinstance(item, Message):
                lm15_messages.append(item)
            elif isinstance(item, Response):
                lm15_messages.append(item.message)
            else:
                raise TypeError(
                    f"Typed lm(...) items must be str, dspy.System/User/Assistant/"
                    f"ToolResult, lm15.Message, or lm15.Response — got {type(item).__name__}. "
                    "A dspy.ToolCall goes inside dspy.Assistant(...)."
                )

        engine, wire_model = self._engine_for(endpoint)
        request = Request(
            model=wire_model,
            messages=tuple(lm15_messages),
            system="\n\n".join(system_texts) or None,
            config=self._config(request_kwargs),
        )
        return engine, request, request_kwargs

    def _dict_to_typed(self, message: dict[str, Any]) -> Any:
        """Lift one legacy role/content dict into the typed vocabulary."""
        role, content = message.get("role"), message.get("content")
        if not isinstance(content, str):
            raise ValueError(f"Message content must be text, got {type(content).__name__}.")
        if role == "system":
            return System(content)
        if role in ("user", "assistant", "developer"):
            return Message(role=role, parts=(TextPart(content),))
        raise ValueError(
            f"Unsupported message role {role!r}. The chat convenience takes "
            "system/user/assistant/developer text messages."
        )

    def _record_typed(self, request: Request, request_kwargs: dict[str, Any], response: Response) -> None:
        self.history.append(
            {
                "model": self.model,
                "messages": list(request.messages),
                "kwargs": request_kwargs,
                "outputs": [response.text] if response.text is not None else [],
                "response": response,
                "timestamp": time.time(),
            }
        )

    def _complete(self, engine: Any, request: Request) -> Response:
        """The buffered transport seam: every non-stream path funnels here."""
        return engine.complete(request)

    def _stream(self, engine: Any, request: Request) -> Any:
        """The streaming transport seam (DummyLM replays its script)."""
        return engine.stream(request)

    def _guard_events(self, engine: Any, request: Request) -> Any:
        """Yield stream events; map provider failures to the typed `LMError`."""
        try:
            for event in self._stream(engine, request):
                if event.type == "error":
                    raise LMError(
                        f"LM stream for {self.model!r} failed "
                        f"({event.error.code}): {event.error.message}"
                    )
                yield event
        except LM15Error as e:
            raise self._wrap(e) from e

    def _coerce_messages(self, messages: list[dict[str, Any]] | None, prompt: str | None) -> list[dict[str, Any]]:
        if (messages is None) == (prompt is None):
            raise ValueError("Pass exactly one of `messages` or `prompt`.")
        if prompt is not None:
            return [{"role": "user", "content": prompt}]
        return messages

    def _request(self, messages: list[dict[str, Any]], request_kwargs: dict[str, Any]) -> list[str]:
        request_kwargs = dict(request_kwargs)
        endpoint = {k: request_kwargs.pop(k) for k in _ENDPOINT_FIELDS if request_kwargs.get(k) is not None}
        try:
            engine, wire_model = self._engine_for(endpoint)
            request = self._build_request(messages, request_kwargs, model=wire_model)
            response = self._complete(engine, request)
        except LM15Error as e:
            raise self._wrap(e) from e

        text = response.text
        if text is None:
            raise LMError(
                f"LM {self.model!r} returned no text content "
                f"(finish_reason={response.finish_reason!r}). The dict-in/strings-out "
                "call speaks text-only chat; for tool calls and richer parts use "
                "`lm.complete(lm15.Request(...))`."
            )
        return [text]

    def _engine_for(self, endpoint: dict[str, Any]) -> tuple[Any, str]:
        """Pick the engine and wire model for one call.

        No endpoint config — or an explicitly passed router, which always
        wins: the router as-is, routing `self.model`. Otherwise
        `api_key` alone builds a private router that binds the key to
        the resolved provider, and `api_base`/`base_url` pins the
        RESOLVED provider's adapter at that URL — the endpoint moves,
        the dialect does not. `"anthropic:claude-..."` with `api_base`
        speaks Anthropic to your URL; an unresolvable bare model with
        `api_base` defaults to an OpenAI-compatible chat server, the
        self-hosted convention.
        """
        if not endpoint or self.router is not default_router():
            return self.router, self.model

        cache_key = tuple(sorted(endpoint.items()))
        cached = getattr(self, "_pinned", {}).get(cache_key)
        if cached is not None:
            return cached

        base_url = endpoint.get("api_base") or endpoint.get("base_url")
        api_key = endpoint.get("api_key")
        try:
            resolution = default_router().resolve(self.model)
        except LM15Error:
            resolution = None

        if base_url is not None:
            from lm15.router import ADAPTERS, CHAT_PRESET_ROUTES

            # lm15's transport honors HTTP(S)_PROXY/NO_PROXY, so a
            # mandated proxy (e.g. the FlexIR egress broker) is respected.
            extra: dict[str, Any] = {}
            if resolution is None:
                cls, wire_model = ADAPTERS["openai-chat"], self.model
            else:
                wire_model = resolution.model
                cls = ADAPTERS.get(resolution.provider)
                if cls is None:  # chat-compat preset (ollama, vllm, ...)
                    cls = ADAPTERS["openai-chat"]
                    route = CHAT_PRESET_ROUTES.get(resolution.provider)
                    if route is not None:
                        extra["compat"] = route.provider
            engine: Any = cls(api_key=api_key or "", base_url=base_url, **extra)
        else:
            provider = resolution.provider if resolution is not None else self.model
            engine = LMRouter(RouterConfig(api_keys={provider: api_key}))
            wire_model = self.model  # the private router re-resolves it

        pinned = self.__dict__.setdefault("_pinned", {})
        pinned[cache_key] = (engine, wire_model)
        return engine, wire_model

    def _build_request(
        self, messages: list[dict[str, Any]], request_kwargs: dict[str, Any], *, model: str | None = None
    ) -> Request:
        system_texts: list[str] = []
        lm15_messages: list[Message] = []
        for m in messages:
            role, content = m.get("role"), m.get("content")
            if not isinstance(content, str):
                raise ValueError(
                    f"Message content must be text, got {type(content).__name__}. For "
                    "images, tool results, or other parts, build an lm15.Request and "
                    "call `lm.complete(request)`."
                )
            if role == "system":
                system_texts.append(content)
            elif role in ("user", "assistant", "developer"):
                lm15_messages.append(Message(role=role, parts=(TextPart(content),)))
            else:
                raise ValueError(
                    f"Unsupported message role {role!r}. The chat convenience takes "
                    "system/user/assistant/developer text messages."
                )

        return Request(
            model=model if model is not None else self.model,
            messages=tuple(lm15_messages),
            system="\n\n".join(system_texts) or None,
            config=self._config(request_kwargs),
        )

    def _config(self, request_kwargs: dict[str, Any]) -> Config:
        """Split kwargs into universal `Config` fields and provider extensions."""
        config_kwargs = {k: v for k, v in request_kwargs.items() if k in _CONFIG_FIELDS}
        extensions = {k: v for k, v in request_kwargs.items() if k not in _CONFIG_FIELDS}
        if extensions:
            config_kwargs["extensions"] = extensions
        return Config(**config_kwargs)

    def _wrap(self, e: LM15Error) -> LMError:
        return LMError(f"LM request to {self.model!r} failed ({e.code}): {e}")

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, capabilities={self.capabilities})"


class _RecordingStream(ResponseStream):
    """An lm15 `ResponseStream` that records LM history once, on clean completion.

    Streamed and buffered calls are observationally identical afterward:
    the record lands through the same shape as a buffered typed call,
    and only when the stream finished without failure.
    """

    def __init__(self, events: Any, request: Request, record: Any) -> None:
        super().__init__(events, request)
        self._record = record
        self._recorded = False

    def events(self) -> Any:
        yield from super().events()
        # Reached only when the stream completed cleanly — a failure
        # propagates out of the `yield from` and skips the record.
        if not self._recorded:
            self._recorded = True
            self._record(self.response)
