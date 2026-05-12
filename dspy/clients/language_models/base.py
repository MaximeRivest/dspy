"""Normalized language model contract for DSPy."""

from __future__ import annotations

import copy
import datetime
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator

from typing_extensions import Self

from dspy.clients.language_models.features import FeatureStatus, LMFeatureReporter, feature_status
from dspy.dsp.utils import settings
from dspy.utils.callback import ACTIVE_CALL_ID
from dspy.utils.exceptions import LMError
from dspy.utils.inspect_history import pretty_print_history

if TYPE_CHECKING:
    from dspy.clients.language_models.types import (
        AsyncLMStream,
        LMRequest,
        LMResponse,
        LMStream,
        LMStreamEvent,
    )
    from dspy.utils.callback import BaseCallback


MAX_HISTORY_SIZE = 10_000
GLOBAL_LANGUAGE_MODEL_HISTORY: list[Mapping[str, Any]] = []

logger = logging.getLogger(__name__)

_REQUEST_FEATURE_HOOKS = (
    ("text", "map_request_text", "map_request_text() is implemented."),
    ("input_image", "map_request_input_image", "map_request_input_image() is implemented."),
    ("input_audio", "map_request_input_audio", "map_request_input_audio() is implemented."),
    ("input_file", "map_request_input_file", "map_request_input_file() is implemented."),
    ("tools", "map_request_tools", "map_request_tools() is implemented."),
    ("tool_choice", "map_request_tool_choice", "map_request_tool_choice() is implemented."),
    ("assistant_tool_calls", "map_request_assistant_tool_calls", "map_request_assistant_tool_calls() is implemented."),
    ("tool_results", "map_request_tool_results", "map_request_tool_results() is implemented."),
    ("response_schema", "map_request_response_schema", "map_request_response_schema() is implemented."),
    ("reasoning_config", "map_request_reasoning_config", "map_request_reasoning_config() is implemented."),
    ("prompt_cache", "map_request_prompt_cache", "map_request_prompt_cache() is implemented."),
    ("logprobs", "map_request_logprobs", "map_request_logprobs() is implemented."),
    ("multiple_outputs", "map_request_multiple_outputs", "map_request_multiple_outputs() is implemented."),
    ("provider_extensions", "map_request_provider_extensions", "map_request_provider_extensions() is implemented."),
)

_RESPONSE_FEATURE_HOOKS = (
    ("text", "map_response_text", "map_response_text() is implemented."),
    ("reasoning", "map_response_reasoning", "map_response_reasoning() is implemented."),
    ("tool_calls", "map_response_tool_calls", "map_response_tool_calls() is implemented."),
    ("citations", "map_response_citations", "map_response_citations() is implemented."),
    ("output_image", "map_response_output_image", "map_response_output_image() is implemented."),
    ("output_audio", "map_response_output_audio", "map_response_output_audio() is implemented."),
    ("output_file", "map_response_output_file", "map_response_output_file() is implemented."),
    ("refusal", "map_response_refusal", "map_response_refusal() is implemented."),
)


@dataclass(frozen=True)
class LMCapabilities:
    """The model features a language model can use for its current backend.

    Capabilities are intentionally descriptive, not prescriptive. They tell
    adapters and modules which native paths are worth trying. A concrete
    backend still owns the final request validation because deployments can
    differ even when model names look the same.

    Attributes:
        function_calling: Whether the model can return native tool calls.
        reasoning: Whether the model can return native reasoning content.
        response_schema: Whether the model can enforce a structured response
            schema, such as JSON schema or a Pydantic model.
        streaming: Whether the model can stream normalized DSPy events.
        input_image: Whether request messages may contain image parts.
        input_audio: Whether request messages may contain audio parts.
        input_file: Whether request messages may contain file parts.
        output_image: Whether responses may contain generated image parts.
        output_audio: Whether responses may contain generated audio parts.
        tool_results: Whether request messages may contain tool-result parts.
        extensions: Extra backend-specific capability flags.
    """

    function_calling: bool = False
    reasoning: bool = False
    response_schema: bool = False
    streaming: bool = False
    input_image: bool = False
    input_audio: bool = False
    input_file: bool = False
    output_image: bool = False
    output_audio: bool = False
    tool_results: bool = False
    extensions: dict[str, Any] = field(default_factory=dict)


class LanguageModel:
    """Call a language model with a normalized DSPy request.

    Subclass `LanguageModel` when you want a model implementation to behave
    like a native DSPy LM. Users call the object with friendly inputs such as
    strings, `dspy.User(...)` messages, or an `LMRequest`. The base class
    normalizes those inputs, calls `forward(request)`, records history, and
    returns an `LMResponse`.

    Concrete subclasses implement `forward()`. They translate `LMRequest` into
    their provider or runtime format, run the model, and return `LMResponse`.

    Args:
        model: The model name or deployment identifier used by this LM.
        cache: Whether this LM should use DSPy's cache by default.
        callbacks: Optional callbacks attached to this LM instance. Callback
            execution is wired by DSPy's callback layer when this class is
            exported as a public LM type.
        **kwargs: Default request configuration, such as `temperature`,
            `max_tokens`, or provider-specific values.

    Examples:
        Define a small custom LM:
        ```python
        import dspy


        class EchoLM(dspy.LanguageModel):
            def forward(self, request: dspy.LMRequest) -> dspy.LMResponse:
                return dspy.LMResponse.from_text("hello", model=request.model)


        lm = EchoLM(model="test/echo")
        response = lm("Say hello")
        print(response.text)
        ```

    See Also:
        [`dspy.LM`][dspy.LM]
        [`dspy.LMRequest`][dspy.LMRequest]
        [`dspy.LMResponse`][dspy.LMResponse]
    """

    def __init__(
        self,
        model: str,
        *,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        **kwargs: Any,
    ):
        self.model = model
        self.cache = cache
        self.callbacks = callbacks or []
        self.kwargs = dict(kwargs)
        self.history: list[dict[str, Any]] = []
        self.features = LMFeatureReporter(self)

    # ---------------------------------------------------------------------
    # Model and deployment metadata
    # ---------------------------------------------------------------------

    @property
    def capabilities(self) -> LMCapabilities:
        """The native features available for this model instance."""
        return self.get_capabilities()

    def get_capabilities(self) -> LMCapabilities:
        """Return the native features available for this model instance.

        Subclasses should override this when capability checks depend on the
        model name, provider, deployment, or runtime configuration.
        """
        return LMCapabilities()

    # ---------------------------------------------------------------------
    # Core execution hooks for concrete LanguageModel implementations
    # ---------------------------------------------------------------------

    def forward(self, request: LMRequest) -> LMResponse:
        """Run one normalized language model request.

        Subclasses must implement this method. It should translate the
        normalized request into the backend's format, call the model, and return
        a normalized `LMResponse`.

        Args:
            request: The normalized request to run.

        Returns:
            The normalized response returned by the model.
        """
        raise NotImplementedError("Subclasses must implement forward(request).")

    async def aforward(self, request: LMRequest) -> LMResponse:
        """Run one normalized language model request asynchronously.

        Subclasses that support native async inference should override this
        method. The default raises `NotImplementedError` rather than silently
        running sync inference in an event loop.
        """
        raise NotImplementedError("Subclasses must implement aforward(request) for async calls.")

    def forward_stream(self, request: LMRequest) -> Iterator[LMStreamEvent]:
        """Run one normalized language model request as a stream of events."""
        raise NotImplementedError(f"{type(self).__name__} does not support streaming.")

    async def aforward_stream(self, request: LMRequest) -> AsyncIterator[LMStreamEvent]:
        """Run one normalized language model request as an async stream of events."""
        raise NotImplementedError(f"{type(self).__name__} does not support async streaming.")

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        """Map a provider exception to a DSPy exception.

        Subclasses should override this when their provider raises native
        exceptions for conditions DSPy understands, such as context-window
        failures. The default returns the original exception unchanged.
        """
        return error

    # ---------------------------------------------------------------------
    # Normalized request mapping hooks
    # ---------------------------------------------------------------------

    def map_request_text(self, value: Any) -> Any:
        """Map request text to a provider request value."""
        raise NotImplementedError

    def map_request_input_image(self, value: Any) -> Any:
        """Map an `LMImagePart` request part to a provider request value."""
        raise NotImplementedError

    def map_request_input_audio(self, value: Any) -> Any:
        """Map an `LMAudioPart` request part to a provider request value."""
        raise NotImplementedError

    def map_request_input_file(self, value: Any) -> Any:
        """Map an `LMFilePart` request part to a provider request value."""
        raise NotImplementedError

    def map_request_tools(self, value: Any) -> Any:
        """Map an `LMToolSpec` request value to a provider request value."""
        raise NotImplementedError

    def map_request_tool_choice(self, value: Any) -> Any:
        """Map `LMConfig.tool_choice` to a provider request value."""
        raise NotImplementedError

    def map_request_assistant_tool_calls(self, value: Any) -> Any:
        """Map an assistant `LMToolCallPart` to a provider request value."""
        raise NotImplementedError

    def map_request_tool_results(self, value: Any) -> Any:
        """Map an `LMToolResultPart` to a provider request value."""
        raise NotImplementedError

    def map_request_response_schema(self, value: Any) -> Any:
        """Map `LMConfig.response_format` to a provider request value."""
        raise NotImplementedError

    def map_request_reasoning_config(self, value: Any) -> Any:
        """Map `LMConfig.reasoning` to a provider request value."""
        raise NotImplementedError

    def map_request_prompt_cache(self, value: Any) -> Any:
        """Map provider-side prompt/token cache controls to a provider request value."""
        raise NotImplementedError

    def map_request_logprobs(self, value: Any) -> Any:
        """Map `LMConfig.logprobs` to a provider request value."""
        raise NotImplementedError

    def map_request_multiple_outputs(self, value: Any) -> Any:
        """Map `LMConfig.n` to a provider request value."""
        raise NotImplementedError

    def map_request_provider_extensions(self, value: Any) -> Any:
        """Map `LMConfig.extensions` to provider request values."""
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Normalized response mapping hooks
    # ---------------------------------------------------------------------

    def map_response_text(self, value: Any) -> Any:
        """Map provider text output to an `LMTextPart`."""
        raise NotImplementedError

    def map_response_reasoning(self, value: Any) -> Any:
        """Map provider reasoning output to an `LMThinkingPart`."""
        raise NotImplementedError

    def map_response_tool_calls(self, value: Any) -> Any:
        """Map provider tool-call output to an `LMToolCallPart`."""
        raise NotImplementedError

    def map_response_citations(self, value: Any) -> Any:
        """Map provider citation output to an `LMCitationPart`."""
        raise NotImplementedError

    def map_response_output_image(self, value: Any) -> Any:
        """Map provider image output to an `LMImagePart`."""
        raise NotImplementedError

    def map_response_output_audio(self, value: Any) -> Any:
        """Map provider audio output to an `LMAudioPart`."""
        raise NotImplementedError

    def map_response_output_file(self, value: Any) -> Any:
        """Map provider file output to an `LMFilePart`."""
        raise NotImplementedError

    def map_response_refusal(self, value: Any) -> Any:
        """Map provider refusal output to an `LMRefusalPart`."""
        raise NotImplementedError

    # ---------------------------------------------------------------------
    # Implementation support introspection
    # ---------------------------------------------------------------------

    def get_request_feature_statuses(self) -> dict[str, FeatureStatus]:
        """Infer normalized request shapes this LM implementation supports."""
        return {
            name: self._feature_status_from_hook(f"request.{name}", hook, supported_evidence=evidence)
            for name, hook, evidence in _REQUEST_FEATURE_HOOKS
        }

    def get_response_feature_statuses(self) -> dict[str, FeatureStatus]:
        """Infer normalized response shapes this LM implementation supports."""
        statuses = {
            name: self._feature_status_from_hook(f"response.{name}", hook, supported_evidence=evidence)
            for name, hook, evidence in _RESPONSE_FEATURE_HOOKS
        }
        statuses["usage"] = feature_status(
            "response.usage", "unknown", "No LMResponse with usage has been observed yet."
        )
        statuses["cost"] = feature_status("response.cost", "unknown", "No LMResponse with cost has been observed yet.")
        return statuses

    # ---------------------------------------------------------------------
    # State, serialization, and lifecycle hooks
    # ---------------------------------------------------------------------

    def dump_state(self) -> dict[str, Any]:
        """Return a sanitized reconstruction state for this LM.

        The default state works for simple constructor-only LMs. Subclasses with
        clients, local weights, auth objects, or other runtime resources should
        override this method and pair it with `load_state()`.
        """
        filtered_kwargs = {
            key: value for key, value in self.kwargs.items() if not key.startswith("api_") and key != "api_key"
        }
        return {
            "model": self.model,
            "cache": self.cache,
            **filtered_kwargs,
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Self:
        """Reconstruct this LM from `dump_state()` output."""
        return cls(**state)

    def copy(self, **overrides: Any) -> Self:
        """Return a copy of this LM with updated inference defaults.

        Runtime observations such as history are reset on the copy. Subclasses
        that hold non-copyable clients, sessions, model weights, or device
        handles should override this method and preserve the same semantics.
        """
        new_instance = copy.deepcopy(self)
        new_instance.history = []
        new_instance.features = LMFeatureReporter(new_instance)

        for key, value in overrides.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            if key in getattr(new_instance, "kwargs", {}) or not hasattr(self, key):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value

        return new_instance

    # ---------------------------------------------------------------------
    # Request support validation
    # ---------------------------------------------------------------------

    def validate_request(self, request: LMRequest):
        """Return whether this implementation can map every shape used by `request`.

        This is an implementation-support check, not a model-capability check.
        It answers whether this `LanguageModel` subclass can faithfully
        translate the normalized DSPy request shape without silently dropping
        fields. Use `capabilities` for model- or deployment-specific feature
        hints such as whether a particular model can use tools or images.
        """
        return self.features.validate_request(request)

    def _feature_status_from_hook(self, name: str, hook_name: str, *, supported_evidence: str) -> FeatureStatus:
        if name in {"request.text", "response.text"} and self._method_overridden("forward"):
            return feature_status(name, "inferred", "forward() is implemented for text generation.")
        if self._method_overridden(hook_name):
            return feature_status(name, "inferred", supported_evidence)
        return feature_status(name, "unsupported", f"{hook_name}() is not implemented.")

    def _method_overridden(self, method_name: str) -> bool:
        method = getattr(type(self), method_name, None)
        base_method = getattr(LanguageModel, method_name, None)
        return method is not None and base_method is not None and method is not base_method

    def supports_request(self, request: LMRequest) -> bool:
        """Return whether this LM supports every shape used by `request`."""
        return bool(self.validate_request(request))

    def require_request_support(self, request: LMRequest) -> None:
        """Raise if this LM cannot map every shape used by `request`."""
        support = self.validate_request(request)
        if support:
            return
        from dspy.utils.exceptions import LMUnsupportedFeatureError

        raise LMUnsupportedFeatureError(
            "This LanguageModel does not support the normalized request shape.",
            model=self.model,
            features=[issue.feature for issue in support.issues],
            issues=[issue.message for issue in support.issues],
        )

    # ---------------------------------------------------------------------
    # Public call API
    # ---------------------------------------------------------------------

    def __call__(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMResponse:
        """Normalize a direct LM call and return an `LMResponse`.

        Args:
            *items: Positional content such as text, `dspy.Image`, message
                constructors like `dspy.User(...)`, or an `LMRequest`.
            prompt: Optional text prompt keyword.
            messages: Optional chat messages accepted at the public boundary
                and normalized immediately.
            request: An explicit normalized request.
            **kwargs: Request configuration that overrides this LM's defaults.

        Returns:
            An `LMResponse` with normalized outputs, usage, cost, cache status,
            and provider metadata when available.
        """
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        call_id = self._start_lm_callbacks(
            callbacks,
            request=normalized_request,
            raw_inputs=self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs),
        )
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        result = None
        exception = None
        try:
            self.require_request_support(normalized_request)
            response = self._forward_with_cache(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def acall(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMResponse:
        """Normalize an async LM call and return an `LMResponse`."""
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        call_id = self._start_lm_callbacks(
            callbacks,
            request=normalized_request,
            raw_inputs=self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs),
        )
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        result = None
        exception = None
        try:
            self.require_request_support(normalized_request)
            response = await self._aforward_with_cache(normalized_request)
            result = self._finalize_response(normalized_request, response)
            return result
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, normalized_request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    def normalize_request(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMRequest:
        """Normalize public call inputs into an `LMRequest`.

        This method is deliberately small and delegates most parsing to
        `LMRequest`. The request type owns the content vocabulary; the language
        model owns defaults such as `model`, `cache`, and generation kwargs.
        """
        lm_types = _import_lm_types()
        request_cls = lm_types.LMRequest

        if request is None and items and isinstance(items[0], request_cls):
            request = items[0]
            items = items[1:]

        if request is not None:
            if prompt is not None or messages is not None or items:
                raise ValueError(
                    "Pass either an LMRequest or direct-call inputs, not both. "
                    "Use call kwargs to override request config."
                )
            return self._override_request(request, **kwargs)

        merged_kwargs = {**self.kwargs, **kwargs}
        merged_kwargs.setdefault("cache", self.cache)

        if hasattr(request_cls, "from_call"):
            return request_cls.from_call(
                model=self.model,
                items=items,
                prompt=prompt,
                messages=messages,
                **merged_kwargs,
            )

        if hasattr(request_cls, "from_prompt_or_messages"):
            if items:
                raise TypeError(
                    "Positional LM items require LMRequest.from_call(). "
                    "Implement that constructor in dspy.clients.language_models.types."
                )
            return request_cls.from_prompt_or_messages(
                model=self.model,
                prompt=prompt,
                messages=messages,
                **merged_kwargs,
            )

        raise TypeError("LMRequest must define from_call() or from_prompt_or_messages().")

    def stream(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> LMStream:
        """Normalize a direct LM call and return a stream with `.result()`."""
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        try:
            self.require_request_support(normalized_request)
            self._require_stream_support(async_=False)
        except Exception as error:
            self._observe_failed_stream_construction(
                normalized_request,
                error,
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            )
            raise
        lm_types = _import_lm_types()
        return lm_types.LMStream(
            request=normalized_request,
            events=self._callback_wrapped_stream_events(
                normalized_request,
                self.forward_stream(normalized_request),
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    def astream(
        self,
        *items: Any,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        request: LMRequest | None = None,
        **kwargs: Any,
    ) -> AsyncLMStream:
        """Normalize a direct LM call and return an async stream with `.result()`.

        `astream()` returns an async iterator directly, so callers can write
        `async for event in lm.astream(...)` without first awaiting the stream.
        """
        normalized_request = self.normalize_request(
            *items,
            prompt=prompt,
            messages=messages,
            request=request,
            **kwargs,
        )
        callbacks = self._get_active_callbacks()
        raw_inputs = self._raw_callback_inputs(items=items, prompt=prompt, messages=messages, kwargs=kwargs)
        try:
            self.require_request_support(normalized_request)
            self._require_stream_support(async_=True)
        except Exception as error:
            self._observe_failed_stream_construction(
                normalized_request,
                error,
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            )
            raise
        lm_types = _import_lm_types()
        return lm_types.AsyncLMStream(
            request=normalized_request,
            events=self._callback_wrapped_astream_events(
                normalized_request,
                self.aforward_stream(normalized_request),
                callbacks=callbacks,
                raw_inputs=raw_inputs,
            ),
            finalize=self._finalize_response,
        )

    # ---------------------------------------------------------------------
    # Callback machinery
    # ---------------------------------------------------------------------

    def _get_active_callbacks(self) -> list[BaseCallback]:
        """Return global and instance callbacks for this LM call."""
        return list(settings.get("callbacks", []) or []) + list(getattr(self, "callbacks", []) or [])

    def _raw_callback_inputs(
        self,
        *,
        items: tuple[Any, ...],
        prompt: str | None,
        messages: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        raw: dict[str, Any] = {"items": items, "prompt": prompt, "messages": messages, "kwargs": kwargs}
        return _sanitize_callback_value(raw)

    def _start_lm_callbacks(
        self,
        callbacks: list[BaseCallback],
        *,
        request: LMRequest,
        raw_inputs: dict[str, Any],
    ) -> str | None:
        if not callbacks:
            return None

        call_id = uuid.uuid4().hex
        inputs = {"request": _sanitize_lm_request_for_callbacks(request), "raw": raw_inputs}
        for callback in callbacks:
            try:
                callback.on_lm_start(call_id=call_id, instance=self, inputs=inputs)
            except Exception as error:
                logger.warning("Error when calling callback %s: %s", callback, error)
        return call_id

    def _end_lm_callbacks(
        self,
        callbacks: list[BaseCallback],
        *,
        call_id: str | None,
        outputs: LMResponse | None,
        exception: Exception | None,
    ) -> None:
        if not callbacks or call_id is None:
            return

        for callback in callbacks:
            try:
                callback.on_lm_end(call_id=call_id, outputs=outputs, exception=exception)
            except Exception as error:
                logger.warning("Error when applying callback %s's LM end handler: %s", callback, error)

    def _observe_failed_stream_construction(
        self,
        request: LMRequest,
        error: Exception,
        *,
        callbacks: list[BaseCallback],
        raw_inputs: dict[str, Any],
    ) -> None:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)
        try:
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=None, exception=error)
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)

    def _callback_wrapped_stream_events(
        self,
        request: LMRequest,
        events: Iterator[LMStreamEvent],
        *,
        callbacks: list[BaseCallback],
        raw_inputs: dict[str, Any],
    ) -> Iterator[LMStreamEvent]:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        builder = _import_lm_types().LMOutputBuilder()
        result = None
        exception = None
        try:
            for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    async def _callback_wrapped_astream_events(
        self,
        request: LMRequest,
        events: AsyncIterator[LMStreamEvent],
        *,
        callbacks: list[BaseCallback],
        raw_inputs: dict[str, Any],
    ) -> AsyncIterator[LMStreamEvent]:
        call_id = self._start_lm_callbacks(callbacks, request=request, raw_inputs=raw_inputs)
        parent_call_id = ACTIVE_CALL_ID.get()
        if call_id is not None:
            ACTIVE_CALL_ID.set(call_id)

        builder = _import_lm_types().LMOutputBuilder()
        result = None
        exception = None
        try:
            async for event in events:
                built = builder.apply(event)
                if built is not None:
                    result = built
                yield event
        except Exception as error:
            normalized_error = self._normalize_and_observe_error(error, request)
            exception = normalized_error
            if normalized_error is error:
                raise
            raise normalized_error from error
        finally:
            if call_id is not None:
                ACTIVE_CALL_ID.set(parent_call_id)
            self._end_lm_callbacks(callbacks, call_id=call_id, outputs=result, exception=exception)

    # ---------------------------------------------------------------------
    # History and internal execution machinery
    # ---------------------------------------------------------------------

    def inspect_history(self, n: int = 1, file: Any | None = None) -> None:
        """Print recent LM interactions recorded on this instance."""
        pretty_print_history(self.history, n, file=file)

    def update_history(self, entry: dict[str, Any]) -> None:
        """Append one normalized interaction to DSPy's LM history stores."""
        if settings.disable_history:
            return

        if len(GLOBAL_LANGUAGE_MODEL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_LANGUAGE_MODEL_HISTORY.pop(0)
        GLOBAL_LANGUAGE_MODEL_HISTORY.append(entry)

        if settings.max_history_size != 0:
            if len(self.history) >= settings.max_history_size:
                self.history.pop(0)
            self.history.append(entry)

        for module in settings.caller_modules or []:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)

    def _forward_with_cache(self, request: LMRequest) -> LMResponse:
        if not _request_cache_enabled(request, self.cache):
            return self.forward(request)
        response = _cached_language_model_forward(
            cache_request=self._cache_request(request),
            lm=self,
            request=request,
        )
        return _prepare_cached_lm_response(response)

    async def _aforward_with_cache(self, request: LMRequest) -> LMResponse:
        if not _request_cache_enabled(request, self.cache):
            return await self.aforward(request)
        response = await _cached_language_model_aforward(
            cache_request=self._cache_request(request),
            lm=self,
            request=request,
        )
        return _prepare_cached_lm_response(response)

    def _cache_request(self, request: LMRequest) -> dict[str, Any]:
        return {
            "lm_class": f"{type(self).__module__}.{type(self).__qualname__}",
            "lm_state": _sanitize_cache_value(self.dump_state()),
            "request": _sanitize_cache_value(_model_dump_for_cache(request)),
        }

    def _override_request(self, request: LMRequest, **kwargs: Any) -> LMRequest:
        if not kwargs:
            return request

        if hasattr(request, "with_config_overrides"):
            return request.with_config_overrides(**kwargs)

        if hasattr(request, "model_copy"):
            update = _request_config_update(request, kwargs)
            return request.model_copy(update=update, deep=True)

        raise TypeError("LMRequest overrides require with_config_overrides() or Pydantic model_copy().")

    def _require_stream_support(self, *, async_: bool) -> None:
        feature = self.features.async_streaming if async_ else self.features.streaming
        if feature:
            return
        from dspy.utils.exceptions import LMUnsupportedFeatureError

        name = "async_streaming" if async_ else "streaming"
        raise LMUnsupportedFeatureError(
            f"This LanguageModel implementation does not support {name}.",
            model=self.model,
            features=[name],
            issues=[feature.evidence or f"{name} is not supported."],
        )

    def _finalize_response(self, request: LMRequest, response: LMResponse) -> LMResponse:
        self._observe_response_features(response)
        self._track_usage(response)

        if not settings.disable_history:
            lm_types = _import_lm_types()
            entry = lm_types.LMHistoryEntry(
                request=request,
                response=response,
                timestamp=datetime.datetime.now().isoformat(),
                uuid=str(uuid.uuid4()),
                model_type=getattr(self, "model_type", None),
            )
            self.update_history(entry)

        return response

    def _observe_response_features(self, response: LMResponse) -> None:
        usage = _response_usage_as_dict(response)
        if usage:
            self.features.observe("usage", "Latest LMResponse included usage.")
            self.features.observe("response.usage", "Latest LMResponse included usage.")

        if getattr(response, "cost", None) is not None:
            self.features.observe("cost", "Latest LMResponse included cost.")
            self.features.observe("response.cost", "Latest LMResponse included cost.")

    def _normalize_and_observe_error(self, error: Exception, request: LMRequest) -> Exception:
        normalized_error = self.normalize_error(error, request)
        if isinstance(normalized_error, LMError):
            self.features.observe_error(normalized_error)
        return normalized_error

    def _track_usage(self, response: LMResponse) -> None:
        if getattr(response, "cache_hit", False):
            return
        if not settings.usage_tracker:
            return

        usage = _response_usage_as_dict(response)
        if usage:
            settings.usage_tracker.add_usage(self.model, usage)


def inspect_history(n: int = 1, file: Any | None = None) -> None:
    """Print recent interactions from all normalized language models."""
    pretty_print_history(GLOBAL_LANGUAGE_MODEL_HISTORY, n, file=file)


def _prepare_cached_lm_response(response: Any) -> Any:
    if getattr(response, "cache_hit", False) and hasattr(response, "cost"):
        response.cost = None
    return response


def _cached_language_model_forward(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    def run(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
        return lm.forward(request)

    return run(cache_request=cache_request, lm=lm, request=request)


async def _cached_language_model_aforward(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
    from dspy.clients.cache import request_cache

    @request_cache(cache_arg_name="cache_request", ignored_args_for_cache_key=["lm", "request"])
    async def run(cache_request: dict[str, Any], lm: LanguageModel, request: Any) -> Any:
        return await lm.aforward(request)

    return await run(cache_request=cache_request, lm=lm, request=request)


def _import_lm_types():
    try:
        from dspy.clients.language_models import types as lm_types
    except ImportError as exc:
        raise ImportError(
            "dspy.clients.language_models.types must be implemented before LanguageModel can "
            "normalize calls. Define LMRequest, LMResponse, and related types there."
        ) from exc
    return lm_types


def _request_cache_enabled(request: Any, default: bool) -> bool:
    config = getattr(request, "config", None)
    cache = getattr(config, "cache", None)
    enabled = getattr(cache, "enabled", None)
    return default if enabled is None else bool(enabled)


def _model_dump_for_cache(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return value


def _sanitize_cache_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if key_text in {"api_key", "api_base", "base_url", "authorization", "x_api_key"}:
                continue
            if key_text.startswith("api_"):
                continue
            sanitized[key] = _sanitize_cache_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_cache_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_cache_value(item) for item in value]
    return value


def _request_config_update(request: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    config = getattr(request, "config", None)
    if config is None:
        return {"config": kwargs}
    if hasattr(config, "model_copy"):
        return {"config": config.model_copy(update=kwargs, deep=True)}
    if isinstance(config, dict):
        return {"config": {**config, **kwargs}}
    raise TypeError("Cannot override config on this LMRequest object.")


def _response_usage_as_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "usage_as_dict"):
        return response.usage_as_dict()

    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return dict(usage)


def _sanitize_lm_request_for_callbacks(request: Any) -> Any:
    config = getattr(request, "config", None)
    if config is not None and hasattr(config, "model_copy"):
        config = config.model_copy(
            update={"extensions": _sanitize_callback_value(getattr(config, "extensions", {}) or {})},
            deep=True,
        )
        return request.model_copy(
            update={
                "config": config,
                "metadata": _sanitize_callback_value(getattr(request, "metadata", {}) or {}),
            },
            deep=True,
        )
    return _sanitize_callback_value(request)


def _sanitize_callback_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        sanitized = {}
        for key, item in value.items():
            key_text = str(key).lower().replace("-", "_")
            if key_text == "api_key" or key_text.startswith("api_") or key_text in {"authorization", "x_api_key"}:
                sanitized[key] = "<redacted>"
            else:
                sanitized[key] = _sanitize_callback_value(item)
        return sanitized
    if isinstance(value, tuple):
        return tuple(_sanitize_callback_value(item) for item in value)
    if isinstance(value, list):
        return [_sanitize_callback_value(item) for item in value]
    return value
