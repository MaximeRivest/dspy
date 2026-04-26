import asyncio
import datetime
import logging
import threading
import uuid
from dataclasses import dataclass
from typing import Any, TextIO

from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.dsp.utils import settings
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.inspect_history import pretty_print_history

logger = logging.getLogger(__name__)

MAX_HISTORY_SIZE = 10_000
GLOBAL_HISTORY = []

LMOutput = dict[str, Any] | str
_OUTPUT_METADATA_KEYS = {"usage", "cost", "model", "response_model", "response", "_hidden_params", "cache_hit"}


@dataclass
class LMResponse:
    """Return LM outputs with provider metadata from a custom backend.

    Use `LMResponse` when your backend already has normalized DSPy outputs and
    you also want history and usage tracking to include provider metadata. The
    `outputs` field accepts the same text, dictionary, or list shapes that
    `BaseLM.forward()` accepts directly.

    Args:
        outputs: Generated text, one output dictionary, or a list of outputs.
            Output dictionaries may include `text`, `reasoning_content`,
            `tool_calls`, `citations`, or `logprobs`.
        usage: Token usage from the provider, such as `prompt_tokens`,
            `completion_tokens`, and `total_tokens`.
        cost: Provider cost for this request. DSPy stores this in history as
            `entry["cost"]`.
        model: Provider model that served the request. DSPy stores this in
            history as `entry["response_model"]`.
        response: Optional raw provider response to keep in history.
        cache_hit: Whether this result came from a cache. Usage is not added to
            DSPy's usage tracker when this is `True`.

    Examples:
    Return text and usage from a custom backend:
    ```python
    import dspy


    class AcmeLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            result = acme_generate(prompt=prompt, messages=messages)
            return dspy.LMResponse(
                outputs=result.text,
                usage={"prompt_tokens": result.input_tokens, "completion_tokens": result.output_tokens},
                cost=result.cost_usd,
                model=result.model,
                response=result,
            )
    ```
    """

    outputs: LMOutput | list[LMOutput]
    usage: Any | None = None
    cost: float | None = None
    model: str | None = None
    response: Any | None = None
    cache_hit: bool = False

    @property
    def _hidden_params(self) -> dict[str, Any] | None:
        if self.cost is None:
            return None
        return {"response_cost": self.cost}


class BaseLM:
    """Implement a custom language model backend for DSPy.

    Subclass `BaseLM` when you want DSPy modules to call a model that is not
    handled by the built-in `dspy.LM` router. In most cases, implement only
    `forward()`. `BaseLM` supplies `__call__`, `acall`, callbacks, history,
    copying, and provider lifecycle hooks.

    The easiest `forward()` return value is the model text as a string. For
    multiple generations, return a list of strings. When you need output-level
    metadata, return a dictionary or list of dictionaries with keys like `text`,
    `reasoning_content`, `tool_calls`, `citations`, or `logprobs`. When you also
    need provider metadata such as token usage, cost, cache hits, or the served
    model name, return an `LMResponse` or a dictionary with an `outputs` key.
    For backward compatibility, `BaseLM` also accepts OpenAI-shaped chat, text
    completion, or Responses API objects.

    Examples:
    Minimal backend returning text:
    ```python
    import dspy


    class EchoLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            messages, params = self.prepare_request(prompt, messages, **kwargs)
            return f"Echo: {messages[-1]['content']}"


    lm = EchoLM("echo")
    assert lm("hello") == ["Echo: hello"]
    ```

    Backend that calls an HTTP API and returns normalized text:
    ```python
    import httpx
    import dspy


    class AcmeLM(dspy.BaseLM):
        def __init__(self, model, api_key, **kwargs):
            super().__init__(model, **kwargs)
            self.api_key = api_key

        def forward(self, prompt=None, messages=None, **kwargs):
            messages, params = self.prepare_request(prompt, messages, **kwargs)
            response = httpx.post(
                "https://api.acme.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "messages": messages, **params},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]


    dspy.configure(lm=AcmeLM("acme/small", api_key="..."))
    result = dspy.Predict("question -> answer")(question="What is DSPy?")
    print(result.answer)
    ```

    Return output metadata when your backend exposes it:
    ```python
    import dspy


    class ReasoningLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return {
                "text": "[[ ## answer ## ]]\\nParis\\n\\n[[ ## completed ## ]]",
                "reasoning_content": "The question asks for France's capital.",
            }
    ```

    Return provider metadata for history and usage tracking:
    ```python
    import dspy


    class UsageLM(dspy.BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            return dspy.LMResponse(
                outputs="[[ ## answer ## ]]\\nParis\\n\\n[[ ## completed ## ]]",
                usage={"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
                cost=0.0004,
                model="acme/small-2026-04-01",
            )
    ```

    Register a backend for `dspy.LM(...)`:
    ```python
    from dspy.clients.lm import LM, register_lm_backend


    @register_lm_backend
    def route_acme(model, *args, **kwargs):
        return AcmeLM if model.startswith("acme/") else None


    lm = LM("acme/small", api_key="...")
    assert isinstance(lm.backend, AcmeLM)
    ```
    """

    def __init__(
        self,
        model,
        model_type="chat",
        temperature=0.0,
        max_tokens=1000,
        cache=True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Create a backend instance.

        Args:
            model: Provider model name or deployment identifier.
            model_type: Response shape to expect when `forward()` returns an
                OpenAI-shaped object. Use `"chat"`, `"text"`, or `"responses"`.
            temperature: Default sampling temperature stored in `self.kwargs`.
            max_tokens: Default maximum output tokens stored in `self.kwargs`.
            cache: Whether this backend should use caching when it implements a
                cache-aware transport.
            callbacks: Per-LM callbacks. Global callbacks still come from
                `dspy.settings`.
            num_retries: Retry count for backends that support retrying.
            provider: Provider object for launch, kill, fine-tune, and
                reinforcement-learning hooks.
            finetuning_model: Optional model name to fine-tune instead of
                `model`.
            launch_kwargs: Default launch arguments used by `launch()`.
            train_kwargs: Default fine-tuning arguments used by `finetune()`.
            **kwargs: Extra default request parameters. They are stored in
                `self.kwargs` and merged by `prepare_request()`.

        Examples:
        Call `super().__init__()` from a custom backend:
        ```python
        import dspy


        class AcmeLM(dspy.BaseLM):
            def __init__(self, model, api_key, **kwargs):
                super().__init__(model, **kwargs)
                self.api_key = api_key
        ```
        """
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.callbacks = callbacks or []
        self.history = []
        self.num_retries = num_retries
        self.provider = provider or self.infer_provider()
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        self._warned_zero_temp_rollout = False

    @property
    def supports_function_calling(self) -> bool:
        """Whether the model supports function calling (tool use)."""
        return False

    @property
    def supports_reasoning(self) -> bool:
        """Whether the model supports native reasoning (extended thinking)."""
        return False

    @property
    def supports_response_schema(self) -> bool:
        """Whether the model supports structured output via response schema."""
        return False

    @property
    def supported_params(self) -> set[str]:
        """Set of supported OpenAI-style parameter names for the model."""
        return set()

    def prepare_request(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Build messages and merged request parameters for `forward()`.

        Use this helper in custom backends to handle the common DSPy calling
        convention: callers may pass either a plain prompt or a chat-style list
        of messages, and call-time keyword arguments override defaults stored in
        `self.kwargs`.

        Args:
            prompt: Plain text prompt. Used only when `messages` is `None`.
            messages: Chat-style messages. If omitted, `prompt` becomes a user
                message.
            **kwargs: Call-time request parameters such as `temperature` or
                `max_tokens`.

        Returns:
            A pair `(messages, params)`. `messages` is always a list of message
            dictionaries. `params` is `{**self.kwargs, **kwargs}`.

        Examples:
        Use `prepare_request()` in a custom backend:
        ```python
        import dspy


        class AcmeLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                messages, params = self.prepare_request(prompt, messages, **kwargs)
                return call_acme_api(model=self.model, messages=messages, **params)
        ```
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        return messages, {**self.kwargs, **kwargs}

    @staticmethod
    def _response_value(response, key: str, default=None):
        if isinstance(response, dict):
            return response.get(key, default)
        return getattr(response, key, default)

    @classmethod
    def _raw_response(cls, response):
        if isinstance(response, LMResponse):
            return response.response if response.response is not None else response
        if isinstance(response, dict) and "outputs" in response:
            raw_response = response.get("response")
            return response if raw_response is None else raw_response
        return response

    @classmethod
    def _metadata_value(cls, response, key: str, default=None):
        value = cls._response_value(response, key)
        if value is not None:
            return value

        raw_response = cls._raw_response(response)
        if raw_response is response:
            return default
        return cls._response_value(raw_response, key, default)

    @classmethod
    def _metadata_dict(cls, response, key: str) -> dict[str, Any]:
        value = cls._metadata_value(response, key, {}) or {}
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "model_dump"):
            return value.model_dump()
        try:
            return dict(value)
        except (TypeError, ValueError):
            return {
                attr: cls._response_value(value, attr)
                for attr in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens")
                if cls._response_value(value, attr) is not None
            }

    @classmethod
    def _response_cost(cls, response):
        if cls._metadata_value(response, "cache_hit", False):
            return None

        cost = cls._metadata_value(response, "cost")
        if cost is not None:
            return cost

        hidden_params = cls._metadata_value(response, "_hidden_params", {}) or {}
        return cls._response_value(hidden_params, "response_cost")

    @classmethod
    def _response_model(cls, response, default_model: str):
        return cls._metadata_value(
            response,
            "response_model",
            cls._metadata_value(response, "model", default_model),
        )

    @classmethod
    def _is_lm_response(cls, response) -> bool:
        if isinstance(response, LMResponse):
            return True
        return (
            isinstance(response, dict)
            and "outputs" in response
            and "choices" not in response
            and "output" not in response
        )

    @classmethod
    def _lm_response_outputs(cls, response):
        return cls._response_value(response, "outputs")

    @staticmethod
    def _is_processed_output(output) -> bool:
        if isinstance(output, str):
            return True
        if not isinstance(output, dict):
            return False
        if "choices" in output or "output" in output or "outputs" in output:
            return False
        return any(key in output for key in ("text", "reasoning_content", "tool_calls", "citations", "logprobs"))

    @classmethod
    def _normalize_processed_outputs(cls, response) -> list[LMOutput]:
        values = response if isinstance(response, list) else [response]
        outputs = []
        for value in values:
            if isinstance(value, dict):
                value = {k: v for k, v in value.items() if k not in _OUTPUT_METADATA_KEYS}
                if "text" not in value:
                    value = {"text": None, **value}
            outputs.append(value)
        return outputs

    @classmethod
    def _is_processed_outputs(cls, response) -> bool:
        if isinstance(response, (str, dict)):
            return cls._is_processed_output(response)
        if isinstance(response, list):
            return all(cls._is_processed_output(output) for output in response)
        return False

    def _track_usage(self, response, usage: dict[str, Any]) -> None:
        if not usage or self._metadata_value(response, "cache_hit", False):
            return
        if settings.usage_tracker:
            settings.usage_tracker.add_usage(self.model, usage)

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        merged_kwargs = {**self.kwargs, **kwargs}

        if self._is_lm_response(response):
            outputs = self._normalize_processed_outputs(self._lm_response_outputs(response))
        elif self._is_processed_outputs(response):
            outputs = self._normalize_processed_outputs(response)
        elif self.model_type == "responses":
            outputs = self._process_response(response)
        else:
            outputs = self._process_completion(response, merged_kwargs)

        usage = self._metadata_dict(response, "usage")
        self._track_usage(response, usage)

        if settings.disable_history:
            return outputs

        # Logging, with removed api key & where `cost` is None on cache hit.
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
        entry = {
            "prompt": prompt,
            "messages": messages,
            "kwargs": kwargs,
            "response": self._raw_response(response),
            "outputs": outputs,
            "usage": usage,
            "cost": self._response_cost(response),
            "timestamp": datetime.datetime.now().isoformat(),
            "uuid": str(uuid.uuid4()),
            "model": self.model,
            "response_model": self._response_model(response, self.model),
            "model_type": self.model_type,
        }

        self.update_history(entry)

        return outputs

    @with_callbacks
    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[dict[str, Any] | str]:
        """Call the backend and return normalized LM outputs.

        Args:
            prompt: Plain text prompt for direct LM calls.
            messages: Chat-style messages. DSPy adapters pass these.
            **kwargs: Request parameters passed to `forward()`.

        Returns:
            A list of strings or dictionaries. DSPy adapters parse these into
            predictions.

        Examples:
        Call a custom backend directly:
        ```python
        import dspy


        class EchoLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                messages, params = self.prepare_request(prompt, messages, **kwargs)
                return f"Echo: {messages[-1]['content']}"


        lm = EchoLM("echo")
        outputs = lm("hello")
        assert outputs == ["Echo: hello"]
        ```
        """
        response = self.forward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)

        return outputs

    @with_callbacks
    async def acall(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ) -> list[dict[str, Any] | str]:
        """Call the backend asynchronously and return normalized LM outputs.

        Args:
            prompt: Plain text prompt for direct LM calls.
            messages: Chat-style messages. DSPy adapters pass these.
            **kwargs: Request parameters passed to `aforward()`.

        Returns:
            A list of strings or dictionaries. DSPy adapters parse these into
            predictions.

        Examples:
        Await a custom backend call:
        ```python
        import dspy


        class EchoLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                messages, params = self.prepare_request(prompt, messages, **kwargs)
                return f"Echo: {messages[-1]['content']}"


        lm = EchoLM("echo")
        outputs = await lm.acall("hello")
        assert outputs == ["Echo: hello"]
        ```
        """
        response = await self.aforward(prompt=prompt, messages=messages, **kwargs)
        outputs = self._process_lm_response(response, prompt, messages, **kwargs)
        return outputs

    def forward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        """Call the model and return generated output.

        Subclasses must implement this method. Use `prepare_request()` to merge
        `prompt`, `messages`, default parameters, and call-time parameters.

        Return one of these shapes:

        - a string for one generation;
        - a list of strings for multiple generations;
        - a dictionary like `{"text": "...", "reasoning_content": "..."}`;
        - a list of such dictionaries;
        - an `LMResponse` with outputs plus usage, cost, cache, or model data;
        - a dictionary with an `outputs` key plus usage, cost, cache, or model data;
        - an OpenAI-shaped chat, text completion, or Responses API object.

        Args:
            prompt: Plain text prompt. DSPy passes this for direct LM calls.
            messages: Chat-style message dictionaries. DSPy adapters usually
                pass this.
            **kwargs: Request parameters supplied by the caller or module.

        Returns:
            Generated output in one of the shapes listed above. `BaseLM.__call__`
            converts it to a list of strings or dictionaries for DSPy adapters.

        Raises:
            dspy.ContextWindowExceededError: When the request fails because the
                input exceeds the model's context window. DSPy adapters and
                modules rely on this error to trigger fallback behavior. Catch
                your provider's native context-length error and re-raise this
                DSPy error.

        Examples:
        Return text directly:
        ```python
        import dspy


        class AcmeLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                messages, params = self.prepare_request(prompt, messages, **kwargs)
                return acme_generate(self.model, messages, **params)
        ```

        Return provider metadata with the text:
        ```python
        import dspy


        class AcmeReasoningLM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                result = acme_generate_with_reasoning(prompt=prompt, messages=messages)
                return dspy.LMResponse(
                    outputs={"text": result.answer, "reasoning_content": result.reasoning},
                    usage=result.usage,
                    cost=result.cost,
                    model=result.model,
                )
        ```
        """
        raise NotImplementedError("Subclasses must implement this method.")

    async def aforward(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs
    ):
        """Call the model asynchronously.

        Override this when your client has a native async API. If you only
        implement `forward()`, the default `aforward()` runs it in a worker
        thread so `lm.acall(...)` still works.

        Args:
            prompt: Plain text prompt. DSPy passes this for direct LM calls.
            messages: Chat-style message dictionaries. DSPy adapters usually
                pass this.
            **kwargs: Request parameters supplied by the caller or module.

        Returns:
            The same shapes supported by `forward()`.

        Examples:
        Use a native async client:
        ```python
        import dspy


        class AcmeLM(dspy.BaseLM):
            async def aforward(self, prompt=None, messages=None, **kwargs):
                messages, params = self.prepare_request(prompt, messages, **kwargs)
                response = await acme_client.generate(self.model, messages, **params)
                return response.text
        ```
        """
        return await asyncio.to_thread(self.forward, prompt=prompt, messages=messages, **kwargs)

    def copy(self, **kwargs):
        """Returns a copy of the language model with possibly updated parameters.

        Any provided keyword arguments update the corresponding attributes or LM kwargs of
        the copy. For example, ``lm.copy(rollout_id=1, temperature=1.0)`` returns an LM whose
        requests use a different rollout ID at non-zero temperature to bypass cache collisions.
        """

        import copy

        new_instance = copy.deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(new_instance, key, value)
            if (key in self.kwargs) or (not hasattr(self, key)):
                if value is None:
                    new_instance.kwargs.pop(key, None)
                else:
                    new_instance.kwargs[key] = value
        if hasattr(new_instance, "_warned_zero_temp_rollout"):
            new_instance._warned_zero_temp_rollout = False

        return new_instance

    def inspect_history(self, n: int = 1, file: "TextIO | None" = None) -> None:
        pretty_print_history(self.history, n, file=file)

    def launch(self, launch_kwargs: dict[str, Any] | None = None):
        """Launch the model through its provider, if the provider supports launching."""
        launch_kwargs = {**self.launch_kwargs, **(launch_kwargs or {})}
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs: dict[str, Any] | None = None):
        """Stop the model through its provider, if the provider supports launching."""
        launch_kwargs = {**self.launch_kwargs, **(launch_kwargs or {})}
        self.provider.kill(self, launch_kwargs)

    def finetune(
        self,
        train_data: list[dict[str, Any]],
        train_data_format: TrainDataFormat | None,
        train_kwargs: dict[str, Any] | None = None,
    ) -> TrainingJob:
        """Start a provider-backed fine-tuning job for this LM."""
        if not self.provider.finetunable:
            raise ValueError(
                f"Provider {self.provider} does not support fine-tuning, please specify your provider by explicitly "
                "setting `provider` when creating the `dspy.LM` instance. For example, "
                "`dspy.LM('openai/gpt-4.1-mini-2025-04-14', provider=dspy.OpenAIProvider())`."
            )

        def thread_function_wrapper():
            return self._run_finetune_job(job)

        thread = threading.Thread(target=thread_function_wrapper)
        train_kwargs = train_kwargs or self.train_kwargs
        model_to_finetune = self.finetuning_model or self.model
        job = self.provider.TrainingJob(
            thread=thread,
            model=model_to_finetune,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs,
        )
        thread.start()

        return job

    def reinforce(self, train_kwargs) -> ReinforceJob:
        """Start a provider-backed reinforcement-learning job for this LM."""
        err = f"Provider {self.provider} does not implement the reinforcement learning interface."
        assert self.provider.reinforceable, err

        job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
        job.initialize()
        return job

    def _run_finetune_job(self, job: TrainingJob):
        # TODO(enhance): We should listen for keyboard interrupts somewhere.
        # Requires TrainingJob.cancel() to be implemented for each provider.
        try:
            model = self.provider.finetune(
                job=job,
                model=job.model,
                train_data=job.train_data,
                train_data_format=job.train_data_format,
                train_kwargs=job.train_kwargs,
            )
            lm = self.copy(model=model)
            job.set_result(lm)
        except Exception as err:
            logger.error(err)
            job.set_result(err)

    def infer_provider(self) -> Provider:
        """Return the provider used for lifecycle hooks.

        The base implementation returns a no-op provider. Override this in
        backends that know how to launch, stop, fine-tune, or reinforce their
        models.

        Examples:
        Use the default no-op provider:
        ```python
        lm = EchoLM("echo")
        assert lm.provider.finetunable is False
        ```
        """
        return Provider()

    def dump_state(self):
        """Return serializable backend state.

        Override this in custom backends when your constructor needs extra
        arguments beyond the common `BaseLM` fields. Return a dictionary that
        `YourBackend(**state)` can consume.

        The base implementation saves shared LM fields and request defaults from
        `self.kwargs`. It excludes `api_key` so credentials are not written to
        disk in plain text.

        Returns:
            A dictionary of constructor arguments that can rebuild this backend.

        Examples:
        Save a custom constructor argument:
        ```python
        import dspy


        class AcmeLM(dspy.BaseLM):
            def __init__(self, model, api_base, **kwargs):
                super().__init__(model, **kwargs)
                self.api_base = api_base

            def dump_state(self):
                return super().dump_state() | {"api_base": self.api_base}
        ```
        """
        state_keys = [
            "model",
            "model_type",
            "cache",
            "num_retries",
            "finetuning_model",
            "launch_kwargs",
            "train_kwargs",
        ]
        # Exclude api_key from kwargs to prevent API keys from being saved in plain text.
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != "api_key"}
        return {key: getattr(self, key) for key in state_keys} | filtered_kwargs

    def _check_truncation(self, results):
        choices = self._response_value(results, "choices", []) or []
        was_truncated = any(self._response_value(choice, "finish_reason") == "length" for choice in choices)
        if self.model_type != "responses" and was_truncated:
            max_tokens = self.kwargs.get("max_tokens", self.kwargs.get("max_completion_tokens"))
            logger.warning(
                f"LM response was truncated due to exceeding max_tokens={max_tokens}. "
                "You can inspect the latest LM interactions with `dspy.inspect_history()`. "
                "To avoid truncation, consider passing a larger max_tokens when setting up dspy.LM. "
                f"You may also consider increasing the temperature (currently {self.kwargs.get('temperature')}) "
                " if the reason for truncation is repetition."
            )

    def update_history(self, entry):
        if settings.disable_history:
            return

        # Global LM history
        if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
            GLOBAL_HISTORY.pop(0)

        GLOBAL_HISTORY.append(entry)

        if settings.max_history_size == 0:
            return

        # dspy.LM.history
        if len(self.history) >= settings.max_history_size:
            self.history.pop(0)

        self.history.append(entry)

        # Per-module history
        caller_modules = settings.caller_modules or []
        for module in caller_modules:
            if len(module.history) >= settings.max_history_size:
                module.history.pop(0)
            module.history.append(entry)

    def _process_completion(self, response, merged_kwargs):
        """Process an OpenAI-shaped chat/text completion response."""
        outputs = []
        for choice in self._response_value(response, "choices", []) or []:
            output = {}
            message = self._response_value(choice, "message")

            if message is not None:
                output["text"] = self._response_value(message, "content")
                reasoning_content = self._response_value(message, "reasoning_content")
                if reasoning_content:
                    output["reasoning_content"] = reasoning_content

                tool_calls = self._response_value(message, "tool_calls")
                if tool_calls:
                    output["tool_calls"] = tool_calls
            else:
                output["text"] = self._response_value(choice, "text")

            if merged_kwargs.get("logprobs"):
                output["logprobs"] = self._response_value(choice, "logprobs")

            citations = self._extract_citations_from_response(choice)
            if citations:
                output["citations"] = citations

            outputs.append(output)

        if all(len(output) == 1 for output in outputs):
            # Return a list if every output only has "text" key.
            outputs = [output["text"] for output in outputs]
        return outputs

    def _extract_citations_from_response(self, choice):
        """Extract provider-specific citation data from a completion choice, if available."""
        return None

    def _process_response(self, response):
        """Process an OpenAI Responses API-shaped response."""
        text_outputs = []
        tool_calls = []
        reasoning_contents = []

        for output_item in self._response_value(response, "output", []) or []:
            output_item_type = self._response_value(output_item, "type")
            if output_item_type == "message":
                for content_item in self._response_value(output_item, "content", []) or []:
                    text_outputs.append(self._response_value(content_item, "text", ""))
            elif output_item_type == "function_call":
                if hasattr(output_item, "model_dump"):
                    tool_calls.append(output_item.model_dump())
                elif isinstance(output_item, dict):
                    tool_calls.append(dict(output_item))
                else:
                    tool_calls.append(dict(getattr(output_item, "__dict__", {})))
            elif output_item_type == "reasoning":
                content = self._response_value(output_item, "content", []) or []
                summary = self._response_value(output_item, "summary", []) or []
                for content_item in content or summary:
                    reasoning_contents.append(self._response_value(content_item, "text", ""))

        result = {}
        if len(text_outputs) > 0:
            result["text"] = "".join(text_outputs)
        if len(tool_calls) > 0:
            result["tool_calls"] = tool_calls
        if len(reasoning_contents) > 0:
            result["reasoning_content"] = "".join(reasoning_contents)
        # All `response.output` items map to one answer, so we return a list of size 1.
        return [result]


def inspect_history(n: int = 1, file: "TextIO | None" = None) -> None:
    """The global history shared across all LMs.

    Args:
        n: Number of recent entries to display. Defaults to 1.
        file: An optional file-like object to write output to. When
            provided, ANSI color codes are automatically disabled.
            Defaults to `None` (prints to stdout).
    """
    pretty_print_history(GLOBAL_HISTORY, n, file=file)
