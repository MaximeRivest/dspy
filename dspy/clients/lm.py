"""DSPy language-model wrapper backed by lm15.

``LM`` is the primary way users configure a language model.  Every call
is routed through lm15 — no intermediate ChatCompletion format, no
litellm.  The adapter hands this class dict messages; it converts them
to lm15, calls lm15, and returns extracted outputs.
"""

import datetime
import logging
import re
import threading
import uuid
import warnings
from typing import Any, Literal

import dspy
from dspy.clients.cache import request_cache
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback, with_callbacks
from dspy.utils.exceptions import ContextWindowExceededError
from dspy.utils.inspect_history import pretty_print_history

from .base_lm import BaseLM, GLOBAL_HISTORY, MAX_HISTORY_SIZE, _update_history

logger = logging.getLogger(__name__)


class LM(BaseLM):
    """A language model backed by lm15 for use with DSPy modules."""

    def __init__(
        self,
        model: str,
        model_type: Literal["chat", "text", "responses"] = "chat",
        temperature: float | None = None,
        max_tokens: int | None = None,
        cache: bool = True,
        callbacks: list[BaseCallback] | None = None,
        num_retries: int = 3,
        provider: Provider | None = None,
        finetuning_model: str | None = None,
        launch_kwargs: dict[str, Any] | None = None,
        train_kwargs: dict[str, Any] | None = None,
        use_developer_role: bool = False,
        **kwargs,
    ):
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.provider = provider or self._infer_provider()
        self.callbacks = callbacks or []
        self.history = []
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.use_developer_role = use_developer_role
        self._warned_zero_temp_rollout = False

        family = model.split("/")[-1].lower() if "/" in model else model.lower()
        is_reasoning = re.match(
            r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$", family)

        if is_reasoning:
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise ValueError("OpenAI reasoning models require temperature=1.0/None and max_tokens>=16000/None.")
            self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)
        if self.kwargs.get("rollout_id") is None:
            self.kwargs.pop("rollout_id", None)
        self._warn_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))

    # -- Capabilities (delegated to lm15) ----------------------------------

    @property
    def supports_function_calling(self):
        from dspy.clients import _lm15
        return _lm15.supports_function_calling(self.model)

    @property
    def supports_reasoning(self):
        from dspy.clients import _lm15
        return _lm15.supports_reasoning(self.model)

    @property
    def supports_response_schema(self):
        from dspy.clients import _lm15
        return _lm15.supports_response_schema(self.model)

    @property
    def supported_params(self):
        from dspy.clients import _lm15
        return _lm15.supported_params(self.model)

    # -- Core call: messages → lm15 → outputs ------------------------------

    def _warn_rollout(self, temp, rid):
        if not self._warned_zero_temp_rollout and rid is not None and temp == 0:
            warnings.warn("rollout_id has no effect when temperature=0.", stacklevel=3)
            self._warned_zero_temp_rollout = True

    def _merge_kwargs(self, kwargs):
        kw = dict(kwargs)
        cache = kw.pop("cache", self.cache)
        kw = {**self.kwargs, **kw}
        self._warn_rollout(kw.get("temperature"), kw.get("rollout_id"))
        if kw.get("rollout_id") is None:
            kw.pop("rollout_id", None)
        return kw, cache

    @with_callbacks
    def __call__(self, prompt=None, messages=None, **kwargs) -> list[dict[str, Any] | str]:
        from dspy.clients import _lm15

        kw, cache = self._merge_kwargs(kwargs)
        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]

        multimodal = kw.pop("_multimodal_parts", None)
        kw["_model_type"] = self.model_type

        def _do_complete(*, request, **_ignored):
            """Cache-compatible wrapper."""
            return _lm15.complete(
                model=self.model, messages=request["messages"],
                num_retries=self.num_retries,
                multimodal_parts=request.get("_multimodal_parts"),
                **request["kwargs"],
            )

        fn = _do_complete
        if cache:
            fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=["api_key", "api_base", "base_url"],
            )(fn)

        try:
            resp, outputs = fn(request={"messages": messages, "_multimodal_parts": multimodal, "kwargs": kw})
        except _lm15.ContextWindowError as e:
            raise ContextWindowExceededError(model=self.model) from e

        # Usage tracking
        if not settings.disable_history:
            usage = {"prompt_tokens": resp.usage.input_tokens,
                     "completion_tokens": resp.usage.output_tokens,
                     "total_tokens": resp.usage.total_tokens}
        else:
            usage = {}

        if dspy.settings.usage_tracker and usage:
            settings.usage_tracker.add_usage(self.model, usage)

        if not settings.disable_history:
            safe_kw = {k: v for k, v in kwargs.items() if not k.startswith("api_")}
            entry = {
                "prompt": prompt, "messages": messages, "kwargs": safe_kw,
                "response": resp, "outputs": outputs, "usage": usage,
                "timestamp": datetime.datetime.now().isoformat(),
                "uuid": str(uuid.uuid4()),
                "model": self.model, "response_model": resp.model or self.model,
                "model_type": self.model_type,
            }
            _update_history(self, entry)

        return outputs

    @with_callbacks
    async def acall(self, prompt=None, messages=None, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        # Strip the @with_callbacks decorator behavior by calling the inner logic
        return await loop.run_in_executor(
            None, lambda: self.__call__.__wrapped__(self, prompt=prompt, messages=messages, **kwargs))

    # -- Backward-compat forward() for custom BaseLM subclasses ------------
    # (LM itself never calls these; they exist so that code doing
    #  ``lm.forward(...)`` or ``isinstance(lm, BaseLM)`` still works.)

    def forward(self, prompt=None, messages=None, **kwargs):
        return self(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt=None, messages=None, **kwargs):
        return await self.acall(prompt=prompt, messages=messages, **kwargs)

    # -- Fine-tuning / reinforcement (provider-specific) -------------------

    def launch(self, launch_kwargs=None):
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs=None):
        self.provider.kill(self, launch_kwargs)

    def finetune(self, train_data, train_data_format=None, train_kwargs=None) -> TrainingJob:
        if not self.provider.finetunable:
            raise ValueError(f"Provider {self.provider} does not support fine-tuning.")
        def _run():
            self._run_finetune_job(job)
        thread = threading.Thread(target=_run)
        train_kwargs = train_kwargs or self.train_kwargs
        job = self.provider.TrainingJob(
            thread=thread, model=self.finetuning_model or self.model,
            train_data=train_data, train_data_format=train_data_format, train_kwargs=train_kwargs)
        thread.start()
        return job

    def reinforce(self, train_kwargs) -> ReinforceJob:
        assert self.provider.reinforceable
        job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
        job.initialize()
        return job

    def _run_finetune_job(self, job):
        try:
            model = self.provider.finetune(
                job=job, model=job.model, train_data=job.train_data,
                train_data_format=job.train_data_format, train_kwargs=job.train_kwargs)
            job.set_result(self.copy(model=model))
        except Exception as e:
            logger.error(e); job.set_result(e)

    def _infer_provider(self):
        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        return Provider()

    def dump_state(self):
        keys = ["model","model_type","cache","num_retries","finetuning_model","launch_kwargs","train_kwargs"]
        filtered = {k: v for k, v in self.kwargs.items() if k != "api_key"}
        return {k: getattr(self, k) for k in keys} | filtered
