"""DSPy language-model wrapper backed by lm15.

This module contains the ``LM`` class — the primary way users configure a
language model for DSPy.  Under the hood every call is routed through lm15,
which handles provider resolution, retries, streaming, multimodal parts,
tool calling, and cost tracking.  DSPy only adds its own caching layer and
the fine-tuning / reinforcement interfaces on top.
"""

import logging
import re
import threading
import warnings
from typing import Any, Literal

import dspy
from dspy.clients.cache import request_cache
from dspy.clients.openai import OpenAIProvider
from dspy.clients.provider import Provider, ReinforceJob, TrainingJob
from dspy.clients.utils_finetune import TrainDataFormat
from dspy.dsp.utils.settings import settings
from dspy.utils.callback import BaseCallback
from dspy.utils.exceptions import ContextWindowExceededError

from .base_lm import BaseLM

logger = logging.getLogger(__name__)


class LM(BaseLM):
    """A language model supporting chat or text completion requests for use with DSPy modules."""

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
        # Remember to update LM.copy() if you modify the constructor!
        self.model = model
        self.model_type = model_type
        self.cache = cache
        self.provider = provider or self.infer_provider()
        self.callbacks = callbacks or []
        self.history = []
        self.num_retries = num_retries
        self.finetuning_model = finetuning_model
        self.launch_kwargs = launch_kwargs or {}
        self.train_kwargs = train_kwargs or {}
        self.use_developer_role = use_developer_role
        self._warned_zero_temp_rollout = False

        model_family = model.split("/")[-1].lower() if "/" in model else model.lower()
        is_reasoning = re.match(
            r"^(?:o[1345](?:-(?:mini|nano|pro))?(?:-\d{4}-\d{2}-\d{2})?|gpt-5(?!-chat)(?:-.*)?)$",
            model_family,
        )

        if is_reasoning:
            if (temperature and temperature != 1.0) or (max_tokens and max_tokens < 16000):
                raise ValueError(
                    "OpenAI's reasoning models require temperature=1.0 or None and max_tokens >= 16000 or None."
                )
            self.kwargs = dict(temperature=temperature, max_completion_tokens=max_tokens, **kwargs)
        else:
            self.kwargs = dict(temperature=temperature, max_tokens=max_tokens, **kwargs)

        if self.kwargs.get("rollout_id") is None:
            self.kwargs.pop("rollout_id", None)
        self._warn_zero_temp_rollout(self.kwargs.get("temperature"), self.kwargs.get("rollout_id"))

    # ------------------------------------------------------------------
    # Backend — all capability queries delegated to lm15
    # ------------------------------------------------------------------

    def _get_backend(self):
        from dspy.clients import _lm15
        return _lm15

    @property
    def supports_function_calling(self) -> bool:
        return self._get_backend().supports_function_calling(self.model)

    @property
    def supports_reasoning(self) -> bool:
        return self._get_backend().supports_reasoning(self.model)

    @property
    def supports_response_schema(self) -> bool:
        return self._get_backend().supports_response_schema(self.model)

    @property
    def supported_params(self) -> set[str]:
        return self._get_backend().supported_params(self.model)

    # ------------------------------------------------------------------
    # Request building & completion
    # ------------------------------------------------------------------

    def _warn_zero_temp_rollout(self, temperature, rollout_id):
        if not self._warned_zero_temp_rollout and rollout_id is not None and temperature == 0:
            warnings.warn("rollout_id has no effect when temperature=0.", stacklevel=3)
            self._warned_zero_temp_rollout = True

    def _build_request(self, prompt, messages, kwargs):
        kwargs = dict(kwargs)
        cache = kwargs.pop("cache", self.cache)
        messages = messages or [{"role": "user", "content": prompt}]
        if self.use_developer_role and self.model_type == "responses":
            messages = [{**m, "role": "developer"} if m.get("role") == "system" else m for m in messages]
        kwargs = {**self.kwargs, **kwargs}
        self._warn_zero_temp_rollout(kwargs.get("temperature"), kwargs.get("rollout_id"))
        if kwargs.get("rollout_id") is None:
            kwargs.pop("rollout_id", None)
        return dict(model=self.model, messages=messages, **kwargs), cache

    def _wrap_with_cache(self, fn, cache):
        if cache:
            fn = request_cache(
                cache_arg_name="request",
                ignored_args_for_cache_key=["api_key", "api_base", "base_url"],
            )(fn)
        return fn

    def forward(self, prompt=None, messages=None, **kwargs):
        request, cache = self._build_request(prompt, messages, kwargs)
        backend = self._get_backend()
        fn = self._wrap_with_cache(backend.complete_request, cache)

        try:
            results = fn(request=request, model_type=self.model_type, num_retries=self.num_retries)
        except backend.ContextWindowError as e:
            raise ContextWindowExceededError(model=self.model) from e

        if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
            settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results

    async def aforward(self, prompt=None, messages=None, **kwargs):
        request, cache = self._build_request(prompt, messages, kwargs)
        backend = self._get_backend()
        fn = self._wrap_with_cache(backend.acomplete_request, cache)

        try:
            results = await fn(request=request, model_type=self.model_type, num_retries=self.num_retries)
        except backend.ContextWindowError as e:
            raise ContextWindowExceededError(model=self.model) from e

        if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
            settings.usage_tracker.add_usage(self.model, dict(results.usage))
        return results

    # ------------------------------------------------------------------
    # Fine-tuning / reinforcement (provider-specific, kept as-is)
    # ------------------------------------------------------------------

    def launch(self, launch_kwargs=None):
        self.provider.launch(self, launch_kwargs)

    def kill(self, launch_kwargs=None):
        self.provider.kill(self, launch_kwargs)

    def finetune(self, train_data, train_data_format=None, train_kwargs=None) -> TrainingJob:
        if not self.provider.finetunable:
            raise ValueError(
                f"Provider {self.provider} does not support fine-tuning. "
                "Set `provider=` explicitly, e.g. dspy.LM('openai/gpt-4.1-mini', provider=dspy.OpenAIProvider())."
            )

        def _run():
            return self._run_finetune_job(job)

        thread = threading.Thread(target=_run)
        train_kwargs = train_kwargs or self.train_kwargs
        job = self.provider.TrainingJob(
            thread=thread,
            model=self.finetuning_model or self.model,
            train_data=train_data,
            train_data_format=train_data_format,
            train_kwargs=train_kwargs,
        )
        thread.start()
        return job

    def reinforce(self, train_kwargs) -> ReinforceJob:
        assert self.provider.reinforceable, f"Provider {self.provider} does not support reinforcement."
        job = self.provider.ReinforceJob(lm=self, train_kwargs=train_kwargs)
        job.initialize()
        return job

    def _run_finetune_job(self, job):
        try:
            model = self.provider.finetune(
                job=job, model=job.model, train_data=job.train_data,
                train_data_format=job.train_data_format, train_kwargs=job.train_kwargs,
            )
            job.set_result(self.copy(model=model))
        except Exception as err:
            logger.error(err)
            job.set_result(err)

    def infer_provider(self) -> Provider:
        if OpenAIProvider.is_provider_model(self.model):
            return OpenAIProvider()
        return Provider()

    def dump_state(self):
        state_keys = ["model", "model_type", "cache", "num_retries", "finetuning_model", "launch_kwargs", "train_kwargs"]
        filtered_kwargs = {k: v for k, v in self.kwargs.items() if k != "api_key"}
        return {key: getattr(self, key) for key in state_keys} | filtered_kwargs
