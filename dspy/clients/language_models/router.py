"""Route `dspy.LM` to concrete normalized language model backends."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from typing_extensions import Self

from dspy.clients.language_models.base import LanguageModel
from dspy.clients.language_models.features import FeatureStatus, LMFeatureReporter
from dspy.clients.language_models.types import LMRequest, LMResponse, LMStreamEvent

LMBackendFactory = Callable[..., LanguageModel | None]

_BACKEND_FACTORIES: list[LMBackendFactory] = []


def register_lm_backend(factory: LMBackendFactory) -> LMBackendFactory:
    """Register a backend factory used by `dspy.LM`.

    A factory receives the same constructor arguments as `LM`. Return a
    `LanguageModel` when it owns the model, or `None` to let the next factory
    try.
    """
    _BACKEND_FACTORIES.append(factory)
    return factory


class LMRouter(LanguageModel):
    """Route one public `dspy.LM` instance to a concrete backend."""

    # The router owns public LM lifecycle: callbacks, cache, history, and copy.
    # The backend owns provider/protocol behavior and exposes exact feature
    # support through the `LanguageModel` hooks.

    def __init__(
        self,
        model: str | None = None,
        *,
        backend: LanguageModel | None = None,
        model_type: Literal["chat", "text", "responses"] = "chat",
        cache: bool = True,
        callbacks: list[Any] | None = None,
        **kwargs: Any,
    ):
        if backend is None:
            if model is None:
                raise TypeError("LM requires `model` unless `backend` is provided.")
            backend = _route_lm_backend(
                model,
                model_type=model_type,
                cache=cache,
                callbacks=callbacks,
                **kwargs,
            )
        self.backend = backend
        self.callbacks = list(callbacks if callbacks is not None else getattr(backend, "callbacks", []) or [])
        self.history: list[dict[str, Any]] = []
        self.features = LMFeatureReporter(self)

    @property
    def model(self) -> str:
        return self.backend.model

    @model.setter
    def model(self, value: str) -> None:
        self.backend.model = value

    @property
    def cache(self) -> bool:
        return self.backend.cache

    @cache.setter
    def cache(self, value: bool) -> None:
        self.backend.cache = value

    @property
    def kwargs(self) -> dict[str, Any]:
        return self.backend.kwargs

    @kwargs.setter
    def kwargs(self, value: dict[str, Any]) -> None:
        self.backend.kwargs = value

    @property
    def model_type(self) -> str | None:
        return getattr(self.backend, "model_type", None)

    @property
    def capabilities(self):
        return self.backend.capabilities

    def get_capabilities(self):
        return self.backend.get_capabilities()

    def get_request_feature_statuses(self) -> dict[str, FeatureStatus]:
        return self.backend.get_request_feature_statuses()

    def get_response_feature_statuses(self) -> dict[str, FeatureStatus]:
        return self.backend.get_response_feature_statuses()

    def validate_request(self, request: LMRequest):
        return self.backend.validate_request(request)

    def forward(self, request: LMRequest) -> LMResponse:
        return self.backend.forward(request)

    async def aforward(self, request: LMRequest) -> LMResponse:
        return await self.backend.aforward(request)

    def forward_stream(self, request: LMRequest):
        yield from self.backend.forward_stream(request)

    async def aforward_stream(self, request: LMRequest):
        async for event in self.backend.aforward_stream(request):
            yield event

    def normalize_error(self, error: Exception, request: LMRequest) -> Exception:
        return self.backend.normalize_error(error, request)

    def dump_state(self) -> dict[str, Any]:
        return {
            "backend_class": f"{type(self.backend).__module__}.{type(self.backend).__qualname__}",
            "backend_state": self.backend.dump_state(),
        }

    @classmethod
    def load_state(cls, state: dict[str, Any]) -> Self:
        backend_class = _import_object(state["backend_class"])
        backend = backend_class.load_state(state["backend_state"])
        return cls(backend=backend)

    def copy(self, **overrides: Any) -> Self:
        return type(self)(backend=self.backend.copy(**overrides))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)


LM = LMRouter


def _route_lm_backend(
    model: str,
    *args: Any,
    model_type: Literal["chat", "text", "responses"] = "chat",
    **kwargs: Any,
) -> LanguageModel:
    # `model_type` selects DSPy's default LiteLLM protocol backend. Custom
    # factories should not receive it as a generation kwarg by accident.
    for factory in reversed(_BACKEND_FACTORIES):
        backend = factory(model, *args, **kwargs)
        if backend is not None:
            return backend
    return _default_litellm_backend(model, *args, model_type=model_type, **kwargs)


def _default_litellm_backend(
    model: str,
    *,
    model_type: Literal["chat", "text", "responses"] = "chat",
    **kwargs: Any,
) -> LanguageModel:
    from dspy.clients.language_models.litellm import LiteLLMChatLM, LiteLLMResponsesLM, LiteLLMTextLM

    backend_cls = {
        "chat": LiteLLMChatLM,
        "text": LiteLLMTextLM,
        "responses": LiteLLMResponsesLM,
    }.get(model_type)
    if backend_cls is None:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    return backend_cls(model=model, **kwargs)


def _import_object(path: str) -> Any:
    module_name, _, qualname = path.rpartition(".")
    if not module_name:
        raise ImportError(f"Cannot import object from {path!r}.")
    import importlib

    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj
