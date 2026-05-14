"""Route `dspy.LM` to concrete normalized language model backends."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any, Literal

from dspy.clients.language_models.base import LanguageModel

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


def LMRouter(  # noqa: N802
    model: str | None = None,
    *args: Any,
    backend: LanguageModel | None = None,
    **kwargs: Any,
) -> LanguageModel:
    """Create a concrete normalized language model backend.

    Args:
        model: Model name or deployment identifier. Required unless `backend`
            is supplied.
        *args: Positional arguments forwarded to registered backend constructors.
        backend: Optional prebuilt backend. When supplied, returned as-is.
        **kwargs: Constructor arguments for the selected backend.

    Returns:
        A concrete `LanguageModel` instance.
    """
    if backend is not None:
        return backend
    if model is None:
        raise TypeError("LM requires `model` unless `backend` is provided.")
    protocol = _deprecated_model_type(kwargs)
    return _route_lm_backend(model, *args, protocol=protocol, **kwargs)


LM = LMRouter


def _deprecated_model_type(kwargs: dict[str, Any]) -> Literal["chat", "text", "responses"]:
    model_type = kwargs.pop("model_type", None)
    if model_type is None:
        return "chat"
    warnings.warn(
        "`model_type` is deprecated for `dspy.LM` and `dspy.LMRouter`. "
        "Instantiate `OpenAIResponsesLM` or `OpenAICompletionsLM` directly when you need a specific OpenAI API.",
        DeprecationWarning,
        stacklevel=3,
    )
    if model_type not in {"chat", "text", "responses"}:
        raise ValueError(f"Unsupported model_type: {model_type!r}")
    return model_type


def _route_lm_backend(
    model: str,
    *args: Any,
    protocol: Literal["chat", "text", "responses"] = "chat",
    **kwargs: Any,
) -> LanguageModel:
    # `protocol` selects DSPy's default built-in OpenAI backend. Custom
    # registered backends should not receive public routing options by accident.
    for factory in reversed(_BACKEND_FACTORIES):
        backend = factory(model, *args, **kwargs)
        if backend is not None:
            return backend
    return _default_builtin_backend(model, *args, protocol=protocol, **kwargs)


def _default_builtin_backend(
    model: str,
    *,
    protocol: Literal["chat", "text", "responses"] = "chat",
    **kwargs: Any,
) -> LanguageModel:
    from dspy.clients.language_models.anthropic import AnthropicLM
    from dspy.clients.language_models.gemini import GenAILM
    from dspy.clients.language_models.openai import OpenAICompletionsLM, OpenAIResponsesLM

    provider = model.split("/", 1)[0] if "/" in model else "openai"
    if provider == "anthropic":
        return AnthropicLM(model=model, **kwargs)
    if provider in {"gemini", "google", "genai"}:
        return GenAILM(model=model, **kwargs)

    if protocol == "responses":
        return OpenAIResponsesLM(model=model, **kwargs)
    if protocol == "chat":
        return OpenAICompletionsLM(model=model, protocol="chat", **kwargs)
    if protocol == "text":
        return OpenAICompletionsLM(model=model, protocol="text", **kwargs)
    raise ValueError(f"Unsupported protocol: {protocol!r}")
