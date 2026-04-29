"""Create model handles through the stable `dspy.LM` entry point.

`dspy.LM(...)` is the convenient constructor for complete `BaseLM`
implementations. The selected implementation must generate text and may also
support `launch()`, `kill()`, `finetune()`, or `reinforce()`. By default, DSPy
uses the LiteLLM implementation. Community libraries can register resolvers
that select their own `BaseLM` subclasses for model names or constructor
arguments.

Examples:
Default LiteLLM-backed LM:
```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

assert isinstance(lm, dspy.LM)
```

Register several backends:
```python
from dspy.clients.base_lm import BaseLM
from dspy.clients.lm import LM, register_lm_backend


class AcmeLM(BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        ...


class TinkerLM(BaseLM):
    def forward(self, prompt=None, messages=None, **kwargs):
        ...


@register_lm_backend
def route_acme(model, *args, **kwargs):
    return AcmeLM if model.startswith("acme/") else None


@register_lm_backend
def route_tinker(model, *args, **kwargs):
    return TinkerLM if model.startswith("tinker/") else None


acme_lm = LM("acme/small")
tinker_lm = LM("tinker/research")

assert isinstance(acme_lm, LM)
assert isinstance(acme_lm.backend, AcmeLM)
assert isinstance(tinker_lm.backend, TinkerLM)
```
"""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from dspy.clients.base_lm import BaseLM

LMBackendResolver = Callable[..., type[BaseLM] | None]
_LM_BACKEND_RESOLVERS: list[LMBackendResolver] = []
_DEFAULT_BACKEND = "dspy.clients.litellmlm:LiteLLMLM"
_DEFAULT_BACKEND_MODULE = _DEFAULT_BACKEND.partition(":")[0]
_BACKEND_ATTRS = {
    "model",
    "model_type",
    "cache",
    "callbacks",
    "history",
    "num_retries",
    "provider",
    "finetuning_model",
    "launch_kwargs",
    "train_kwargs",
    "kwargs",
    "use_developer_role",
    "_warned_zero_temp_rollout",
}


@dataclass(frozen=True)
class LMBackendProvenance:
    """Record how `dspy.LM(...)` selected its backend.

    Use this object when you need to inspect which resolver or explicit backend
    produced an `LM` instance. The backend itself is available as `lm.backend`.

    Attributes:
        backend_cls: The `BaseLM` subclass selected for the instance.
        backend_name: Fully qualified name of `backend_cls`.
        resolver_name: Name of the resolver that selected the backend, or
            `None` when the backend was explicit or the default fallback.
        explicit_backend: Whether the user supplied `backend=` directly.

    Examples:
    Inspect backend provenance:
    ```python
    import dspy

    lm = dspy.LM("openai/gpt-4o-mini")

    assert lm.backend_provenance.backend_name.endswith("LiteLLMLM")
    assert lm.backend_provenance.explicit_backend is False
    ```
    """

    backend_cls: type[BaseLM]
    backend_name: str
    resolver_name: str | None = None
    explicit_backend: bool = False


def _backend_name(backend_cls: type[BaseLM]) -> str:
    return f"{backend_cls.__module__}.{backend_cls.__qualname__}"


def _backend_path(backend_cls: type[BaseLM]) -> str:
    if "<locals>" in backend_cls.__qualname__:
        raise ValueError(
            "LM backends must be defined at module scope to be serialized. "
            f"Received non-importable backend {backend_cls!r}."
        )
    return f"{backend_cls.__module__}:{backend_cls.__qualname__}"


def _resolver_name(resolver: LMBackendResolver) -> str:
    return f"{resolver.__module__}.{getattr(resolver, '__qualname__', resolver.__name__)}"


def _import_backend(path: str) -> type[BaseLM]:
    module_name, _, object_name = path.partition(":")
    if not module_name or not object_name:
        raise ValueError(f"Backend path must be of the form 'module:object', but received {path!r}.")

    obj = import_module(module_name)
    for part in object_name.split("."):
        obj = getattr(obj, part)
    return obj


def _get_default_backend_cls() -> type[BaseLM]:
    return _validate_backend_cls(_import_backend(_DEFAULT_BACKEND))


def __getattr__(name: str):
    # Backward-compatible access for helper names that historically lived in
    # dspy.clients.lm. The default backend module owns those implementations;
    # this branch is only reached when callers request a legacy helper.
    module = import_module(_DEFAULT_BACKEND_MODULE)
    if name in getattr(module, "__all__", ()):
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def register_lm_backend(resolver: LMBackendResolver, *, prepend: bool = True) -> LMBackendResolver:
    """Register a resolver that routes `dspy.LM(...)` to an LM implementation.

    Resolver functions receive the same constructor arguments passed to `LM`.
    Return a `BaseLM` subclass to handle the model, or `None` to let later
    resolvers try. Community packages can call this at import time to plug a
    complete model handle into the public `dspy.LM` constructor.

    Args:
        resolver: Callable returning a `BaseLM` subclass or `None`.
        prepend: If `True`, try this resolver before existing resolvers.

    Returns:
        The resolver. This lets you use `register_lm_backend` as a decorator.

    Examples:
    Register by calling the function:
    ```python
    from dspy.clients.base_lm import BaseLM
    from dspy.clients.lm import LM, register_lm_backend


    class AcmeLM(BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            ...


    def acme_resolver(model, *args, **kwargs):
        return AcmeLM if model.startswith("acme/") else None


    register_lm_backend(acme_resolver)
    lm = LM("acme/small")

    assert isinstance(lm, LM)
    assert isinstance(lm.backend, AcmeLM)
    ```

    Register several backends:
    ```python
    @register_lm_backend
    def route_acme(model, *args, **kwargs):
        return AcmeLM if model.startswith("acme/") else None


    @register_lm_backend
    def route_tinker(model, *args, **kwargs):
        return TinkerLM if model.startswith("tinker/") else None


    acme_lm = LM("acme/small")
    tinker_lm = LM("tinker/research")
    default_lm = LM("openai/gpt-4o-mini")  # Falls back to LiteLLM.
    ```

    Control precedence when two resolvers can handle the same model:
    ```python
    register_lm_backend(route_general_acme, prepend=False)
    register_lm_backend(route_experimental_acme, prepend=True)

    # route_experimental_acme is tried first.
    lm = LM("acme/small")
    ```
    """
    if prepend:
        _LM_BACKEND_RESOLVERS.insert(0, resolver)
    else:
        _LM_BACKEND_RESOLVERS.append(resolver)
    return resolver


def unregister_lm_backend(resolver: LMBackendResolver) -> None:
    """Remove a previously registered LM backend resolver.

    Use this in tests, notebooks, or plugin shutdown code when a resolver should
    no longer affect `dspy.LM(...)`.

    Args:
        resolver: The resolver function that was passed to
            `register_lm_backend`.

    Examples:
    Register a resolver temporarily:
    ```python
    from dspy.clients.lm import LM, register_lm_backend, unregister_lm_backend

    register_lm_backend(acme_resolver)
    lm = LM("acme/small")

    unregister_lm_backend(acme_resolver)
    ```
    """
    _LM_BACKEND_RESOLVERS.remove(resolver)


def _validate_backend_cls(backend_cls: type[BaseLM]) -> type[BaseLM]:
    if not isinstance(backend_cls, type) or not issubclass(backend_cls, BaseLM):
        raise TypeError(f"LM backend must be a BaseLM subclass, but received {backend_cls!r}.")
    lm_cls = globals().get("LM")
    if lm_cls is not None and issubclass(backend_cls, lm_cls):
        raise TypeError("LM cannot route to itself; return a concrete BaseLM subclass instead.")
    return backend_cls


def _resolve_lm_backend(
    model: str,
    *args: Any,
    backend: type[BaseLM] | str | None = None,
    **kwargs: Any,
) -> tuple[type[BaseLM], LMBackendProvenance]:
    if backend is not None:
        backend_cls = _validate_backend_cls(_import_backend(backend) if isinstance(backend, str) else backend)
        return backend_cls, LMBackendProvenance(
            backend_cls=backend_cls,
            backend_name=_backend_name(backend_cls),
            explicit_backend=True,
        )

    for resolver in list(_LM_BACKEND_RESOLVERS):
        backend_cls = resolver(model, *args, **kwargs)
        if backend_cls is not None:
            backend_cls = _validate_backend_cls(backend_cls)
            return backend_cls, LMBackendProvenance(
                backend_cls=backend_cls,
                backend_name=_backend_name(backend_cls),
                resolver_name=_resolver_name(resolver),
                explicit_backend=False,
            )

    backend_cls = _get_default_backend_cls()
    return backend_cls, LMBackendProvenance(
        backend_cls=backend_cls,
        backend_name=_backend_name(backend_cls),
        explicit_backend=False,
    )


def _configure_backend_for_lm_module(backend: BaseLM) -> None:
    configure = getattr(backend, "_configure_public_lm_wrapper", None)
    if configure is not None:
        configure(module_name=__name__)


def _route_builtin_lm(model: str, *args: Any, **kwargs: Any) -> type[BaseLM] | None:
    if model.startswith(("local:", "openai/local:", "huggingface/")):
        from dspy.clients.lm_local import LocalLM

        return LocalLM
    if model.startswith("databricks:"):
        from dspy.clients.databricks import DatabricksLM

        return DatabricksLM
    if model.startswith("openai-compatible:"):
        from dspy.clients.openai_compatible import OpenAICompatibleLM

        return OpenAICompatibleLM
    return None


register_lm_backend(_route_builtin_lm, prepend=False)


class LM(BaseLM):
    """Create a model handle that DSPy programs can use.

    `LM` keeps the convenient `dspy.LM(...)` entry point. It resolves the model
    name to a concrete `BaseLM` implementation, stores it in `lm.backend`, and
    delegates generation plus optional lifecycle and training methods to that
    implementation. The default implementation is `LiteLLMLM`.

    Args:
        model: Model identifier such as `"openai/gpt-4o-mini"`,
            `"local:Qwen/Qwen2.5-7B-Instruct"`, or
            `"openai-compatible:llama3.2"`.
        *args: Positional arguments passed to the selected implementation.
        backend: Optional explicit `BaseLM` subclass or import path. When set,
            this bypasses registered resolvers.
        **kwargs: Keyword arguments passed to the selected implementation, such
            as `temperature`, `max_tokens`, `cache`, `base_url`, or
            implementation-specific options.

    Examples:
    Configure the default LiteLLM-backed LM:
    ```python
    import dspy
    from dspy.clients.litellmlm import LiteLLMLM

    lm = dspy.LM("openai/gpt-4o-mini", temperature=0.0)
    dspy.configure(lm=lm)

    assert isinstance(lm, dspy.LM)
    assert isinstance(lm.backend, LiteLLMLM)
    ```

    Route to an explicit backend:
    ```python
    from dspy.clients.lm import LM
    from dspy.clients.litellmlm import LiteLLMLM

    lm = LM("openai/gpt-4o-mini", backend=LiteLLMLM)

    assert isinstance(lm, LM)
    assert isinstance(lm.backend, LiteLLMLM)
    assert lm.backend_provenance.explicit_backend is True
    ```

    Route with a community resolver:
    ```python
    from dspy.clients.base_lm import BaseLM
    from dspy.clients.lm import LM, register_lm_backend


    class AcmeLM(BaseLM):
        def forward(self, prompt=None, messages=None, **kwargs):
            ...


    @register_lm_backend
    def route_acme(model, *args, **kwargs):
        return AcmeLM if model.startswith("acme/") else None


    lm = LM("acme/small")

    assert isinstance(lm, LM)
    assert isinstance(lm.backend, AcmeLM)
    ```

    Register multiple community backends:
    ```python
    @register_lm_backend
    def route_acme(model, *args, **kwargs):
        return AcmeLM if model.startswith("acme/") else None


    @register_lm_backend
    def route_tinker(model, *args, **kwargs):
        return TinkerLM if model.startswith("tinker/") else None


    lm1 = LM("acme/small")
    lm2 = LM("tinker/research")
    lm3 = LM("openai/gpt-4o-mini")  # Falls back to LiteLLM.

    assert isinstance(lm1.backend, AcmeLM)
    assert isinstance(lm2.backend, TinkerLM)
    assert isinstance(lm3.backend, LiteLLMLM)
    ```
    """

    def __init__(self, model: str, *args: Any, backend: type[BaseLM] | str | None = None, **kwargs: Any):
        backend_cls, provenance = _resolve_lm_backend(model, *args, backend=backend, **kwargs)
        backend_instance = backend_cls(model, *args, **kwargs)
        _configure_backend_for_lm_module(backend_instance)

        object.__setattr__(self, "_backend", backend_instance)
        object.__setattr__(self, "backend_provenance", provenance)

    @property
    def backend(self) -> BaseLM:
        """Return the concrete backend instance used by this `LM`.

        Examples:
        Inspect the routed backend:
        ```python
        import dspy
        from dspy.clients.litellmlm import LiteLLMLM

        lm = dspy.LM("openai/gpt-4o-mini")
        assert isinstance(lm.backend, LiteLLMLM)
        ```
        """
        return self._backend

    @property
    def backend_cls(self) -> type[BaseLM]:
        """Return the backend class selected for this `LM`.

        Examples:
        Compare the selected backend class:
        ```python
        import dspy
        from dspy.clients.litellmlm import LiteLLMLM

        lm = dspy.LM("openai/gpt-4o-mini")
        assert lm.backend_cls is LiteLLMLM
        ```
        """
        return self.backend_provenance.backend_cls

    def __getattr__(self, name: str) -> Any:
        backend = self.__dict__.get("_backend")
        if backend is not None:
            return getattr(backend, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        backend = self.__dict__.get("_backend")
        if backend is not None and name in _BACKEND_ATTRS:
            setattr(backend, name, value)
        else:
            object.__setattr__(self, name, value)

    @property
    def supports_function_calling(self) -> bool:
        """Whether the selected backend supports native function calling.

        Examples:
        Ask the selected backend for tool-call support:
        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        supports_tools = lm.supports_function_calling
        ```
        """
        return self.backend.supports_function_calling

    @property
    def supports_reasoning(self) -> bool:
        """Whether the selected backend supports native reasoning output.

        Examples:
        Ask the selected backend for reasoning support:
        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        supports_reasoning = lm.supports_reasoning
        ```
        """
        return self.backend.supports_reasoning

    @property
    def supports_response_schema(self) -> bool:
        """Whether the selected backend supports structured response schemas.

        Examples:
        Ask the selected backend for schema support:
        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        supports_schema = lm.supports_response_schema
        ```
        """
        return self.backend.supports_response_schema

    @property
    def supported_params(self) -> set[str]:
        """Return OpenAI-style parameters supported by the selected backend.

        Examples:
        Check whether JSON response formatting is supported:
        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        if "response_format" in lm.supported_params:
            print("schema or JSON response format is available")
        ```
        """
        return self.backend.supported_params

    def forward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
        """Call the selected backend synchronously.

        Args:
            prompt: Optional plain-text prompt. When provided without messages,
                the backend turns it into a user message.
            messages: Optional chat-style message list.
            **kwargs: Request parameters passed to the backend.

        Examples:
        Call an LM directly:
        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        outputs = lm("What is DSPy?")
        ```
        """
        return self.backend.forward(prompt=prompt, messages=messages, **kwargs)

    async def aforward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
        """Call the selected backend asynchronously.

        Args:
            prompt: Optional plain-text prompt. When provided without messages,
                the backend turns it into a user message.
            messages: Optional chat-style message list.
            **kwargs: Request parameters passed to the backend.

        Examples:
        Await an LM call:
        ```python
        import dspy

        lm = dspy.LM("openai/gpt-4o-mini")
        outputs = await lm.acall("What is DSPy?")
        ```
        """
        return await self.backend.aforward(prompt=prompt, messages=messages, **kwargs)

    def _process_lm_response(self, response, prompt, messages, **kwargs):
        return self.backend._process_lm_response(response, prompt, messages, **kwargs)

    def launch(self, launch_kwargs: dict[str, Any] | None = None):
        """Start resources needed by the selected LM implementation."""
        return self.backend.launch(launch_kwargs)

    def kill(self, launch_kwargs: dict[str, Any] | None = None):
        """Release resources owned by the selected LM implementation."""
        return self.backend.kill(launch_kwargs)

    def finetune(self, *args: Any, **kwargs: Any):
        """Fine-tune the selected LM implementation and return its job."""
        return self.backend.finetune(*args, **kwargs)

    def reinforce(self, *args: Any, **kwargs: Any):
        """Start reinforcement training on the selected LM implementation."""
        return self.backend.reinforce(*args, **kwargs)

    def copy(self, **kwargs):
        backend_copy = self.backend.copy(**kwargs)
        _configure_backend_for_lm_module(backend_copy)

        new_instance = object.__new__(type(self))
        object.__setattr__(new_instance, "_backend", backend_copy)
        object.__setattr__(new_instance, "backend_provenance", self.backend_provenance)
        return new_instance

    def dump_state(self):
        """Return serializable LM state.

        This wraps the selected backend's `dump_state()`. When the backend is
        not the default DSPy backend, the returned state includes a `backend`
        key with the backend import path so `LM(**state)` can reconstruct it.

        For custom backends, the backend class must live at module scope so it
        has a stable import path. Loading that path from saved state may import
        code, so DSPy treats it as unsafe unless the module is already imported
        or the caller opts in with `allow_unsafe_lm_state=True`.

        Examples:
        Dump a custom backend with its path:
        ```python
        state = lm.dump_state()
        backend_path = state.get("backend")
        ```
        """
        state = self.backend.dump_state()
        backend_path = _backend_path(self.backend_cls)
        if backend_path != _DEFAULT_BACKEND:
            state["backend"] = backend_path
        return state


__all__ = [
    "LM",
    "LMBackendProvenance",
    "register_lm_backend",
    "unregister_lm_backend",
]
