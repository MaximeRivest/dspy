import logging
import os
import warnings
from pathlib import Path
from typing import Any

import litellm

from dspy.clients import language_models as _language_models
from dspy.clients.base_lm import BaseLM
from dspy.clients.cache import Cache
from dspy.clients.embedding import Embedder
from dspy.clients.language_models import __all__ as _language_model_all
from dspy.clients.language_models.router import LMRouter, register_lm_backend
from dspy.clients.lm import LM as _legacy_lm_cls  # noqa: N811
from dspy.clients.provider import Provider, TrainingJob

_WARNED_EXPERIMENTAL_LM_ROUTER = False
_LANGUAGE_MODEL_LAZY_NAMES = set()
_LANGUAGE_MODEL_CLIENT_EXPORTS = tuple(_language_model_all)
for _name in _LANGUAGE_MODEL_CLIENT_EXPORTS:
    if _name not in _LANGUAGE_MODEL_LAZY_NAMES:
        globals()[_name] = getattr(_language_models, _name)


def __getattr__(name: str):
    if name in _LANGUAGE_MODEL_LAZY_NAMES:
        return getattr(_language_models, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

class LM(_legacy_lm_cls):
    """Create DSPy's public language model.

    By default this is the existing LiteLLM-backed legacy LM class and remains
    subclassable for existing custom LMs. Set `dspy.configure(experimental=True)`
    or use `dspy.context(experimental=True)` to route direct `dspy.LM(...)`
    construction to a normalized `LanguageModel` backend.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        from dspy.dsp.utils import settings

        if cls is LM and settings.get("experimental", False):
            global _WARNED_EXPERIMENTAL_LM_ROUTER
            if not _WARNED_EXPERIMENTAL_LM_ROUTER:
                warnings.warn(
                    "`dspy.LM(...)` is using the experimental normalized LM router because "
                    "`dspy.settings.experimental=True`. This path was introduced experimentally in DSPy 3.3. "
                    "It returns a provider-specific `LanguageModel`, not the legacy LiteLLM-backed "
                    "`dspy.clients.lm.LM`, and is planned to become the default in DSPy 3.5.",
                    FutureWarning,
                    stacklevel=2,
                )
                _WARNED_EXPERIMENTAL_LM_ROUTER = True
            return LMRouter(*args, **kwargs)

        return super().__new__(cls)


logger = logging.getLogger(__name__)

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
DISK_CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))  # 30 GB default


def inspect_history(n: int = 1, file: Any | None = None) -> None:
    """Print recent interactions from legacy and normalized language models.

    Args:
        n: Number of recent history entries to display. Defaults to 1.
        file: Optional file-like object to write output to. When provided,
            ANSI color codes are automatically disabled.
    """
    from dspy.clients.base_lm import GLOBAL_HISTORY
    from dspy.clients.language_models.base import GLOBAL_LANGUAGE_MODEL_HISTORY
    from dspy.utils.inspect_history import pretty_print_history

    history = [*GLOBAL_HISTORY, *GLOBAL_LANGUAGE_MODEL_HISTORY]
    history.sort(key=lambda entry: entry.get("timestamp", ""))
    pretty_print_history(history, n, file=file)


def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int = 1000000,
    restrict_pickle: bool = False,
    safe_types: list[type[Any]] | None = None,
):
    """Configure the cache for DSPy.

    Args:
        enable_disk_cache: Whether to enable on-disk cache.
        enable_memory_cache: Whether to enable in-memory cache.
        disk_cache_dir: The directory to store the on-disk cache.
        disk_size_limit_bytes: The size limit of the on-disk cache.
        memory_max_entries: The maximum number of entries in the in-memory cache. To allow the cache to grow without
                            bounds, set this parameter to `math.inf` or a similar value.
        restrict_pickle: When True, restrict pickle deserialization to a known-safe
            set of types. When False (default), use unrestricted pickle.
        safe_types: Additional types to allow when restrict_pickle is True.
    """

    DSPY_CACHE = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
        restrict_pickle=restrict_pickle,
        safe_types=safe_types,
    )

    import dspy

    # Update the reference to point to the new cache
    dspy.cache = DSPY_CACHE


litellm.telemetry = False
litellm.cache = None  # By default we disable LiteLLM cache and use DSPy on-disk cache.


def _get_dspy_cache():
    disk_cache_dir = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
    disk_cache_limit = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))

    try:
        _dspy_cache = Cache(
            enable_disk_cache=True,
            enable_memory_cache=True,
            disk_cache_dir=disk_cache_dir,
            disk_size_limit_bytes=disk_cache_limit,
            memory_max_entries=1000000,
        )
    except Exception as e:
        # If cache creation fails (e.g., in AWS Lambda), create a memory-only cache
        logger.warning("Failed to initialize disk cache, falling back to memory-only cache: %s", e)
        _dspy_cache = Cache(
            enable_disk_cache=False,
            enable_memory_cache=True,
            disk_cache_dir=disk_cache_dir,
            disk_size_limit_bytes=disk_cache_limit,
            memory_max_entries=1000000,
        )
    return _dspy_cache


DSPY_CACHE = _get_dspy_cache()


def configure_litellm_logging(level: str = "ERROR"):
    """Configure LiteLLM logging to the specified level."""
    # Litellm uses a global logger called `verbose_logger` to control all loggings.
    from litellm._logging import verbose_logger

    numeric_logging_level = getattr(logging, level)

    verbose_logger.setLevel(numeric_logging_level)
    for h in verbose_logger.handlers:
        h.setLevel(numeric_logging_level)


def enable_litellm_logging():
    litellm.suppress_debug_info = False
    configure_litellm_logging("DEBUG")


def disable_litellm_logging():
    litellm.suppress_debug_info = True
    configure_litellm_logging("ERROR")


# By default, we disable LiteLLM logging for clean logging
disable_litellm_logging()

__all__ = [
    "BaseLM",
    "LM",
    "LMRouter",
    "register_lm_backend",
    "Provider",
    "TrainingJob",
    "inspect_history",
    "Embedder",
    "enable_litellm_logging",
    "disable_litellm_logging",
    "configure_cache",
    *_LANGUAGE_MODEL_CLIENT_EXPORTS,
]
