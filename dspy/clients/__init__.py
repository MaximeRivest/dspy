import logging
import os
from pathlib import Path

from dspy.clients.base_lm import BaseLM, inspect_history
from dspy.clients.cache import Cache
from dspy.clients.embedding import Embedder
from dspy.clients.lm import LM
from dspy.clients.provider import Provider, TrainingJob

logger = logging.getLogger(__name__)

DISK_CACHE_DIR = os.environ.get("DSPY_CACHEDIR") or os.path.join(Path.home(), ".dspy_cache")
DISK_CACHE_LIMIT = int(os.environ.get("DSPY_CACHE_LIMIT", 3e10))  # 30 GB default


def configure_cache(
    enable_disk_cache: bool | None = True,
    enable_memory_cache: bool | None = True,
    disk_cache_dir: str | None = DISK_CACHE_DIR,
    disk_size_limit_bytes: int | None = DISK_CACHE_LIMIT,
    memory_max_entries: int = 1000000,
):
    """Configure the cache for DSPy."""
    DSPY_CACHE = Cache(
        enable_disk_cache,
        enable_memory_cache,
        disk_cache_dir,
        disk_size_limit_bytes,
        memory_max_entries,
    )

    import dspy
    dspy.cache = DSPY_CACHE


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


# Backward-compat stubs — these were litellm-specific, now no-ops.
def enable_litellm_logging():
    """No-op. lm15 backend does not use litellm."""
    pass


def disable_litellm_logging():
    """No-op. lm15 backend does not use litellm."""
    pass


__all__ = [
    "BaseLM",
    "LM",
    "Provider",
    "TrainingJob",
    "inspect_history",
    "Embedder",
    "enable_litellm_logging",
    "disable_litellm_logging",
    "configure_cache",
]
