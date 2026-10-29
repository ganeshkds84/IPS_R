"""Reproducible seed management.

:func:`seed_everything` seeds Python's :mod:`random`, NumPy's legacy global RNG
and, when installed, PyTorch (CPU and CUDA). Prefer explicit generators created
with :func:`make_rng` and per-component seeds from :func:`derive_seed` over
relying on global state.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import random
from dataclasses import dataclass
from typing import Final

import numpy as np

from ips_r.core.exceptions import SeedError
from ips_r.core.log import get_logger

logger = get_logger(__name__)

MAX_SEED: Final = 2**32 - 1


@dataclass(frozen=True, slots=True)
class SeedState:
    """Record of a seeding operation, stored with run metadata.

    Attributes:
        seed: The global seed that was applied.
        deterministic: Whether deterministic algorithms were requested.
        libraries: Libraries that were successfully seeded.
    """

    seed: int
    deterministic: bool
    libraries: tuple[str, ...]


def validate_seed(seed: int) -> int:
    """Validate that ``seed`` is an integer in ``[0, 2**32 - 1]``.

    Raises:
        SeedError: If the seed has the wrong type or is out of range.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SeedError(f"Seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise SeedError(f"Seed must be in [0, {MAX_SEED}], got {seed}")
    return seed


def _seed_torch(seed: int, deterministic: bool) -> bool:
    """Seed PyTorch if it is installed. Returns whether it was seeded."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        logger.debug("seed.torch.unavailable")
        return False
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Required by cuBLAS for deterministic matmuls on CUDA >= 10.2.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as exc:
        raise SeedError(f"Failed to seed torch: {exc}") from exc
    return True


def seed_everything(seed: int, *, deterministic: bool = False) -> SeedState:
    """Seed all known sources of randomness.

    ``PYTHONHASHSEED`` is exported so that *child* processes are reproducible;
    it cannot change hashing in the already-running interpreter.

    Args:
        seed: Global seed in ``[0, 2**32 - 1]``.
        deterministic: Additionally request deterministic algorithms (PyTorch).

    Returns:
        A :class:`SeedState` describing what was seeded.

    Raises:
        SeedError: If the seed is invalid or a library fails to seed.
    """
    seed = validate_seed(seed)
    libraries: list[str] = []

    random.seed(seed)
    libraries.append("random")
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    libraries.append("numpy")

    if _seed_torch(seed, deterministic):
        libraries.append("torch")

    state = SeedState(seed=seed, deterministic=deterministic, libraries=tuple(libraries))
    logger.info(
        "seed.applied", seed=seed, deterministic=deterministic, libraries=list(state.libraries)
    )
    return state


def derive_seed(base_seed: int, *keys: str | int) -> int:
    """Derive a stable child seed from a base seed and identifying keys.

    The result depends only on the inputs (not on ``PYTHONHASHSEED``), so
    components, folds or workers get independent yet reproducible streams.

    Example:
        >>> derive_seed(42, "retriever", 0) == derive_seed(42, "retriever", 0)
        True

    Raises:
        SeedError: If ``base_seed`` is invalid.
    """
    validate_seed(base_seed)
    payload = ":".join([str(base_seed), *(f"{type(k).__name__}={k}" for k in keys)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (MAX_SEED + 1)


def make_rng(seed: int) -> np.random.Generator:
    """Create an independent NumPy :class:`~numpy.random.Generator`.

    Raises:
        SeedError: If the seed is invalid.
    """
    return np.random.default_rng(validate_seed(seed))
