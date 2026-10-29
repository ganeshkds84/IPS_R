"""Tests for reproducible seed management."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest

from ips_r.core.exceptions import SeedError
from ips_r.core.seed import MAX_SEED, derive_seed, make_rng, seed_everything


def _draw() -> tuple[float, float]:
    return random.random(), float(np.random.rand())


def test_seed_everything_is_reproducible() -> None:
    seed_everything(123)
    first = _draw()
    seed_everything(123)
    assert _draw() == first


def test_seed_state_records_libraries() -> None:
    state = seed_everything(5, deterministic=True)
    assert state.seed == 5
    assert state.deterministic is True
    assert {"random", "numpy"} <= set(state.libraries)


@pytest.mark.parametrize("bad", [-1, MAX_SEED + 1, True, 1.5, "3"])
def test_invalid_seeds_are_rejected(bad: Any) -> None:
    with pytest.raises(SeedError):
        seed_everything(bad)


def test_derive_seed_is_stable_and_key_sensitive() -> None:
    assert derive_seed(42, "retriever", 0) == derive_seed(42, "retriever", 0)
    assert derive_seed(42, "retriever", 0) != derive_seed(42, "retriever", 1)
    assert derive_seed(42, "1") != derive_seed(42, 1)
    assert 0 <= derive_seed(7, "x") <= MAX_SEED


def test_make_rng_is_independent_and_reproducible() -> None:
    a, b = make_rng(1), make_rng(1)
    assert a.integers(0, 1000, 5).tolist() == b.integers(0, 1000, 5).tolist()
