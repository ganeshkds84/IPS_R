"""Shared pytest fixtures: environment isolation, logging reset and config helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pytest
from omegaconf import DictConfig

from ips_r.config import load_config
from ips_r.core.log import _remove_handlers, clear_context

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip ``IPS_R_*`` variables and protect process-global seeding state."""
    for key in list(os.environ):
        if key.startswith("IPS_R_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    yield
    clear_context()
    _remove_handlers()


@pytest.fixture
def config_dir() -> Path:
    """Path to the repository's Hydra config directory."""
    return CONFIG_DIR


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., DictConfig]:
    """Return a factory composing the real config with all paths redirected to ``tmp_path``."""

    def _make(overrides: Sequence[str] = ()) -> DictConfig:
        base = [f"paths.root='{tmp_path.as_posix()}'"]
        return load_config([*base, *overrides], config_dir=CONFIG_DIR)

    return _make
