"""Tests for Hydra config composition and the structured schema."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from ips_r.config import AppConfig, as_app_config, load_config, save_config, to_yaml
from ips_r.config.schema import COMPONENT_KEYS
from ips_r.core.exceptions import ConfigurationError

MakeConfig = Callable[..., DictConfig]


def test_default_config_composes(make_config: MakeConfig, tmp_path: Path) -> None:
    cfg = make_config()
    assert cfg.experiment.name == "default"
    assert cfg.seed.value == 42
    assert Path(cfg.paths.data) == tmp_path / "data"
    for key in COMPONENT_KEYS:
        assert cfg[key].name is None


def test_overrides_are_applied(make_config: MakeConfig) -> None:
    cfg = make_config(["seed.value=7", "logging.level=DEBUG"])
    assert cfg.seed.value == 7
    assert cfg.logging.level == "DEBUG"


def test_experiment_preset(make_config: MakeConfig) -> None:
    cfg = make_config(["experiment=example"])
    assert cfg.experiment.name == "example"
    assert cfg.seed.value == 1234


def test_environment_variables_feed_config(
    make_config: MakeConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("IPS_R_SEED", "99")
    monkeypatch.setenv("IPS_R_CACHE_DIR", str(tmp_path / "elsewhere"))
    cfg = make_config()
    assert cfg.seed.value == 99
    assert Path(cfg.paths.cache) == tmp_path / "elsewhere"


@pytest.mark.parametrize("override", ["not_a_key=1", "seed.typo=1"])
def test_unknown_key_is_rejected(make_config: MakeConfig, override: str) -> None:
    with pytest.raises(ConfigurationError, match="Failed to compose"):
        make_config([override])


def test_type_mismatch_is_rejected(make_config: MakeConfig) -> None:
    with pytest.raises(ConfigurationError):
        make_config(["seed.value=not-an-int"])


def test_component_params_are_free_form(make_config: MakeConfig) -> None:
    cfg = make_config(["+retriever.params.k1=1.2"])
    assert cfg.retriever.params.k1 == 1.2


def test_missing_config_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        load_config(config_dir=tmp_path)


def test_as_app_config_returns_typed_object(make_config: MakeConfig) -> None:
    app = as_app_config(make_config())
    assert isinstance(app, AppConfig)
    assert app.seed.value == 42


def test_save_and_reload_round_trip(make_config: MakeConfig, tmp_path: Path) -> None:
    cfg = make_config()
    path = save_config(cfg, tmp_path / "snap" / "config.yaml")
    reloaded = OmegaConf.load(path)
    assert reloaded == cfg
    assert "seed:" in to_yaml(cfg)
