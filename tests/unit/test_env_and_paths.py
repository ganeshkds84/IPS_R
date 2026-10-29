"""Tests for environment-variable handling and project path discovery."""

from __future__ import annotations

from pathlib import Path

import pytest

from ips_r.core.env import (
    get_env,
    get_env_bool,
    get_env_int,
    load_env_file,
    load_settings,
)
from ips_r.core.exceptions import ConfigurationError, EnvironmentConfigError
from ips_r.core.paths import ProjectPaths, find_config_dir, find_project_root


def test_get_env_handles_unset_and_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPS_R_EMPTY", "")
    assert get_env("IPS_R_EMPTY", "fallback") == "fallback"
    with pytest.raises(EnvironmentConfigError):
        get_env("IPS_R_MISSING", required=True)


@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("Yes", True), ("off", False)])
def test_get_env_bool(monkeypatch: pytest.MonkeyPatch, raw: str, expected: bool) -> None:
    monkeypatch.setenv("IPS_R_FLAG", raw)
    assert get_env_bool("IPS_R_FLAG") is expected


def test_malformed_values_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPS_R_FLAG", "maybe")
    monkeypatch.setenv("IPS_R_NUM", "ten")
    with pytest.raises(EnvironmentConfigError):
        get_env_bool("IPS_R_FLAG")
    with pytest.raises(EnvironmentConfigError):
        get_env_int("IPS_R_NUM")


def test_load_env_file_populates_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("IPS_R_LOG_LEVEL=debug\nIPS_R_SEED=11\n", encoding="utf-8")
    monkeypatch.setenv("IPS_R_SEED", "5")  # existing variables win by default
    # Register IPS_R_LOG_LEVEL with monkeypatch so the value loaded from the file is undone.
    monkeypatch.setenv("IPS_R_LOG_LEVEL", "placeholder")
    monkeypatch.delenv("IPS_R_LOG_LEVEL")
    assert load_env_file(env_file) == env_file
    settings = load_settings()
    assert settings.log_level == "DEBUG"
    assert settings.seed == 5


def test_missing_env_file(tmp_path: Path) -> None:
    assert load_env_file(tmp_path / "absent.env") is None
    with pytest.raises(EnvironmentConfigError):
        load_env_file(tmp_path / "absent.env", required=True)


def test_find_project_root_walks_up(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text("{}", encoding="utf-8")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    assert find_project_root(nested) == tmp_path.resolve()
    assert find_config_dir(tmp_path) == (tmp_path / "configs").resolve()


def test_env_root_takes_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPS_R_ROOT", str(tmp_path))
    assert find_project_root() == tmp_path.resolve()
    monkeypatch.setenv("IPS_R_ROOT", str(tmp_path / "nope"))
    with pytest.raises(ConfigurationError):
        find_project_root()


def test_find_config_dir_requires_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError):
        find_config_dir(tmp_path)


def test_project_paths_resolve_and_ensure(tmp_path: Path) -> None:
    paths = ProjectPaths.from_mapping(
        {
            "root": str(tmp_path),
            "data": "data",
            "cache": "cache",
            "models": str(tmp_path / "m"),
            "outputs": "out",
            "experiments": "experiments",
        }
    )
    paths.ensure()
    assert paths.data == (tmp_path / "data").resolve()
    assert paths.models.is_dir()
    assert paths.outputs.is_dir()
    assert set(paths.as_dict()) == {"root", "data", "cache", "models", "outputs", "experiments"}


def test_project_paths_missing_key(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="missing"):
        ProjectPaths.from_mapping({"root": str(tmp_path)})
