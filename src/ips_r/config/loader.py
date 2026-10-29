"""Compose, validate and serialise Hydra configurations.

The Hydra *compose API* is used rather than ``@hydra.main`` so configuration can
be loaded from the CLI, tests, notebooks and scripts alike, without Hydra
taking over the working directory or logging.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
from hydra.errors import HydraException
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

from ips_r.config.schema import AppConfig
from ips_r.core.exceptions import ConfigurationError
from ips_r.core.log import get_logger
from ips_r.core.paths import find_config_dir, find_project_root

logger = get_logger(__name__)

SCHEMA_NAME = "base_config"
DEFAULT_CONFIG_NAME = "config"
_registered = False


def _project_root_resolver() -> str:
    """OmegaConf resolver ``${ips_r.root:}`` returning the project root as a POSIX path."""
    return find_project_root().as_posix()


def register_configs() -> None:
    """Register the structured schema and custom resolvers (idempotent)."""
    global _registered  # noqa: PLW0603 - one-time process-wide registration
    if _registered:
        return
    ConfigStore.instance().store(name=SCHEMA_NAME, node=AppConfig)
    OmegaConf.register_new_resolver("ips_r.root", _project_root_resolver, replace=True)
    _registered = True
    logger.debug("config.registered", schema=SCHEMA_NAME)


def load_config(
    overrides: Sequence[str] | None = None,
    *,
    config_dir: str | Path | None = None,
    config_name: str = DEFAULT_CONFIG_NAME,
    resolve: bool = True,
) -> DictConfig:
    """Compose the configuration.

    Args:
        overrides: Hydra override strings, e.g. ``["seed.value=7", "experiment=example"]``.
        config_dir: Directory containing ``config.yaml``. Auto-detected when omitted.
        config_name: Primary config file name without extension.
        resolve: Resolve all interpolations eagerly so the config is a stable snapshot.

    Returns:
        The composed configuration, validated against :class:`AppConfig`.

    Raises:
        ConfigurationError: If composition, validation or interpolation fails.
    """
    register_configs()
    directory = Path(config_dir).resolve() if config_dir is not None else find_config_dir()
    override_list = list(overrides or [])
    try:
        with initialize_config_dir(config_dir=str(directory), version_base="1.3", job_name="ips_r"):
            cfg = compose(config_name=config_name, overrides=override_list)
        if resolve:
            OmegaConf.resolve(cfg)
    except (HydraException, OmegaConfBaseException, ValueError, OSError) as exc:
        raise ConfigurationError(
            f"Failed to compose config '{config_name}' from {directory} "
            f"with overrides {override_list}: {exc}"
        ) from exc
    logger.debug("config.loaded", config_dir=str(directory), overrides=override_list)
    return cfg


def as_app_config(cfg: DictConfig) -> AppConfig:
    """Convert a composed config into a typed :class:`AppConfig` instance.

    Raises:
        ConfigurationError: If the config does not satisfy the schema.
    """
    try:
        merged = OmegaConf.merge(OmegaConf.structured(AppConfig), cfg)
        obj = OmegaConf.to_object(merged)
    except OmegaConfBaseException as exc:
        raise ConfigurationError(f"Config does not match AppConfig schema: {exc}") from exc
    if not isinstance(obj, AppConfig):
        raise ConfigurationError(f"Expected AppConfig, got {type(obj).__name__}")
    return obj


def to_yaml(cfg: DictConfig, *, resolve: bool = True) -> str:
    """Render a config as YAML.

    Raises:
        ConfigurationError: If interpolations cannot be resolved.
    """
    try:
        return OmegaConf.to_yaml(cfg, resolve=resolve)
    except OmegaConfBaseException as exc:
        raise ConfigurationError(f"Cannot render config: {exc}") from exc


def save_config(cfg: DictConfig, path: str | Path) -> Path:
    """Write a resolved YAML snapshot of ``cfg`` to ``path``.

    Raises:
        ConfigurationError: If the file cannot be written.
    """
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(to_yaml(cfg), encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"Cannot write config to {target}: {exc}") from exc
    logger.debug("config.saved", path=str(target))
    return target
