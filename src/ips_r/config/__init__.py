"""Hydra-based configuration: structured schema, composition and serialisation."""

from ips_r.config.loader import (
    as_app_config,
    load_config,
    register_configs,
    save_config,
    to_yaml,
)
from ips_r.config.schema import (
    AppConfig,
    ComponentConfig,
    ExperimentConfig,
    LoggingConfig,
    PathsConfig,
    RunConfig,
    SeedConfig,
)

__all__ = [
    "AppConfig",
    "ComponentConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "PathsConfig",
    "RunConfig",
    "SeedConfig",
    "as_app_config",
    "load_config",
    "register_configs",
    "save_config",
    "to_yaml",
]
