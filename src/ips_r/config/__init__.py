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
    PreparationConfig,
    RunConfig,
    SeedConfig,
    StatisticsConfig,
    ValidationConfig,
)

__all__ = [
    "AppConfig",
    "ComponentConfig",
    "ExperimentConfig",
    "LoggingConfig",
    "PathsConfig",
    "PreparationConfig",
    "RunConfig",
    "SeedConfig",
    "StatisticsConfig",
    "ValidationConfig",
    "as_app_config",
    "load_config",
    "register_configs",
    "save_config",
    "to_yaml",
]
