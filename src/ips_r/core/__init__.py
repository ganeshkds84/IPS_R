"""Cross-cutting infrastructure: errors, logging, environment, paths, seeds, registries."""

from ips_r.core.exceptions import (
    ComponentError,
    ComponentNotReadyError,
    ConfigurationError,
    DataError,
    DataValidationError,
    EnvironmentConfigError,
    IPSRError,
    PDFParseError,
    PipelineError,
    RegistryError,
    SeedError,
)
from ips_r.core.log import bind_context, clear_context, configure_logging, get_logger
from ips_r.core.registry import Registry
from ips_r.core.seed import SeedState, derive_seed, make_rng, seed_everything

__all__ = [
    "ComponentError",
    "ComponentNotReadyError",
    "ConfigurationError",
    "DataError",
    "DataValidationError",
    "EnvironmentConfigError",
    "IPSRError",
    "PDFParseError",
    "PipelineError",
    "Registry",
    "RegistryError",
    "SeedError",
    "SeedState",
    "bind_context",
    "clear_context",
    "configure_logging",
    "derive_seed",
    "get_logger",
    "make_rng",
    "seed_everything",
]
