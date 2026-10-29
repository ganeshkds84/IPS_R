"""Structured-config schema validated by OmegaConf at composition time.

The YAML files under ``configs/`` are merged on top of :class:`AppConfig`, so
typos in keys and type mismatches fail fast. Component-specific options live
in the free-form ``params`` mapping of each :class:`ComponentConfig`.
"""

# NOTE: no ``from __future__ import annotations`` - OmegaConf inspects real annotations.

from dataclasses import dataclass, field
from typing import Any

from omegaconf import MISSING

from ips_r.core.log import get_logger

logger = get_logger(__name__)

COMPONENT_KEYS: tuple[str, ...] = (
    "dataset",
    "retriever",
    "candidate_generator",
    "decision_policy",
    "reranker",
    "generator",
    "evaluator",
)
"""Config keys of the pluggable components, in pipeline order."""


@dataclass
class ExperimentConfig:
    """Experiment identity, used for output directories and run metadata."""

    name: str = "default"
    description: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class PathsConfig:
    """Filesystem locations. Relative paths are resolved against ``root``."""

    root: str = MISSING
    data: str = MISSING
    cache: str = MISSING
    models: str = MISSING
    outputs: str = MISSING
    experiments: str = MISSING


@dataclass
class LoggingConfig:
    """Logging behaviour for a run."""

    level: str = "INFO"
    format: str = "console"
    to_file: bool = True


@dataclass
class SeedConfig:
    """Global random seed settings."""

    value: int = 42
    deterministic: bool = False


@dataclass
class RunConfig:
    """Per-invocation run settings."""

    output_dir: str | None = None
    dry_run: bool = False


@dataclass
class ComponentConfig:
    """Selection of a registered component implementation.

    Attributes:
        name: Registry name of the implementation, or ``None`` to disable the component.
        params: Keyword configuration passed to the implementation's constructor.
    """

    name: str | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class AppConfig:
    """Root configuration schema."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    run: RunConfig = field(default_factory=RunConfig)
    dataset: ComponentConfig = field(default_factory=ComponentConfig)
    retriever: ComponentConfig = field(default_factory=ComponentConfig)
    candidate_generator: ComponentConfig = field(default_factory=ComponentConfig)
    decision_policy: ComponentConfig = field(default_factory=ComponentConfig)
    reranker: ComponentConfig = field(default_factory=ComponentConfig)
    generator: ComponentConfig = field(default_factory=ComponentConfig)
    evaluator: ComponentConfig = field(default_factory=ComponentConfig)
    pipeline: ComponentConfig = field(default_factory=ComponentConfig)
