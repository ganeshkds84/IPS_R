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
    "chunker",
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
class ValidationConfig:
    """Rules applied by :class:`ips_r.data.validation.DatasetValidator`."""

    check_encoding: bool = True
    check_metadata: bool = True
    min_year: int = 1500
    max_year: int = 2100
    max_issues_per_code: int = 100


@dataclass
class StatisticsConfig:
    """Options of :class:`ips_r.data.statistics.StatisticsAccumulator`."""

    word_pattern: str = r"\w+"
    lowercase: bool = True
    top_k_terms: int = 25


@dataclass
class PreparationConfig:
    """Behaviour of :func:`ips_r.data.preparation.prepare_dataset` (``ips-r prepare``).

    Attributes:
        output_subdir: Directory inside the run directory receiving the artefacts.
        drop_invalid: Exclude records with error-severity issues from the outputs.
        fail_on_error: Fail the run (after writing reports) if validation finds errors.
        write_documents: Write ``documents.jsonl``.
        write_questions: Write ``questions.jsonl``.
        write_chunks: Write ``chunks.jsonl`` (requires a chunker).
        validation: Validation rules.
        statistics: Statistics options.
    """

    output_subdir: str = "dataset"
    drop_invalid: bool = True
    fail_on_error: bool = False
    write_documents: bool = True
    write_questions: bool = True
    write_chunks: bool = True
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    statistics: StatisticsConfig = field(default_factory=StatisticsConfig)


@dataclass
class AppConfig:
    """Root configuration schema."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: SeedConfig = field(default_factory=SeedConfig)
    run: RunConfig = field(default_factory=RunConfig)
    preparation: PreparationConfig = field(default_factory=PreparationConfig)
    dataset: ComponentConfig = field(default_factory=ComponentConfig)
    chunker: ComponentConfig = field(default_factory=ComponentConfig)
    retriever: ComponentConfig = field(default_factory=ComponentConfig)
    candidate_generator: ComponentConfig = field(default_factory=ComponentConfig)
    decision_policy: ComponentConfig = field(default_factory=ComponentConfig)
    reranker: ComponentConfig = field(default_factory=ComponentConfig)
    generator: ComponentConfig = field(default_factory=ComponentConfig)
    evaluator: ComponentConfig = field(default_factory=ComponentConfig)
    pipeline: ComponentConfig = field(default_factory=ComponentConfig)
